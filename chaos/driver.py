"""In-process crash driver.

One scenario = one fault stream, one crash schedule and one planner
temperature, replayed against any config. Each incarnation reopens the
journal and gets a fresh planner with a new seed (a re-sampled LLM is a
different draw); a crash is a `SimulatedCrash` raised from a crash point.

The agent and the platform share one VirtualClock, so deadlines and delayed
requests play out in simulated time and a run stays deterministic.

A scenario can also freeze its first incarnation instead of crashing it: the
incarnation blocks on a thread at a hook point, the driver presumes it dead
and runs replacements to the end, then wakes it up. That is the "zombie"
case: a hung worker that comes back after its run was taken over.
"""

from __future__ import annotations

import tempfile
import threading
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
from fastapi.testclient import TestClient  # noqa: E402

from agents.campaign import CampaignGoal, ScriptedPlanner, campaign_tools
from chaos.invariants import Check, check
from harness.clock import RealClock, VirtualClock
from harness.journal import Journal
from harness.runtime import DEADLINE_MARGIN, Config, Outcome, Runtime, SimulatedCrash, approve, resolve
from harness.tools import ToolRegistry
from platform_sim.app import create_app
from platform_sim.faults import FaultConfig

MAX_INCARNATIONS = 12


@dataclass(frozen=True)
class Scenario:
    seed: int
    faults: FaultConfig = FaultConfig()
    # Crashes in order: each (point, nth) fires at the nth pass of `point` after
    # the previous crash. It stays armed across restarts that aren't crashes
    # (escalations, suspensions), so pausing for an operator never skips one.
    crashes: tuple[tuple[str, int], ...] = ()
    temperature: float = 0.0
    # (point, nth): freeze the first incarnation there and wake it once the run is over.
    freeze: tuple[str, int] | None = None


class CrashAt:
    """Crash hook: raise at the nth pass of `point` (optionally only at one step)."""

    def __init__(self, point: str, nth: int = 1, step: int | None = None) -> None:
        self.point, self.nth, self.step = point, nth, step
        self.seen = 0
        self.fired_at: int | None = None

    def __call__(self, point: str, step: int) -> None:
        if point != self.point or (self.step is not None and step != self.step):
            return
        self.seen += 1
        if self.seen == self.nth:
            self.fired_at = step
            raise SimulatedCrash(f"{point} at step {step}")


class FreezeAt:
    """Hook that blocks its thread at the nth pass of `point` until released."""

    def __init__(self, point: str, nth: int = 1) -> None:
        self.point, self.nth = point, nth
        self.seen = 0
        self.frozen = False
        self.paused = threading.Event()  # set once frozen, or once the run returns without freezing
        self.release = threading.Event()

    def __call__(self, point: str, step: int) -> None:
        if point != self.point:
            return
        self.seen += 1
        if self.seen == self.nth:
            self.frozen = True
            self.paused.set()
            self.release.wait()


class Zombie:
    """An incarnation on its own thread that may freeze, be presumed dead, and come back."""

    def __init__(self, freeze: tuple[str, int], incarnation: Callable[[Any], Outcome]) -> None:
        self.hook = FreezeAt(*freeze)
        self.outcome: Outcome | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._body, args=(incarnation,), daemon=True)

    def _body(self, incarnation) -> None:
        try:
            self.outcome = incarnation(self.hook)
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed silently
            self.error = exc
        finally:
            self.hook.paused.set()

    def start(self) -> Outcome | None:
        """Run until it freezes (None) or returns (its outcome)."""
        self.thread.start()
        self.hook.paused.wait()
        if self.hook.frozen:
            return None
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.outcome

    def resume(self) -> str:
        """Wake it up and let it run to wherever it stops; returns how it ended."""
        self.hook.release.set()
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            return "hung"
        if self.error is not None:
            return f"error: {type(self.error).__name__}"
        return self.outcome.status


def audit_operator(client, run_id: str, escalation: dict, clock=None) -> tuple[bool, Any]:
    """Resolve an escalation the way a human would: check the platform's dashboard.

    A person looks only after the call's deadline has passed and any request
    still in flight has landed (drained), then reads the platform's audit log
    for the escalated call's trace id. It models a person checking by hand;
    escalations are a real cost.
    """
    clock = clock or RealClock()
    if escalation.get("deadline") is not None:
        clock.sleep(escalation["deadline"] + DEADLINE_MARGIN - clock.now())
    client.post("/_admin/drain")
    effects = client.get("/_admin/effects", params={"run_id": run_id}).json()["effects"]
    hits = [e for e in effects if e["trace_id"] == escalation["trace_id"]]
    return (True, hits[-1]["payload"]) if hits else (False, None)


@dataclass
class RunReport:
    config: str
    seed: int
    temperature: float
    status: str = "incarnation_limit"
    incarnations: int = 0
    crashes: int = 0
    escalations: int = 0
    approvals: int = 0
    planner_calls: int = 0
    conflicts: int = 0  # 409s returned to the agent (positional keys should never see one)
    froze: bool = False  # a frozen first incarnation was presumed dead, then woken
    zombie: str | None = None  # how the woken incarnation ended
    escalation_causes: Counter = field(default_factory=Counter)  # (cause, verdict) -> count
    check: Check | None = None
    events: list = field(default_factory=list, repr=False)

    @property
    def finished(self) -> bool:
        return self.status == "finished"


def planner_seed(seed: int, incarnation: int) -> int:
    return seed * 1_000 + incarnation


def escalation_causes(events: list, crashes: list[tuple[str, int]]) -> Counter:
    """(cause, verdict) for every escalation; the verdict is the operator's resolution."""
    crashes = list(crashes)
    out: Counter = Counter()
    for i, e in enumerate(events):
        if e.kind != "status" or e.payload.get("status") != "escalated":
            continue
        step, reason = e.payload["step"], e.payload["reason"]
        if reason == "ambiguous response":
            last = [x for x in events[:i] if x.kind == "attempt" and x.step == step][-1]
            cause = f"ambiguous reply ({last.payload['status']})"
        elif reason.startswith("outcome unknown"):
            # Only these crash points leave an intent without a result behind.
            crash = next((c for c in crashes if c[1] == step and c[0] in ("after_intent", "after_send")),
                         None)
            if crash:
                crashes.remove(crash)
            cause = f"restart after a crash {crash[0]}" if crash else "restart"
        else:
            cause = reason.split(":")[0]
        nxt = next((x for x in events[i + 1:] if x.step == step and x.kind in ("result", "intent_voided")),
                   None)
        verdict = "unresolved" if nxt is None else "applied" if nxt.kind == "result" else "not applied"
        out[(cause, verdict)] += 1
    return out


def run_scenario(
    scenario: Scenario,
    config: Config,
    *,
    tools: ToolRegistry | None = None,
    goal: CampaignGoal = CampaignGoal(),
    journal_dir: str | Path | None = None,
    durability: str = "full",
    hook_factory: Callable[[int], Callable | None] | None = None,
    operator: Callable = audit_operator,
    keep_events: bool = False,
    planner_factory: Callable[[int, float], Any] = ScriptedPlanner,
) -> RunReport:
    """Run one scenario under one config to completion (or the incarnation limit)."""
    tools = tools or campaign_tools()
    run_id = f"run-{scenario.seed}"
    report = RunReport(config.name, scenario.seed, scenario.temperature)
    schedule = (CrashAt(point, nth) for point, nth in scenario.crashes)
    armed = next(schedule, None)
    fired: list[tuple[str, int]] = []
    clock = VirtualClock()

    with tempfile.TemporaryDirectory(dir=journal_dir) as tmp, \
            TestClient(create_app(faults=scenario.faults, clock=clock)) as client:
        path = Path(tmp) / "journal.db"

        def incarnation(hook, planner) -> Outcome:
            with Journal(path, durability) as journal:
                runtime = Runtime(journal, tools, client, config, crash_hook=hook, clock=clock)
                return runtime.run(run_id, goal, planner)

        zombie = zombie_planner = None
        for n in range(MAX_INCARNATIONS):
            planner = planner_factory(planner_seed(scenario.seed, n), scenario.temperature)
            report.incarnations += 1
            if n == 0 and scenario.freeze:
                zombie, zombie_planner = Zombie(scenario.freeze, lambda h, p=planner: incarnation(h, p)), planner
                outcome = zombie.start()
                if outcome is None:  # frozen: presumed dead, so a replacement takes over
                    report.froze = True
                    continue
            else:
                hook = hook_factory(n) if hook_factory else armed
                try:
                    outcome = incarnation(hook, planner)
                except SimulatedCrash:
                    report.crashes += 1
                    if isinstance(hook, CrashAt) and hook.fired_at is not None:
                        fired.append((hook.point, hook.fired_at))
                    armed = next(schedule, None)
                    continue
                finally:
                    report.planner_calls += planner.calls
            if outcome.status == "escalated":
                report.escalations += 1
                applied, value = operator(client, run_id, outcome.detail, clock=clock)
                with Journal(path, durability) as journal:
                    resolve(journal, run_id, applied, value)
                continue
            if outcome.status == "awaiting_approval":
                report.approvals += 1
                with Journal(path, durability) as journal:
                    approve(journal, run_id, outcome.detail["step"], outcome.detail["args_hash"])
                continue
            if outcome.status == "suspended":
                continue
            report.status = outcome.status
            break

        if zombie is not None:
            if report.froze:
                report.zombie = zombie.resume()
            report.planner_calls += zombie_planner.calls
        client.post("/_admin/drain")  # whatever is still in flight lands (or is dropped) now

        with Journal(path, durability) as journal:
            events = journal.events(run_id)
        if keep_events:
            report.events = events
        report.escalation_causes = escalation_causes(events, fired)
        approved = None
        if config.approvals:
            approved = {f"{run_id}/{e.step}" for e in events if e.kind == "approval"}
        effects = client.get("/_admin/effects", params={"run_id": run_id}).json()["effects"]
        requests = client.get("/_admin/requests", params={"run_id": run_id}).json()["requests"]
        report.conflicts = sum(1 for r in requests if r["status"] == 409)
        report.check = check(effects, goal.expected_effects, goal.expected_spend, approved)
    return report
