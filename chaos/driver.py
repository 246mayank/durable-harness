"""In-process crash driver.

One scenario = one fault stream, one crash schedule and one planner
temperature, replayed against any config. Each incarnation reopens the
journal and gets a fresh planner with a new seed (a re-sampled LLM is a
different draw); a crash is a `SimulatedCrash` raised from a crash point.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agents.campaign import CampaignGoal, ScriptedPlanner, campaign_tools
from chaos.invariants import Check, check
from harness.journal import Journal
from harness.runtime import Config, Runtime, SimulatedCrash, approve, resolve
from harness.tools import ToolRegistry
from platform_sim.app import create_app
from platform_sim.faults import FaultConfig

MAX_INCARNATIONS = 12


@dataclass(frozen=True)
class Scenario:
    seed: int
    faults: FaultConfig = FaultConfig()
    # One entry per incarnation: crash at the nth time that incarnation passes `point`.
    crashes: tuple[tuple[str, int], ...] = ()
    temperature: float = 0.0


class CrashAt:
    """Crash hook: raise at the nth pass of `point` (optionally only at one step)."""

    def __init__(self, point: str, nth: int = 1, step: int | None = None) -> None:
        self.point, self.nth, self.step = point, nth, step
        self.seen = 0

    def __call__(self, point: str, step: int) -> None:
        if point != self.point or (self.step is not None and step != self.step):
            return
        self.seen += 1
        if self.seen == self.nth:
            raise SimulatedCrash(f"{point} at step {step}")


def audit_operator(client, run_id: str, escalation: dict) -> tuple[bool, Any]:
    """Resolve an escalation the way a human would: check the platform's dashboard.

    This reads the platform's audit log for the escalated call's trace id. It
    models a person looking the call up by hand; escalations are a real cost.
    """
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
    check: Check | None = None
    events: list = field(default_factory=list, repr=False)

    @property
    def finished(self) -> bool:
        return self.status == "finished"


def planner_seed(seed: int, incarnation: int) -> int:
    return seed * 1_000 + incarnation


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
) -> RunReport:
    """Run one scenario under one config to completion (or the incarnation limit)."""
    tools = tools or campaign_tools()
    run_id = f"run-{scenario.seed}"
    report = RunReport(config.name, scenario.seed, scenario.temperature)
    if hook_factory is None:
        def hook_factory(n: int):
            return CrashAt(*scenario.crashes[n]) if n < len(scenario.crashes) else None

    with tempfile.TemporaryDirectory(dir=journal_dir) as tmp, \
            TestClient(create_app(faults=scenario.faults)) as client:
        path = Path(tmp) / "journal.db"
        for n in range(MAX_INCARNATIONS):
            planner = ScriptedPlanner(planner_seed(scenario.seed, n), scenario.temperature)
            with Journal(path, durability) as journal:
                runtime = Runtime(journal, tools, client, config, crash_hook=hook_factory(n))
                report.incarnations += 1
                try:
                    outcome = runtime.run(run_id, goal, planner)
                except SimulatedCrash:
                    report.crashes += 1
                    continue
                finally:
                    report.planner_calls += planner.calls
                if outcome.status == "escalated":
                    report.escalations += 1
                    applied, value = operator(client, run_id, outcome.detail)
                    resolve(journal, run_id, applied, value)
                    continue
                if outcome.status == "awaiting_approval":
                    report.approvals += 1
                    approve(journal, run_id, outcome.detail["step"], outcome.detail["args_hash"])
                    continue
                if outcome.status == "suspended":
                    continue
                report.status = outcome.status
                break

        with Journal(path, durability) as journal:
            events = journal.events(run_id)
        if keep_events:
            report.events = events
        approved = None
        if config.approvals:
            approved = {f"{run_id}/{e.step}" for e in events if e.kind == "approval"}
        effects = client.get("/_admin/effects", params={"run_id": run_id}).json()["effects"]
        requests = client.get("/_admin/requests", params={"run_id": run_id}).json()["requests"]
        report.conflicts = sum(1 for r in requests if r["status"] == 409)
        report.check = check(effects, goal.expected_effects, goal.expected_spend, approved)
    return report
