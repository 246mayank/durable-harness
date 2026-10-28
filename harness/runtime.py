"""The durable agent loop.

A run is a fold over its journal. Each step is written in the order
decision -> intent -> attempt* -> result, and a restart resumes from the
first step without a result:

    crash point      journal holds              on restart
    after_decision   decision                   execute the journaled decision
    after_intent     decision, intent           outcome unknown: recover by effect class
    after_send       decision, intent           same (the platform may have applied it)
    after_result     decision, ..., result      step done: ask the planner for the next

A decided step is never re-asked. The planner is consulted only when no step
is in progress, so a resumed run makes exactly the calls it had decided on,
with the same arguments and the same positional keys.

Two more rules cover calls that may still be in flight:

- Deadlines. Every send carries a deadline, journaled before the send, after
  which the platform won't process it. A lookup that finds nothing proves
  nothing while a send might still land, so recovery waits until the step's
  latest deadline has passed before it trusts a negative lookup.
- Fencing. Each incarnation takes the run over with a new epoch, and every
  journal write and every send checks it. An incarnation that was presumed
  dead and wakes up can't journal or send again; its only possible stray send
  is one it had already passed the check for, and that send carries a
  journaled deadline that the new owner waits out before it acts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from harness.clock import RealClock
from harness.errors import Ambiguous, Rejected, ToolError, Unresolvable
from harness.journal import Fenced, Journal, RunState, StepState
from harness.keys import KeyMode, args_hash, make_key
from harness.tools import CallContext, Effect, Tool, ToolRegistry

CRASH_POINTS = ("after_decision", "after_intent", "after_send", "after_result")
# Hooks also fire at "before_send", after the fence check and just before the request goes out.
# It isn't a crash point (a crash there looks like after_intent), but it is the one gap a
# presumed-dead incarnation can be frozen in and still send from when it wakes.
MAX_STEPS = 40
DEADLINE = 0.5  # seconds a send has to be processed; waited out before a negative lookup
DEADLINE_MARGIN = 0.05  # slack for clock reads on either side

CrashHook = Callable[[str, int], None]


class SimulatedCrash(BaseException):
    """Raised by a crash hook. A BaseException, so no `except Exception` can swallow it."""


class Planner(Protocol):
    calls: int

    def decide(self, goal: Any, history: list[dict]) -> dict:
        """Return {"tool": name, "args": {...}} or {"finish": True}."""


@dataclass(frozen=True)
class Config:
    name: str
    replay: bool
    keys: KeyMode
    retries: int
    effect_aware: bool
    approvals: bool = False
    budget_cap: int | None = None
    deadline: float | None = None  # seconds; None sends no deadline and never waits
    fencing: bool = False

    def __post_init__(self) -> None:
        if self.effect_aware and self.keys is KeyMode.NONE:
            raise ValueError("effect-aware recovery re-sends idempotent calls, so it needs keys")
        if self.fencing and not self.replay:
            raise ValueError("fencing guards a shared journal; without replay there is none")


CONFIGS = {
    c.name: c
    for c in (
        Config("naive", replay=False, keys=KeyMode.NONE, retries=0, effect_aware=False),
        Config("naive_retry", replay=False, keys=KeyMode.NONE, retries=5, effect_aware=False),
        Config("keys_only", replay=False, keys=KeyMode.ARGS, retries=5, effect_aware=False),
        Config("harness", replay=True, keys=KeyMode.POSITIONAL, retries=5, effect_aware=True,
               deadline=DEADLINE, fencing=True),
    )
}


@dataclass(frozen=True)
class Outcome:
    """Why run() returned: finished, escalated, suspended, awaiting_approval, gave_up or fenced."""

    status: str
    detail: dict = field(default_factory=dict)


_ABSENT = object()  # a lookup found no trace of the effect


class Runtime:
    def __init__(
        self,
        journal: Journal,
        tools: ToolRegistry,
        client: httpx.Client,
        config: Config,
        *,
        crash_hook: CrashHook | None = None,
        backoff: float = 0.0,
        clock=None,
    ) -> None:
        # Without replay the log lives in memory only, so a restart loses it and
        # the agent starts again from the goal. That is the baseline under test.
        self.journal = journal if config.replay else Journal(":memory:")
        self.tools = tools
        self.client = client
        self.config = config
        self.crash_hook = crash_hook
        self.backoff = backoff
        self.clock = clock or RealClock()
        self.epoch: int | None = None

    # ---- the loop -----------------------------------------------------------

    def run(self, run_id: str, goal: Any, planner: Planner) -> Outcome:
        try:
            return self._run(run_id, goal, planner)
        except Fenced as exc:
            # A newer incarnation owns the run. Journal nothing more; just stop.
            return Outcome("fenced", {"epoch": exc.epoch, "current": exc.current})

    def _run(self, run_id: str, goal: Any, planner: Planner) -> Outcome:
        if self.config.fencing:
            self.epoch = self.journal.acquire(run_id)  # before reading, so no older write is missed
        state = self.journal.load_state(run_id)
        self._emit(state, "incarnation", None, {"n": state.incarnations + 1, "config": self.config.name,
                                                "epoch": self.epoch})
        if state.status["status"] in ("finished", "escalated", "awaiting_approval"):
            return Outcome(state.status["status"], state.status)

        while True:
            step = state.current
            if step is None:
                if len(state.steps) >= MAX_STEPS:
                    return self._stop(state, "gave_up", {"reason": "step limit"})
                self._check_fence(state)  # don't pay for a planner call we couldn't journal
                decision = planner.decide(goal, state.history())
                step = self._emit(state, "decision", len(state.steps), decision)
                self._hook("after_decision", step.index)
            if step.decision.get("finish"):
                self._emit(state, "result", step.index, {"status": "ok", "value": None, "via": "finish"})
                return self._stop(state, "finished", {"steps": len(state.steps)})
            stop = self._execute(state, step)
            if stop is not None:
                return stop

    def _execute(self, state: RunState, step: StepState) -> Outcome | None:
        tool = self.tools.get(step.decision.get("tool", ""))
        if tool is None:
            return self._fail(state, step, Rejected(f"unknown tool {step.decision.get('tool')!r}"))
        try:
            args = tool.args.model_validate(step.decision.get("args", {}))
        except ValidationError as exc:
            return self._fail(state, step, Rejected(f"invalid arguments: {exc.errors()}"))

        if step.intent is not None:
            return self._recover(state, step, tool, args)

        arg_dict = args.model_dump()
        if self.config.approvals and tool.effect is Effect.UNSAFE:
            digest = args_hash(tool.name, arg_dict)
            if step.approved_hash != digest:
                return self._stop(
                    state, "awaiting_approval",
                    {"step": step.index, "tool": tool.name, "args": arg_dict, "args_hash": digest},
                )
        if self.config.budget_cap is not None and tool.spend is not None:
            reserved = self._reserved_spend(state) + tool.spend(args)
            if reserved > self.config.budget_cap:
                return self._fail(
                    state, step,
                    Rejected(f"budget cap: {reserved} > {self.config.budget_cap} (committed + pending)"),
                )

        ctx = CallContext(state.run_id, step.index, make_key(
            self.config.keys, state.run_id, step.index, tool.name, arg_dict))
        if tool.prepare is not None:
            try:
                precondition = self._read(lambda: tool.prepare(self.client, args, ctx))
            except ToolError as err:
                return self._fail(state, step, err)
            ctx = replace(ctx, precondition=precondition)

        ctx = replace(ctx, deadline=self._new_deadline())
        self._emit(state, "intent", step.index, {
            "tool": tool.name, "args": arg_dict, "key": ctx.key,
            "effect": tool.effect.value, "precondition": ctx.precondition, "deadline": ctx.deadline,
        })
        self._hook("after_intent", step.index)
        return self._send(state, step, tool, args, ctx)

    # ---- calling and recovering -----------------------------------------------

    def _send(self, state: RunState, step: StepState, tool: Tool, args: Any,
              ctx: CallContext, in_doubt: bool = False) -> Outcome | None:
        """Send with retries. `in_doubt`: an earlier attempt may already have applied."""
        aware = self.config.effect_aware
        ambiguous = in_doubt
        last: ToolError | None = None
        for n in range(self.config.retries + 1):
            if n and self.backoff:
                self.clock.sleep(self.backoff * 2 ** (n - 1))
            ctx = self._before_send(state, step, ctx)
            self._hook("before_send", step.index)
            try:
                value = tool.execute(self.client, args, ctx)
            except ToolError as err:
                self._hook("after_send", step.index)
                self._emit(state, "attempt", step.index,
                           {"n": n, "outcome": err.kind, "status": err.status, "error": str(err)})
                last = err
                if isinstance(err, Rejected):
                    return self._fail(state, step, err)
                if isinstance(err, Ambiguous):
                    ambiguous = True
                    if aware and tool.effect is Effect.UNSAFE:
                        return self._escalate(state, step, ctx, "ambiguous response")
                    if aware and tool.effect is Effect.RECONCILABLE:
                        found = self._reconcile(state, step, tool, args, ctx)
                        if found is not _ABSENT:
                            return found
                continue
            self._hook("after_send", step.index)
            self._emit(state, "attempt", step.index, {"n": n, "outcome": "ok"})
            return self._succeed(state, step, value, via="call")

        if aware and ambiguous and tool.effect is not Effect.READ:
            # Never hand an unknown outcome to the planner: it would re-plan the
            # call as a new step, under a new key. Stop; the next incarnation
            # recovers this step from its intent.
            return self._stop(state, "suspended", {"step": step.index, "reason": "retries exhausted"})
        return self._fail(state, step, last)

    def _recover(self, state: RunState, step: StepState, tool: Tool, args: Any) -> Outcome | None:
        """Resume a step whose intent was journaled but whose result was not."""
        intent = step.intent
        ctx = CallContext(state.run_id, step.index, intent["key"], intent.get("precondition"),
                          step.deadline)
        if not self.config.effect_aware or tool.effect in (Effect.READ, Effect.IDEMPOTENT):
            # Even if every re-send is now refused outright, the original may have
            # applied: the step stays in doubt and must not go back to the planner.
            return self._send(state, step, tool, args, ctx, in_doubt=True)
        if tool.effect is Effect.RECONCILABLE:
            found = self._reconcile(state, step, tool, args, ctx)
            return self._send(state, step, tool, args, ctx) if found is _ABSENT else found
        return self._escalate(state, step, ctx, "outcome unknown after restart")

    def _reconcile(self, state: RunState, step: StepState, tool: Tool, args: Any,
                   ctx: CallContext) -> Any:
        """Look the effect up. Returns _ABSENT, None (recorded as done) or an Outcome.

        Finding nothing proves nothing while a send may still be in flight, so
        first wait until no send of this step can be processed any more.
        """
        self._await_deadline(ctx.deadline)
        try:
            read_ctx = replace(ctx, deadline=None)  # a read isn't bound by the write's deadline
            found = self._read(lambda: tool.lookup(self.client, args, read_ctx))
        except ToolError:
            return self._stop(state, "suspended", {"step": step.index, "reason": "lookup failed"})
        except Unresolvable as exc:
            return self._escalate(state, step, ctx, f"lookup inconclusive: {exc}")
        if found is None:
            return _ABSENT
        return self._succeed(state, step, found, via="lookup")

    def _read(self, fn: Callable[[], Any]) -> Any:
        """Run a side-effect-free call, retrying anything but a rejection."""
        for n in range(self.config.retries + 1):
            try:
                return fn()
            except Rejected:
                raise
            except ToolError:
                if n == self.config.retries:
                    raise
                if self.backoff:
                    self.clock.sleep(self.backoff * 2 ** n)

    def _reserved_spend(self, state: RunState) -> int:
        """Committed spend plus spend whose outcome is still unknown."""
        total = 0
        for s in state.steps:
            tool = self.tools.get(s.decision.get("tool", ""))
            if tool is None or tool.spend is None or s.intent is None:
                continue
            if s.result is None or s.result["status"] == "ok":
                total += tool.spend(tool.args.model_validate(s.intent["args"]))
        return total

    # ---- deadlines and fencing ----------------------------------------------------

    def _new_deadline(self) -> float | None:
        return self.clock.now() + self.config.deadline if self.config.deadline else None

    def _await_deadline(self, deadline: float | None) -> None:
        if deadline is not None:
            self.clock.sleep(deadline + DEADLINE_MARGIN - self.clock.now())

    def _check_fence(self, state: RunState) -> None:
        if self.epoch is not None:
            self.journal.check_epoch(state.run_id, self.epoch)

    def _before_send(self, state: RunState, step: StepState, ctx: CallContext) -> CallContext:
        """Fence check, plus a fresh deadline (journaled first) once the current one has passed."""
        self._check_fence(state)
        if self.config.deadline and (ctx.deadline is None or self.clock.now() >= ctx.deadline):
            ctx = replace(ctx, deadline=self._new_deadline())
            self._emit(state, "deadline", step.index, {"deadline": ctx.deadline})
        return ctx

    # ---- journaling helpers -----------------------------------------------------

    def _emit(self, state: RunState, kind: str, step: int | None, payload: dict) -> StepState | None:
        state.apply(self.journal.append(state.run_id, kind, step, payload, epoch=self.epoch))
        return state.steps[step] if step is not None else None

    def _hook(self, point: str, step: int) -> None:
        if self.crash_hook is not None:
            self.crash_hook(point, step)

    def _succeed(self, state: RunState, step: StepState, value: Any, via: str) -> None:
        self._emit(state, "result", step.index, {"status": "ok", "value": value, "via": via})
        self._hook("after_result", step.index)
        return None

    def _fail(self, state: RunState, step: StepState, err: ToolError | None) -> None:
        err = err or ToolError("failed")
        self._emit(state, "result", step.index,
                   {"status": "error", "error_class": err.kind, "error": str(err), "body": err.body})
        self._hook("after_result", step.index)
        return None

    def _escalate(self, state: RunState, step: StepState, ctx: CallContext, reason: str) -> Outcome:
        return self._stop(state, "escalated", {
            "step": step.index, "tool": step.decision["tool"], "args": step.decision["args"],
            "trace_id": ctx.trace_id, "reason": reason, "deadline": ctx.deadline,
        })

    def _stop(self, state: RunState, status: str, detail: dict) -> Outcome:
        self._emit(state, "status", None, {"status": status, **detail})
        return Outcome(status, detail)


# ---- operator actions ---------------------------------------------------------


def resolve(journal: Journal, run_id: str, applied: bool, value: Any = None) -> None:
    """An operator's answer to an escalation.

    applied=True records `value` as the step's result; applied=False voids the
    intent, so the next run re-executes the step.
    """
    state = journal.load_state(run_id)
    if state.status["status"] != "escalated":
        raise ValueError(f"run {run_id} is not escalated (status: {state.status['status']})")
    step = state.status["step"]
    if applied:
        journal.append(run_id, "result", step, {"status": "ok", "value": value, "via": "operator"})
    else:
        journal.append(run_id, "intent_voided", step, {"reason": "operator: not applied"})
    journal.append(run_id, "status", None, {"status": "running", "resolved_step": step})


def approve(journal: Journal, run_id: str, step: int, digest: str) -> None:
    """Approve one unsafe call. The approval binds to its arguments' hash."""
    journal.append(run_id, "approval", step, {"args_hash": digest})
    journal.append(run_id, "status", None, {"status": "running", "approved_step": step})
