"""Crash at every point x every step: the harness must be correct every time."""

import pytest

from chaos.driver import CrashAt, Scenario, run_scenario
from harness.runtime import CONFIGS, CRASH_POINTS

STEPS = range(9)  # 8 tool calls + finish


@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("point", CRASH_POINTS)
def test_single_crash(point, step, temperature):
    report = run_scenario(
        Scenario(seed=step, temperature=temperature),
        CONFIGS["harness"],
        hook_factory=lambda n: CrashAt(point, step=step) if n == 0 else None,
    )
    assert report.finished
    assert report.check.clean, report.check
    assert report.conflicts == 0
    # A decided step is never re-asked: same planner calls as a clean run.
    assert report.planner_calls == 9


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("point", CRASH_POINTS)
def test_crash_again_during_recovery(point, step):
    # The resumed incarnation crashes at the same place before it can finish recovering.
    report = run_scenario(
        Scenario(seed=100 + step, temperature=0.7),
        CONFIGS["harness"],
        hook_factory=lambda n: CrashAt(point, step=step) if n < 3 else None,
    )
    assert report.finished
    assert report.check.clean, report.check
    assert report.planner_calls == 9
