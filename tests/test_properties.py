"""Property-based crash schedules: the harness's safety invariants hold for any of them.

Schedules mix network faults (delayed delivery included), crashes, and a first
incarnation that may hang and come back as a zombie after the run is over.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agents.campaign import campaign_tools
from chaos.driver import Scenario, run_scenario
from harness.runtime import CONFIGS, CRASH_POINTS
from platform_sim.faults import FaultConfig

probability = st.floats(0, 0.2, allow_nan=False)
scenarios = st.builds(
    Scenario,
    seed=st.integers(0, 10_000),
    faults=st.builds(FaultConfig, p_429=probability, p_5xx=probability, p_lost_ack=probability,
                     p_delayed=probability, seed=st.integers(0, 2**31 - 1)),
    crashes=st.lists(st.tuples(st.sampled_from(CRASH_POINTS), st.integers(1, 10)), max_size=4).map(tuple),
    temperature=st.sampled_from([0.0, 0.3, 0.7, 1.0]),
    freeze=st.none() | st.tuples(st.sampled_from((*CRASH_POINTS, "before_send")), st.integers(1, 10)),
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=scenarios, conditional_writes=st.booleans())
def test_harness_never_duplicates_or_overspends(scenario, conditional_writes):
    report = run_scenario(scenario, CONFIGS["harness"], tools=campaign_tools(conditional_writes),
                          keep_events=True)
    assert report.check.duplicate_total == 0, report.check
    assert report.check.excess_spend == 0
    assert report.conflicts == 0
    if report.finished:
        assert report.check.missing_total == 0
    # The planner is asked exactly once per journaled decision: never re-asked.
    # (A fenced zombie checks its fence before asking, so it never adds an unjournaled call.)
    assert report.planner_calls == sum(1 for e in report.events if e.kind == "decision")
    if report.froze:
        assert report.zombie == "fenced"
