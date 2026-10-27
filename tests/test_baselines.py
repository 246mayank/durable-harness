"""The baselines fail the way the design says they should."""

from chaos.driver import CrashAt, Scenario, run_scenario
from harness.runtime import CONFIGS
from platform_sim.faults import FaultConfig

ALLOCATE_STEP = 5


def crash_once(point: str, step: int):
    return lambda n: CrashAt(point, step=step) if n == 0 else None


def total_duplicates(config: str, scenarios, **kw) -> int:
    return sum(run_scenario(s, CONFIGS[config], **kw).check.duplicate_total for s in scenarios)


def test_lost_acks_duplicate_under_naive_retry_but_not_harness():
    scenarios = [Scenario(seed=s, faults=FaultConfig(p_lost_ack=0.3, seed=s)) for s in range(10)]
    assert total_duplicates("naive_retry", scenarios) > 0
    assert total_duplicates("harness", scenarios) == 0


def test_restart_without_replay_redoes_everything():
    report = run_scenario(Scenario(seed=1), CONFIGS["naive"],
                          hook_factory=crash_once("after_result", ALLOCATE_STEP))
    assert report.finished
    assert report.check.duplicates == {"campaign": 1, "ad": 3, "email_batch": 1, "allocation": 1}


def test_keys_only_at_temperature_zero_duplicates_only_unkeyable_effects():
    # The re-planned run produces the same arguments, so keyed calls replay;
    # the email batch and the allocation have no key support and go out twice.
    report = run_scenario(Scenario(seed=1), CONFIGS["keys_only"],
                          hook_factory=crash_once("after_result", ALLOCATE_STEP))
    assert report.finished
    assert report.check.duplicates == {"campaign": 0, "ad": 0, "email_batch": 1, "allocation": 1}


def test_keys_only_duplicates_ads_once_the_planner_resamples():
    # At temperature > 0 the restarted planner writes different ad copy:
    # new arguments, new args-hash keys, new ads.
    scenarios = [Scenario(seed=s, temperature=0.7) for s in range(10)]
    hook = crash_once("after_result", 3)
    ad_dups = sum(
        run_scenario(s, CONFIGS["keys_only"], hook_factory=hook).check.duplicates["ad"] for s in scenarios
    )
    assert ad_dups > 0
    assert total_duplicates("harness", scenarios, hook_factory=hook) == 0


def test_harness_never_reuses_a_key_with_different_arguments():
    for seed in range(10):
        scenario = Scenario(seed=seed, temperature=0.7,
                            faults=FaultConfig(p_lost_ack=0.2, p_5xx=0.1, p_429=0.1, seed=seed),
                            crashes=(("after_send", 3), ("after_intent", 2)))
        assert run_scenario(scenario, CONFIGS["harness"]).conflicts == 0
