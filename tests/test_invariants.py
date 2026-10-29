"""The invariant checker counts from the platform's effects, never the agent's view."""

from chaos.invariants import check

EXPECTED = {"campaign": 1, "ad": 3, "email_batch": 1, "allocation": 2}


def effect(kind: str, step: int, amount: int | None = None) -> dict:
    return {"kind": kind, "trace_id": f"run-1/{step}", "amount": amount}


CLEAN_RUN = [effect("campaign", 0), effect("ad", 1), effect("ad", 2), effect("ad", 3), effect("email_batch", 4),
             effect("allocation", 5, 500), effect("allocation", 7, 200)]


def test_a_clean_run_is_clean():
    assert check(CLEAN_RUN, EXPECTED, 700).clean


def test_a_step_applied_twice_is_told_apart_from_a_new_decision():
    resent = check([*CLEAN_RUN, effect("ad", 2)], EXPECTED, 700)  # step 2 landed again
    assert resent.duplicates["ad"] == 1 and resent.repeated_steps == 1 and not resent.clean
    decided = check([*CLEAN_RUN, effect("ad", 8)], EXPECTED, 700)  # a fourth ad, decided as step 8
    assert decided.duplicates["ad"] == 1 and decided.repeated_steps == 0
