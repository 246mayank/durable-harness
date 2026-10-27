from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from agents.campaign import CampaignGoal, ScriptedPlanner, campaign_tools
from chaos.driver import CrashAt, Scenario, run_scenario
from harness.journal import Journal
from harness.runtime import CONFIGS, Runtime, approve
from platform_sim.app import create_app
from platform_sim.faults import FaultConfig

HARNESS = CONFIGS["harness"]
ALLOCATE_STEP = 5  # campaign, ad, ad, ad, email, allocate(500), metrics, boost, finish


def crash_once(point: str, step: int):
    return lambda n: CrashAt(point, step=step) if n == 0 else None


@pytest.mark.parametrize("config", CONFIGS.values(), ids=list(CONFIGS))
def test_fault_free_run_is_exact_for_every_config(config):
    report = run_scenario(Scenario(seed=1), config)
    assert report.finished
    assert report.check.counts == {"campaign": 1, "ad": 3, "email_batch": 1, "allocation": 2}
    assert report.check.spend == 700
    assert report.planner_calls == 9
    assert report.check.clean


class Finisher:
    """A planner that finishes as soon as it is asked: runs only the journaled step."""

    calls = 0

    def decide(self, goal, history):
        self.calls += 1
        return {"finish": True}


def test_lost_ack_on_email_suspends_then_reconciles_by_lookup():
    goal = CampaignGoal()
    with TestClient(create_app(faults=FaultConfig(p_lost_ack=1.0))) as client, Journal() as journal:
        journal.append("r", "decision", 0, {"tool": "send_email_batch", "args": {
            "segment": goal.email_segment, "subject": "s", "body": "b"}})
        # Every response is lost, lookups included, so the step can't be settled.
        first = Runtime(journal, campaign_tools(), client, HARNESS).run("r", goal, Finisher())
        assert first.status == "suspended"
        client.post("/_admin/faults", json={})
        second = Runtime(journal, campaign_tools(), client, HARNESS).run("r", goal, Finisher())
        assert second.status == "finished"
        assert journal.load_state("r").steps[0].result["via"] == "lookup"
        effects = client.get("/_admin/effects").json()["effects"]
        assert [e["kind"] for e in effects] == ["email_batch"]


def test_crash_after_send_on_unsafe_call_escalates_and_operator_confirms():
    report = run_scenario(Scenario(seed=2), HARNESS, hook_factory=crash_once("after_send", ALLOCATE_STEP),
                          keep_events=True)
    assert report.finished and report.check.clean
    assert report.escalations == 1
    results = [e for e in report.events if e.kind == "result" and e.step == ALLOCATE_STEP]
    assert [r.payload["via"] for r in results] == ["operator"]


def test_crash_after_intent_on_unsafe_call_escalates_and_operator_voids():
    report = run_scenario(Scenario(seed=3), HARNESS, hook_factory=crash_once("after_intent", ALLOCATE_STEP),
                          keep_events=True)
    assert report.finished and report.check.clean
    assert report.escalations == 1
    kinds = [e.kind for e in report.events if e.step == ALLOCATE_STEP]
    assert kinds == ["decision", "intent", "intent_voided", "intent", "attempt", "result"]


def test_ambiguous_idempotent_call_suspends_rather_than_replanning():
    config = replace(HARNESS, retries=0)
    suspended = 0
    for seed in range(6):
        report = run_scenario(Scenario(seed=seed, faults=FaultConfig(p_5xx=0.2, seed=seed)), config,
                              keep_events=True)
        assert report.finished and report.check.clean
        # An unknown outcome of a side-effecting call never reaches the planner.
        # (A failed read may: re-planning a read is harmless.)
        failed = {e.step for e in report.events if e.kind == "result" and e.payload["status"] == "error"}
        decided = {e.step: e.payload.get("tool") for e in report.events if e.kind == "decision"}
        assert {decided[s] for s in failed} <= {"get_metrics"}
        suspended += sum(1 for e in report.events
                         if e.kind == "status" and e.payload["status"] == "suspended")
    assert suspended > 0


def test_approval_gate_binds_to_arguments():
    config = replace(HARNESS, name="harness_approvals", approvals=True)
    report = run_scenario(Scenario(seed=5), config)
    assert report.finished and report.check.clean
    assert report.approvals == 2 and report.check.unapproved == 0

    with TestClient(create_app()) as client, Journal() as journal:
        runtime = Runtime(journal, campaign_tools(), client, config)
        outcome = runtime.run("r", CampaignGoal(), ScriptedPlanner())
        assert outcome.status == "awaiting_approval"
        approve(journal, "r", outcome.detail["step"], "hash-of-different-args")
        again = Runtime(journal, campaign_tools(), client, config).run("r", CampaignGoal(), ScriptedPlanner())
        assert again.status == "awaiting_approval"  # an approval for other arguments doesn't count
        spent = [e for e in client.get("/_admin/effects").json()["effects"] if e["kind"] == "allocation"]
        assert spent == []


def test_budget_reservation_refuses_calls_over_the_cap():
    config = replace(HARNESS, name="harness_budget", budget_cap=600)
    report = run_scenario(Scenario(seed=6), config)
    assert report.check.spend == 500  # the $200 boost would exceed the cap
    assert report.status == "gave_up"


def test_conditional_writes_resolve_lost_acks_without_escalating():
    tools = campaign_tools(conditional_writes=True)
    for seed in range(8):
        scenario = Scenario(seed=seed, faults=FaultConfig(p_lost_ack=0.3, p_5xx=0.1, seed=seed))
        plain = run_scenario(scenario, HARNESS)
        cas = run_scenario(scenario, HARNESS, tools=tools)
        assert plain.check.clean and cas.check.clean
        assert cas.escalations == 0
    crashed = run_scenario(Scenario(seed=9), HARNESS, tools=tools,
                           hook_factory=crash_once("after_send", ALLOCATE_STEP))
    assert crashed.finished and crashed.check.clean and crashed.escalations == 0
