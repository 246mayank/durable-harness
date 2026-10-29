"""Parallel tool calls: one decision, several calls, each recovered from its own intent."""

import functools

import pytest
from fastapi.testclient import TestClient

from agents.campaign import CampaignGoal, ScriptedPlanner, campaign_tools
from chaos.bench import VARIANTS
from chaos.driver import CrashAt, Scenario, run_scenario
from harness.clock import VirtualClock
from harness.journal import Journal
from harness.runtime import CONFIGS, CRASH_POINTS, Runtime, resolve
from platform_sim.app import create_app
from platform_sim.faults import Fault

PARALLEL = functools.partial(ScriptedPlanner, parallel=True)
STEPS = range(9)  # campaign, [3 ads, email, allocate 500], metrics, boost, finish
BATCH = range(1, 6)


def test_independent_calls_are_decided_at_once_and_their_intents_journaled_together():
    report = run_scenario(Scenario(seed=1), CONFIGS["harness"], planner_factory=PARALLEL, keep_events=True)
    assert report.finished and report.check.clean
    assert report.planner_calls == 5  # 9 calls: campaign, one batch of 5, metrics, boost, finish
    batch = [e for e in report.events if e.kind == "decision"][1]
    assert len(batch.payload["calls"]) == len(BATCH)
    intents = [e for e in report.events if e.kind == "intent" and e.step in BATCH]
    assert [e.step for e in intents] == list(BATCH)
    assert len({e.ts for e in intents}) == 1  # one transaction
    first_send = min(e.seq for e in report.events if e.kind == "attempt" and e.step in BATCH)
    assert all(e.seq < first_send for e in intents)  # every intent on file before any call goes out


@pytest.mark.parametrize("config", ["harness", "harness_cas"])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("point", CRASH_POINTS)
def test_single_crash_with_parallel_calls(point, step, temperature, config):
    runtime_config, conditional = VARIANTS[config]
    report = run_scenario(
        Scenario(seed=step, temperature=temperature), runtime_config, tools=campaign_tools(conditional),
        planner_factory=PARALLEL, hook_factory=lambda n: CrashAt(point, step=step) if n == 0 else None,
    )
    assert report.finished
    assert report.check.clean, report.check
    assert report.conflicts == 0
    assert report.planner_calls == 5  # a decided batch is never re-asked


@pytest.mark.parametrize("config, escalations", [("harness", 1), ("harness_cas", 0)])
def test_a_crash_after_the_batch_intents_recovers_each_call_by_its_effect_class(config, escalations):
    # All five calls are in doubt at once. The ads are re-sent under their keys,
    # the email is looked up, and the allocation escalates unless its
    # conditional write can tell whether it applied.
    runtime_config, conditional = VARIANTS[config]
    report = run_scenario(Scenario(seed=1), runtime_config, tools=campaign_tools(conditional),
                          planner_factory=PARALLEL,
                          hook_factory=lambda n: CrashAt("after_intent", step=1) if n == 0 else None)
    assert report.crashes == 1 and report.finished and report.check.clean
    assert report.escalations == escalations


class FundAndAdvertise:
    """Creates a campaign, then decides an allocation and an ad at once, then finishes."""

    calls = 0

    def decide(self, goal, history):
        self.calls += 1
        done = {h["tool"]: h for h in history if h["status"] == "ok"}
        if "create_campaign" not in done:
            return {"tool": "create_campaign", "args": {"name": "n", "product": "p", "audience": "a"}}
        cid = done["create_campaign"]["value"]["id"]
        if "allocate_budget" not in done:
            return {"calls": [{"tool": "allocate_budget", "args": {"campaign_id": cid, "amount": 500}},
                              {"tool": "create_ad", "args": {"campaign_id": cid, "headline": "h", "body": "b"}}]}
        return {"finish": True}


def test_calls_a_stopped_run_never_sent_are_voided_not_left_in_doubt():
    clock, goal, planner = VirtualClock(), CampaignGoal(), FundAndAdvertise()
    with TestClient(create_app(clock=clock)) as client, Journal(":memory:") as journal:
        def run():
            return Runtime(journal, campaign_tools(), client, CONFIGS["harness"], clock=clock).run(
                "r", goal, planner)

        def count(kind):
            return sum(e["kind"] == kind for e in client.get("/_admin/effects").json()["effects"])

        client.app.state.platform.faults.force(Fault.NONE, Fault.LOST_ACK)  # the allocation's ack is lost
        assert run().status == "escalated"
        voided = [e for e in journal.events("r") if e.kind == "intent_voided"]
        assert [(e.step, e.payload["reason"]) for e in voided] == [(2, "not sent: run stopped")]
        assert count("ad") == 0  # the ad was never sent, so it is not in doubt

        resolve(journal, "r", applied=True, value={"id": "alloc"})  # the operator finds the allocation
        assert run().status == "finished"
        assert (count("allocation"), count("ad")) == (1, 1)
