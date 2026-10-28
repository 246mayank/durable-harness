"""Calls that are still in flight: delayed delivery, deadlines, zombie incarnations, fencing."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from agents.campaign import CampaignGoal, campaign_tools
from chaos.driver import Scenario, run_scenario
from harness.clock import VirtualClock
from harness.journal import Journal
from harness.runtime import CONFIGS, Runtime
from platform_sim.app import create_app
from platform_sim.faults import Fault

HARNESS = CONFIGS["harness"]
NO_DEADLINE = replace(HARNESS, name="harness_nodeadline", deadline=None)
NO_FENCE = replace(HARNESS, name="harness_nofence", fencing=False)
EMAIL_STEP, ALLOCATE_STEP = 4, 5  # campaign, ad, ad, ad, email, allocate(500), metrics, boost, finish


class Finisher:
    """A planner that finishes as soon as it is asked: runs only the journaled step."""

    calls = 0

    def decide(self, goal, history):
        self.calls += 1
        return {"finish": True}


def effects(client, kind):
    client.post("/_admin/drain")
    return [e for e in client.get("/_admin/effects").json()["effects"] if e["kind"] == kind]


@pytest.mark.parametrize("delay", [0.3, 0.8], ids=["lands-before-deadline", "lands-after-deadline"])
@pytest.mark.parametrize("config, batches", [(HARNESS, 1), (NO_DEADLINE, 2)], ids=["deadline", "no-deadline"])
def test_negative_lookup_waits_until_no_send_can_land(config, batches, delay):
    # The email POST is answered 504 but is still in flight. Looking it up right
    # away finds nothing and re-sends: a second batch once the original lands.
    # Waiting out the deadline first either finds the original or guarantees
    # the platform will drop it.
    clock = VirtualClock()
    goal = CampaignGoal()
    with TestClient(create_app(clock=clock)) as client, Journal() as journal:
        client.app.state.platform.faults.force(Fault.DELAYED, delay=delay)
        journal.append("r", "decision", 0, {"tool": "send_email_batch", "args": {
            "segment": goal.email_segment, "subject": "s", "body": "b"}})
        outcome = Runtime(journal, campaign_tools(), client, config, clock=clock).run("r", goal, Finisher())
        assert outcome.status == "finished"
        assert len(effects(client, "email_batch")) == batches


def test_conditional_write_is_immune_to_delayed_delivery_even_without_deadlines():
    # The delayed original carries expected_total=0; once the re-send has
    # committed, the total is 500 and the original fails its precondition.
    clock = VirtualClock()
    with TestClient(create_app(clock=clock)) as client, Journal() as journal:
        cid = client.post("/v1/campaigns", json={"name": "n", "product": "p", "audience": "a"}).json()["id"]
        # First request is the precondition read, the second the allocation itself.
        client.app.state.platform.faults.force(Fault.NONE, Fault.DELAYED, delay=0.3)
        journal.append("r", "decision", 0, {"tool": "allocate_budget",
                                            "args": {"campaign_id": cid, "amount": 500}})
        outcome = Runtime(journal, campaign_tools(conditional_writes=True), client, NO_DEADLINE,
                          clock=clock).run("r", CampaignGoal(), Finisher())
        assert outcome.status == "finished"
        assert [e["amount"] for e in effects(client, "allocation")] == [500]


@pytest.mark.parametrize("point", ["after_decision", "after_intent", "after_send", "after_result"])
def test_fencing_stops_a_zombie_incarnation(point):
    # Incarnation 0 hangs at its 5th pass of `point`, is presumed dead, and
    # wakes up after its replacement has finished the run.
    scenario = Scenario(seed=1, freeze=(point, 5))
    fenced = run_scenario(scenario, HARNESS)
    assert fenced.froze and fenced.zombie == "fenced"
    assert fenced.finished and fenced.check.clean
    unfenced = run_scenario(scenario, NO_FENCE)
    assert unfenced.froze and unfenced.check.duplicate_total > 0


@pytest.mark.parametrize("step", [EMAIL_STEP, ALLOCATE_STEP], ids=["email", "allocation"])
def test_a_zombie_frozen_after_its_fence_check_needs_the_deadline(step):
    # Frozen between the fence check and the send, the zombie still sends when
    # it wakes. Only the deadline it carries (which the new owner waited out
    # before acting) keeps that send from landing.
    scenario = Scenario(seed=1, freeze=("before_send", step + 1))  # one attempt per step, no faults
    with_deadline = run_scenario(scenario, HARNESS)
    assert with_deadline.froze and with_deadline.check.clean
    fencing_only = run_scenario(scenario, NO_DEADLINE)
    assert fencing_only.froze and fencing_only.check.duplicate_total == 1
