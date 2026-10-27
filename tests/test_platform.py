from fastapi.testclient import TestClient

from platform_sim.app import create_app
from platform_sim.faults import FaultConfig

CAMPAIGN = {"name": "Q4 launch", "product": "Ledgerly", "audience": "CFOs"}


def client(**faults) -> TestClient:
    return TestClient(create_app(faults=FaultConfig(**faults)))


def effects(c: TestClient, kind: str | None = None) -> list[dict]:
    rows = c.get("/_admin/effects").json()["effects"]
    return [e for e in rows if kind is None or e["kind"] == kind]


def test_repeated_key_replays_stored_response():
    c = client()
    first = c.post("/v1/campaigns", json=CAMPAIGN, headers={"Idempotency-Key": "k1"})
    second = c.post("/v1/campaigns", json=CAMPAIGN, headers={"Idempotency-Key": "k1"})
    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    assert second.headers["idempotent-replayed"] == "true"
    assert len(effects(c, "campaign")) == 1


def test_key_reuse_with_different_body_is_409():
    c = client()
    c.post("/v1/campaigns", json=CAMPAIGN, headers={"Idempotency-Key": "k1"})
    other = c.post("/v1/campaigns", json={**CAMPAIGN, "name": "other"}, headers={"Idempotency-Key": "k1"})
    assert other.status_code == 409
    assert len(effects(c, "campaign")) == 1


def test_no_key_gives_duplicates():
    c = client()
    c.post("/v1/campaigns", json=CAMPAIGN)
    c.post("/v1/campaigns", json=CAMPAIGN)
    assert len(effects(c, "campaign")) == 2


def test_lost_ack_commits_then_returns_504():
    c = client(p_lost_ack=1.0)
    r = c.post("/v1/campaigns", json=CAMPAIGN, headers={"Idempotency-Key": "k1"})
    assert r.status_code == 504
    assert len(effects(c, "campaign")) == 1
    # A retry with the same key does not create a second campaign.
    c.post("/_admin/faults", json={})
    retry = c.post("/v1/campaigns", json=CAMPAIGN, headers={"Idempotency-Key": "k1"})
    assert retry.status_code == 201 and retry.headers["idempotent-replayed"] == "true"
    assert len(effects(c, "campaign")) == 1


def test_5xx_and_429_do_not_commit():
    for faults, codes in (({"p_5xx": 1.0}, {500, 503}), ({"p_429": 1.0}, {429})):
        c = client(**faults)
        for _ in range(5):
            assert c.post("/v1/campaigns", json=CAMPAIGN).status_code in codes
        assert effects(c) == []


def test_admin_endpoints_are_never_faulted():
    c = client(p_5xx=1.0)
    assert c.get("/_admin/effects").status_code == 200
    assert c.get("/_admin/faults").json()["rolls"] == 0


def test_fault_sequence_is_deterministic_per_seed():
    def sequence(seed: int) -> list[str]:
        c = client(p_429=0.2, p_5xx=0.2, p_lost_ack=0.2, seed=seed)
        for _ in range(40):
            c.post("/v1/campaigns", json=CAMPAIGN)
        return [r["fault"] for r in c.get("/_admin/requests").json()["requests"]]

    assert sequence(3) == sequence(3)
    assert sequence(3) != sequence(4)
    assert {"none", "429", "5xx", "lost_ack"} <= set(sequence(3))


def test_email_batches_are_found_by_client_ref():
    c = client()
    email = {"segment": "trial_users", "subject": "Hi", "body": "...", "client_ref": "ref-1"}
    c.post("/v1/email/batches", json=email)
    assert len(c.get("/v1/email/batches", params={"client_ref": "ref-1"}).json()["items"]) == 1
    assert c.get("/v1/email/batches", params={"client_ref": "ref-2"}).json()["items"] == []
    # No key support: a resend is a second batch.
    c.post("/v1/email/batches", json=email, headers={"Idempotency-Key": "ignored"})
    assert len(effects(c, "email_batch")) == 2


def test_allocations_are_unsafe_but_support_a_precondition():
    c = client()
    cid = c.post("/v1/campaigns", json=CAMPAIGN).json()["id"]
    url = f"/v1/campaigns/{cid}/allocations"
    assert c.post(url, json={"amount": 500}).json()["total"] == 500
    assert c.post(url, json={"amount": 500}).json()["total"] == 1000
    stale = c.post(url, json={"amount": 200, "expected_total": 500})
    assert stale.status_code == 412
    assert c.post(url, json={"amount": 200, "expected_total": 1000}).status_code == 201
    assert c.get(url).json()["total"] == 1200


def test_ads_require_an_existing_campaign():
    c = client()
    r = c.post("/v1/campaigns/cmp_404/ads", json={"headline": "h", "body": "b"},
               headers={"Idempotency-Key": "k"})
    assert r.status_code == 404
    assert effects(c) == []


def test_metrics_are_deterministic_per_copy():
    c = client()
    cid = c.post("/v1/campaigns", json=CAMPAIGN).json()["id"]
    for headline in ("a", "b", "c"):
        c.post(f"/v1/campaigns/{cid}/ads", json={"headline": headline, "body": "x"})
    m1 = c.get(f"/v1/campaigns/{cid}/metrics").json()
    m2 = c.get(f"/v1/campaigns/{cid}/metrics").json()
    assert m1 == m2 and len(m1["ads"]) == 3
    assert all(0.005 <= ad["ctr"] <= 0.05 for ad in m1["ads"])


def test_effects_are_attributed_to_run_and_trace():
    c = client()
    c.post("/v1/campaigns", json=CAMPAIGN, headers={"X-Run-Id": "r1", "X-Trace-Id": "r1/0"})
    c.post("/v1/campaigns", json=CAMPAIGN, headers={"X-Run-Id": "r2"})
    rows = c.get("/_admin/effects", params={"run_id": "r1"}).json()["effects"]
    assert [(e["run_id"], e["trace_id"]) for e in rows] == [("r1", "r1/0")]
