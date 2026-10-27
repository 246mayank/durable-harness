"""Simulated ad/email platform: ground truth plus seeded fault injection.

The endpoints cover all four effect classes the harness distinguishes:

    POST /v1/campaigns                    idempotent   (honours Idempotency-Key)
    POST /v1/campaigns/{id}/ads           idempotent   (honours Idempotency-Key)
    POST /v1/email/batches                reconcilable (carries client_ref)
    GET  /v1/email/batches?client_ref=    ...its lookup
    POST /v1/campaigns/{id}/allocations   unsafe       (no key, no lookup)
    GET  /v1/campaigns/{id}/metrics       read

Admin endpoints under /_admin are never faulted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from platform_sim.db import PlatformDB, next_id
from platform_sim.faults import Fault, FaultConfig, FaultInjector

SEGMENT_SIZES = {"trial_users": 1200, "enterprise_leads": 340, "churned_accounts": 780}
IMPRESSIONS_PER_AD = 10_000


class CampaignIn(BaseModel):
    name: str
    product: str
    audience: str


class AdIn(BaseModel):
    headline: str
    body: str


class EmailBatchIn(BaseModel):
    segment: str
    subject: str
    body: str
    client_ref: str | None = None


class AllocationIn(BaseModel):
    amount: int = Field(gt=0)
    ad_id: str | None = None
    # Optional precondition: apply only if the campaign's current total matches.
    expected_total: int | None = None


def ctr_for(headline: str, body: str) -> float:
    """Deterministic click-through rate in [0.5%, 5%] derived from the copy."""
    digest = hashlib.sha256(f"{headline}\n{body}".encode()).digest()
    return round(0.005 + int.from_bytes(digest[:4], "big") / 2**32 * 0.045, 5)


def _canonical_hash(path: str, body: dict) -> str:
    return hashlib.sha256(json.dumps([path, body], sort_keys=True).encode()).hexdigest()


class FaultMiddleware:
    """Pure-ASGI middleware applying one fault roll per non-admin request.

    429 and 5xx short-circuit before the endpoint runs, so nothing commits.
    A lost ack runs the endpoint to completion (it commits) and then replaces
    the response with a 504.
    """

    def __init__(self, app, platform: "Platform") -> None:
        self.app = app
        self.platform = platform

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"].startswith("/_admin"):
            await self.app(scope, receive, send)
            return
        fault, status = self.platform.faults.roll()
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        self.platform.log_request(
            scope["method"], scope["path"], headers.get("x-run-id"), headers.get("x-trace-id"), fault
        )
        if fault in (Fault.RATE_LIMIT, Fault.SERVER_ERROR):
            extra = {"Retry-After": "0"} if status == 429 else None
            await JSONResponse({"error": fault.value}, status, headers=extra)(scope, receive, send)
            return
        if fault is Fault.LOST_ACK:

            async def discard(message) -> None:
                pass

            await self.app(scope, receive, discard)
            await JSONResponse({"error": "gateway timeout"}, 504)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class Platform:
    def __init__(self, db_path: str = ":memory:", faults: FaultConfig | None = None) -> None:
        self.db = PlatformDB(db_path)
        self.faults = FaultInjector(faults)

    def log_request(self, method, path, run_id, trace_id, fault: Fault) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO requests (method, path, run_id, trace_id, fault) VALUES (?,?,?,?,?)",
                (method, path, run_id, trace_id, fault.value),
            )

    def idempotent(
        self, request: Request, key: str | None, body: dict, create: Callable
    ) -> JSONResponse:
        """Run `create` once per Idempotency-Key.

        A repeated key with the same body replays the stored response; the
        same key with a different body is a client bug and returns 409. Only
        2xx results are stored, so a 404 can be retried with the same key.
        """
        body_hash = _canonical_hash(request.url.path, body)
        with self.db.tx() as conn:
            if key:
                row = conn.execute(
                    "SELECT body_hash, status, response FROM idem WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    if row["body_hash"] != body_hash:
                        return JSONResponse(
                            {"error": "idempotency_key_reused", "key": key}, 409
                        )
                    return JSONResponse(
                        json.loads(row["response"]),
                        row["status"],
                        headers={"Idempotent-Replayed": "true"},
                    )
            status, payload = create(conn)
            if key and 200 <= status < 300:
                conn.execute(
                    "INSERT INTO idem (key, body_hash, status, response) VALUES (?,?,?,?)",
                    (key, body_hash, status, json.dumps(payload)),
                )
        return JSONResponse(payload, status)


def _record(conn, kind, object_id, request: Request, payload: dict, amount: int | None = None):
    conn.execute(
        "INSERT INTO effects (kind, object_id, run_id, trace_id, amount, payload)"
        " VALUES (?,?,?,?,?,?)",
        (
            kind,
            object_id,
            request.headers.get("x-run-id"),
            request.headers.get("x-trace-id"),
            amount,
            json.dumps(payload, sort_keys=True),
        ),
    )


def _campaign_exists(conn, campaign_id: str) -> bool:
    return conn.execute("SELECT 1 FROM campaigns WHERE id = ?", (campaign_id,)).fetchone() is not None


def _not_found(campaign_id: str) -> tuple[int, dict]:
    return 404, {"error": "campaign_not_found", "campaign_id": campaign_id}


def create_app(db_path: str = ":memory:", faults: FaultConfig | None = None) -> FastAPI:
    platform = Platform(db_path, faults)
    app = FastAPI(title="platform_sim")
    app.state.platform = platform
    app.add_middleware(FaultMiddleware, platform=platform)

    @app.post("/v1/campaigns", status_code=201)
    def create_campaign(
        body: CampaignIn, request: Request, idempotency_key: str | None = Header(None)
    ):
        def create(conn):
            cid = next_id(conn, "campaigns", "cmp")
            conn.execute(
                "INSERT INTO campaigns VALUES (?,?,?,?)",
                (cid, body.name, body.product, body.audience),
            )
            out = {"id": cid, **body.model_dump()}
            _record(conn, "campaign", cid, request, out)
            return 201, out

        return platform.idempotent(request, idempotency_key, body.model_dump(), create)

    @app.post("/v1/campaigns/{campaign_id}/ads", status_code=201)
    def create_ad(
        campaign_id: str, body: AdIn, request: Request, idempotency_key: str | None = Header(None)
    ):
        def create(conn):
            if not _campaign_exists(conn, campaign_id):
                return _not_found(campaign_id)
            ad_id = next_id(conn, "ads", "ad")
            conn.execute(
                "INSERT INTO ads VALUES (?,?,?,?)", (ad_id, campaign_id, body.headline, body.body)
            )
            out = {"id": ad_id, "campaign_id": campaign_id, **body.model_dump()}
            _record(conn, "ad", ad_id, request, out)
            return 201, out

        return platform.idempotent(request, idempotency_key, body.model_dump(), create)

    @app.post("/v1/email/batches", status_code=201)
    def send_email_batch(body: EmailBatchIn, request: Request):
        # No Idempotency-Key support: a resend is a second batch of emails.
        with platform.db.tx() as conn:
            batch_id = next_id(conn, "email_batches", "eml")
            recipients = SEGMENT_SIZES.get(body.segment, 500)
            conn.execute(
                "INSERT INTO email_batches VALUES (?,?,?,?,?,?)",
                (batch_id, body.segment, body.subject, body.body, body.client_ref, recipients),
            )
            out = {"id": batch_id, "recipients": recipients, **body.model_dump()}
            _record(conn, "email_batch", batch_id, request, out)
        return JSONResponse(out, 201)

    @app.get("/v1/email/batches")
    def find_email_batches(client_ref: str):
        rows = platform.db.query(
            "SELECT id, segment, subject, body, client_ref, recipients FROM email_batches"
            " WHERE client_ref = ? ORDER BY id",
            (client_ref,),
        )
        return {"items": rows}

    @app.post("/v1/campaigns/{campaign_id}/allocations", status_code=201)
    def allocate_budget(campaign_id: str, body: AllocationIn, request: Request):
        # No key and no lookup by reference: the outcome of a lost response
        # cannot be determined from the API alone.
        with platform.db.tx() as conn:
            if not _campaign_exists(conn, campaign_id):
                status, out = _not_found(campaign_id)
                return JSONResponse(out, status)
            (total,) = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM allocations WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if body.expected_total is not None and body.expected_total != total:
                return JSONResponse(
                    {"error": "precondition_failed", "expected_total": body.expected_total,
                     "total": total},
                    412,
                )
            alloc_id = next_id(conn, "allocations", "alc")
            conn.execute(
                "INSERT INTO allocations VALUES (?,?,?,?)",
                (alloc_id, campaign_id, body.amount, body.ad_id),
            )
            out = {
                "id": alloc_id,
                "campaign_id": campaign_id,
                "amount": body.amount,
                "ad_id": body.ad_id,
                "total": total + body.amount,
            }
            _record(conn, "allocation", alloc_id, request, out, amount=body.amount)
        return JSONResponse(out, 201)

    @app.get("/v1/campaigns/{campaign_id}/allocations")
    def get_allocations(campaign_id: str):
        rows = platform.db.query(
            "SELECT id, amount, ad_id FROM allocations WHERE campaign_id = ? ORDER BY rowid",
            (campaign_id,),
        )
        return {"campaign_id": campaign_id, "total": sum(r["amount"] for r in rows), "items": rows}

    @app.get("/v1/campaigns/{campaign_id}/metrics")
    def get_metrics(campaign_id: str):
        ads = platform.db.query(
            "SELECT id, headline, body FROM ads WHERE campaign_id = ? ORDER BY rowid",
            (campaign_id,),
        )
        if not ads and not platform.db.query("SELECT 1 FROM campaigns WHERE id = ?", (campaign_id,)):
            status, out = _not_found(campaign_id)
            return JSONResponse(out, status)
        rows = []
        for ad in ads:
            ctr = ctr_for(ad["headline"], ad["body"])
            rows.append(
                {
                    "ad_id": ad["id"],
                    "impressions": IMPRESSIONS_PER_AD,
                    "clicks": round(IMPRESSIONS_PER_AD * ctr),
                    "ctr": ctr,
                }
            )
        return {"campaign_id": campaign_id, "ads": rows}

    # ---- admin (never faulted) -------------------------------------------

    @app.get("/_admin/faults")
    def get_faults():
        return {**platform.faults.config.as_dict(), "rolls": platform.faults.rolls}

    @app.post("/_admin/faults")
    def set_faults(config: dict):
        platform.faults.configure(FaultConfig(**config))
        return platform.faults.config.as_dict()

    @app.get("/_admin/effects")
    def get_effects(run_id: str | None = None):
        sql = "SELECT seq, kind, object_id, run_id, trace_id, amount, payload FROM effects"
        rows = (
            platform.db.query(sql + " WHERE run_id = ? ORDER BY seq", (run_id,))
            if run_id
            else platform.db.query(sql + " ORDER BY seq")
        )
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return {"effects": rows}

    @app.get("/_admin/requests")
    def get_requests(run_id: str | None = None):
        sql = "SELECT seq, method, path, run_id, trace_id, fault FROM requests"
        if run_id:
            return {"requests": platform.db.query(sql + " WHERE run_id = ? ORDER BY seq", (run_id,))}
        return {"requests": platform.db.query(sql + " ORDER BY seq")}

    @app.post("/_admin/reset")
    def reset(config: dict | None = None):
        platform.db.reset()
        platform.faults.configure(FaultConfig(**(config or {})))
        return {"ok": True}

    return app
