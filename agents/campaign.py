"""B2B campaign agent: tools, goal and a scripted planner.

The planner decides from the goal plus observed history only, as an LLM
would. It has no memory of its own, so after a restart without replay it
starts over, and with temperature > 0 a restarted planner is a different
draw: different ad copy, a different email subject, sometimes a different
step order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from harness.errors import Ambiguous, Rejected, Unresolvable, request
from harness.tools import Effect, ToolRegistry, tool

# ---- tools ------------------------------------------------------------------------


class CreateCampaignArgs(BaseModel):
    name: str
    product: str
    audience: str


class CreateAdArgs(BaseModel):
    campaign_id: str
    headline: str
    body: str


class SendEmailArgs(BaseModel):
    segment: str
    subject: str
    body: str


class AllocateArgs(BaseModel):
    campaign_id: str
    amount: int = Field(gt=0)
    ad_id: str | None = None


class MetricsArgs(BaseModel):
    campaign_id: str


def _keyed_headers(ctx) -> dict[str, str]:
    headers = ctx.headers()
    if ctx.key:
        headers["Idempotency-Key"] = ctx.key
    return headers


@tool(Effect.IDEMPOTENT, CreateCampaignArgs)
def create_campaign(client, args: CreateCampaignArgs, ctx):
    """Create a campaign for a product and audience."""
    return request(client, "POST", "/v1/campaigns", json=args.model_dump(), headers=_keyed_headers(ctx))


@tool(Effect.IDEMPOTENT, CreateAdArgs)
def create_ad(client, args: CreateAdArgs, ctx):
    """Create one ad (headline + body) in a campaign."""
    return request(
        client, "POST", f"/v1/campaigns/{args.campaign_id}/ads",
        json={"headline": args.headline, "body": args.body}, headers=_keyed_headers(ctx),
    )


@tool(Effect.RECONCILABLE, SendEmailArgs)
def send_email_batch(client, args: SendEmailArgs, ctx):
    """Send one email batch to a segment."""
    # The platform has no idempotency keys here; the key travels as client_ref
    # so the batch can be found again.
    return request(
        client, "POST", "/v1/email/batches",
        json={**args.model_dump(), "client_ref": ctx.key}, headers=ctx.headers(),
    )


@send_email_batch.reconciler
def _find_email_batch(client, args: SendEmailArgs, ctx):
    if ctx.key is None:
        raise Unresolvable("no client_ref to look up")
    found = request(client, "GET", "/v1/email/batches", params={"client_ref": ctx.key},
                    headers=ctx.headers())["items"]
    return found[0] if found else None


@tool(Effect.UNSAFE, AllocateArgs, spend=lambda a: a.amount)
def allocate_budget(client, args: AllocateArgs, ctx):
    """Add budget to a campaign, optionally boosting one ad."""
    return request(
        client, "POST", f"/v1/campaigns/{args.campaign_id}/allocations",
        json={"amount": args.amount, "ad_id": args.ad_id}, headers=ctx.headers(),
    )


@tool(Effect.READ, MetricsArgs)
def get_metrics(client, args: MetricsArgs, ctx):
    """Read per-ad impressions, clicks and CTR."""
    return request(client, "GET", f"/v1/campaigns/{args.campaign_id}/metrics", headers=ctx.headers())


# Conditional-write variant: the same API call plus an `expected_total`
# precondition captured before the intent. A lost response is then checkable
# (total unchanged -> not applied; total + amount -> applied), and a re-send
# can never double-apply because its precondition would fail.
@tool(Effect.RECONCILABLE, AllocateArgs, name="allocate_budget", spend=lambda a: a.amount)
def allocate_budget_cas(client, args: AllocateArgs, ctx):
    """Add budget to a campaign, optionally boosting one ad."""
    try:
        return request(
            client, "POST", f"/v1/campaigns/{args.campaign_id}/allocations",
            json={"amount": args.amount, "ad_id": args.ad_id,
                  "expected_total": ctx.precondition["total"]},
            headers=ctx.headers(),
        )
    except Rejected as err:
        if err.status == 412:  # the total moved: possibly our own earlier attempt
            raise Ambiguous("precondition failed", err.status, err.body) from err
        raise


@allocate_budget_cas.preparer
def _allocation_total(client, args: AllocateArgs, ctx):
    return {"total": _current_total(client, args, ctx)}


@allocate_budget_cas.reconciler
def _find_allocation(client, args: AllocateArgs, ctx):
    before = ctx.precondition["total"]
    total = _current_total(client, args, ctx)
    if total == before:
        return None
    if total == before + args.amount:
        return {"campaign_id": args.campaign_id, "amount": args.amount, "ad_id": args.ad_id,
                "total": total}
    raise Unresolvable(f"total is {total}; expected {before} or {before + args.amount}")


def _current_total(client, args: AllocateArgs, ctx) -> int:
    return request(client, "GET", f"/v1/campaigns/{args.campaign_id}/allocations",
                   headers=ctx.headers())["total"]


def campaign_tools(conditional_writes: bool = False) -> ToolRegistry:
    allocate = allocate_budget_cas if conditional_writes else allocate_budget
    return ToolRegistry.of([create_campaign, create_ad, send_email_batch, allocate, get_metrics])


# ---- goal -------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignGoal:
    product: str = "Ledgerly"
    audience: str = "finance leads at mid-market SaaS companies"
    n_ads: int = 3
    email_segment: str = "trial_users"
    initial_budget: int = 500
    boost: int = 200

    @property
    def expected_effects(self) -> dict[str, int]:
        return {"campaign": 1, "ad": self.n_ads, "email_batch": 1, "allocation": 2}

    @property
    def expected_spend(self) -> int:
        return self.initial_budget + self.boost


# ---- scripted planner ---------------------------------------------------------------

AD_ANGLES = [
    ("Close the books in 3 days", "{product} automates reconciliation for {audience}."),
    ("Month-end without the all-nighters", "{product} matches every transaction while you sleep."),
    ("Real-time cash, every entity", "{product} gives {audience} one live view of cash."),
    ("Audit-ready, every single day", "{product} keeps a clean trail your auditors will love."),
    ("Cut AP costs by 40%", "{product} turns invoices into approvals in minutes."),
    ("From spreadsheets to sign-off", "{product} replaces the close checklist with workflows."),
    ("Forecast with confidence", "{product} rolls actuals into forecasts automatically."),
    ("Your controller will thank you", "{product}: fewer manual journals, faster close."),
]

SUBJECTS = [
    "Your trial: close the books faster with {product}",
    "3 workflows {audience} set up in week one",
    "How finance teams cut month-end to 3 days",
    "A faster close starts here",
]


@dataclass
class _Plan:
    ads: list[tuple[str, str]]
    subject: str
    email_first: bool


@dataclass
class ScriptedPlanner:
    seed: int = 0
    temperature: float = 0.0
    calls: int = 0
    _plan: _Plan | None = field(default=None, repr=False)

    def decide(self, goal: CampaignGoal, history: list[dict]) -> dict:
        self.calls += 1
        plan = self._sample(goal)
        ok = [h for h in history if h["status"] == "ok"]

        campaign = next((h["value"] for h in ok if h["tool"] == "create_campaign"), None)
        if campaign is None:
            return _call("create_campaign", "start a campaign",
                         name=f"{goal.product} Q4 pipeline", product=goal.product,
                         audience=goal.audience)
        cid = campaign["id"]

        ads = [h["value"] for h in ok if h["tool"] == "create_ad" and h["args"]["campaign_id"] == cid]
        emailed = any(h["tool"] == "send_email_batch" for h in ok)
        email = _call("send_email_batch", "announce to the segment", segment=goal.email_segment,
                      subject=plan.subject, body=f"See what {goal.product} does for your close.")
        if plan.email_first and not emailed:
            return email
        if len(ads) < goal.n_ads:
            headline, body = plan.ads[len(ads)]
            return _call("create_ad", f"ad {len(ads) + 1} of {goal.n_ads}",
                         campaign_id=cid, headline=headline, body=body)
        if not emailed:
            return email

        allocations = [h["args"] for h in ok if h["tool"] == "allocate_budget"]
        if not any(a.get("ad_id") is None for a in allocations):
            return _call("allocate_budget", "fund the campaign", campaign_id=cid,
                         amount=goal.initial_budget)
        if not any(a.get("ad_id") for a in allocations):
            metrics = next((h["value"] for h in reversed(ok) if h["tool"] == "get_metrics"), None)
            ours = {ad["id"] for ad in ads}
            rows = [r for r in (metrics or {}).get("ads", []) if r["ad_id"] in ours]
            if len(rows) < len(ours):
                return _call("get_metrics", "find the best ad", campaign_id=cid)
            best = max(rows, key=lambda r: r["ctr"])
            return _call("allocate_budget", f"boost the best ad (ctr {best['ctr']:.2%})",
                         campaign_id=cid, amount=goal.boost, ad_id=best["ad_id"])
        return {"finish": True, "note": "campaign live, emailed, funded and boosted"}

    def _sample(self, goal: CampaignGoal) -> _Plan:
        """Draw this planner's copy and ordering once, from its own seed."""
        if self._plan is not None:
            return self._plan
        rng, t = random.Random(self.seed), self.temperature
        fmt = {"product": goal.product, "audience": goal.audience}
        chosen: list[int] = []
        for i in range(goal.n_ads):
            options = [a for a in range(len(AD_ANGLES)) if a not in chosen]
            default = i if i in options else options[0]
            chosen.append(rng.choice([a for a in options if a != default] or options)
                          if rng.random() < t else default)
        subject = SUBJECTS[rng.randrange(1, len(SUBJECTS))] if rng.random() < t else SUBJECTS[0]
        self._plan = _Plan(
            ads=[(AD_ANGLES[a][0], AD_ANGLES[a][1].format(**fmt)) for a in chosen],
            subject=subject.format(**fmt),
            email_first=rng.random() < t / 2,
        )
        return self._plan


def _call(tool_name: str, note: str, **args) -> dict:
    return {"tool": tool_name, "args": args, "note": note}
