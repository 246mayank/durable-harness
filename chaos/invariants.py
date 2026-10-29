"""Invariants checked against the platform's ground truth, never the agent's view."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

KINDS = ("campaign", "ad", "email_batch", "allocation")


@dataclass(frozen=True)
class Check:
    counts: dict[str, int]
    duplicates: dict[str, int]  # side effects beyond the expected count, per kind
    missing: dict[str, int]  # expected side effects that never happened, per kind
    spend: int
    excess_spend: int
    unapproved: int | None  # unsafe effects with no approval; None if the gate is off
    # Effects from a step that had already taken effect (same trace id): a
    # recovery failure. A duplicate from a new step is a new planner decision.
    # Step numbers only identify a step across incarnations in configs that
    # replay the journal; without replay they restart at 0.
    repeated_steps: int = 0

    @property
    def duplicate_total(self) -> int:
        return sum(self.duplicates.values())

    @property
    def missing_total(self) -> int:
        return sum(self.missing.values())

    @property
    def clean(self) -> bool:
        return (self.duplicate_total == 0 and self.missing_total == 0
                and self.excess_spend == 0 and not self.unapproved and self.repeated_steps == 0)


def check(
    effects: list[dict],
    expected: dict[str, int],
    expected_spend: int,
    approved_traces: set[str] | None = None,
) -> Check:
    counts = Counter(e["kind"] for e in effects)
    per_step = Counter(e["trace_id"] for e in effects if e["trace_id"])
    spend = sum(e["amount"] or 0 for e in effects if e["kind"] == "allocation")
    unapproved = None
    if approved_traces is not None:
        unapproved = sum(
            1 for e in effects if e["kind"] == "allocation" and e["trace_id"] not in approved_traces
        )
    return Check(
        counts={k: counts.get(k, 0) for k in KINDS},
        duplicates={k: max(0, counts.get(k, 0) - expected[k]) for k in KINDS},
        missing={k: max(0, expected[k] - counts.get(k, 0)) for k in KINDS},
        spend=spend,
        excess_spend=max(0, spend - expected_spend),
        unapproved=unapproved,
        repeated_steps=sum(n - 1 for n in per_step.values()),
    )
