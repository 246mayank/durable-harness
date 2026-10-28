"""Print a run's journal as a timeline.

    python -m harness.trace runs/journal.db [run_id]
"""

from __future__ import annotations

import argparse
import json

from harness.journal import Event, Journal


def _short(value, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _call(decision: dict) -> str:
    args = ", ".join(f"{k}={_short(v, 28)}" for k, v in decision.get("args", {}).items())
    return f"{decision['tool']}({args})"


def format_event(e: Event) -> str:
    p = e.payload
    step = f"step {e.step:<2}" if e.step is not None else "       "
    if e.kind == "incarnation":
        epoch = f", epoch {p['epoch']}" if p.get("epoch") else ""
        return f"-- incarnation {p.get('n')} ({p.get('config')}{epoch}) " + "-" * 30
    if e.kind == "status":
        detail = {k: v for k, v in p.items() if k not in ("status", "args")}
        return f"   == {p['status']} {_short(detail, 90) if detail else ''}"
    if e.kind == "decision":
        return f"{step} decide   " + ("finish" if p.get("finish") else _call(p))
    if e.kind == "intent":
        key = (p.get("key") or "-")[:10]
        pre = f" precondition={p['precondition']}" if p.get("precondition") else ""
        return f"{step} intent   key={key} effect={p.get('effect')}{pre}"
    if e.kind == "attempt":
        status = f" {p['status']}" if p.get("status") else ""
        return f"{step} attempt  #{p.get('n')} {p.get('outcome')}{status}"
    if e.kind == "result":
        if p["status"] == "ok":
            return f"{step} result   ok via {p.get('via')} -> {_short(p.get('value'))}"
        return f"{step} result   {p.get('error_class')}: {_short(p.get('error'))}"
    if e.kind == "intent_voided":
        return f"{step} voided   {p.get('reason')}"
    if e.kind == "approval":
        return f"{step} approved args_hash={p['args_hash'][:10]}"
    if e.kind == "deadline":
        return f"{step} renewed  deadline (the last one passed before this send)"
    return f"{step} {e.kind} {_short(p)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("journal")
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    with Journal(args.journal) as journal:
        for run_id in [args.run_id] if args.run_id else journal.run_ids():
            print(f"# {run_id}")
            for event in journal.events(run_id):
                print(format_event(event))


if __name__ == "__main__":
    main()
