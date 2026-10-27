"""Run the campaign agent against a platform over HTTP.

    python -m platform_sim --port 8700 --p-lost-ack 0.1
    python -m agents run --run-id demo --die-at after_send:6   # hard exit mid-run
    python -m agents run --run-id demo                         # resumes from the journal
    python -m agents resolve --run-id demo --from-audit        # if it escalated
    python -m harness.trace runs/journal.db demo

Exit codes: 0 finished, 3 escalated, 4 suspended, 5 awaiting approval, 6 gave up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import httpx

from agents.campaign import CampaignGoal, ScriptedPlanner, campaign_tools
from harness.journal import Journal
from harness.runtime import CONFIGS, Runtime, approve, resolve

EXIT = {"finished": 0, "escalated": 3, "suspended": 4, "awaiting_approval": 5, "gave_up": 6}


def die_at(spec: str):
    """Crash hook that ends the process on the spot: no unwinding, no cleanup."""
    point, nth = spec.split(":")
    seen = 0

    def hook(p: str, step: int) -> None:
        nonlocal seen
        if p == point:
            seen += 1
            if seen == int(nth):
                print(f"dying at {p} (step {step})", file=sys.stderr, flush=True)
                os._exit(137)

    return hook


def cmd_run(args) -> int:
    config = CONFIGS[args.config]
    if args.approvals:
        config = replace(config, approvals=True)
    Path(args.journal).parent.mkdir(parents=True, exist_ok=True)
    with Journal(args.journal) as journal, \
            httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        seed = args.seed
        if seed is None:  # a re-sampled planner per incarnation
            seed = journal.load_state(args.run_id).incarnations
        runtime = Runtime(journal, campaign_tools(args.conditional_writes), client, config,
                          crash_hook=die_at(args.die_at) if args.die_at else None,
                          backoff=args.backoff)
        print("READY", flush=True)
        outcome = runtime.run(args.run_id, CampaignGoal(), ScriptedPlanner(seed, args.temperature))
    print(json.dumps({"status": outcome.status, **outcome.detail}), flush=True)
    return EXIT[outcome.status]


def cmd_resolve(args) -> int:
    with Journal(args.journal) as journal:
        if args.from_audit:
            from chaos.driver import audit_operator

            escalation = journal.load_state(args.run_id).status
            with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
                applied, value = audit_operator(client, args.run_id, escalation)
        else:
            applied, value = args.applied == "yes", json.loads(args.value) if args.value else None
        resolve(journal, args.run_id, applied, value)
    print(f"resolved: applied={applied}")
    return 0


def cmd_approve(args) -> int:
    with Journal(args.journal) as journal:
        status = journal.load_state(args.run_id).status
        if status["status"] != "awaiting_approval":
            print(f"nothing to approve (status: {status['status']})", file=sys.stderr)
            return 1
        print(f"approving step {status['step']}: {status['tool']} {json.dumps(status['args'])}")
        approve(journal, args.run_id, status["step"], status["args_hash"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--journal", default="runs/journal.db")
    parser.add_argument("--base-url", default="http://127.0.0.1:8700")
    parser.add_argument("--timeout", type=float, default=5.0)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--config", choices=sorted(CONFIGS), default="harness")
    run.add_argument("--seed", type=int, help="planner seed (default: the incarnation number)")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--die-at", metavar="POINT:NTH", help="hard-exit at the nth pass of a crash point")
    run.add_argument("--conditional-writes", action="store_true")
    run.add_argument("--approvals", action="store_true")
    run.add_argument("--backoff", type=float, default=0.05)
    run.set_defaults(fn=cmd_run)

    res = sub.add_parser("resolve")
    res.add_argument("--run-id", required=True)
    group = res.add_mutually_exclusive_group(required=True)
    group.add_argument("--applied", choices=("yes", "no"))
    group.add_argument("--from-audit", action="store_true", help="check the platform's audit log")
    res.add_argument("--value", help="JSON result to record when --applied yes")
    res.set_defaults(fn=cmd_resolve)

    app = sub.add_parser("approve")
    app.add_argument("--run-id", required=True)
    app.set_defaults(fn=cmd_approve)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
