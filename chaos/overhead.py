"""What durability costs: the journal's time per tool call.

    python -m chaos.overhead --runs 300

Times clean campaign runs (scripted planner, in-process platform, no faults,
no crashes) under the harness with its journal in memory and on disk at each
durability level, in a shuffled order each round. Also times bare journal
appends. The in-process platform answers in about a millisecond, so this
isolates the journal's cost; a real tool call over the network takes far
longer. Wall-clock numbers depend on the machine and its disk, so unlike
results.md this file is not reproducible bit for bit.
"""

from __future__ import annotations

import argparse
import platform
import random
import sqlite3
import statistics
import tempfile
import time
import warnings
from pathlib import Path

from agents.campaign import CampaignGoal, ScriptedPlanner, campaign_tools
from chaos.bench import RESULTS, _git_sha, _table, update_readme
from harness.clock import VirtualClock
from harness.journal import Journal
from harness.runtime import CONFIGS, Runtime
from platform_sim.app import create_app
from platform_sim.faults import FaultConfig

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
from fastapi.testclient import TestClient  # noqa: E402

MODES = ("memory", "normal", "send", "full")
LABELS = {"memory": "in memory (no durability)", "normal": "`normal`: no fsync",
          "send": "`send` (default): fsync before each send", "full": "`full`: fsync every write"}
TOOL_LATENCY_MS = 100  # a fast remote API, for the projection only


def _journal(mode: str, directory: Path, name: str) -> Journal:
    return Journal(":memory:") if mode == "memory" else Journal(directory / f"{name}-{mode}.db", mode)


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _p(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def time_runs(runs: int, directory: Path, warmup: int = 10) -> dict[str, dict]:
    """Per mode: seconds per tool call for each run, and journal writes and fsyncs in total."""
    tools, goal, config, clock = campaign_tools(), CampaignGoal(), CONFIGS["harness"], VirtualClock()
    journals = {mode: _journal(mode, directory, "runs") for mode in MODES}
    rng = random.Random(0)
    out = {mode: {"per_call": [], "writes": 0, "fsyncs": 0, "calls": 0} for mode in MODES}
    try:
        with TestClient(create_app(faults=FaultConfig(), clock=clock)) as client:
            for i in range(warmup + runs):
                for mode in rng.sample(MODES, len(MODES)):  # shuffled, so no mode always follows another
                    journal, run_id = journals[mode], f"{mode}-{i}"
                    runtime = Runtime(journal, tools, client, config, clock=clock)
                    fsyncs = journal.fsyncs
                    started = time.perf_counter()
                    outcome = runtime.run(run_id, goal, ScriptedPlanner(seed=i))
                    elapsed = time.perf_counter() - started
                    if outcome.status != "finished":
                        raise RuntimeError(f"{run_id}: {outcome}")
                    events = journal.events(run_id)
                    calls = sum(1 for e in events if e.kind == "intent")
                    if i >= warmup:
                        out[mode]["per_call"].append(elapsed / calls)
                        out[mode]["writes"] += len(events)
                        out[mode]["fsyncs"] += journal.fsyncs - fsyncs
                        out[mode]["calls"] += calls
    finally:
        for journal in journals.values():
            journal.close()
    return out


def time_appends(n: int, directory: Path) -> dict[str, list[float]]:
    """Seconds per committed intent append, per mode (under `send` an intent is a write that fsyncs)."""
    payload = {"tool": "create_ad", "args": {"campaign_id": "cmp_1", "headline": "h" * 40, "body": "b" * 120},
               "key": "k" * 64, "effect": "idempotent", "precondition": None, "deadline": 1_000_000.5}
    out = {}
    for mode in MODES:
        with _journal(mode, directory, "appends") as j:
            times = []
            for _ in range(n):
                started = time.perf_counter()
                j.append("bench", "intent", None, payload)
                times.append(time.perf_counter() - started)
            out[mode] = times
    return out


def render(runs: dict[str, dict], appends: dict[str, list[float]], n_runs: int, n_appends: int,
           directory: Path) -> str:
    base = statistics.median(runs["memory"]["per_call"])
    rows = []
    for mode in MODES:
        per_call = runs[mode]["per_call"]
        median = statistics.median(per_call)
        rows.append([LABELS[mode], _ms(median), _ms(_p(per_call, 0.95)), _ms(median - base),
                     f"{runs[mode]['writes'] / runs[mode]['calls']:.1f}",
                     f"{runs[mode]['fsyncs'] / runs[mode]['calls']:.1f}"])
    append_rows = [[LABELS[m], _ms(statistics.median(appends[m])), _ms(_p(appends[m], 0.99))] for m in MODES]
    added = {m: statistics.median(runs[m]["per_call"]) - base for m in ("send", "full")}
    fsyncs = {m: runs[m]["fsyncs"] / runs[m]["calls"] for m in ("send", "full")}
    machine = (f"{platform.system()} {platform.release()}, {platform.machine()}, Python "
               f"{platform.python_version()}, SQLite {sqlite3.sqlite_version}; journal files in "
               f"`{directory.anchor or directory}` (system temp directory)")
    return "\n".join([
        "# What durability costs",
        "",
        f"Generated by `python -m chaos.overhead --runs {n_runs}` at commit `{_git_sha()}`. "
        "Do not edit by hand. Wall-clock timings: they vary by machine and from run to run.",
        "",
        f"Machine: {machine}.",
        "",
        f"{n_runs} clean campaign runs per row (8 tool calls each, scripted planner, in-process platform, "
        "no faults or crashes), the rows run in a shuffled order each round. Time per tool call covers the whole "
        "step: planner, journal writes, fence checks and the in-process HTTP call.",
        "",
        _table(["harness journal", "ms per tool call (median)", "p95", "added over in-memory",
                "journal writes per tool call", "fsyncs per tool call"], rows),
        "",
        f"Bare intent appends, {n_appends} per row, each its own committed transaction:",
        "",
        _table(["journal", "ms per append (median)", "p99"], append_rows),
        "",
        f"Fsyncing every write (`full`) costs {fsyncs['full']:.1f} fsyncs and {_ms(added['full'])} ms per "
        f"tool call here. Fsyncing only before each send (`send`) costs {fsyncs['send']:.1f} and "
        f"{_ms(added['send'])} ms, with the same guarantee that every call that went out has its intent "
        f"and deadline on disk. Against a tool that takes {TOOL_LATENCY_MS} ms (a fast remote API), that "
        f"is {added['send'] * 1000 / TOOL_LATENCY_MS:.0%} on top of each call instead of "
        f"{added['full'] * 1000 / TOOL_LATENCY_MS:.0%}: a projection from the numbers above, not a "
        "measurement. An LLM decision takes far longer than either.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--appends", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        runs = time_runs(args.runs, directory)
        appends = time_appends(args.appends, directory)
        text = render(runs, appends, args.runs, args.appends, directory)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overhead.md").write_text(text, encoding="utf-8", newline="\n")
    if args.out.resolve() == RESULTS:
        update_readme("overhead", text.split("\n", 2)[2].strip())
    print(text)


if __name__ == "__main__":
    main()
