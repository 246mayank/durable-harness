"""Benchmark: every seeded scenario against every config; writes results/results.md.

    python -m chaos.bench --scenarios 500 --seed 0

The output files are generated; never edit them by hand.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean

from agents.campaign import campaign_tools
from chaos.driver import RunReport, Scenario, run_scenario
from chaos.invariants import KINDS
from chaos.scenarios import generate
from harness.runtime import CONFIGS

RESULTS = Path(__file__).resolve().parent.parent / "results"
README = RESULTS.parent / "README.md"

# name -> (runtime config, conditional writes?)
BENCH = {
    "naive": (CONFIGS["naive"], False),
    "naive_retry": (CONFIGS["naive_retry"], False),
    "keys_only": (CONFIGS["keys_only"], False),
    "harness": (CONFIGS["harness"], False),
    "harness_cas": (replace(CONFIGS["harness"], name="harness_cas"), True),
}

DESCRIPTIONS = {
    "naive": "no replay, no keys, no retries: errors go back to the planner",
    "naive_retry": "naive + 5 blind retries",
    "keys_only": "naive_retry + idempotency keys hashed from the arguments",
    "harness": "journal replay + positional keys + effect-class recovery",
    "harness_cas": "harness + conditional writes (`expected_total`) on allocations",
}


def _run(task: tuple[Scenario, str, str]) -> RunReport:
    scenario, name, durability = task
    config, conditional = BENCH[name]
    return run_scenario(scenario, config, tools=campaign_tools(conditional), durability=durability)


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=RESULTS.parent, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "harness", "agents",
                                "platform_sim", "chaos"], capture_output=True, text=True,
                               cwd=RESULTS.parent).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def summarize(reports: list[RunReport]) -> dict:
    n = len(reports)
    finished = [r for r in reports if r.finished]
    return {
        "runs": n,
        "finished": len(finished),
        "dup_runs": sum(1 for r in reports if r.check.duplicate_total),
        "dups": sum(r.check.duplicate_total for r in reports),
        "excess": sum(r.check.excess_spend for r in reports),
        "missing": sum(r.check.missing_total for r in finished),
        "escalations": sum(r.escalations for r in reports),
        "esc_runs": sum(1 for r in reports if r.escalations),
        "planner_calls": mean(r.planner_calls for r in reports) if reports else 0,
        "crashes": sum(r.crashes for r in reports),
        "conflicts": sum(r.conflicts for r in reports),
        "by_kind": {k: sum(r.check.duplicates[k] for r in reports) for k in KINDS},
    }


def summary_table(stats: dict[str, dict]) -> str:
    return _table(
        ["config", "finished", "runs with a duplicate", "duplicate effects", "excess spend",
         "missing (finished runs)", "escalations", "planner calls / run", "crashes injected", "409s"],
        [[f"`{name}`", _pct(s["finished"], s["runs"]), _pct(s["dup_runs"], s["runs"]), s["dups"],
          f"${s['excess']:,}", s["missing"], f"{s['escalations']} in {s['esc_runs']} runs",
          f"{s['planner_calls']:.1f}", s["crashes"], s["conflicts"]]
         for name, s in stats.items()],
    )


def kind_table(stats: dict[str, dict]) -> str:
    return _table(["config", *KINDS],
                  [[f"`{name}`", *(s["by_kind"][k] for k in KINDS)] for name, s in stats.items()])


def update_readme(section: str, body: str) -> None:
    """Rewrite the generated block between `<!-- section:start -->` and `<!-- section:end -->`."""
    start, end = f"<!-- {section}:start -->", f"<!-- {section}:end -->"
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    README.write_text(f"{head}{start}\n{body}\n{end}{tail}", encoding="utf-8", newline="\n")


def render(by_config: dict[str, list[RunReport]], scenarios: list[Scenario], args) -> str:
    n = len(scenarios)
    stats = {name: summarize(rs) for name, rs in by_config.items()}
    out = [
        "# Results",
        "",
        f"Generated by `python -m chaos.bench --scenarios {n} --seed {args.seed}` at commit "
        f"`{_git_sha()}`. Journal durability: `synchronous={args.durability.upper()}`.",
        "Do not edit by hand.",
        "",
        f"Each of the **{n} seeded scenarios** fixes a fault stream (429 / 5xx-before-commit / "
        "commit-then-504, each p in [0, 0.15]), a crash schedule (0-3 in-process crashes at "
        "named crash points) and a planner temperature (0, 0.3 or 0.7). Every scenario is "
        "replayed against every config on a fresh simulated platform. All numbers are counted "
        "from the platform's own audit log, not from what the agent believes.",
        "",
        "## Summary",
        "",
        summary_table(stats),
        "",
        "A clean run makes 9 planner calls and leaves 1 campaign, 3 ads, 1 email batch and "
        "2 allocations ($700).",
        "",
        "## Duplicate effects by kind",
        "",
        kind_table(stats),
        "",
        "`keys_only` dedupes campaigns and ads only while a re-planned run reproduces the same "
        "arguments; email batches and allocations have no key support at all.",
        "",
        "## By planner temperature",
        "",
    ]
    temps = sorted({s.temperature for s in scenarios})
    rows = []
    for name, reports in by_config.items():
        row = [f"`{name}`"]
        for t in temps:
            sub = [r for r in reports if r.temperature == t]
            row.append(f"{sum(1 for r in sub if r.check.duplicate_total)}/{len(sub)} runs, "
                       f"{sum(r.check.duplicates['ad'] + r.check.duplicates['campaign'] for r in sub)} keyed")
        rows.append(row)
    out += [
        _table(["config", *(f"temp {t}" for t in temps)], rows),
        "",
        "Cells: runs with any duplicate / runs, and duplicates of *keyed* effects (campaigns + ads). "
        "Above temperature 0, a restarted planner writes new ad copy, so argument-hash keys stop "
        "matching.",
        "",
        "## Faults only vs. with crashes",
        "",
    ]
    rows = []
    for name, reports in by_config.items():
        calm = [r for r in reports if not r.crashes]
        crashed = [r for r in reports if r.crashes]
        rows.append([f"`{name}`",
                     f"{sum(r.check.duplicate_total for r in calm)} in {len(calm)} runs",
                     f"{sum(r.check.duplicate_total for r in crashed)} in {len(crashed)} runs"])
    out += [
        _table(["config", "duplicates, no crash", "duplicates, ≥1 crash"], rows),
        "",
        "## Configs",
        "",
        *(f"- `{name}`: {DESCRIPTIONS[name]}" for name in by_config),
        "",
        "Escalations are resolved by an operator that reads the platform's audit log, which "
        "models a person checking the dashboard. Each one is a human interruption: a cost, "
        "not a free fix.",
        "",
    ]
    return "\n".join(out)


def render_resume_numbers(by_config: dict[str, list[RunReport]], n: int) -> str:
    s = {name: summarize(rs) for name, rs in by_config.items()}
    return "\n".join([
        "# Numbers for the resume bullets",
        "",
        "Generated with results.md; copy numbers from here only.",
        "",
        f"- N (seeded scenarios per config): **{n}**",
        f"- harness duplicate side effects: **{s['harness']['dups']}** "
        f"(excess spend ${s['harness']['excess']:,})",
        f"- X, retrying agent loop (`naive_retry`) duplicate side effects: **{s['naive_retry']['dups']}** "
        f"across {s['naive_retry']['dup_runs']} runs (excess spend ${s['naive_retry']['excess']:,})",
        f"- Y, idempotency keys alone (`keys_only`) duplicate side effects: **{s['keys_only']['dups']}** "
        f"across {s['keys_only']['dup_runs']} runs",
        f"  - of which on *keyed* endpoints (campaigns + ads), caused by a re-planned step producing "
        f"new arguments: **{s['keys_only']['by_kind']['campaign'] + s['keys_only']['by_kind']['ad']}**",
        f"  - of which on endpoints with no key support (email batches + allocations): "
        f"**{s['keys_only']['by_kind']['email_batch'] + s['keys_only']['by_kind']['allocation']}**",
        f"- harness escalations to a human: **{s['harness']['escalations']}** "
        f"in {s['harness']['esc_runs']} runs; with conditional writes: "
        f"**{s['harness_cas']['escalations']}**" if "harness_cas" in s else "",
        "",
        "Wording rules: say \"simulated platform\", \"seeded scenarios\" and \"built and evaluated\". "
        "The numbers above come from the scripted planner with in-process crash injection; the "
        "hard-kill numbers (separate OS processes) are in sigkill.md and the small real-LLM run "
        "is in llm.md, so cite those files for those claims. Never \"exactly-once\", "
        "\"production\" or \"deployed\".",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--durability", choices=("full", "normal"), default="full")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    scenarios = generate(args.scenarios, args.seed)
    tasks = [(s, name, args.durability) for name in BENCH for s in scenarios]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(_run, tasks, chunksize=8))
    by_config: dict[str, list[RunReport]] = defaultdict(list)
    for report in reports:
        by_config[report.config].append(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(render(by_config, scenarios, args), encoding="utf-8",
                                         newline="\n")
    (args.out / "resume_numbers.md").write_text(render_resume_numbers(by_config, len(scenarios)),
                                                encoding="utf-8", newline="\n")
    if args.out.resolve() == RESULTS:
        stats = {name: summarize(rs) for name, rs in by_config.items()}
        update_readme("results", "\n".join([
            f"{len(scenarios)} seeded scenarios x {len(BENCH)} configs, in-process crash injection, "
            f"commit `{_git_sha()}`. Generated by `python -m chaos.bench`; full tables in "
            "[results/results.md](results/results.md).",
            "",
            summary_table(stats),
            "",
            "Duplicate effects by kind:",
            "",
            kind_table(stats),
        ]))
    with open(args.out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["config", "seed", "temperature", "status", "incarnations", "crashes",
                         "escalations", "planner_calls", "conflicts", *(f"dup_{k}" for k in KINDS),
                         *(f"missing_{k}" for k in KINDS), "spend"])
        for r in reports:
            writer.writerow([r.config, r.seed, r.temperature, r.status, r.incarnations, r.crashes,
                             r.escalations, r.planner_calls, r.conflicts,
                             *(r.check.duplicates[k] for k in KINDS),
                             *(r.check.missing[k] for k in KINDS), r.check.spend])
    print(f"{len(reports)} runs in {time.perf_counter() - started:.0f}s -> {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
