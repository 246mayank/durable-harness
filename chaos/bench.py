"""Benchmark: every seeded scenario against every config; writes results/results.md.

    python -m chaos.bench --scenarios 500 --zombies 200 --seed 0

Three experiments share one process pool:

- main: the seeded scenarios (faults incl. delayed delivery, crashes, temperature)
  against the baselines and the harness;
- delayed delivery: the same scenarios against the harness with deadlines off;
- zombies: scenarios whose first incarnation hangs and comes back after the run
  is over, against the harness with fencing and/or deadlines off.

The output files are generated; never edit them by hand.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean

from agents.campaign import campaign_tools
from chaos.driver import RunReport, Scenario, run_scenario
from chaos.invariants import KINDS
from chaos.scenarios import generate, generate_zombies
from harness.runtime import CONFIGS

RESULTS = Path(__file__).resolve().parent.parent / "results"
README = RESULTS.parent / "README.md"

_H = CONFIGS["harness"]
# name -> (runtime config, conditional writes?)
VARIANTS = {
    "naive": (CONFIGS["naive"], False),
    "naive_retry": (CONFIGS["naive_retry"], False),
    "keys_only": (CONFIGS["keys_only"], False),
    "harness": (_H, False),
    "harness_cas": (replace(_H, name="harness_cas"), True),
    "harness_nodeadline": (replace(_H, name="harness_nodeadline", deadline=None), False),
    "harness_cas_nodeadline": (replace(_H, name="harness_cas_nodeadline", deadline=None), True),
    "harness_nofence": (replace(_H, name="harness_nofence", fencing=False), False),
    "harness_neither": (replace(_H, name="harness_neither", fencing=False, deadline=None), False),
}
BENCH = ["naive", "naive_retry", "keys_only", "harness", "harness_cas"]
NO_DEADLINE = ["harness_nodeadline", "harness_cas_nodeadline"]  # main scenarios, deadlines off
ZOMBIE = ["harness_neither", "harness_nofence", "harness_nodeadline", "harness", "harness_cas"]

DESCRIPTIONS = {
    "naive": "no replay, no keys, no retries: errors go back to the planner",
    "naive_retry": "naive + 5 blind retries",
    "keys_only": "naive_retry + idempotency keys hashed from the arguments",
    "harness": "journal replay + positional keys + effect-class recovery + deadlines + fencing",
    "harness_cas": "harness + conditional writes (`expected_total`) on allocations",
    "harness_nodeadline": "harness without deadlines: lookups and the operator don't wait",
    "harness_cas_nodeadline": "harness_cas without deadlines",
    "harness_nofence": "harness without fencing: a woken incarnation can keep journaling",
    "harness_neither": "harness without fencing or deadlines",
}


def _run(task: tuple[Scenario, str, str, str]) -> tuple[str, RunReport]:
    scenario, name, durability, experiment = task
    config, conditional = VARIANTS[name]
    return experiment, run_scenario(scenario, config, tools=campaign_tools(conditional),
                                    durability=durability)


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


def upper_bound(n: int, confidence: float = 0.95) -> float:
    """One-sided upper bound on a rate after 0 events in n trials (exact; ~3/n)."""
    return 1 - (1 - confidence) ** (1 / n)


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
        "froze": sum(r.froze for r in reports),
        "fenced": sum(r.zombie == "fenced" for r in reports),
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


def bounds(stats: dict[str, dict]) -> str:
    lines = []
    for name, s in stats.items():
        if s["runs"] and not s["dup_runs"]:
            lines.append(f"- `{name}`: 0 of {s['runs']} runs had a duplicate, so with 95% confidence "
                         f"the per-run rate under this scenario mix is below "
                         f"{100 * upper_bound(s['runs']):.2f}%.")
    return "\n".join(lines)


def escalation_table(reports: list[RunReport]) -> str:
    causes: Counter = Counter()
    for r in reports:
        causes.update(r.escalation_causes)
    names = sorted({c for c, _ in causes}, key=lambda c: -sum(n for (k, _), n in causes.items() if k == c))
    rows = [[c, causes[(c, "not applied")], causes[(c, "applied")],
             sum(n for (k, _), n in causes.items() if k == c)] for c in names]
    total = [sum(causes[(c, v)] for c in names) for v in ("not applied", "applied")]
    rows.append(["**total**", *total, sum(total)])
    return _table(["cause", "had not happened", "had happened", "escalations"], rows)


def escalation_summary(reports: list[RunReport], cas_escalations: int) -> str:
    causes: Counter = Counter()
    for r in reports:
        causes.update(r.escalation_causes)
    total = sum(causes.values())
    if not total:
        return "The harness did not escalate in this scenario mix."
    false_alarms = sum(n for (_, verdict), n in causes.items() if verdict == "not applied")
    replies = sum(n for (cause, _), n in causes.items() if cause.startswith("ambiguous reply"))
    return (f"{false_alarms} of {total} escalations ({100 * false_alarms / total:.0f}%) were false "
            f"alarms: the call had never happened. {replies} ({100 * replies / total:.0f}%) came from "
            "an ambiguous reply during a live call, which only the platform can disambiguate: with "
            f"conditional writes (`harness_cas`) there are {cas_escalations}. A restart after a crash "
            "between the intent and the send can't be avoided by journaling more, because the intent "
            "is already the last write before the send.")


def zombie_table(stats: dict[str, dict]) -> str:
    return _table(
        ["config", "zombies (hung, then woke)", "zombie stopped by its fence", "runs with a duplicate",
         "duplicate effects", "excess spend", "finished"],
        [[f"`{name}`", s["froze"], s["fenced"], _pct(s["dup_runs"], s["runs"]), s["dups"],
          f"${s['excess']:,}", _pct(s["finished"], s["runs"])] for name, s in stats.items()],
    )


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


def render_inflight(main: dict[str, list[RunReport]], zombies: dict[str, list[RunReport]],
                    n_zombies: int, heading: str = "###") -> str:
    """The delayed-delivery and zombie sections (also the README's `inflight` block)."""
    delayed = {name: summarize(main[name]) for name in ("harness", "harness_nodeadline", "harness_cas",
                                                        "harness_cas_nodeadline")}
    return "\n".join([
        f"{heading} Delayed delivery: deadlines on and off",
        "",
        "Same scenarios as above; some requests stay in flight after their 504 and land later. "
        "Without deadlines, a lookup that runs before the original lands finds nothing, and the "
        "re-send makes a second email batch. Conditional writes are immune: the late original "
        "fails its `expected_total` check.",
        "",
        _table(["config", "runs with a duplicate", "duplicate effects", "email batches", "allocations",
                "escalations"],
               [[f"`{name}`", _pct(s["dup_runs"], s["runs"]), s["dups"], s["by_kind"]["email_batch"],
                 s["by_kind"]["allocation"], s["escalations"]] for name, s in delayed.items()]),
        "",
        f"{heading} Zombie incarnations: fencing and deadlines on and off",
        "",
        f"{n_zombies} scenarios with the same fault mix and no crashes. Instead, the first "
        "incarnation hangs at a random hook point; a replacement takes the run over and finishes "
        "it; then the hung one wakes up and carries on. `before_send` (after the fence check, "
        "before the request leaves) is among the hang points.",
        "",
        zombie_table({name: summarize(rs) for name, rs in zombies.items()}),
        "",
        "Fencing stops a woken incarnation from journaling or sending again. Deadlines cover the "
        "one send it may already have been about to make: by the time it goes out, its deadline "
        "has passed, because the new owner waited it out before acting on a negative lookup or an "
        "operator's check.",
    ])


def render(main: dict[str, list[RunReport]], zombies: dict[str, list[RunReport]],
           scenarios: list[Scenario], n_zombies: int, args) -> str:
    n = len(scenarios)
    stats = {name: summarize(main[name]) for name in BENCH}
    out = [
        "# Results",
        "",
        f"Generated by `python -m chaos.bench --scenarios {n} --zombies {n_zombies} --seed {args.seed}` "
        f"at commit `{_git_sha()}`. Journal durability: `synchronous={args.durability.upper()}`.",
        "Do not edit by hand.",
        "",
        f"Each of the **{n} seeded scenarios** fixes a fault stream (429 / 5xx-before-commit / "
        "commit-then-504 / delayed delivery, each p in [0, 0.15]), a crash schedule (0-3 "
        "in-process crashes at named crash points) and a planner temperature (0, 0.3 or 0.7). "
        "Every scenario is replayed against every config on a fresh simulated platform. All "
        "numbers are counted from the platform's own audit log, after every request still in "
        "flight has landed or been dropped, not from what the agent believes.",
        "",
        "## Summary",
        "",
        summary_table(stats),
        "",
        "A clean run makes 9 planner calls and leaves 1 campaign, 3 ads, 1 email batch and "
        "2 allocations ($700).",
        "",
        bounds(stats),
        "",
        "## Duplicate effects by kind",
        "",
        kind_table(stats),
        "",
        "`keys_only` dedupes campaigns and ads only while a re-planned run reproduces the same "
        "arguments; email batches and allocations have no key support at all.",
        "",
        "## Where the harness's escalations come from",
        "",
        "Each escalation is a human check. The verdict is what the operator found in the "
        "platform's audit log.",
        "",
        escalation_table(main["harness"]),
        "",
        escalation_summary(main["harness"], stats["harness_cas"]["escalations"]),
        "",
        "## By planner temperature",
        "",
    ]
    temps = sorted({s.temperature for s in scenarios})
    rows = []
    for name in BENCH:
        row = [f"`{name}`"]
        for t in temps:
            sub = [r for r in main[name] if r.temperature == t]
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
    for name in BENCH:
        calm = [r for r in main[name] if not r.crashes]
        crashed = [r for r in main[name] if r.crashes]
        rows.append([f"`{name}`",
                     f"{sum(r.check.duplicate_total for r in calm)} in {len(calm)} runs",
                     f"{sum(r.check.duplicate_total for r in crashed)} in {len(crashed)} runs"])
    out += [
        _table(["config", "duplicates, no crash", "duplicates, ≥1 crash"], rows),
        "",
        "## Calls still in flight",
        "",
        render_inflight(main, zombies, n_zombies),
        "",
        "## Configs",
        "",
        *(f"- `{name}`: {DESCRIPTIONS[name]}" for name in VARIANTS),
        "",
        "Escalations are resolved by an operator that waits out the call's deadline, lets any "
        "request still in flight land, and then reads the platform's audit log: a person checking "
        "the dashboard. Each one is a human interruption: a cost, not a free fix.",
        "",
    ]
    return "\n".join(out)


def render_resume_numbers(main: dict[str, list[RunReport]], zombies: dict[str, list[RunReport]],
                          n: int, n_zombies: int) -> str:
    s = {name: summarize(rs) for name, rs in main.items()}
    z = {name: summarize(rs) for name, rs in zombies.items()}
    return "\n".join([
        "# Numbers for the resume bullets",
        "",
        "Generated with results.md; copy numbers from here only.",
        "",
        f"- N (seeded scenarios per config): **{n}**",
        f"- harness duplicate side effects: **{s['harness']['dups']}** "
        f"(excess spend ${s['harness']['excess']:,}); 95% upper bound on the per-run duplicate rate: "
        f"**{100 * upper_bound(n):.2f}%**",
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
        f"**{s['harness_cas']['escalations']}**",
        f"- delayed delivery: harness without deadlines: **{s['harness_nodeadline']['dups']}** duplicate "
        f"effects in {s['harness_nodeadline']['dup_runs']} runs; with deadlines: "
        f"**{s['harness']['dups']}**",
        f"- zombie incarnations ({n_zombies} scenarios, {z['harness']['froze']} zombies): no fencing or "
        f"deadlines **{z['harness_neither']['dups']}** duplicate effects; fencing only "
        f"**{z['harness_nodeadline']['dups']}**; both **{z['harness']['dups']}**",
        "",
        "Wording rules: say \"simulated platform\", \"seeded scenarios\" and \"built and evaluated\". "
        "The numbers above come from the scripted planner with in-process crash injection; the "
        "hard-kill numbers (separate OS processes) are in sigkill.md, the fault-rate sweep in "
        "sweep.md, and the small real-LLM run in llm.md, so cite those files for those claims. "
        "Never \"exactly-once\", \"production\" or \"deployed\".",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--zombies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--durability", choices=("full", "normal"), default="full")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    scenarios = generate(args.scenarios, args.seed)
    zombie_scenarios = generate_zombies(args.zombies, args.seed)
    tasks = [(s, name, args.durability, "main") for name in BENCH + NO_DEADLINE for s in scenarios]
    tasks += [(s, name, args.durability, "zombie") for name in ZOMBIE for s in zombie_scenarios]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run, tasks, chunksize=8))
    main_runs: dict[str, list[RunReport]] = defaultdict(list)
    zombie_runs: dict[str, list[RunReport]] = defaultdict(list)
    for experiment, report in results:
        (main_runs if experiment == "main" else zombie_runs)[report.config].append(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(render(main_runs, zombie_runs, scenarios, args.zombies, args),
                                         encoding="utf-8", newline="\n")
    (args.out / "resume_numbers.md").write_text(
        render_resume_numbers(main_runs, zombie_runs, len(scenarios), args.zombies),
        encoding="utf-8", newline="\n")
    if args.out.resolve() == RESULTS:
        stats = {name: summarize(main_runs[name]) for name in BENCH}
        update_readme("results", "\n".join([
            f"{len(scenarios)} seeded scenarios x {len(BENCH)} configs, in-process crash injection, "
            f"commit `{_git_sha()}`. Generated by `python -m chaos.bench`; full tables in "
            "[results/results.md](results/results.md).",
            "",
            summary_table(stats),
            "",
            bounds(stats),
            "",
            "Duplicate effects by kind:",
            "",
            kind_table(stats),
        ]))
        update_readme("inflight", render_inflight(main_runs, zombie_runs, args.zombies, heading="####"))
        update_readme("escalations", escalation_table(main_runs["harness"]))
    with open(args.out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["experiment", "config", "seed", "temperature", "status", "incarnations",
                         "crashes", "froze", "zombie", "escalations", "planner_calls", "conflicts",
                         *(f"dup_{k}" for k in KINDS), *(f"missing_{k}" for k in KINDS), "spend"])
        for experiment, r in results:
            writer.writerow([experiment, r.config, r.seed, r.temperature, r.status, r.incarnations,
                             r.crashes, int(r.froze), r.zombie or "", r.escalations, r.planner_calls,
                             r.conflicts, *(r.check.duplicates[k] for k in KINDS),
                             *(r.check.missing[k] for k in KINDS), r.check.spend])
    print(f"{len(results)} runs in {time.perf_counter() - started:.0f}s -> {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
