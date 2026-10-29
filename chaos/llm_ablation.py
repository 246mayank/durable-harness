"""Real-LLM ablation: an LLM planner in place of the scripted one.

    python -m chaos.llm_ablation --scenarios 20                                   # Gemini
    python -m chaos.llm_ablation --provider ollama --scenarios 50 --control       # local model
    python -m chaos.llm_ablation --provider ollama --scenarios 50 --control --parallel

With --parallel the model may return several tool calls per reply, and the
harness journals them as one decision of several calls; the results go to
their own file (llm_parallel.md, llm_local_parallel.md).

Each scenario crashes once, after the result of a step between the first ad
and the email, with light lost-ack faults. keys_only re-plans from the goal
after the crash; the harness resumes from its journal. With --control, the
harness also runs each scenario with no faults and no crashes: any
"duplicate" there is the model's own planning mistake, which separates
model mistakes from recovery failures in aggregate. Run by run, a step that
took effect twice is a recovery failure; any other duplicate is a new step
the model decided. Small N, slow or rate-limited models:
a check that the scripted results carry over to real models, not a benchmark.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path

from agents.campaign import campaign_tools
from agents.llm_planner import PlannerError, gemini_factory, ollama_factory
from chaos.bench import RESULTS, _git_sha, _pct, _table, update_readme
from chaos.driver import Scenario, run_scenario
from chaos.invariants import KINDS
from harness.runtime import CONFIGS
from platform_sim.faults import FaultConfig

ABLATION = ("keys_only", "harness")
CONTROL = "harness, no faults or crashes"
PROVIDERS = {  # provider -> (default model, output file / README block, how the model is reached)
    "gemini": ("gemini-3.5-flash-lite", "llm", "Gemini function calling"),
    "ollama": ("qwen2.5:3b", "llm_local", "Ollama tool calling, running locally on CPU (a reply "
               "without a tool call ends the run, as in a standard agent loop, so goal steps the "
               "model skips count as missing)"),
}


def scenarios(n: int, seed: int, temperature: float) -> list[Scenario]:
    rng = random.Random(seed)
    return [
        Scenario(
            seed=10_000 + i,
            faults=FaultConfig(p_lost_ack=0.05, seed=rng.randrange(2**31)),
            crashes=(("after_result", rng.randint(2, 5)),),
            temperature=temperature,
        )
        for i in range(n)
    ]


def batching(events: list) -> tuple[int, int]:
    """(tool calls the model decided in multi-call replies, all tool calls it decided), from the journal."""
    batched = total = 0
    for e in events:
        if e.kind == "decision" and not e.payload.get("finish"):
            n = len(e.payload.get("calls") or [e.payload])
            total += n
            batched += n if n > 1 else 0
    return batched, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    parser.add_argument("--model", help="default: gemini-3.5-flash-lite / qwen2.5:3b")
    parser.add_argument("--scenarios", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-interval", type=float, default=4.0, help="Gemini only: seconds between requests")
    parser.add_argument("--control", action="store_true", help="also run the harness with no faults or crashes")
    parser.add_argument("--parallel", action="store_true", help="let the model return several calls per reply")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    default_model, block, via = PROVIDERS[args.provider]
    model = args.model or default_model
    block += "_parallel" if args.parallel else ""

    tools = campaign_tools()
    factory = (gemini_factory(tools, model, min_interval=args.min_interval, parallel=args.parallel)
               if args.provider == "gemini" else ollama_factory(tools, model, parallel=args.parallel))
    configs = (*ABLATION, CONTROL) if args.control else ABLATION
    rows, skipped = [], []
    stopped = None
    started = time.perf_counter()
    for scenario in scenarios(args.scenarios, args.seed, args.temperature):
        group = []
        try:
            for name in ABLATION:
                group.append(run_scenario(scenario, CONFIGS[name], tools=tools, planner_factory=factory,
                                          keep_events=True))
            if args.control:
                calm = replace(scenario, faults=FaultConfig(), crashes=())
                report = run_scenario(calm, CONFIGS["harness"], tools=tools, planner_factory=factory,
                                      keep_events=True)
                report.config = CONTROL
                group.append(report)
        except PlannerError as exc:  # the model never made a tool call: skip, and say so
            skipped.append(scenario.seed)
            print(f"scenario {scenario.seed} skipped: {exc}", flush=True)
            continue
        except RuntimeError as exc:  # quota or availability: keep only complete groups
            stopped = f"stopped after {len(rows) // len(configs)} scenarios: {str(exc)[:120]}"
            print(stopped, flush=True)
            break
        rows += group
        for report in group:
            print(f"scenario {scenario.seed} {report.config}: {report.status}, "
                  f"dups {report.check.duplicates}, steps applied twice {report.check.repeated_steps}, "
                  f"planner calls {report.planner_calls}", flush=True)
    if not rows:
        raise SystemExit(stopped or "no runs completed")
    n = len(rows) // len(configs)

    table = []
    for name in configs:
        rs = [r for r in rows if r.config == name]
        replays = CONFIGS["harness" if name == CONTROL else name].replay
        table.append([
            f"`{name}`", len(rs), _pct(sum(r.finished for r in rs), len(rs)),
            _pct(sum(1 for r in rs if r.check.duplicate_total), len(rs)),
            *(sum(r.check.duplicates[k] for r in rs) for k in KINDS),
            sum(r.check.repeated_steps for r in rs) if replays else "n/a",
            sum(r.check.missing_total for r in rs if r.finished),
            f"{sum(r.planner_calls for r in rs) / len(rs):.1f}",
        ])
        if args.parallel:  # only a replayed journal keeps every decision
            batched, total = map(sum, zip(*(batching(r.events) for r in rs), strict=True))
            table[-1].append(_pct(batched, total) if replays else "n/a")
    command = (f"python -m chaos.llm_ablation --provider {args.provider} --model {model} "
               f"--scenarios {args.scenarios} --seed {args.seed}" + (" --control" if args.control else "")
               + (" --parallel" if args.parallel else ""))
    notes = []
    if stopped:
        notes.append(stopped)
    if skipped:
        notes.append(f"{len(skipped)} skipped because the model made no tool call: {skipped}")
    text = "\n".join([
        "# Real-LLM ablation (small N)" + (", parallel tool calls" if args.parallel else ""),
        "",
        f"Generated by `{command}` at commit `{_git_sha()}`. Do not edit by hand.",
        "",
        f"{n} scenarios x {len(configs)} configs completed" + (f" ({'; '.join(notes)})." if notes else "."),
        "",
        f"Planner: `{model}` via {via}, at temperature {args.temperature}, deciding from the goal and "
        "observed history only"
        + (". The model may return several tool calls per reply, and is told it may when they don't "
           "depend on each other; the harness journals such a reply as one decision of several calls"
           if args.parallel else "")
        + ". Every scenario crashes once after a step between the first ad and "
        "the email (in-process crash injection), with 5% lost acks. LLM output is not reproducible "
        "run to run, and N is small: a check that the scripted-planner results carry over, not a "
        "benchmark.",
        "",
        _table(["config", "runs", "finished", "runs with a duplicate", *(f"dup {k}" for k in KINDS),
                "steps applied twice", "missing (finished runs)", "planner calls / run",
                *(["tool calls decided in multi-call replies"] if args.parallel else [])], table),
        "",
        "A duplicate here can also be the model's own planning mistake (say, a fourth ad). "
        "`steps applied twice` tells the two apart: it counts effects from a step that had already "
        "taken effect, which only a re-send can cause. Any other duplicate is a new step the model "
        "decided. Without a journal to replay, `keys_only` "
        "numbers its steps from 0 again after a crash, so the column doesn't apply to it."
        + (f" The `{CONTROL}` row measures the model's mistakes alone: same scenarios, same model, "
           "nothing to recover from." if args.control else ""),
        "",
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{block}.md").write_text(text, encoding="utf-8", newline="\n")
    if args.out.resolve() == RESULTS:
        update_readme(block, text.split("\n", 2)[2].strip())
    print(f"{len(rows)} runs in {time.perf_counter() - started:.0f}s -> {args.out / f'{block}.md'}")


if __name__ == "__main__":
    main()
