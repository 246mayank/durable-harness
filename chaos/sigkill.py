"""Hard-kill runner: the agent and the platform as separate OS processes.

    python -m chaos.sigkill --scenarios 40 --seed 0

The platform runs as one uvicorn server. Every agent incarnation is a fresh
`python -m agents run` process that the runner kills from outside after a
random delay (Popen.kill: SIGKILL on POSIX, TerminateProcess on Windows). A
kill can land mid-request or mid-commit, and nothing in the agent gets to
clean up. Kill timing is real wall-clock time, so the output is not
bit-for-bit reproducible the way results.md is.
"""

from __future__ import annotations

import argparse
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from agents.campaign import CampaignGoal
from chaos.bench import RESULTS, _git_sha, _pct, _table, update_readme
from chaos.driver import audit_operator, planner_seed
from chaos.invariants import check
from chaos.scenarios import generate
from harness.journal import Journal
from harness.runtime import resolve

MAX_INCARNATIONS = 12
CONFIGS = {  # name -> (agent --config, conditional writes)
    "naive_retry": ("naive_retry", False),
    "keys_only": ("keys_only", False),
    "harness": ("harness", False),
    "harness_cas": ("harness", True),
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_platform(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "platform_sim", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/_admin/faults", timeout=1)
            return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("platform did not start")


def launch(base_url: str, journal: Path, run_id: str, config: str, conditional: bool,
           seed: int, temperature: float) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "agents", "--journal", str(journal), "--base-url", base_url,
           "run", "--run-id", run_id, "--config", config, "--seed", str(seed),
           "--temperature", str(temperature), "--backoff", "0"]
    if conditional:
        cmd.append("--conditional-writes")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if proc.stdout.readline().strip() != "READY":  # imports done; the run starts now
        proc.wait()
        raise RuntimeError("agent failed to start")
    return proc


def calibrate(base_url: str, tmp: Path, name: str) -> float:
    """Median wall time of a clean run of a config after READY: that config's kill window."""
    config, conditional = CONFIGS[name]
    httpx.post(f"{base_url}/_admin/reset", json={})
    times = []
    for i in range(3):
        proc = launch(base_url, tmp / f"calibrate-{name}-{i}.db", f"calibrate-{name}-{i}", config,
                      conditional, 0, 0.0)
        started = time.perf_counter()
        proc.wait()
        times.append(time.perf_counter() - started)
    return sorted(times)[1]


def run_one(client: httpx.Client, base_url: str, scenario, name: str, tmp: Path,
            rng: random.Random, window: float) -> dict:
    config, conditional = CONFIGS[name]
    client.post("/_admin/reset", json=scenario.faults.as_dict())
    run_id = f"{name}-{scenario.seed}"
    journal = tmp / f"{run_id}.db"
    kills_left = len(scenario.crashes)
    row = {"config": name, "kills": 0, "escalations": 0, "status": "incarnation_limit"}
    for n in range(MAX_INCARNATIONS):
        proc = launch(base_url, journal, run_id, config, conditional,
                      planner_seed(scenario.seed, n), scenario.temperature)
        if kills_left:
            try:
                proc.wait(timeout=rng.uniform(0, window))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                row["kills"] += 1
                kills_left -= 1
                continue
        code = proc.wait()
        if code == 3:
            row["escalations"] += 1
            with Journal(journal) as j:
                applied, value = audit_operator(client, run_id, j.load_state(run_id).status)
                resolve(j, run_id, applied, value)
            continue
        if code == 4:
            continue
        row["status"] = {0: "finished", 6: "gave_up"}.get(code, f"exit {code}")
        break
    goal = CampaignGoal()
    client.post("/_admin/drain")  # whatever is still in flight lands (or is dropped) now
    effects = client.get("/_admin/effects", params={"run_id": run_id}).json()["effects"]
    row["check"] = check(effects, goal.expected_effects, goal.expected_spend)
    return row


def render(rows: list[dict], n: int, args, window: dict[str, float]) -> str:
    out = [
        "# Hard-kill results",
        "",
        f"Generated by `python -m chaos.sigkill --scenarios {n} --seed {args.seed}` at commit "
        f"`{_git_sha()}`. Do not edit by hand.",
        "",
        "The agent and the simulated platform run as separate OS processes. Each agent "
        "incarnation that is scheduled to die is killed from outside (`Popen.kill`: SIGKILL "
        "on POSIX, TerminateProcess on Windows) after a uniform random delay between 0 and that "
        "config's own clean-run duration (measured first: "
        + ", ".join(f"`{k}` {v * 1000:.0f} ms" for k, v in window.items())
        + "), so kills land at a uniformly random point of the run. Kill timing is wall-clock, so "
        "these numbers are not bit-for-bit reproducible. Faults (delayed delivery included, in "
        "real seconds) and temperatures come from the same seeded scenarios as results.md; "
        "deadlines are checked against the wall clock both processes share.",
        "",
    ]
    table = []
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        table.append([
            f"`{name}`", len(rs), sum(r["kills"] for r in rs),
            _pct(sum(r["status"] == "finished" for r in rs), len(rs)),
            _pct(sum(1 for r in rs if r["check"].duplicate_total), len(rs)),
            sum(r["check"].duplicate_total for r in rs),
            f"${sum(r['check'].excess_spend for r in rs):,}",
            sum(r["check"].missing_total for r in rs if r["status"] == "finished"),
            sum(r["escalations"] for r in rs),
        ])
    out += [_table(["config", "runs", "kills landed", "finished", "runs with a duplicate",
                    "duplicate effects", "excess spend", "missing (finished runs)", "escalations"],
                   table), ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scenarios", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = start_platform(port)
    rng = random.Random(args.seed)
    rows = []
    try:
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(base_url=base_url, timeout=10) as client:
            window = {name: calibrate(base_url, Path(tmp), name) for name in CONFIGS}
            # Scenarios with at least one scheduled crash; each becomes a hard kill.
            scenarios = [s for s in generate(args.scenarios * 3, args.seed) if s.crashes][: args.scenarios]
            for scenario in scenarios:
                for name in CONFIGS:
                    rows.append(run_one(client, base_url, scenario, name, Path(tmp), rng, window[name]))
                print(f"scenario {scenario.seed}: "
                      + ", ".join(f"{r['config']}={r['check'].duplicate_total}" for r in rows[-len(CONFIGS):]),
                      flush=True)
    finally:
        server.kill()
    args.out.mkdir(parents=True, exist_ok=True)
    text = render(rows, len(scenarios), args, window)
    (args.out / "sigkill.md").write_text(text, encoding="utf-8", newline="\n")
    if args.out.resolve() == RESULTS:
        update_readme("sigkill", text.split("\n", 2)[2].strip())
    print(f"-> {args.out / 'sigkill.md'}")


if __name__ == "__main__":
    main()
