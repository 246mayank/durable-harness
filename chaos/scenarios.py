"""Seeded scenario generation: one fault stream, one crash schedule, one temperature.

Zombie scenarios have no crashes; their first incarnation hangs instead and
comes back after a replacement has finished the run.
"""

from __future__ import annotations

import random
from dataclasses import replace

from chaos.driver import Scenario
from harness.runtime import CRASH_POINTS
from platform_sim.faults import FaultConfig

TEMPERATURES = (0.0, 0.3, 0.7)
MAX_FAULT_P = 0.15
FREEZE_POINTS = (*CRASH_POINTS, "before_send")


def generate(n: int, seed: int = 0) -> list[Scenario]:
    rng = random.Random(seed)
    scenarios = []
    for i in range(n):
        faults = FaultConfig(
            p_429=round(rng.uniform(0, MAX_FAULT_P), 3),
            p_5xx=round(rng.uniform(0, MAX_FAULT_P), 3),
            p_lost_ack=round(rng.uniform(0, MAX_FAULT_P), 3),
            p_delayed=round(rng.uniform(0, MAX_FAULT_P), 3),
            seed=rng.randrange(2**31),
        )
        n_crashes = rng.choices((0, 1, 2, 3), weights=(25, 35, 25, 15))[0]
        crashes = tuple((rng.choice(CRASH_POINTS), rng.randint(1, 8)) for _ in range(n_crashes))
        scenarios.append(Scenario(i, faults, crashes, rng.choice(TEMPERATURES)))
    return scenarios


def generate_zombies(n: int, seed: int = 0) -> list[Scenario]:
    """The same fault streams and temperatures, with a hang-and-return instead of crashes."""
    rng = random.Random(f"zombies-{seed}")
    return [replace(s, crashes=(), freeze=(rng.choice(FREEZE_POINTS), rng.randint(1, 8)))
            for s in generate(n, seed)]
