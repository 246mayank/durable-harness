"""Seeded scenario generation: one fault stream, one crash schedule, one temperature."""

from __future__ import annotations

import random

from chaos.driver import Scenario
from harness.runtime import CRASH_POINTS
from platform_sim.faults import FaultConfig

TEMPERATURES = (0.0, 0.3, 0.7)
MAX_FAULT_P = 0.15


def generate(n: int, seed: int = 0) -> list[Scenario]:
    rng = random.Random(seed)
    scenarios = []
    for i in range(n):
        faults = FaultConfig(
            p_429=round(rng.uniform(0, MAX_FAULT_P), 3),
            p_5xx=round(rng.uniform(0, MAX_FAULT_P), 3),
            p_lost_ack=round(rng.uniform(0, MAX_FAULT_P), 3),
            seed=rng.randrange(2**31),
        )
        n_crashes = rng.choices((0, 1, 2, 3), weights=(25, 35, 25, 15))[0]
        crashes = tuple((rng.choice(CRASH_POINTS), rng.randint(1, 8)) for _ in range(n_crashes))
        scenarios.append(Scenario(i, faults, crashes, rng.choice(TEMPERATURES)))
    return scenarios
