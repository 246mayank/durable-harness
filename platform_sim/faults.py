"""Seeded fault injection for the simulated platform.

Every non-admin request consumes exactly one roll from a seeded RNG, so a
given seed produces the same fault sequence for the same request sequence.
"""

from __future__ import annotations

import enum
import random
from dataclasses import asdict, dataclass


class Fault(enum.Enum):
    NONE = "none"
    RATE_LIMIT = "429"  # refused before any processing
    SERVER_ERROR = "5xx"  # failed before commit
    LOST_ACK = "lost_ack"  # committed, then the response is lost (504)


@dataclass(frozen=True)
class FaultConfig:
    p_429: float = 0.0
    p_5xx: float = 0.0
    p_lost_ack: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        probs = (self.p_429, self.p_5xx, self.p_lost_ack)
        if any(p < 0 for p in probs) or sum(probs) > 1:
            raise ValueError(f"invalid fault probabilities: {probs}")

    def as_dict(self) -> dict:
        return asdict(self)


class FaultInjector:
    def __init__(self, config: FaultConfig | None = None) -> None:
        self.configure(config or FaultConfig())

    def configure(self, config: FaultConfig) -> None:
        self.config = config
        self.rolls = 0
        self._rng = random.Random(config.seed)

    def roll(self) -> tuple[Fault, int]:
        """Return the fault for the next request and the status code it implies."""
        r = self._rng.random()
        self.rolls += 1
        c = self.config
        if r < c.p_429:
            return Fault.RATE_LIMIT, 429
        r -= c.p_429
        if r < c.p_5xx:
            return Fault.SERVER_ERROR, 500 if r < c.p_5xx / 2 else 503
        r -= c.p_5xx
        if r < c.p_lost_ack:
            return Fault.LOST_ACK, 504
        return Fault.NONE, 0
