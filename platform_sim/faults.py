"""Seeded fault injection for the simulated platform.

Every non-admin request consumes exactly one roll from a seeded RNG, so a
given seed produces the same fault sequence for the same request sequence.
A delayed request also draws its delay from the same RNG.
"""

from __future__ import annotations

import enum
import random
from collections import deque
from dataclasses import asdict, dataclass


class Fault(enum.Enum):
    NONE = "none"
    RATE_LIMIT = "429"  # refused before any processing
    SERVER_ERROR = "5xx"  # failed before commit
    LOST_ACK = "lost_ack"  # committed, then the response is lost (504)
    DELAYED = "delayed"  # 504 now; the request is still in flight and is processed later


STATUS = {Fault.NONE: 0, Fault.RATE_LIMIT: 429, Fault.SERVER_ERROR: 500, Fault.LOST_ACK: 504,
          Fault.DELAYED: 504}


@dataclass(frozen=True)
class FaultConfig:
    p_429: float = 0.0
    p_5xx: float = 0.0
    p_lost_ack: float = 0.0
    seed: int = 0
    p_delayed: float = 0.0
    max_delay: float = 1.0  # seconds; a delayed request is processed uniform(0, max_delay) later

    def __post_init__(self) -> None:
        probs = (self.p_429, self.p_5xx, self.p_lost_ack, self.p_delayed)
        if any(p < 0 for p in probs) or sum(probs) > 1:
            raise ValueError(f"invalid fault probabilities: {probs}")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")

    def as_dict(self) -> dict:
        return asdict(self)


class FaultInjector:
    def __init__(self, config: FaultConfig | None = None) -> None:
        self.configure(config or FaultConfig())

    def configure(self, config: FaultConfig) -> None:
        self.config = config
        self.rolls = 0
        self._rng = random.Random(config.seed)
        self._forced: deque[tuple[Fault, float | None]] = deque()
        self._next_delay: float | None = None

    def force(self, *faults: Fault, delay: float | None = None) -> None:
        """Make the next requests draw these faults, in order; `delay` fixes a delayed one's delay."""
        self._forced.extend((f, delay) for f in faults)

    def roll(self) -> tuple[Fault, int]:
        """Return the fault for the next request and the status code it implies."""
        r = self._rng.random()
        self.rolls += 1
        if self._forced:  # the roll above is still consumed, so the seeded stream stays aligned
            fault, self._next_delay = self._forced.popleft()
            return fault, STATUS[fault]
        c = self.config
        if r < c.p_429:
            return Fault.RATE_LIMIT, 429
        r -= c.p_429
        if r < c.p_5xx:
            return Fault.SERVER_ERROR, 500 if r < c.p_5xx / 2 else 503
        r -= c.p_5xx
        if r < c.p_lost_ack:
            return Fault.LOST_ACK, 504
        r -= c.p_lost_ack
        if r < c.p_delayed:
            return Fault.DELAYED, 504
        return Fault.NONE, 0

    def delay(self) -> float:
        """How long a delayed request stays in flight."""
        if self._next_delay is not None:
            delay, self._next_delay = self._next_delay, None
            return delay
        return self._rng.uniform(0, self.config.max_delay)
