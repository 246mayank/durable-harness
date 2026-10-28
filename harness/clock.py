"""Clocks shared by the runtime and the simulated platform.

Deadlines only mean something if both sides read the same clock. Separate
processes on one machine share `RealClock`. In-process runs share one
`VirtualClock`, where sleeping advances time instantly, so waiting out a
deadline costs nothing and a run is deterministic.
"""

from __future__ import annotations

import time


class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def advance_to(self, t: float) -> None:
        """Real time can't be moved; callers that need to wait use sleep()."""


class VirtualClock:
    """Simulated seconds. Moves only when someone sleeps or advances it."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += seconds

    def advance_to(self, t: float) -> None:
        self.t = max(self.t, t)
