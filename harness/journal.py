"""Append-only run journal in SQLite (WAL mode), and the state folded from it.

Per step, events are written in the order

    decision -> intent (before the call) -> attempt* -> result (after the call)

Each event is its own committed transaction, so once `append` returns the
event survives a process kill.

Durability: `synchronous=FULL` (the default) fsyncs the WAL on every commit,
so an intent is on disk before the call it describes goes out, even across
an OS crash or power loss. `synchronous=NORMAL` skips that fsync. It still
survives a process kill, because the committed frames are in the OS page
cache, but the last few events can be lost on power failure, and a lost
intent for a call that did go out is exactly the case the journal exists to
prevent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KINDS = frozenset(
    {"decision", "intent", "attempt", "result", "status", "incarnation", "intent_voided", "approval"}
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    run_id  TEXT    NOT NULL,
    seq     INTEGER NOT NULL,
    kind    TEXT    NOT NULL,
    step    INTEGER,
    payload TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    kind: str
    step: int | None
    payload: dict
    ts: float


@dataclass
class StepState:
    index: int
    decision: dict
    intent: dict | None = None
    attempts: list[dict] = field(default_factory=list)
    result: dict | None = None
    voided: int = 0
    approved_hash: str | None = None

    @property
    def done(self) -> bool:
        return self.result is not None


@dataclass
class RunState:
    run_id: str
    steps: list[StepState] = field(default_factory=list)
    status: dict = field(default_factory=lambda: {"status": "new"})
    incarnations: int = 0

    def apply(self, event: Event) -> None:
        kind, p = event.kind, event.payload
        if kind == "incarnation":
            self.incarnations += 1
        elif kind == "status":
            self.status = p
        elif kind == "decision":
            if event.step != len(self.steps):
                raise ValueError(f"decision for step {event.step}, expected {len(self.steps)}")
            self.steps.append(StepState(event.step, p))
        else:
            step = self.steps[event.step]
            if kind == "intent":
                step.intent = p
            elif kind == "attempt":
                step.attempts.append(p)
            elif kind == "result":
                step.result = p
            elif kind == "intent_voided":
                step.intent = None
                step.voided += 1
            elif kind == "approval":
                step.approved_hash = p["args_hash"]

    @property
    def current(self) -> StepState | None:
        """The step in progress: decided but without a result."""
        if self.steps and not self.steps[-1].done:
            return self.steps[-1]
        return None

    def history(self) -> list[dict]:
        """What the planner sees: each completed step's call and observation."""
        return [
            {
                "step": s.index,
                "tool": s.decision.get("tool"),
                "args": s.decision.get("args"),
                **s.result,
            }
            for s in self.steps
            if s.done and not s.decision.get("finish")
        ]


class Journal:
    def __init__(self, path: str | Path = ":memory:", durability: str = "full") -> None:
        if durability not in ("full", "normal"):
            raise ValueError("durability must be 'full' or 'normal'")
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA synchronous={durability.upper()}")
        self._conn.executescript(SCHEMA)

    def append(self, run_id: str, kind: str, step: int | None, payload: dict[str, Any]) -> Event:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        body = json.dumps(payload, sort_keys=True)
        ts = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                (seq,) = self._conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?", (run_id,)
                ).fetchone()
                self._conn.execute(
                    "INSERT INTO events (run_id, seq, kind, step, payload, ts) VALUES (?,?,?,?,?,?)",
                    (run_id, seq, kind, step, body, ts),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")
        return Event(run_id, seq, kind, step, json.loads(body), ts)

    def events(self, run_id: str) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, seq, kind, step, payload, ts FROM events"
                " WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [Event(r[0], r[1], r[2], r[3], json.loads(r[4]), r[5]) for r in rows]

    def run_ids(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id")]

    def load_state(self, run_id: str) -> RunState:
        state = RunState(run_id)
        for event in self.events(run_id):
            state.apply(event)
        return state

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
