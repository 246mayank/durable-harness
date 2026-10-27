"""SQLite ground truth for the simulated platform.

The platform's own database is the source of truth the invariant checker
reads. It is deliberately separate from the agent's journal.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY, name TEXT, product TEXT, audience TEXT
);
CREATE TABLE IF NOT EXISTS ads (
    id TEXT PRIMARY KEY, campaign_id TEXT, headline TEXT, body TEXT
);
CREATE TABLE IF NOT EXISTS email_batches (
    id TEXT PRIMARY KEY, segment TEXT, subject TEXT, body TEXT,
    client_ref TEXT, recipients INTEGER
);
CREATE INDEX IF NOT EXISTS email_batches_ref ON email_batches(client_ref);
CREATE TABLE IF NOT EXISTS allocations (
    id TEXT PRIMARY KEY, campaign_id TEXT, amount INTEGER, ad_id TEXT
);
-- Stored responses for Idempotency-Key replay. Written in the same
-- transaction as the effect, so a key is recorded if and only if the effect is.
CREATE TABLE IF NOT EXISTS idem (
    key TEXT PRIMARY KEY, body_hash TEXT, status INTEGER, response TEXT
);
-- Audit log of every committed side effect, attributed to the caller.
CREATE TABLE IF NOT EXISTS effects (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT, object_id TEXT, run_id TEXT, trace_id TEXT,
    amount INTEGER, payload TEXT
);
-- Every non-admin request and the fault it drew.
CREATE TABLE IF NOT EXISTS requests (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    method TEXT, path TEXT, run_id TEXT, trace_id TEXT, fault TEXT
);
"""

TABLES = ("campaigns", "ads", "email_batches", "allocations", "idem", "effects", "requests")


class PlatformDB:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """One serialized transaction; rolled back if the body raises."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params)]

    def reset(self) -> None:
        with self.tx() as conn:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM sqlite_sequence")

    def close(self) -> None:
        self._conn.close()


def next_id(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return f"{prefix}_{count + 1}"
