import sqlite3

import pytest

from harness.journal import Journal
from harness.keys import KeyMode, make_key


def test_events_survive_reopen(tmp_path):
    path = tmp_path / "j.db"
    with Journal(path) as j:
        j.append("r1", "incarnation", None, {"n": 1})
        j.append("r1", "decision", 0, {"tool": "create_campaign", "args": {"name": "x"}})
        j.append("r1", "intent", 0, {"tool": "create_campaign", "key": "k0"})
    with Journal(path) as j:
        events = j.events("r1")
        state = j.load_state("r1")
    assert [e.seq for e in events] == [1, 2, 3]
    assert [e.kind for e in events] == ["incarnation", "decision", "intent"]
    assert state.incarnations == 1
    assert state.current.index == 0 and state.current.intent == {"tool": "create_campaign", "key": "k0"}


def test_seq_is_per_run():
    j = Journal()
    j.append("a", "incarnation", None, {})
    j.append("b", "incarnation", None, {})
    j.append("a", "incarnation", None, {})
    assert [e.seq for e in j.events("a")] == [1, 2]
    assert [e.seq for e in j.events("b")] == [1]
    assert j.run_ids() == ["a", "b"]


def test_fold_tracks_step_lifecycle():
    j = Journal()
    j.append("r", "decision", 0, {"tool": "t", "args": {}})
    j.append("r", "intent", 0, {"key": "k"})
    j.append("r", "attempt", 0, {"n": 0, "outcome": "ambiguous"})
    j.append("r", "intent_voided", 0, {"reason": "operator"})
    state = j.load_state("r")
    step = state.current
    assert step.intent is None and step.voided == 1 and len(step.attempts) == 1
    j.append("r", "intent", 0, {"key": "k"})
    j.append("r", "result", 0, {"status": "ok", "value": {"id": 1}})
    state = j.load_state("r")
    assert state.current is None
    assert state.history() == [{"step": 0, "tool": "t", "args": {}, "status": "ok", "value": {"id": 1}}]


def test_rejects_unknown_kinds_and_out_of_order_decisions():
    j = Journal()
    with pytest.raises(ValueError):
        j.append("r", "bogus", None, {})
    j.append("r", "decision", 1, {"tool": "t"})
    with pytest.raises(ValueError):
        j.load_state("r")


@pytest.mark.parametrize("durability, level", [("full", 2), ("normal", 1)])
def test_durability_sets_synchronous(tmp_path, durability, level):
    with Journal(tmp_path / "j.db", durability):
        pass
    j = Journal(tmp_path / "j.db", durability)
    assert j._conn.execute("PRAGMA synchronous").fetchone()[0] == level
    assert j._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    j.close()


def test_appends_are_committed_immediately(tmp_path):
    path = tmp_path / "j.db"
    j = Journal(path)
    j.append("r", "incarnation", None, {})
    # A second connection sees the event without the first one closing.
    other = sqlite3.connect(path)
    assert other.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    other.close()
    j.close()


def test_key_modes():
    a = make_key(KeyMode.ARGS, "r", 3, "create_ad", {"headline": "a"})
    b = make_key(KeyMode.ARGS, "r", 3, "create_ad", {"headline": "b"})
    p1 = make_key(KeyMode.POSITIONAL, "r", 3, "create_ad", {"headline": "a"})
    p2 = make_key(KeyMode.POSITIONAL, "r", 3, "create_ad", {"headline": "b"})
    assert make_key(KeyMode.NONE, "r", 3, "create_ad", {}) is None
    assert a != b  # args keys follow the arguments...
    assert p1 == p2  # ...positional keys follow the step
    assert p1 != make_key(KeyMode.POSITIONAL, "r", 4, "create_ad", {})
    assert p1 != make_key(KeyMode.POSITIONAL, "other-run", 3, "create_ad", {})
