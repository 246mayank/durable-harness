"""Idempotency keys.

    none        no key at all
    args        hash(run_id, tool, args): stable only while the arguments are
    positional  hash(run_id, step): stable for a step whatever its arguments

Positional keys are correct only together with replay: the journal pins
step N to one decision, so a resumed step re-sends the same arguments under
the same key. Without replay, step N after a restart may be a different
call, and a positional key would collide with it.
"""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any


class KeyMode(str, enum.Enum):
    NONE = "none"
    ARGS = "args"
    POSITIONAL = "positional"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def args_hash(tool: str, args: dict) -> str:
    return hashlib.sha256(canonical([tool, args]).encode()).hexdigest()[:32]


def make_key(mode: KeyMode, run_id: str, step: int, tool: str, args: dict) -> str | None:
    if mode is KeyMode.NONE:
        return None
    if mode is KeyMode.ARGS:
        material = canonical(["args", run_id, tool, args])
    else:
        material = canonical(["positional", run_id, step])
    return hashlib.sha256(material.encode()).hexdigest()[:32]
