"""Tool registry with typed arguments and effect classes.

    effect        example            when the outcome is unknown
    read          get_metrics        re-send
    idempotent    create_ad          re-send with the same key
    reconcilable  send_email_batch   look up by client_ref; send only if absent
    unsafe        allocate_budget    escalate to a human; never guess
"""

from __future__ import annotations

import enum
import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel


class Effect(str, enum.Enum):
    READ = "read"
    IDEMPOTENT = "idempotent"
    RECONCILABLE = "reconcilable"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class CallContext:
    run_id: str
    step: int
    key: str | None
    # Captured once before the intent is written and journaled with it, so a
    # resumed step sends the same precondition (see conditional writes).
    precondition: dict | None = None

    @property
    def trace_id(self) -> str:
        return f"{self.run_id}/{self.step}"

    def headers(self) -> dict[str, str]:
        return {"X-Run-Id": self.run_id, "X-Trace-Id": self.trace_id}


Execute = Callable[[httpx.Client, Any, CallContext], Any]
# Returns the existing effect, None if it is absent, or raises Unresolvable.
Lookup = Callable[[httpx.Client, Any, CallContext], Any]
Prepare = Callable[[httpx.Client, Any, CallContext], dict]


@dataclass
class Tool:
    name: str
    args: type[BaseModel]
    effect: Effect
    execute: Execute
    description: str = ""
    lookup: Lookup | None = None
    prepare: Prepare | None = None
    # Spend committed by a successful call, for budget reservation.
    spend: Callable[[Any], int] | None = None

    def reconciler(self, fn: Lookup) -> Lookup:
        self.lookup = fn
        return fn

    def preparer(self, fn: Prepare) -> Prepare:
        self.prepare = fn
        return fn

    def schema(self) -> dict:
        """Function-calling schema for an LLM planner."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args.model_json_schema(),
        }


def tool(effect: Effect, args: type[BaseModel], *, name: str | None = None,
         spend: Callable[[Any], int] | None = None) -> Callable[[Execute], Tool]:
    def wrap(fn: Execute) -> Tool:
        return Tool(name or fn.__name__, args, effect, fn, inspect.getdoc(fn) or "", spend=spend)

    return wrap


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    @classmethod
    def of(cls, tools: Iterable[Tool]) -> "ToolRegistry":
        registry = cls()
        for t in tools:
            if t.effect is Effect.RECONCILABLE and t.lookup is None:
                raise ValueError(f"reconcilable tool {t.name!r} needs a lookup")
            registry.tools[t.name] = t
        return registry

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]
