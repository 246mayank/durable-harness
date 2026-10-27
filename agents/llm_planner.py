"""A real-LLM planner (Gemini function calling) behind the same interface as ScriptedPlanner.

Each decision sends the goal plus the observed history and asks for exactly
one tool call. The model has no other memory, so a restart without replay
re-plans from scratch, and at temperature > 0 it writes new copy each time.

Reads GEMINI_API_KEY from the environment and sends it only as a request
header to Google's API.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from agents.campaign import CampaignGoal
from harness.tools import ToolRegistry

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM = """You are a B2B marketing agent operating an ad and email platform through tools.

Goal, for {product} aimed at {audience}. Do each of these exactly once:
1. Create a campaign named "{product} Q4 pipeline" (product "{product}", audience "{audience}").
2. Create {n_ads} distinct ads in that campaign. Write your own headline and body for each.
3. Send one email batch to segment "{segment}". Write your own subject and body.
4. Allocate an initial budget of {initial} to the campaign (no ad_id).
5. Read the campaign metrics.
6. Allocate {boost} to the ad with the highest CTR (set ad_id).
Then call finish.

Call exactly one tool per turn. Use ids returned by earlier calls. If a call
failed, decide from its error whether to retry it."""


def _gemini_schema(schema: dict, defs: dict | None = None) -> dict:
    """Reduce a Pydantic JSON schema to the OpenAPI subset Gemini accepts."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return _gemini_schema(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:  # Optional[X] -> X, nullable
        options = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return {**_gemini_schema(options[0], defs), "nullable": True}
    out = {k: v for k, v in schema.items() if k in ("type", "description", "enum", "required")}
    if "properties" in schema:
        out["properties"] = {k: _gemini_schema(v, defs) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = _gemini_schema(schema["items"], defs)
    if "exclusiveMinimum" in schema:
        out["description"] = f"must be > {schema['exclusiveMinimum']}"
    return out


class GeminiPlanner:
    def __init__(
        self,
        tools: ToolRegistry,
        seed: int = 0,
        temperature: float = 1.0,
        model: str = "gemini-3.5-flash",
        min_interval: float = 4.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.calls = 0
        self.seed = seed
        self.temperature = temperature
        self.model = model
        self.min_interval = min_interval
        self.client = client or httpx.Client(timeout=60)
        self.declarations = [
            {"name": t.name, "description": t.description,
             "parameters": _gemini_schema(t.args.model_json_schema())}
            for t in tools.tools.values()
        ] + [{"name": "finish", "description": "Call when every goal step is done."}]
        self._last_request = 0.0

    def decide(self, goal: CampaignGoal, history: list[dict]) -> dict:
        self.calls += 1
        system = SYSTEM.format(product=goal.product, audience=goal.audience, n_ads=goal.n_ads,
                               segment=goal.email_segment, initial=goal.initial_budget, boost=goal.boost)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": self._render(history)}]}],
            "tools": [{"functionDeclarations": self.declarations}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {
                "temperature": self.temperature,
                "seed": self.seed,
                "thinkingConfig": {"thinkingLevel": "low"},
            },
        }
        call = self._function_call(self._post(body))
        if call["name"] == "finish":
            return {"finish": True, "note": self.model}
        args = {k: v for k, v in call.get("args", {}).items() if v is not None}
        return {"tool": call["name"], "args": args, "note": self.model}

    @staticmethod
    def _render(history: list[dict]) -> str:
        if not history:
            return "No tool calls yet. Make the first call."
        lines = ["Tool calls so far, oldest first:"]
        for h in history:
            outcome = (f"ok -> {json.dumps(h['value'])}" if h["status"] == "ok"
                       else f"FAILED ({h.get('error_class')}): {h.get('error')}")
            lines.append(f"{h['step']}. {h['tool']}({json.dumps(h['args'])}) {outcome}")
        lines.append("Make the next call.")
        return "\n".join(lines)

    @staticmethod
    def _function_call(response: dict) -> dict:
        for part in response["candidates"][0]["content"]["parts"]:
            if "functionCall" in part:
                return part["functionCall"]
        raise RuntimeError(f"no function call in response: {json.dumps(response)[:300]}")

    def _post(self, body: dict) -> dict:
        key = os.environ["GEMINI_API_KEY"]
        for attempt in range(8):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                r = self.client.post(API.format(model=self.model), json=body,
                                     headers={"x-goog-api-key": key})
            except httpx.TransportError:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 503):
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
        raise RuntimeError("Gemini kept failing")


def gemini_factory(tools: ToolRegistry, model: str = "gemini-3.5-flash", **kwargs: Any):
    """planner_factory(seed, temperature) for chaos.driver.run_scenario."""
    client = httpx.Client(timeout=60)
    return lambda seed, temperature: GeminiPlanner(tools, seed, temperature, model, client=client, **kwargs)
