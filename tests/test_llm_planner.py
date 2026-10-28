"""LLM planners turn a model's tool call into a decision; no network (MockTransport)."""

import json

import httpx
import pytest

from agents.campaign import CampaignGoal, campaign_tools
from agents.llm_planner import GeminiPlanner, OllamaPlanner, PlannerError

HISTORY = [{"step": 0, "tool": "create_campaign", "args": {"name": "n"}, "status": "ok",
            "value": {"id": "cmp_1"}}]


def ollama(replies: list[dict]) -> tuple[OllamaPlanner, list[dict]]:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": replies[len(sent) - 1]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaPlanner(campaign_tools(), seed=3, temperature=0.7, client=client), sent


def test_ollama_tool_call_becomes_a_decision():
    planner, sent = ollama([{"tool_calls": [{"function": {
        "name": "allocate_budget", "arguments": {"campaign_id": "cmp_1", "amount": 500, "ad_id": None}}}]}])
    decision = planner.decide(CampaignGoal(), HISTORY)
    assert decision["tool"] == "allocate_budget"
    assert decision["args"] == {"campaign_id": "cmp_1", "amount": 500}  # nulls dropped
    assert planner.calls == 1
    assert "cmp_1" in sent[0]["messages"][1]["content"]  # the history reached the model
    assert {d["function"]["name"] for d in sent[0]["tools"]} >= {"create_ad", "finish"}


def test_ollama_reply_without_a_tool_call_is_a_final_answer():
    # As in a standard tool-calling loop: no tool call means the model is done.
    planner, sent = ollama([{"content": "All steps are complete."}])
    decision = planner.decide(CampaignGoal(), HISTORY)
    assert decision["finish"] is True and len(sent) == 1
    planner, _ = ollama([{"tool_calls": [{"function": {"name": "finish"}}]}])
    assert planner.decide(CampaignGoal(), HISTORY) == {"finish": True, "note": "qwen2.5:3b"}


def test_gemini_reply_without_a_function_call_is_an_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})))
    with pytest.raises(PlannerError):  # mode ANY should make this impossible; never guess
        GeminiPlanner(campaign_tools(), min_interval=0, client=client).decide(CampaignGoal(), HISTORY)


def test_gemini_function_call_becomes_a_decision(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers["x-goog-api-key"]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "create_ad", "args": {"campaign_id": "cmp_1", "headline": "h",
                                                             "body": "b"}}}]}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    planner = GeminiPlanner(campaign_tools(), min_interval=0, client=client)
    decision = planner.decide(CampaignGoal(), HISTORY)
    assert decision == {"tool": "create_ad", "args": {"campaign_id": "cmp_1", "headline": "h", "body": "b"},
                        "note": "gemini-3.5-flash"}
    assert seen["key"] == "test-key"  # sent as a header, never in the URL
