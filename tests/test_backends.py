"""The HTTP backends against a fake server (no network)."""
import json

import httpx
import pytest

from evoharness import decision as D
from evoharness import llm
from evoharness.interfaces import DecisionUnavailable


@pytest.fixture
def fake_http(monkeypatch):
    calls = []
    replies = []
    real = httpx.Client

    def handler(request):
        body = json.loads(request.content)
        calls.append({"url": str(request.url), "auth": request.headers.get("authorization"), "body": body})
        status, payload = replies.pop(0)
        return httpx.Response(status, json=payload)

    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setenv("TEST_KEY", "sk-test")
    return calls, replies


def tool_reply(name, args, call_id="c1"):
    return 200, {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}], "usage": {"total_tokens": 10}}


def test_openai_compatible_tool_loop(fake_http):
    calls, replies = fake_http
    replies += [tool_reply("lookup", {"x": 1}), tool_reply("final", {"ok": True}, "c2")]
    b = llm.make({"backend": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "m", "api_key_env": "TEST_KEY"})
    final = {"name": "final", "parameters": {"type": "object"}}
    lookup = {"name": "lookup", "parameters": {"type": "object"}}
    args, metas, log = b.call("sys", "user", final, {"lookup": (lookup, lambda a: {"echo": a["x"]})}, max_lookups=2)
    assert args == {"ok": True} and log == [{"tool": "lookup", "args": {"x": 1}}]
    assert calls[0]["url"] == "https://example.invalid/v1/chat/completions" and calls[0]["auth"] == "Bearer sk-test"
    second = calls[1]["body"]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"echo": 1})}


def test_openai_compatible_quota_goes_to_fallback(fake_http):
    calls, replies = fake_http
    replies += [(429, {"error": "rate limit"}), tool_reply("final", {"ok": 2})]
    b = llm.make({"backend": "openai_compatible", "base_url": "https://primary.invalid/v1", "model": "m", "api_key_env": "TEST_KEY",
                  "fallback": {"backend": "openai_compatible", "base_url": "https://fallback.invalid/v1", "model": "m2", "api_key_env": "TEST_KEY"}})
    args, _, _ = b.call("sys", "user", {"name": "final", "parameters": {"type": "object"}})
    assert args == {"ok": 2} and "fallback" in calls[1]["url"]


def test_openai_compatible_account_error(fake_http):
    _, replies = fake_http
    replies += [(401, {"error": "bad key"})]
    b = llm.make({"backend": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "m", "api_key_env": "TEST_KEY"})
    with pytest.raises(llm.AccountError):
        b.call("sys", "user", {"name": "final", "parameters": {"type": "object"}})


QUESTIONS = {"speed": {"instructions": "speed", "options": {"SLOW": "slow", "FAST": "fast"}}}


def test_openai_compatible_decision(fake_http):
    calls, replies = fake_http
    replies += [(200, {"choices": [{"message": {"content": json.dumps({"speed": "FAST"})}}]})]
    m = D.make({"backend": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "m", "api_key_env": "TEST_KEY"})
    assert m.decide({"goal": "g"}, QUESTIONS) == {"speed": "FAST"}
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}


def test_jev_decision_and_fatal_status(fake_http):
    calls, replies = fake_http
    replies += [(200, {"answers": {"speed": {"choice": "SLOW", "probabilities": {"SLOW": 0.7, "FAST": 0.3}}}}), (451, {})]
    m = D.make({"backend": "jev", "endpoint": "https://jev.invalid/decide", "model": "jev", "api_key_env": "TEST_KEY"})
    assert m.decide({"goal": "g"}, QUESTIONS) == {"speed": "SLOW"}
    assert calls[0]["body"]["questions"]["speed"]["criteria"] == {"SLOW": "slow", "FAST": "fast"}
    with pytest.raises(DecisionUnavailable):
        m.decide({"goal": "g"}, QUESTIONS)
