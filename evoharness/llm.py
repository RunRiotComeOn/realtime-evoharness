"""Proposer / verifier backends: one conversation that ends in one forced tool call, optionally after lookup tools.

  openai_compatible  any OpenAI-compatible chat endpoint with tool calls (base_url, model, api_key_env)
  codex              the `codex exec` CLI with the user's own login (model, effort)
  python             your own function "module:function"(system, messages, tools) -> (tool name, arguments);
                     messages is [(role, text)]; used by the tests with a scripted proposer

A backend config may name a `fallback` backend config: after a quota error the calls go there for `fallback_hold_s`.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time


class AccountError(Exception):
    """The API refused the account (auth, balance, model not enabled): stop the run; it resumes at the same round."""


class QuotaError(AccountError):
    """Usage quota or rate limit used up."""


QUOTA_WORDS = ("quota", "limitexceeded", "toomanyrequests", "ratelimit", "rate limit", "usage limit")


def short_args(args):
    s = json.dumps(args, ensure_ascii=False)
    if len(s) < 400:
        return args
    return {k: (v if len(json.dumps(v, ensure_ascii=False)) < 200 else f"({len(v) if isinstance(v, list) else '...'} items)") for k, v in args.items()}


class Backend:
    """Subclasses implement turn(system, messages, tools, force_name) -> (tool name, arguments dict, meta)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def call(self, system, user, final_tool, lookups=None, max_lookups=0, ask_for_tool="Call {name}."):
        """Returns (final tool arguments, metas, lookup log). lookups: {name: (tool spec, handler(args) -> json-able)}."""
        lookups = lookups or {}
        tools = [final_tool] + [spec for spec, _ in lookups.values()]
        messages = [("user", user)]
        metas, log, used = [], [], 0
        while True:
            force = used >= max_lookups or not lookups
            name, args, meta, raw = self.turn(system, messages, tools[:1] if force else tools, final_tool["name"] if force else None)
            metas.append(meta)
            if name is None:   # answered in text
                messages += [("assistant", raw or ""), ("user", ask_for_tool.replace("{name}", final_tool["name"]))]
                used += 1
                continue
            if name == final_tool["name"]:
                return args, metas, log
            if name not in lookups or args is None:
                result = {"error": f"unknown tool or bad arguments: {name}"}
            else:
                try:
                    result = lookups[name][1](args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            log.append({"tool": name, "args": short_args(args or {})})
            used += 1
            messages += [("assistant_tool", {"name": name, "args": args, "raw": raw}), ("tool", {"name": name, "result": result, "raw": raw})]


# ---------------------------------------------------------------- OpenAI-compatible

class OpenAICompatible(Backend):
    _blocked_until = 0.0

    def __init__(self, cfg):
        super().__init__(cfg)
        import httpx
        self.httpx = httpx
        env = cfg.get("api_key_env")
        self.key = os.environ.get(env) if env else None
        if env and not self.key:
            raise AccountError(f"environment variable {env} is not set")
        self.fallback = make(cfg["fallback"]) if cfg.get("fallback") else None

    def turn(self, system, messages, tools, force_name):
        if self.fallback and time.time() < self._blocked_until:
            return self.fallback.turn(system, messages, tools, force_name)
        try:
            return self._turn(system, messages, tools, force_name)
        except QuotaError:
            if not self.fallback:
                raise
            self._blocked_until = time.time() + float(self.cfg.get("fallback_hold_s", 1800))
            return self.fallback.turn(system, messages, tools, force_name)

    def _messages(self, system, messages):
        out = [{"role": "system", "content": system}]
        for role, m in messages:
            if role == "user":
                out.append({"role": "user", "content": m})
            elif role == "assistant":
                out.append({"role": "assistant", "content": m})
            elif role == "assistant_tool":
                out.append({"role": "assistant", "content": "", "tool_calls": [m["raw"]]})
            elif role == "tool":
                out.append({"role": "tool", "tool_call_id": m["raw"]["id"], "content": json.dumps(m["result"], ensure_ascii=False)})
        return out

    def _turn(self, system, messages, tools, force_name):
        body = {"model": self.cfg["model"], "messages": self._messages(system, messages), "max_tokens": self.cfg.get("max_tokens", 16000),
                "tools": [{"type": "function", "function": t} for t in tools],
                "tool_choice": {"type": "function", "function": {"name": force_name}} if force_name else "auto"}
        body.update(self.cfg.get("extra_body") or {})
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        last = None
        for attempt in range(int(self.cfg.get("retries", 2)) + 1):
            t0 = time.monotonic()
            try:
                with self.httpx.Client(trust_env=self.cfg.get("trust_env", True), timeout=self.cfg.get("timeout_s", 600)) as c:
                    r = c.post(url, headers={"Authorization": "Bearer " + self.key} if self.key else {}, json=body)
                text = (r.text or "").lower()
                if r.status_code == 429 or (r.status_code >= 400 and any(w in text for w in QUOTA_WORDS)):
                    raise QuotaError(f"quota/rate limit {r.status_code}: {r.text[:300]}")
                if r.status_code in (401, 402, 403, 404):
                    raise AccountError(f"account error {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                reply = r.json()
                choice = reply["choices"][0]
                m = choice["message"]
                meta = {"seconds": round(time.monotonic() - t0, 1), "usage": reply.get("usage"), "finish_reason": choice.get("finish_reason"), "model": self.cfg["model"]}
                calls = m.get("tool_calls") or []
                if not calls:
                    if force_name:
                        raise ValueError("no tool call in the reply")
                    return None, None, meta, m.get("content")
                call = next((x for x in calls if x["function"]["name"] == force_name), calls[0]) if force_name else calls[0]
                raw_args = (call["function"].get("arguments") or "{}").strip()
                try:
                    args = json.JSONDecoder().raw_decode(raw_args)[0]
                except json.JSONDecodeError as e:
                    raise ValueError(f"tool arguments are not valid JSON ({e}; finish_reason={choice.get('finish_reason')}, {len(raw_args)} chars)")
                return call["function"]["name"], args, meta, {"id": call.get("id"), "type": "function", "function": call["function"]}
            except (self.httpx.HTTPError, ValueError, KeyError) as e:
                last = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"model call failed: {type(last).__name__}: {str(last)[:300]}")


# ---------------------------------------------------------------- text-only backends (codex, python)

def render_conversation(system, messages, tools, force_name):
    def text(role, m):
        if role in ("user", "assistant"):
            return f"[{role.upper()}]\n{m}"
        if role == "assistant_tool":
            return f"[ASSISTANT]\n(called {m['name']} with {json.dumps(m['args'], ensure_ascii=False)})"
        return f"[TOOL RESULT]\n{json.dumps(m['result'], ensure_ascii=False)}"
    tool_text = "\n".join(f"- {t['name']}: {t.get('description', '')}\n  parameters (JSON schema): {json.dumps(t.get('parameters', {}), ensure_ascii=False)}" for t in tools)
    names = [t["name"] for t in tools]
    return (f"{system}\n\n=== TOOLS ===\n{tool_text}\n\n=== CONVERSATION ===\n" + "\n\n".join(text(r, m) for r, m in messages)
            + "\n\n=== YOUR TURN ===\nAnswer with exactly one tool call as a JSON object {\"tool\": <name>, \"arguments_json\": <the arguments as a JSON-encoded string>}; "
            + (f"the tool must be {force_name}. " if force_name else f"the tool must be one of: {', '.join(names)}. ")
            + "Do not run shell commands and do not read files: everything you need is above.")


class Codex(Backend):
    SCHEMA = {"type": "object", "properties": {"tool": {"type": "string"}, "arguments_json": {"type": "string"}},
              "required": ["tool", "arguments_json"], "additionalProperties": False}
    USAGE_WORDS = ("usage limit", "rate limit", "quota", "too many requests", "try again later", "limit reached")
    AUTH_WORDS = ("not logged in", "unauthorized", "401", "please log in", "login required", "invalid token")

    def binary(self):
        for c in (os.environ.get("CODEX_BIN"), shutil.which("codex")):
            if c and os.path.exists(c):
                return c
        raise AccountError("codex CLI not found (set CODEX_BIN)")

    def turn(self, system, messages, tools, force_name):
        prompt = render_conversation(system, messages, tools, force_name) + (f" (run id: {self.cfg['tag']})" if self.cfg.get("tag") else "")
        names = [t["name"] for t in tools]
        last = None
        for attempt in range(int(self.cfg.get("retries", 2)) + 1):
            try:
                with tempfile.TemporaryDirectory(prefix="codex-turn-") as d:
                    schema, out = os.path.join(d, "schema.json"), os.path.join(d, "out.txt")
                    with open(schema, "w") as f:
                        f.write(json.dumps(self.SCHEMA))
                    cmd = [self.binary(), "exec", "-m", self.cfg["model"], "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
                           "-s", "read-only", "-C", d, "-c", f'model_reasoning_effort="{self.cfg.get("effort", "medium")}"', "--output-schema", schema, "-o", out, "-"]
                    t0 = time.monotonic()
                    try:
                        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.cfg.get("timeout_s", 900))
                    except subprocess.TimeoutExpired:
                        raise RuntimeError("codex call timed out")
                    log = (r.stdout or "") + (r.stderr or "")
                    reply = open(out).read().strip() if os.path.exists(out) else ""
                    if r.returncode != 0 or not reply:
                        low = log.lower()
                        if any(w in low for w in self.USAGE_WORDS):
                            raise QuotaError("codex usage/rate limit: " + log[-300:])
                        if any(w in low for w in self.AUTH_WORDS):
                            raise AccountError("codex account error: " + log[-300:])
                        raise RuntimeError(f"codex exec failed (exit {r.returncode}): {log[-300:]}")
                m = re.search(r"tokens used\s*\n?\s*([0-9,]+)", log)
                obj = json.loads(reply)
                if obj.get("tool") not in names:
                    raise ValueError(f"unknown tool {obj.get('tool')!r}")
                args = json.loads(obj.get("arguments_json") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not an object")
                meta = {"seconds": round(time.monotonic() - t0, 1), "usage": {"total_tokens": int(m.group(1).replace(",", "")) if m else None}, "backend": "codex", "model": self.cfg["model"]}
                return obj["tool"], args, meta, None
            except (ValueError, RuntimeError) as e:
                last = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"codex call failed: {type(last).__name__}: {str(last)[:300]}")


class Python(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        mod, _, fn = cfg["callable"].partition(":")
        self.fn = getattr(importlib.import_module(mod), fn)

    def turn(self, system, messages, tools, force_name):
        name, args = self.fn(system, messages, tools)
        if force_name and name != force_name:
            name, args = self.fn(system, messages + [("user", f"You must call {force_name} now.")], tools[:1])
        return name, args, {"backend": "python"}, None


BACKENDS = {"openai_compatible": OpenAICompatible, "codex": Codex, "python": Python}


def make(cfg: dict) -> Backend:
    b = cfg.get("backend")
    if b not in BACKENDS:
        raise ValueError(f"unknown proposer/verifier backend {b!r}; one of {', '.join(BACKENDS)}")
    return BACKENDS[b](cfg)
