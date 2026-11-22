"""Decision-model backends: the fast model that answers the questions at every decision point.

  default            no model: the defaults plus the playbook's advice (a baseline)
  random             a random option for every question, seeded by the request (a baseline)
  openai_compatible  any OpenAI-compatible chat endpoint; the answer is a JSON object {question: LABEL}
  jev                the Jev decision service (answers with a probability per option)
  python             your own function "module:function"(state, questions) -> {question: LABEL}

Keys are read from the environment variable named by `api_key_env`, never from the config file.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random

from .interfaces import DecisionError, DecisionUnavailable

FATAL_STATUS = {401, 402, 403, 451}   # key refused, unpaid, forbidden, not available in this region


def request_key(state: dict, questions: dict) -> str:
    return hashlib.sha256(json.dumps([state, questions], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_answer(ans, questions):
    if not isinstance(ans, dict):
        raise DecisionError("answer is not an object")
    out = {}
    for q, spec in questions.items():
        lab = ans.get(q)
        if lab not in spec["options"]:
            raise DecisionError(f"invalid choice {lab!r} for {q}")
        out[q] = lab
    return out


class RandomDecision:
    def __init__(self, cfg):
        self.seed = cfg.get("rng_seed", 0)

    def decide(self, state, questions):
        rng = random.Random(f"{self.seed}-{request_key(state, questions)}")
        return {q: rng.choice(sorted(s["options"])) for q, s in questions.items()}


class _HTTP:
    def __init__(self, cfg):
        import httpx
        self.httpx = httpx
        self.cfg = cfg
        env = cfg.get("api_key_env")
        self.key = os.environ.get(env) if env else None
        if env and not self.key:
            raise DecisionUnavailable(f"environment variable {env} is not set")
        self.client = httpx.Client(trust_env=False, timeout=cfg.get("timeout_s", 10), proxy=cfg.get("proxy") or os.environ.get(cfg.get("proxy_env") or "") or None)

    def post(self, url, body):
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        try:
            r = self.client.post(url, headers=headers, json=body)
        except self.httpx.HTTPError as e:
            raise DecisionError(type(e).__name__)
        if r.status_code in FATAL_STATUS:
            raise DecisionUnavailable(r.status_code)
        if r.status_code >= 400:
            raise DecisionError(f"HTTP {r.status_code}")
        try:
            return r.json()
        except ValueError:
            raise DecisionError("reply is not JSON")


SYSTEM = ("You are the fast decision layer of a real-time agent. You receive the goal, the upcoming decision point, the live "
          "state and possibly an instruction with recommended answers learned from earlier episodes (follow it unless the "
          "live state clearly says otherwise). Answer every question with exactly one of its option labels. Reply with one "
          "JSON object {question: LABEL} and nothing else.")


class OpenAICompatibleDecision(_HTTP):
    def decide(self, state, questions):
        qs = {q: {"instructions": s["instructions"], "options": s["options"]} for q, s in questions.items()}
        body = {"model": self.cfg["model"], "temperature": self.cfg.get("temperature", 0),
                "messages": [{"role": "system", "content": self.cfg.get("system") or SYSTEM},
                             {"role": "user", "content": json.dumps({"state": state, "questions": qs}, ensure_ascii=False)}],
                "response_format": {"type": "json_object"}, "max_tokens": self.cfg.get("max_tokens", 200)}
        body.update(self.cfg.get("extra_body") or {})
        reply = self.post(self.cfg["base_url"].rstrip("/") + "/chat/completions", body)
        try:
            text = reply["choices"][0]["message"]["content"]
            return check_answer(json.loads(text), questions)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise DecisionError(f"malformed reply: {e}")


class JevDecision(_HTTP):
    def decide(self, state, questions):
        qs = {q: {"type": "choice", "instructions": s["instructions"], "criteria": dict(s["options"])} for q, s in questions.items()}
        body = {"model": self.cfg["model"], "state": json.dumps(state, ensure_ascii=False), "questions": qs}
        reply = self.post(self.cfg["endpoint"], body)
        try:
            out = {}
            for q in questions:
                probs = reply["answers"][q]["probabilities"]
                out[q] = max(probs, key=probs.get)
            return check_answer(out, questions)
        except (KeyError, TypeError, ValueError) as e:
            raise DecisionError(f"malformed reply: {e}")


class PythonDecision:
    def __init__(self, cfg):
        mod, _, fn = cfg["callable"].partition(":")
        self.fn = getattr(importlib.import_module(mod), fn)

    def decide(self, state, questions):
        return check_answer(self.fn(state, questions), questions)


BACKENDS = {"random": RandomDecision, "openai_compatible": OpenAICompatibleDecision, "jev": JevDecision, "python": PythonDecision}


def make(cfg: dict):
    """The decision model for a config, or None for the 'default' backend."""
    b = cfg.get("backend", "default")
    if b == "default":
        return None
    if b not in BACKENDS:
        raise ValueError(f"unknown decision backend {b!r}; one of default, {', '.join(BACKENDS)}")
    return BACKENDS[b](cfg)


def is_model(cfg: dict) -> bool:
    """Whether fallbacks mean the model is not answering (not true of the baselines)."""
    return cfg.get("backend", "default") not in ("default", "random")
