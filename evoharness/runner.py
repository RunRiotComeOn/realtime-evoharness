"""One episode with a harness. Runs in its own process by default:

  python -m evoharness.runner < spec.json      (writes the result to spec["out"])

spec: {"config": {...}, "root": dir, "harness": dir, "scenario": name, "seed": int, "cache": path or null,
       "max_seconds": number or null, "probe": bool, "point": id or null}

Decision timing (config decision.timing):
  blocking   the scenario waits for every answer (no deadline), so an episode measures the harness, not the network;
             identical requests reuse cached answers, so a repeated episode is reproducible.
  deadline   ticks are paced to real time; when a point is due the playbook's advice (plus defaults) is committed at
             once and the model's answer replaces it if it arrives within deadline_ms.
Without an answer the defaults plus the playbook's advice are used ("fallback"); the parent stops the run when too many
decisions fall back, or at once when the model is unavailable (DecisionUnavailable).
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

from . import decision as D
from . import harness as H
from .config import Config
from .interfaces import DecisionError, DecisionUnavailable

PROBE_STEPS = 3


def load_executor_module(hdir: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"executor_{tag}", str(Path(hdir) / "executor.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def questions_of(action_space: dict) -> dict:
    return {q: {"instructions": s["instructions"], "options": dict(s["options"])} for q, s in action_space["questions"].items()}


def summarize(rows: list[dict], dt: float) -> dict:
    out = {"time_s": round(len(rows) * dt, 3)}
    if rows:
        for k, v in rows[0].items():
            if k != "t" and isinstance(v, (int, float)) and not isinstance(v, bool):
                vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                if vals:
                    out["mean_" + k] = round(sum(vals) / len(vals), 3)
    return out


def option_probe(mod, env, ctx, action_space, point_ids):
    """For every question and option: a fingerprint of what the executor does when that option is committed at every
    decision point (its state after commit, except copies of the choices, and act() on the Env's probe observations).
    Two options with the same fingerprint behave the same."""
    import numpy as np
    qs = action_space["questions"]
    dflt = H.defaults(action_space)
    labels = {l for s in qs.values() for l in s["options"]}
    obs_list = env.probe_observations()
    rep = getattr(env, "control_repr", None) or (lambda c: json.dumps(c, sort_keys=True, default=repr))
    base = mod.Executor(ctx, action_space)

    def state_text(ex):
        parts = []
        for k, v in sorted(vars(ex).items()):
            if v is ctx or callable(v) or k in ("choices", "ctx", "defaults"):
                continue
            if isinstance(v, np.ndarray):
                t = hashlib.sha256(np.round(v.astype(float), 6).tobytes()).hexdigest() if v.dtype.kind in "fiub" else repr(v.tolist())
            else:
                try:
                    t = json.dumps(v, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o))
                except Exception:
                    t = repr(v)
            if any(f'"{l}"' in t or f"'{l}'" in t for l in labels):
                continue   # a copy of the choices, not behaviour
            parts.append(k + "=" + t)
        return "|".join(parts)

    def fingerprint(choice):
        try:
            ex = copy.deepcopy(base, {id(ctx): ctx})
        except Exception:
            ex = mod.Executor(ctx, action_space)
        for pid in point_ids:
            ex.commit(pid, dict(choice))
        out = [state_text(ex)]
        for o in obs_list:
            o2 = copy.deepcopy(o)
            for _ in range(PROBE_STEPS):
                c = ex.act(o2)
                r = rep(c)
                out.append(r if isinstance(r, str) else ",".join(f"{float(x):.4f}" for x in r))
        return hashlib.sha256("\n".join(out).encode()).hexdigest()[:16]

    return {q: {lab: fingerprint(dflt | {q: lab}) for lab in s["options"]} for q, s in qs.items()}


class Decider:
    """Builds the request for a decision point, asks the model (cached), falls back to defaults plus advice."""

    def __init__(self, cfg, env, harness, cache):
        self.cfg = cfg["decision"]
        self.env = env
        self.A = harness["action_space"]
        self.P = harness["playbook"]
        self.questions = questions_of(self.A)
        self.defaults = H.defaults(self.A)
        self.cache = cache
        self.new_answers = {}
        self.calls = []
        self.model = D.make(self.cfg)

    def request(self, point, obs):
        live = self.env.live_features(obs, point)
        advice, notes = H.resolve(self.P, dict(point.get("features") or {}) | dict(live))
        state = {"goal": self.A["goal"], "point": self.env.describe_point(point), "live": live}
        if advice:
            state["instruction"] = {"recommended": advice, "matched_rules": notes}
        return state, advice

    def ask(self, state):
        """(choice or None, source). Raises DecisionUnavailable."""
        if self.model is None:
            return None, "default"
        key = D.request_key(state, self.questions)
        if key in self.cache:
            return self.cache[key], "cache"
        if key in self.new_answers:
            return self.new_answers[key], "cache"
        try:
            ans = self.model.decide(state, self.questions)
        except DecisionError as e:
            self.calls.append({"error": str(e)[:120]})
            return None, "fallback"
        self.new_answers[key] = ans
        return ans, ("random" if self.cfg.get("backend") == "random" else "model")

    def record(self, point, source, ms):
        self.calls.append({"point": point["id"], "source": source, "ms": ms})

    def stats(self):
        c = [x for x in self.calls if "source" in x]
        return {"calls": len(c), "model": sum(x["source"] == "model" for x in c), "cached": sum(x["source"] == "cache" for x in c),
                "fallback": sum(x["source"] == "fallback" for x in c), "default": sum(x["source"] == "default" for x in c),
                "random": sum(x["source"] == "random" for x in c),
                "errors": [x["error"] for x in self.calls if "error" in x][:5],
                "mean_ms": round(sum(x["ms"] for x in c) / len(c)) if c else None}


def run(spec: dict) -> dict:
    cfg = Config.from_dict(spec["config"], spec["root"])
    EnvCls = cfg.env_class()
    schema = H.Schema(features=dict(EnvCls.FEATURES), limits=H.Limits.from_config(cfg["limits"], cfg["executor_imports"]))
    h = H.validate(H.load(spec["harness"]), schema)
    dg = H.digest(h)
    out = {"scenario": spec["scenario"], "seed": spec.get("seed", 0), "harness": dg}
    if spec.get("point"):
        out["point"] = spec["point"]
    cache = json.loads(Path(spec["cache"]).read_text()) if spec.get("cache") and Path(spec["cache"]).exists() else {}
    env = EnvCls(**cfg["env_options"])
    dt = float(EnvCls.CONTROL_DT)
    fields = ["t", "point"] + list(EnvCls.TRACE_FIELDS)
    trace_every = int(cfg["trace_every"])
    timing = cfg["decision"]["timing"]
    lead = float(cfg["decision"].get("lead") or 0)
    pool = ThreadPoolExecutor(4) if timing == "deadline" else None
    try:
        obs = env.start_at(spec["scenario"], spec["point"], spec.get("seed", 0)) if spec.get("point") else env.reset(spec["scenario"], spec.get("seed", 0))
        points = env.decision_points()
        ids = [p["id"] for p in points]
        ctx = env.executor_context()
        try:
            mod = load_executor_module(spec["harness"], dg)
            ex = mod.Executor(ctx, h["action_space"])
        except Exception:
            out.update(error="executor_init", traceback=traceback.format_exc()[-2500:])
            return out
        if spec.get("probe") and hasattr(env, "probe_observations"):
            try:
                out["probe"] = option_probe(mod, env, ctx, h["action_space"], ids)
            except Exception:
                out["probe_error"] = traceback.format_exc()[-1500:]
        has_sample = hasattr(ex, "sample")
        if has_sample:
            fields = fields + ["sample"]
        dec = Decider(cfg, env, h, cache)
        decided = {}
        pending = {}   # deadline mode: point id -> (future, t0, advice)

        def decide(p, obs):
            state, advice = dec.request(p, obs)
            t0 = time.monotonic()
            if timing == "deadline" and dec.model is not None:
                provisional = dec.defaults | advice
                ex.commit(p["id"], provisional)
                decided[p["id"]] = {"choice": provisional, "advice": advice, "source": "pending"}
                pending[p["id"]] = (pool.submit(dec.ask, state), t0, advice)
                return
            choice, source = dec.ask(state)
            if choice is None:
                choice = dec.defaults | advice
            dec.record(p, source, round((time.monotonic() - t0) * 1000))
            ex.commit(p["id"], choice)
            decided[p["id"]] = {"choice": choice, "advice": advice}

        def settle(final=False):
            for pid, (fut, t0, advice) in list(pending.items()):
                late = (time.monotonic() - t0) * 1000 > cfg["decision"]["deadline_ms"]
                if fut.done() and not late:
                    choice, source = fut.result()
                    if choice is not None:
                        ex.commit(pid, choice)
                        decided[pid]["choice"] = choice
                    dec.record({"id": pid}, source if choice is not None else "fallback", round((time.monotonic() - t0) * 1000))
                    del pending[pid]
                elif late or final:
                    dec.record({"id": pid}, "fallback", round((time.monotonic() - t0) * 1000))
                    dec.calls[-1]["late"] = True
                    del pending[pid]

        trace, per_point, frame = [], {}, 0
        max_s = spec.get("max_seconds") or EnvCls.MAX_SECONDS
        stage = "executor_commit"
        try:
            for p in points:
                if p["id"] not in decided and env.due(obs, p, lead):
                    decide(p, obs)
            stage = "executor_act"
            wall0 = time.monotonic()
            done = False
            while frame * dt < max_s and not done:
                if pending:
                    settle()
                cur = env.current_point(obs)
                control = ex.act(obs)
                row = env.trace_row(obs, control)
                rd = dict(zip(fields, [round(frame * dt, 3), cur] + list(row)))
                if cur is not None:
                    per_point.setdefault(cur, []).append(rd)
                if frame % trace_every == 0:
                    if has_sample:
                        try:
                            rd = rd | {"sample": {k: round(float(v), 3) for k, v in ex.sample().items()}}
                        except Exception:
                            rd = rd | {"sample": None}
                    trace.append([rd.get(f) for f in fields])
                obs, done = env.step(control)
                frame += 1
                stage = "executor_commit"
                for p in points:
                    if p["id"] not in decided and env.due(obs, p, lead):
                        decide(p, obs)
                stage = "executor_act"
                if timing == "deadline":
                    lag = wall0 + frame * dt - time.monotonic()
                    if lag > 0:
                        time.sleep(lag)
            settle(final=True)
        except DecisionUnavailable:
            raise
        except Exception:
            out.update(error=stage, traceback=traceback.format_exc()[-2500:], t=round(frame * dt, 2))
            out["decisions"] = dec.stats()
            out["new_answers"] = dec.new_answers
            return out
        res = env.result()
        out.update(res)
        summ = getattr(env, "summarize_point", None)
        rows = []
        for p in points:
            pr = per_point.get(p["id"], [])
            s = summ(pr) if summ else summarize(pr, dt)
            s.setdefault("time_s", round(len(pr) * dt, 3))
            d = decided.get(p["id"]) or {}
            rows.append({"id": p["id"], **{k: v for k, v in (p.get("features") or {}).items()}, **s,
                         "choice": d.get("choice"), "advice": d.get("advice")})
        out["points"] = rows if not spec.get("point") else [r for r in rows if r["id"] == spec["point"]]
        out["trace_fields"] = fields
        out["trace"] = trace if not spec.get("point") else [r for r in trace if r[1] == spec["point"]]
        if hasattr(env, "incidents"):
            try:
                dict_rows = [dict(zip(fields, r)) for r in trace]
                out["incidents"] = env.incidents(dict_rows, res)[:4]
            except Exception:
                out["incidents_error"] = traceback.format_exc()[-800:]
        out["decisions"] = dec.stats()
        out["new_answers"] = dec.new_answers
        return out
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
        try:
            env.close()
        except Exception:
            pass


def main():
    spec = json.loads(sys.stdin.read())
    try:
        res = run(spec)
    except DecisionUnavailable as e:
        res = {"scenario": spec.get("scenario"), "fatal": "decision_unavailable", "status": str(e.args[0]) if e.args else ""}
    Path(spec["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["out"]).write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
