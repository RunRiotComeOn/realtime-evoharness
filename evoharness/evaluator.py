"""Episodes of a harness on many scenarios, in parallel, memoised by harness digest; the shared decision cache."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from . import decision as D
from . import harness as H
from .objective import Objective

EPISODE_ATTEMPTS = 3   # an episode process that dies without a result (crash, sleep, timeout) is run again


class RunStopped(Exception):
    """The decision model is not answering (refused, or too many fallbacks): stop instead of measuring defaults."""


def write_atomic(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


class Evaluator:
    def __init__(self, cfg, out, schema: H.Schema):
        self.cfg = cfg
        self.out = Path(out)
        self.schema = schema
        self.obj = Objective(cfg["objective"])
        self.parallel = max(1, int(cfg["parallel"]))
        self.cache_path = self.out / "decision-cache.json"
        self.cache = read_json(self.cache_path, {})
        self.lock = threading.Lock()
        self.dead = {}
        self._smoke = {}
        for d in ("evals", "tmp", "harnesses"):
            (self.out / d).mkdir(parents=True, exist_ok=True)

    # ---- one episode ------------------------------------------------------------------------------------------
    def harness_dir(self, h) -> Path:
        d = H.digest(h)
        p = self.out / "harnesses" / d
        with self.lock:
            if not (p / "executor.py").exists():
                H.save(h, p)
        return p

    def episode(self, h, scenario, seed=0, max_seconds=None, probe=False, point=None) -> dict:
        d = H.digest(h)
        tag = f"{d}-{scenario}-{seed}-{point or ''}-{int(max_seconds or 0)}-{threading.get_ident()}"
        out = self.out / "tmp" / f"{tag}.json"
        spec = {"config": dict(self.cfg), "root": str(self.cfg.root), "harness": str(self.harness_dir(h)), "scenario": scenario,
                "seed": seed, "cache": str(self.cache_path), "max_seconds": max_seconds, "probe": probe, "point": point, "out": str(out)}
        if self.cfg["isolation"] == "inline":
            from . import runner
            from .interfaces import DecisionUnavailable
            try:
                r = runner.run(spec)
            except DecisionUnavailable as e:
                r = {"scenario": scenario, "fatal": "decision_unavailable", "status": str(e.args[0]) if e.args else ""}
        else:
            err = ""
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1])] + self.cfg.sys_path() + [env.get("PYTHONPATH", "")])
            for _ in range(EPISODE_ATTEMPTS):
                out.unlink(missing_ok=True)
                try:
                    p = subprocess.run([sys.executable, "-m", "evoharness.runner"], input=json.dumps(spec), text=True, capture_output=True,
                                       cwd=self.cfg.root, timeout=self.cfg["episode_timeout_s"], env=env)
                    err = p.stderr[-2000:]
                except subprocess.TimeoutExpired:
                    err = "timeout"
                if read_json(out) is not None:
                    break
            r = read_json(out)
            out.unlink(missing_ok=True)
            if r is None:
                return {"scenario": scenario, "seed": seed, "error": "episode_process_failed", "traceback": err}
        self.check_decisions(r)
        with self.lock:
            if r.get("new_answers"):
                self.cache.update(r["new_answers"])
                write_atomic(self.cache_path, json.dumps(self.cache))
        r.pop("new_answers", None)
        return r

    def check_decisions(self, r):
        if r.get("fatal") == "decision_unavailable":
            raise RunStopped(f"the decision model refused the request ({r.get('status')}); check its key, credit and region")
        j = r.get("decisions")
        dcfg = self.cfg["decision"]
        if D.is_model(dcfg) and j and not r.get("error"):
            n = j.get("calls", 0)
            if n >= 4 and j.get("fallback", 0) > float(dcfg.get("max_fallback", 0.2)) * n:
                raise RunStopped(f"the decision model answered only {n - j['fallback']} of {n} decisions on {r.get('scenario')}: {j.get('errors')}")

    # ---- many episodes ------------------------------------------------------------------------------------------
    def evaluate(self, h, scenarios, repeats=1, tag="") -> dict:
        """Episodes of h on every scenario (x repeats), memoised per episode. Returns totals plus the episodes."""
        d = H.digest(h)
        jobs = [(s, k) for s in scenarios for k in range(repeats)]

        def one(job):
            s, k = job
            f = self.out / "evals" / d / f"{s}-r{k}.json"
            if (r := read_json(f)) is not None:
                return r
            r = self.episode(h, s, seed=k)
            if r.get("error") != "episode_process_failed":
                f.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(f, json.dumps(r, ensure_ascii=False))
            return r

        with ThreadPoolExecutor(self.parallel) as ex:
            eps = list(ex.map(one, jobs))
        if repeats > 1:   # one value per scenario: the mean over repeats (unfinished if any repeat is)
            merged = []
            for s in scenarios:
                group = [e for e in eps if e.get("scenario") == s]
                m = dict(group[0])
                fin = all(e.get("finished") and not e.get("error") for e in group)
                m["finished"] = fin
                if fin:
                    m[self.obj.metric] = sum(float(e[self.obj.metric]) for e in group) / len(group)
                m["repeats"] = [e.get(self.obj.metric) for e in group]
                merged.append(m)
            eps = merged
        res = {"digest": d, "tag": tag, "episodes": {e["scenario"]: e for e in eps}} | self.obj.totals(eps)
        return res

    def smoke(self, h) -> dict | None:
        """A few seconds on every training scenario: None if the executor runs everywhere, else the first error.
        Also probes which options have no effect (self.dead[digest])."""
        d = H.digest(h)
        if d in self._smoke:
            return self._smoke[d]
        secs = float(self.cfg["loop"]["smoke_seconds"])
        with ThreadPoolExecutor(self.parallel) as ex:
            eps = list(ex.map(lambda s: self.episode(h, s, max_seconds=secs, probe=True), self.cfg["scenarios"]["train"]))
        self.dead[d] = dead_options(h["decision_interface"], [e.get("probe") for e in eps])
        bad = [e for e in eps if e.get("error")]
        self._smoke[d] = None if not bad else {"scenario": bad[0]["scenario"], "error": bad[0]["error"], "traceback": (bad[0].get("traceback") or "")[-1800:]}
        return self._smoke[d]


def dead_options(decision_interface, probes):
    """Options that change nothing, from per-scenario probes: [question, LABEL] for an option that behaves exactly like
    the question's default everywhere, [question, '*'] for a question whose options all behave the same."""
    probes = [p for p in probes if p]
    out = []
    if not probes:
        return out
    for q, spec in decision_interface["questions"].items():
        if not all(q in p for p in probes):
            continue
        if all(len(set(p[q].values())) == 1 for p in probes):
            out.append([q, "*"])
            continue
        dflt = spec["default"]
        out += [[q, lab] for lab in spec["options"] if lab != dflt and all(p[q].get(lab) == p[q].get(dflt) for p in probes)]
    return out
