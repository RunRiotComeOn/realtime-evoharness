"""Command line:

  evoharness check    CONFIG                      validate the Env, the seed harness, and run one short episode
  evoharness run      CONFIG --out DIR [--rounds N]
  evoharness evaluate CONFIG --harness NAME=DIR [--harness ...] --out DIR [--split test|train|gate] [--repeats N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import harness as H
from .config import Config
from .evaluator import Evaluator, RunStopped
from .llm import AccountError


def load_env_file(path=".env"):
    """KEY=VALUE lines into the environment (keys are never written anywhere by evoharness)."""
    import os
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def schema_of(cfg):
    E = cfg.env_class()
    return E, H.Schema(features=dict(E.FEATURES), limits=H.Limits.from_config(cfg["limits"], cfg["executor_imports"]))


def cmd_check(a):
    cfg = Config.load(a.config)
    E, schema = schema_of(cfg)
    missing = [m for m in ("CONTROL_DT", "MAX_SECONDS", "FEATURES", "EXECUTOR_DOC", "TRACE_FIELDS") if not hasattr(E, m)]
    missing += [m for m in ("reset", "decision_points", "due", "current_point", "step", "result", "executor_context",
                            "describe_point", "live_features", "trace_row", "close") if not callable(getattr(E, m, None))]
    if missing:
        sys.exit(f"{cfg['env']} lacks: {', '.join(missing)}")
    optional = [m for m in ("start_at", "probe_observations", "summarize_point", "incidents", "control_repr") if hasattr(E, m)]
    print(f"env {cfg['env']}: ok; optional capabilities: {', '.join(optional) or 'none'}")
    seed = H.validate(H.load(cfg.path(cfg["seed"])), schema)
    print(f"seed harness {cfg['seed']}: valid ({H.digest(seed)})")
    out = Path(a.out or (cfg.root / ".evoharness-check"))
    ev = Evaluator(cfg, out, schema)
    s = cfg["scenarios"]["train"][0]
    ep = ev.episode(seed, s, max_seconds=a.seconds, probe=True)
    if ep.get("error"):
        sys.exit(f"episode on {s} failed: {ep['error']}\n{ep.get('traceback', '')}")
    print(f"{a.seconds} s on {s}: ok; {len(ep.get('points') or [])} decision points; decisions {ep.get('decisions')}")


def cmd_run(a):
    from .loop import Evolution
    cfg = Config.load(a.config)
    try:
        Evolution(cfg, a.out).run(a.rounds)
    except (RunStopped, AccountError) as e:
        sys.exit(f"run stopped: {e}\n(rerun the same command to resume at the same round)")


def cmd_evaluate(a):
    cfg = Config.load(a.config)
    _, schema = schema_of(cfg)
    scen = cfg.gate_scenarios() if a.split == "gate" else cfg["scenarios"][a.split]
    ev = Evaluator(cfg, a.out, schema)
    report = {}
    for spec in a.harness:
        name, _, path = spec.partition("=")
        h = H.validate(H.load(path), schema)
        r = ev.evaluate(h, scen, a.repeats, name)
        report[name] = {k: r[k] for k in ("digest", "value", "unfinished", "errors", "per_scenario", "metrics")}
        print(json.dumps({name: report[name]["value"], "unfinished": r["unfinished"], "per_scenario": r["per_scenario"]}, ensure_ascii=False))
    Path(a.out, "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


def main(argv=None):
    load_env_file()
    p = argparse.ArgumentParser(prog="evoharness", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("config")
    c.add_argument("--seconds", type=float, default=5.0)
    c.add_argument("--out", default=None)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("config")
    e.add_argument("--harness", action="append", required=True, help="NAME=DIR")
    e.add_argument("--out", required=True)
    e.add_argument("--split", default="test", choices=["test", "train", "gate"])
    e.add_argument("--repeats", type=int, default=1)
    a = p.parse_args(argv)
    {"check": cmd_check, "run": cmd_run, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
