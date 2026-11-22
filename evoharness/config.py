"""Run configuration: one YAML file per experiment. Paths in it are relative to the file."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys

import yaml

DEFAULTS = {
    "env": None,                    # "package.module:ClassName"
    "env_options": {},
    "seed": None,                   # seed harness directory
    "domain_brief": None,           # markdown file with facts about the scenario, shown to the proposer
    "python_path": ["."],           # added to sys.path (relative to the config file) so that `env` imports
    "scenarios": {"train": [], "test": []},
    "objective": {
        "metric": "time_s",
        "direction": "min",         # min | max
        "aggregate": "sum",         # sum | mean
        "failure_value": None,      # an unfinished episode counts as this value
        "description": "",          # one line for the prompts, e.g. "total time over the training scenarios, in seconds"
        "unit": "",
    },
    "gate": {
        "scenarios": "train",       # "train", a list, or {"train_plus": [...]}
        "repeats": 1,
        "min_gain": 0.0,            # in objective units, positive = better
        "min_gain_relative": None,  # or a fraction of the champion's value
        "constraints": [{"metric": "unfinished", "max_increase": 0}, {"metric": "errors", "max": 0}],
        "per_scenario_max_loss": None,
    },
    "decision": {
        "backend": "default",       # default | random | openai_compatible | jev | python
        "timing": "blocking",       # blocking | deadline
        "deadline_ms": 1000,
        "lead": 0.0,
        "max_fallback": 0.2,
    },
    "proposer": {"backend": "openai_compatible"},
    "verifier": {"enabled": True},  # backend defaults to the proposer's
    "loop": {
        "rounds": 40,
        "max_items": 16,
        "repairs": 3,
        "action_space_every": 4,
        "parts": ["action_space", "executor", "playbook"],
        "branches": {"enabled": True, "rounds": 4, "extend": 2, "max": 10, "wire_attempts": 2},
        "tools": {"trace": 4, "trial_episode": 3, "point_trial": 20},
        "smoke_seconds": 10.0,
        "edit_log": 40,
    },
    "limits": {"questions": 8, "options": 6, "rules": 24, "conditions": 3, "text": 300},
    "executor_imports": ["math", "numpy"],
    "prompt_language": "en",
    "parallel": 4,
    "isolation": "process",         # process | inline (inline: no subprocess, for debugging only)
    "episode_timeout_s": 900,
    "trace_every": 3,
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config(dict):
    """A merged configuration dict with a few helpers. `root` is the directory of the config file."""

    def __init__(self, data: dict, root: Path):
        super().__init__(data)
        self.root = Path(root)

    @classmethod
    def load(cls, path, overrides: dict | None = None) -> "Config":
        path = Path(path).resolve()
        data = merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})
        data = merge(data, overrides or {})
        cfg = cls(data, path.parent)
        cfg.check()
        return cfg

    @classmethod
    def from_dict(cls, data: dict, root=".") -> "Config":
        cfg = cls(merge(DEFAULTS, data), Path(root).resolve())
        cfg.check()
        return cfg

    def path(self, p):
        return None if p is None else (self.root / p).resolve()

    def check(self):
        if not self["env"] or ":" not in self["env"]:
            raise ValueError("config: env must be 'module:ClassName'")
        if not self["scenarios"].get("train"):
            raise ValueError("config: scenarios.train is empty")
        o = self["objective"]
        if o["direction"] not in ("min", "max") or o["aggregate"] not in ("sum", "mean"):
            raise ValueError("config: objective.direction must be min|max and aggregate sum|mean")
        if o["failure_value"] is None:
            raise ValueError("config: objective.failure_value is required (the value of an unfinished episode)")
        if self["decision"]["timing"] not in ("blocking", "deadline"):
            raise ValueError("config: decision.timing must be blocking or deadline")

    # ---- derived ------------------------------------------------------------------------------------------
    def sys_path(self) -> list[str]:
        return [str(self.path(p)) for p in self["python_path"]]

    def env_class(self):
        for p in reversed(self.sys_path()):
            if p not in sys.path:
                sys.path.insert(0, p)
        mod, _, name = self["env"].partition(":")
        return getattr(importlib.import_module(mod), name)

    def gate_scenarios(self) -> list[str]:
        g = self["gate"]["scenarios"]
        train = list(self["scenarios"]["train"])
        if g == "train" or g is None:
            return train
        if isinstance(g, dict) and "train_plus" in g:
            return train + [s for s in g["train_plus"] if s not in train]
        return list(g)

    def verifier_backend(self) -> dict:
        v = dict(self["verifier"])
        v.pop("enabled", None)
        return merge(self["proposer"], v) if "backend" not in v else v

    def domain_brief(self) -> str:
        p = self.path(self["domain_brief"])
        return p.read_text() if p else ""
