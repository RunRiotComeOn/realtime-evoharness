"""A harness: the three evolvable parts of a real-time decision agent, kept in one directory.

  decision_interface.json   the Decision Interface: what the decision model is asked at every decision point:
                            {"goal": text, "questions": {name: {"instructions", "default", "options": {LABEL: description}}}}
  executor.py               the Executor: class Executor turns the committed choices into controls, every control tick
  playbook.json             the Playbook: advice shown to the decision model:
                            {"default": {question: LABEL}, "rules": [{"when": [{"field", "op", "value"}], "then": {question: LABEL}, "note"}]}
                            later rules override earlier ones

An edit (an "item" in the proposal's JSON) changes one part:
  {"part": "executor", "op": "replace", "old": exact text (must occur once), "new": text}
  {"part": "executor", "op": "rewrite", "source": the whole file}
  {"part": "decision_interface" | "playbook", "op": "set", "path": [keys...], "value": json}
  {"part": "decision_interface" | "playbook", "op": "delete", "path": [keys...]}
Every item also has an "id", a "rationale" and "depends_on" (ids of the items it needs).
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading

PARTS = ("decision_interface", "executor", "playbook")
LABEL = re.compile(r"^[A-Z][A-Z0-9_]{0,23}$")
NAME = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "vars"}


class HarnessError(ValueError):
    pass


@dataclass
class Limits:
    questions: int = 8
    options: int = 6
    rules: int = 24
    conditions: int = 3
    text: int = 300
    executor_imports: tuple = ("math", "numpy")

    @classmethod
    def from_config(cls, cfg: dict | None, imports=None) -> "Limits":
        cfg = dict(cfg or {})
        if imports is not None:
            cfg["executor_imports"] = tuple(imports)
        return cls(**cfg)


@dataclass
class Schema:
    """What a harness may refer to: the playbook's condition fields (declared by the Env) and the limits."""
    features: dict = field(default_factory=dict)   # name -> description
    limits: Limits = field(default_factory=Limits)


# ---------------------------------------------------------------- files

def load(path) -> dict:
    path = Path(path)
    di = path / "decision_interface.json"
    if not di.exists() and (path / "action_space.json").exists():
        di = path / "action_space.json"   # harnesses saved before the rename
    return {
        "decision_interface": json.loads(di.read_text()),
        "playbook": json.loads((path / "playbook.json").read_text()),
        "executor": (path / "executor.py").read_text(),
    }


def save(h: dict, path) -> Path:
    """Write a harness directory atomically (a kill leaves the old or the new version, never a mix)."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}-{threading.get_ident()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "decision_interface.json").write_text(json.dumps(h["decision_interface"], ensure_ascii=False, indent=1))
    (tmp / "playbook.json").write_text(json.dumps(h["playbook"], ensure_ascii=False, indent=1))
    (tmp / "executor.py").write_text(h["executor"])
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    return path


def digest(h: dict) -> str:
    blob = json.dumps([h["decision_interface"], h["playbook"], h["executor"]], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- validation

def validate_decision_interface(a, schema: Schema):
    lim = schema.limits
    if not isinstance(a, dict) or set(a) - {"goal", "questions"} or not isinstance(a.get("goal"), str) or not a["goal"].strip():
        raise HarnessError("decision_interface must be {'goal': text, 'questions': {...}}")
    qs = a.get("questions")
    if not isinstance(qs, dict) or not 1 <= len(qs) <= lim.questions:
        raise HarnessError(f"decision_interface needs 1-{lim.questions} questions")
    for name, q in qs.items():
        if not NAME.match(name):
            raise HarnessError(f"question name {name!r}: lowercase letters, digits, _")
        if not isinstance(q, dict) or set(q) != {"instructions", "default", "options"}:
            raise HarnessError(f"question {name}: keys must be instructions, default, options")
        if not isinstance(q["instructions"], str) or not q["instructions"].strip() or len(q["instructions"]) > lim.text:
            raise HarnessError(f"question {name}: instructions text (<= {lim.text} chars)")
        opts = q["options"]
        if not isinstance(opts, dict) or not 2 <= len(opts) <= lim.options:
            raise HarnessError(f"question {name}: 2-{lim.options} options")
        for lab, desc in opts.items():
            if not LABEL.match(lab):
                raise HarnessError(f"question {name}: option {lab!r} must be UPPER_CASE")
            if not isinstance(desc, str) or not desc.strip() or len(desc) > lim.text:
                raise HarnessError(f"question {name}: option {lab} needs a description (<= {lim.text} chars)")
        if q["default"] not in opts:
            raise HarnessError(f"question {name}: default {q['default']!r} is not an option")
    return a


def validate_playbook(p, a, schema: Schema):
    lim = schema.limits
    qs = a["questions"]
    if not isinstance(p, dict) or set(p) - {"default", "rules"}:
        raise HarnessError("playbook must be {'default': {...}, 'rules': [...]}")

    def choice(c, where):
        if not isinstance(c, dict):
            raise HarnessError(f"{where}: must be {{question: LABEL}}")
        for q, lab in c.items():
            if q not in qs:
                raise HarnessError(f"{where}: unknown question {q!r}")
            if lab not in qs[q]["options"]:
                raise HarnessError(f"{where}: {lab!r} is not an option of {q}")

    choice(p.get("default", {}), "playbook default")
    rules = p.get("rules", [])
    if not isinstance(rules, list) or len(rules) > lim.rules:
        raise HarnessError(f"playbook rules: a list of at most {lim.rules}")
    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            raise HarnessError(f"rule {i}: must be an object")
        extra = set(r) - {"when", "then", "note"}
        if extra:
            raise HarnessError(f"rule {i}: unexpected keys {sorted(extra)}; a rule has only when, then, note")
        w = r.get("when")
        if not isinstance(w, list) or not 1 <= len(w) <= lim.conditions:
            raise HarnessError(f"rule {i}: 1-{lim.conditions} conditions")
        for c in w:
            if (not isinstance(c, dict) or c.get("field") not in schema.features or c.get("op") not in ("<", ">")
                    or type(c.get("value")) not in (int, float)):
                raise HarnessError(f"rule {i}: condition {c!r} (fields {sorted(schema.features)}, op < or >, numeric value)")
        if not r.get("then"):
            raise HarnessError(f"rule {i}: then must set at least one question")
        choice(r["then"], f"rule {i} then")
    return p


def validate_executor(src, schema: Schema):
    """Static checks only: it compiles, defines class Executor, imports only allowed modules and calls no file,
    eval or import builtins. This is a guard against accidents, not a security sandbox."""
    if not isinstance(src, str) or "class Executor" not in src:
        raise HarnessError("executor.py must define class Executor")
    try:
        tree = ast.parse(src, "executor.py")
        compile(tree, "executor.py", "exec")
    except SyntaxError as e:
        raise HarnessError(f"executor.py does not compile: {e}")
    allowed = set(schema.limits.executor_imports) | {"__future__"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            if n.split(".")[0] not in allowed:
                raise HarnessError(f"executor.py imports {n!r}; allowed: {sorted(allowed - {'__future__'})}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise HarnessError(f"executor.py calls {node.func.id}(), which is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__", "__name__"):
            raise HarnessError(f"executor.py uses {node.attr}, which is not allowed")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Executor" for n in tree.body):
        raise HarnessError("executor.py must define class Executor at the top level")
    return src


def validate(h: dict, schema: Schema) -> dict:
    validate_decision_interface(h["decision_interface"], schema)
    validate_playbook(h["playbook"], h["decision_interface"], schema)
    validate_executor(h["executor"], schema)
    return h


# ---------------------------------------------------------------- items

def apply_item(h: dict, item: dict) -> dict:
    """One change item on a copy of the harness (no validation)."""
    h = copy.deepcopy(h)
    part, op = item.get("part"), item.get("op")
    if part not in PARTS:
        raise HarnessError(f"part must be one of {PARTS}")
    if part == "executor":
        if op == "replace":
            old, new = item.get("old"), item.get("new")
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise HarnessError("executor replace needs old and new text")
            n = h["executor"].count(old)
            if n != 1:
                raise HarnessError(f"executor replace: old text occurs {n} times (must be exactly once)")
            h["executor"] = h["executor"].replace(old, new, 1)
        elif op == "rewrite":
            if not isinstance(item.get("source"), str):
                raise HarnessError("executor rewrite needs source")
            h["executor"] = item["source"]
        else:
            raise HarnessError("executor op must be replace or rewrite")
        return h
    path = item.get("path")
    if not isinstance(path, list) or not path or not all(isinstance(k, (str, int)) for k in path):
        raise HarnessError("path must be a non-empty list of keys")
    node = h[part]
    for k in path[:-1]:
        try:
            node = node[k]
        except (KeyError, IndexError, TypeError):
            raise HarnessError(f"path {path}: {k!r} does not exist")
    last = path[-1]
    if op == "set":
        if "value" not in item:
            raise HarnessError("set needs value")
        if isinstance(node, list):
            if not isinstance(last, int) or not 0 <= last <= len(node):
                raise HarnessError(f"path {path}: list index out of range")
            if last == len(node):
                node.append(item["value"])
            else:
                node[last] = item["value"]
        elif isinstance(node, dict):
            node[last] = item["value"]
        else:
            raise HarnessError(f"path {path}: cannot set inside a {type(node).__name__}")
    elif op == "delete":
        try:
            del node[last]
        except (KeyError, IndexError, TypeError):
            raise HarnessError(f"path {path}: nothing to delete")
    else:
        raise HarnessError("op must be set or delete")
    return h


def apply_items(h: dict, items: list) -> dict:
    for it in items:
        h = apply_item(h, it)
    return h


# ---------------------------------------------------------------- run time

def defaults(decision_interface: dict) -> dict:
    return {q: s["default"] for q, s in decision_interface["questions"].items()}


def resolve(playbook: dict, values: dict):
    """The playbook's advice for one decision point: (choice {question: LABEL}, notes of the matching rules)."""
    out = dict(playbook.get("default", {}))
    notes = []
    for r in playbook.get("rules", []):
        def holds(c):
            v = values.get(c["field"])
            return v is not None and (v < c["value"] if c["op"] == "<" else v > c["value"])
        if all(holds(c) for c in r["when"]):
            out.update(r["then"])
            notes.append(r.get("note", ""))
    return out, notes
