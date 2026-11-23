"""What the proposer and the reviewer see: compact tables of episodes, traces around incidents, readable diffs."""
from __future__ import annotations

import difflib

from . import harness as H

TRACE_ROWS = 60
INCIDENT_BEFORE_S = 3.0
INCIDENT_AFTER_S = 1.0
INCIDENT_ROWS = 45


def table(rows, extra_choice=True):
    """[[keys...], [values...], ...] with the choice as one compact column."""
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k not in ("choice", "advice"):
                keys.append(k)
    out = [keys + (["choice"] if extra_choice else [])]
    for r in rows:
        row = [r.get(k) for k in keys]
        if extra_choice:
            row.append(",".join(f"{q}={v}" for q, v in sorted((r.get("choice") or {}).items())))
        out.append(row)
    return out


def summary(ev, obj, keys=("value", "unfinished", "errors", "per_scenario")):
    s = {k: ev[k] for k in keys if k in ev}
    s["metrics"] = ev.get("metrics", {})
    return s


def episodes_view(ev, obj):
    view = {}
    for s, e in ev["episodes"].items():
        if e.get("error"):
            view[s] = {"error": e["error"], "traceback": (e.get("traceback") or "")[-1500:]}
            continue
        v = {"value": obj.episode_value(e), "finished": e.get("finished")}
        v |= {k: x for k, x in e.items() if isinstance(x, (int, float)) and not isinstance(x, bool) and k not in ("seed",)}
        v["decisions"] = e.get("decisions")
        v["points"] = table(e.get("points") or [])
        view[s] = v
    return view


def item_view(h, item):
    """What an item changes, readable: a diff for code, before/after for JSON."""
    try:
        new = H.apply_item(h, item)
    except H.HarnessError as e:
        return {"error": str(e)}
    if item["part"] == "executor":
        diff = "".join(difflib.unified_diff(h["executor"].splitlines(True), new["executor"].splitlines(True), "executor.py", "executor.py", n=3))
        return {"diff": diff[:12000]}
    node = h[item["part"]]
    for k in item["path"]:
        try:
            node = node[k]
        except (KeyError, IndexError, TypeError):
            node = "(absent)"
            break
    return {"path": item["path"], "before": node, "after": item.get("value", "(deleted)")}


def harness_diff(a, b):
    change = {"executor_diff": "".join(difflib.unified_diff(a["executor"].splitlines(True), b["executor"].splitlines(True), "executor.py", "executor.py", n=3))[:16000]}
    for part in ("decision_interface", "playbook"):
        if a[part] != b[part]:
            change[part] = {"before": a[part], "after": b[part]}
    return change


def thin(rows, n):
    step = max(1, -(-len(rows) // n))
    return rows[::step]


def point_trace(ev, scenario, point_ids):
    """Trace rows of a few decision points of one scenario."""
    ep = ev["episodes"].get(scenario)
    if ep is None:
        return {"error": f"unknown scenario {scenario}; scenarios: {sorted(ev['episodes'])}"}
    fields = ep.get("trace_fields") or []
    ids = [r["id"] for r in ep.get("points") or []]
    out = {"scenario": scenario, "fields": fields, "points": {}}
    for pid in point_ids[:3]:
        if pid not in ids:
            out["points"][pid] = {"error": f"unknown point; points: {ids}"}
            continue
        rows = [r for r in ep.get("trace") or [] if len(r) > 1 and r[1] == pid]
        out["points"][pid] = {"rows": thin(rows, TRACE_ROWS)} if rows else "(no trace: the episode failed or never reached this point)"
    if ep.get("error"):
        out["error"] = ep["error"]
    return out


def incidents(ep, base_ep=None):
    """Trace windows around the incidents an Env reported, if the episode has more of them than base_ep."""
    if not ep or ep.get("error") or not ep.get("trace") or not ep.get("incidents"):
        return []
    if base_ep and not base_ep.get("error") and len(ep.get("incidents") or []) <= len(base_ep.get("incidents") or []):
        return []
    fields = ep.get("trace_fields") or []
    tr = ep["trace"]
    out = []
    for inc in ep["incidents"][:2]:
        t = float(inc.get("t", 0))
        sel = [r for r in tr if t - INCIDENT_BEFORE_S <= r[0] <= t + INCIDENT_AFTER_S]
        before = next((r[1] for r in tr if r[0] >= t - INCIDENT_BEFORE_S + 0.5), None)
        out.append({"scenario": ep["scenario"], "kind": inc.get("kind"), "t": t, "point_before": before, "fields": fields, "rows": thin(sel, INCIDENT_ROWS)})
    return out


def dead_text(dead, decision_interface, lang="en"):
    qs = decision_interface["questions"]
    out = []
    for q, lab in dead:
        if lab == "*":
            out.append(f"{q}: 所有选项在 executor 里的效果完全相同（这个问题没有作用）" if lang == "zh" else f"{q}: all its options make the executor behave the same (the question has no effect)")
        else:
            d = qs.get(q, {}).get("default")
            out.append(f"{q}.{lab}: 效果和默认选项 {d} 完全相同" if lang == "zh" else f"{q}.{lab}: behaves exactly like the default option {d}")
    return out


def point_changes(obj, a, b, threshold):
    """Per scenario, the points whose time differs by at least threshold (b minus a, positive = b slower)."""
    out = {}
    for s, eb in b["episodes"].items():
        ea = a["episodes"].get(s, {})
        if ea.get("error") or eb.get("error"):
            continue
        ta = {r["id"]: r.get("time_s") for r in ea.get("points") or []}
        d = {r["id"]: round(r["time_s"] - ta[r["id"]], 2) for r in eb.get("points") or []
             if r.get("time_s") is not None and ta.get(r["id"]) is not None and abs(r["time_s"] - ta[r["id"]]) >= threshold}
        if d:
            out[s] = d
    return out
