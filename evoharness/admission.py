"""Change items: dependencies, building versions, and the static clean-up of a proposal before anything runs."""
from __future__ import annotations

from . import harness as H


def closure(items, excluded):
    """Ids excluded with everything that depends on them, transitively."""
    out = set(excluded)
    changed = True
    while changed:
        changed = False
        for it in items:
            if it.get("id") not in out and set(d for d in (it.get("depends_on") or []) if isinstance(d, str)) & out:
                out.add(it.get("id"))
                changed = True
    return out


def ancestors(items, iid):
    by = {it["id"]: it for it in items}
    out = set()
    todo = list(by[iid].get("depends_on") or [])
    while todo:
        d = todo.pop()
        if d in by and d not in out:
            out.add(d)
            todo += list(by[d].get("depends_on") or [])
    return out


def build(champion, items, keep, schema):
    return H.validate(H.apply_items(champion, [it for it in items if it["id"] in keep]), schema)


def sanitize(champion, items, schema):
    """Keep the items that are valid; return (items, dropped [{'id', 'reason'}]).
    An item must change the harness and be valid with what it depends on; an undeclared dependency on another item is
    added when the item is valid only together with it; items that conflict when combined, or whose removal breaks the
    others, are dropped (in order)."""
    dropped, seen, out = [], set(), []
    for it in items:
        iid = it.get("id")
        if not isinstance(iid, str) or not iid or iid in seen:
            dropped.append({"id": iid, "reason": "missing or duplicate id"})
            continue
        rat = (it.get("rationale") or "").strip()
        if len(rat) < 15 or "placeholder" in rat.lower():
            dropped.append({"id": iid, "reason": "the rationale must state the evidence and the expected effect"})
            continue
        seen.add(iid)
        out.append(dict(it, depends_on=[d for d in (it.get("depends_on") or []) if isinstance(d, str)]))
    ids = {it["id"] for it in out}
    for it in out:
        bad = [d for d in it["depends_on"] if d not in ids]
        if bad:
            it["depends_on"] = [d for d in it["depends_on"] if d in ids]
            it["missing_deps"] = bad
    valid = []
    for it in out:
        anc = ancestors(out, it["id"])

        def ok(extra=()):
            try:
                before = build(champion, out, anc | set(extra), schema)
                after = build(champion, out, anc | set(extra) | {it["id"]}, schema)
                return H.digest(before) != H.digest(after), None
            except H.HarnessError as e:
                return False, str(e)

        good, err = ok()
        if not good and err:
            for other in out:   # an undeclared dependency?
                if other["id"] != it["id"] and other["id"] not in anc and it["id"] not in ancestors(out, other["id"]):
                    g2, _ = ok({other["id"]} | ancestors(out, other["id"]))
                    if g2:
                        it["depends_on"].append(other["id"])
                        it.setdefault("auto_depends_on", []).append(other["id"])
                        good = True
                        break
        if not good:
            dropped.append({"id": it["id"], "reason": err or "the item does not change the harness"})
            continue
        valid.append(it)
    out = valid
    gone = closure(out, {d["id"] for d in dropped}) - {d["id"] for d in dropped}
    dropped += [{"id": g, "reason": "depends on an invalid item"} for g in gone]
    out = [it for it in out if it["id"] not in gone]
    kept = []   # combine in order; drop what conflicts with the items before it
    for it in out:
        trial = kept + [it]
        try:
            build(champion, trial, {x["id"] for x in trial}, schema)
            kept = trial
        except H.HarnessError as e:
            dropped.append({"id": it["id"], "reason": f"invalid together with the items before it: {e}"})
    out = [it for it in kept if set(it["depends_on"]) <= {x["id"] for x in kept}]
    for it in list(out):   # every leave-one-out version must be valid
        try:
            build(champion, out, {x["id"] for x in out} - closure(out, {it["id"]}), schema)
        except H.HarnessError as e:
            dropped.append({"id": it["id"], "reason": f"removing it makes other items invalid (declare depends_on): {e}"})
            out = [x for x in out if x["id"] not in closure(out, {it["id"]})]
    return out, dropped
