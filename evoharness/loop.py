"""The evolution loop. Each round:

  1. the champion runs every training scenario (memoised by harness digest);
  2. the proposer sees the whole harness, the episodes point by point, the edit log, recent incidents and the options
     without effect, may call lookup tools, and proposes change items (or opens a development branch);
  3. items are applied and validated; errors go back to the proposer (loop.repairs times); a short smoke run on every
     training scenario must not crash, also with any single item left out or added alone;
  4. the candidate (all items), every leave-one-out version and every item alone run the training scenarios;
  5. the verifier (an LLM reviewer, or a numeric rule) admits or rejects each item;
  6. the champion plus the admitted items must pass the gate against the champion; then it becomes the champion.
Development branches: a structural change goes in as one unit and is repaired for a few rounds; it merges (after the
verifier) as soon as it passes the gate. A passing branch with options without effect is first sent back to be wired.
Everything is written to the output directory; a stopped run resumes at its next round with the same command.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from . import harness as H
from . import llm
from . import prompts as PR
from . import views as V
from .admission import ancestors, build, closure, sanitize
from .evaluator import Evaluator, write_atomic
from .objective import Gate, Objective

KEEP_INCIDENTS = 6


class Evolution:
    def __init__(self, cfg, out):
        self.cfg = cfg
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.Env = cfg.env_class()
        self.schema = H.Schema(features=dict(self.Env.FEATURES), limits=H.Limits.from_config(cfg["limits"], cfg["executor_imports"]))
        self.ev = Evaluator(cfg, self.out, self.schema)
        self.obj = Objective(cfg["objective"])
        self.gate = Gate(cfg["gate"], self.obj)
        self.lang = cfg["prompt_language"]
        self.P = PR.get(self.lang)
        self.L = cfg["loop"]
        self.train = list(cfg["scenarios"]["train"])
        self._proposer = self._verifier = None
        self.state_path = self.out / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            seed = H.validate(H.load(cfg.path(cfg["seed"])), self.schema)
            self.state = {"champion": seed, "rounds": [], "log": [], "branch": None, "incidents": []}
        self.save()

    # ---- plumbing -----------------------------------------------------------------------------------------------
    def save(self):
        write_atomic(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=1))

    @property
    def proposer(self):
        if self._proposer is None:
            self._proposer = llm.make(dict(self.cfg["proposer"], tag=self.out.name))
        return self._proposer

    @property
    def verifier(self):
        if self._verifier is None:
            self._verifier = llm.make(dict(self.cfg.verifier_backend(), tag=self.out.name))
        return self._verifier

    def build(self, base, items, keep):
        return build(base, items, keep, self.schema)

    def parts(self):
        return set(self.L.get("parts") or H.PARTS)

    def branches_on(self):
        return bool(self.L["branches"]["enabled"])

    def verifier_on(self):
        return bool(self.cfg["verifier"].get("enabled", True))

    def evaluate(self, h, tag):
        return self.ev.evaluate(h, self.train, 1, tag)

    def gate_eval(self, h, train_ev=None, tag="gate"):
        scen, rep = self.cfg.gate_scenarios(), int(self.cfg["gate"]["repeats"])
        if train_ev is not None and scen == self.train and rep == 1:
            return train_ev
        return self.ev.evaluate(h, scen, rep, tag)

    def passes(self, champ, champ_train_ev, cand, cand_train_ev):
        """(passes the gate, reasons, the candidate's gate evaluation)."""
        ce = self.gate_eval(champ, champ_train_ev, "gate-champion")
        ne = self.gate_eval(cand, cand_train_ev, "gate-candidate")
        ok, why = self.gate.check(ce, ne)
        return ok, why, ne

    def gate_text(self):
        g = self.cfg["gate"]
        parts = [f"scenarios {self.cfg.gate_scenarios() if g['scenarios'] != 'train' else 'the training scenarios'}"]
        need = g.get("min_gain") or 0
        parts.append(f"improves the objective by at least {need}" + (f" or {g['min_gain_relative']:.2%} of it" if g.get("min_gain_relative") else "") if need or g.get("min_gain_relative") else "improves the objective")
        for c in g.get("constraints") or []:
            if "max_increase" in c:
                parts.append(f"{c['metric']} does not increase by more than {c['max_increase']}")
            if "max" in c:
                parts.append(f"{c['metric']} at most {c['max']}")
        if g.get("per_scenario_max_loss") is not None:
            parts.append(f"no scenario loses more than {g['per_scenario_max_loss']}")
        return "; ".join(parts)

    def tools_text(self):
        t = self.L["tools"]
        out = [PR.fill(self.P.TOOL_TRACE, n=t["trace"]), PR.fill(self.P.TOOL_TRIAL, n=t["trial_episode"])]
        if hasattr(self.Env, "start_at"):
            out.append(PR.fill(self.P.TOOL_POINT, n=t["point_trial"]))
        return "\n".join(out)

    def system_proposer(self, mode):
        lim = self.schema.limits
        brief = self.cfg.domain_brief().strip()
        return PR.fill(self.P.PROPOSER, objective=self.obj.phrase(), failure_value=self.obj.failure_value, lead=self.cfg["decision"].get("lead", 0),
                       max_options=lim.options, max_questions=lim.questions, max_conditions=lim.conditions, hz=round(1 / float(self.Env.CONTROL_DT)),
                       executor_doc=self.Env.EXECUTOR_DOC.strip(), max_items=self.L["max_items"], imports=", ".join(lim.executor_imports),
                       features=json.dumps(self.schema.features, ensure_ascii=False), gate=self.gate_text(),
                       domain_brief=("\n" + brief + "\n") if brief else "", mode=mode, tools=self.tools_text())

    def system_verifier(self):
        return PR.fill(self.P.VERIFIER, gate=self.gate_text(), objective=self.obj.phrase(), failure_value=self.obj.failure_value)

    # ---- tools --------------------------------------------------------------------------------------------------
    def propose_tool(self):
        item = {"type": "object", "properties": {"id": {"type": "string"}, "part": {"type": "string", "enum": list(H.PARTS)},
                "op": {"type": "string", "enum": ["replace", "rewrite", "set", "delete"]}, "old": {"type": "string"}, "new": {"type": "string"},
                "source": {"type": "string"}, "path": {"type": "array", "items": {}}, "value": {}, "rationale": {"type": "string"},
                "depends_on": {"type": "array", "items": {"type": "string"}}}, "required": ["id", "part", "op", "rationale"]}
        return {"name": "propose_harness_changes", "description": self.P.PROPOSE_DESC,
                "parameters": {"type": "object", "properties": {"hypothesis": {"type": "string"},
                               "items": {"type": "array", "items": item, "minItems": 1, "maxItems": self.L["max_items"]},
                               "open_branch": {"type": "boolean", "description": self.P.OPEN_BRANCH_DESC}}, "required": ["hypothesis", "items"]}}

    def verify_tool(self):
        return {"name": "verify_items", "description": self.P.VERIFY_DESC,
                "parameters": {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "admit": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["id", "admit", "reason"]}}},
                    "required": ["verdicts"]}}

    def lookups(self, base, base_ev, trials, point_trials):
        items_schema = self.propose_tool()["parameters"]["properties"]["items"]
        out = {
            "get_point_trace": ({"name": "get_point_trace", "description": self.P.TRACE_DESC, "parameters": {"type": "object", "properties": {
                "scenario": {"type": "string"}, "points": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}}, "required": ["scenario", "points"]}},
                lambda x: V.point_trace(base_ev, x.get("scenario"), x.get("points") or [])),
            "trial_episode": ({"name": "trial_episode", "description": self.P.TRIAL_DESC, "parameters": {"type": "object", "properties": {
                "scenario": {"type": "string"}, "items": items_schema}, "required": ["scenario", "items"]}},
                lambda x: self.trial(base, base_ev, x, trials)),
        }
        if hasattr(self.Env, "start_at"):
            out["point_trial"] = ({"name": "point_trial", "description": self.P.POINT_DESC, "parameters": {"type": "object", "properties": {
                "scenario": {"type": "string"}, "point": {"type": "string"}, "items": dict(items_schema, minItems=0),
                "with_trace": {"type": "boolean"}}, "required": ["scenario", "point", "items"]}},
                lambda x: self.point_trial(base, base_ev, x, point_trials))
        return out

    def trial(self, base, base_ev, x, trials):
        if len(trials) >= self.L["tools"]["trial_episode"]:
            return {"error": f"the {self.L['tools']['trial_episode']} trial episodes of this round are used up; submit propose_harness_changes"}
        s = x.get("scenario")
        if s not in base_ev["episodes"]:
            return {"error": f"unknown scenario {s}; scenarios: {sorted(base_ev['episodes'])}"}
        items, dropped = sanitize(base, x.get("items") or [], self.schema)
        if not items:
            return {"error": "no valid items", "problems": dropped}
        h = self.build(base, items, {it["id"] for it in items})
        ep = self.ev.episode(h, s)
        b = base_ev["episodes"][s]
        res = {"scenario": s, "value": self.obj.episode_value(ep), "current_value": self.obj.episode_value(b),
               "improvement": self.obj.gain(self.obj.episode_value(b), self.obj.episode_value(ep)), "finished": ep.get("finished"),
               "items_applied": [it["id"] for it in items]}
        res["metrics"] = {k: v for k, v in ep.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if dropped:
            res["invalid_items"] = dropped
        if ep.get("error"):
            res["error"] = ep["error"]
            res["traceback"] = (ep.get("traceback") or "")[-1500:]
        else:
            ch = V.point_changes(self.obj, {"episodes": {s: b}}, {"episodes": {s: ep}}, 0.05)
            res["point_time_changes_s"] = ch.get(s, {})
            inc = V.incidents(ep, b)
            if inc:
                res["incidents"] = inc
        res["note"] = self.P.NOTE_TRIAL
        trials.append({k: res.get(k) for k in ("scenario", "value", "current_value", "error", "items_applied")})
        return res

    def point_trial(self, base, base_ev, x, point_trials):
        if len(point_trials) >= self.L["tools"]["point_trial"]:
            return {"error": f"the {self.L['tools']['point_trial']} point trials of this round are used up"}
        s, pid = x.get("scenario"), x.get("point")
        ep = base_ev["episodes"].get(s)
        if ep is None:
            return {"error": f"unknown scenario {s}; scenarios: {sorted(base_ev['episodes'])}"}
        ids = [r["id"] for r in ep.get("points") or []]
        if pid not in ids:
            return {"error": f"unknown point {pid}; points: {ids}"}
        raw = x.get("items") or []
        items, dropped = sanitize(base, raw, self.schema) if raw else ([], [])
        if raw and not items:
            return {"error": "no valid items", "problems": dropped}
        h = self.build(base, items, {it["id"] for it in items}) if items else base
        self._pt = getattr(self, "_pt", {})

        def run(v):
            key = (H.digest(v), s, pid)
            if key not in self._pt:
                self._pt[key] = self.ev.episode(v, s, point=pid)
            return self._pt[key]

        with ThreadPoolExecutor(2) as ex:
            cand, cur = list(ex.map(run, [h, base]))

        def view(r):
            row = (r.get("points") or [{}])[0]
            v = {k: x for k, x in row.items() if k not in ("choice", "advice")} | {"finished": r.get("finished")}
            if r.get("error"):
                v |= {"error": r["error"], "traceback": (r.get("traceback") or "")[-1200:]}
            return v

        res = {"scenario": s, "point": pid, "candidate": view(cand), "current": view(cur), "items_applied": [it["id"] for it in items]}
        if dropped:
            res["invalid_items"] = dropped
        ct, bt = res["candidate"].get("time_s"), res["current"].get("time_s")
        if ct is not None and bt is not None:
            res["point_gain_s"] = round(bt - ct, 3)
        if cand.get("trace") and (x.get("with_trace") or cand.get("incidents")):
            res["candidate_trace"] = {"fields": cand.get("trace_fields"), "rows": V.thin(cand["trace"], 40)}
        res["note"] = self.P.NOTE_POINT
        point_trials.append({"scenario": s, "point": pid, "items_applied": res["items_applied"], "candidate_s": ct, "current_s": bt, "error": cand.get("error")})
        return res

    # ---- checks -------------------------------------------------------------------------------------------------
    def new_dead(self, base, h):
        """Options without effect in h that were not already without effect in base."""
        self.ev.smoke(base)
        self.ev.smoke(h)
        had = self.ev.dead.get(H.digest(base), [])
        return [x for x in self.ev.dead.get(H.digest(h), []) if x not in had]

    def runtime_problems(self, champion, items):
        """Smoke runs of the candidate, every leave-one-out version and every item alone. A version that crashes
        because an item needs another one it did not declare gets that dependency added; what still crashes is
        returned. Mutates the items' depends_on. None if everything runs."""
        ids = {it["id"] for it in items}
        by = {it["id"]: it for it in items}
        smoke = lambda keep: self.ev.smoke(self.build(champion, items, keep))
        err = smoke(ids)
        if err:
            return [{"version": "all items"} | err]

        def alone_ok(iid, extra=()):
            keep = ancestors(items, iid) | {iid}
            for e in extra:
                keep |= ancestors(items, e) | {e}
            return smoke(keep) is None

        for it in items:
            if alone_ok(it["id"]):
                continue
            need = next((o["id"] for o in items if o["id"] != it["id"] and it["id"] not in ancestors(items, o["id"]) and alone_ok(it["id"], (o["id"],))), None)
            if need is None:
                keep = ancestors(items, it["id"]) | {it["id"]}
                return [{"version": f"only {it['id']} (and its dependencies) on the current harness", "items_applied": sorted(keep)} | (smoke(keep) or {})]
            it["depends_on"] = list(it.get("depends_on") or []) + [need]
            it.setdefault("auto_depends_on", []).append(need)
        for it in items:
            for _ in range(len(items)):
                keep = ids - closure(items, {it["id"]})
                if not keep or smoke(keep) is None:
                    break
                broken = [j for j in keep if not alone_ok(j)]
                for j in (broken or keep):
                    if it["id"] not in by[j].get("depends_on", []):
                        by[j]["depends_on"] = list(by[j].get("depends_on") or []) + [it["id"]]
                        by[j].setdefault("auto_depends_on", []).append(it["id"])
            keep = ids - closure(items, {it["id"]})
            if keep and smoke(keep) is not None:
                return [{"version": f"without {it['id']}", "items_applied": sorted(keep)} | smoke(keep)]
        return None

    # ---- proposer ---------------------------------------------------------------------------------------------
    def mode_text(self, focus=None, branch=None, wire=False):
        P, b = self.P, self.L["branches"]
        if wire:
            extra = P.WIRE_MODE
        elif branch:
            extra = P.BRANCH_MODE
        else:
            extra = (PR.fill(P.BRANCH_HINT, branch_rounds=b["rounds"], branch_max=b["max"]) if self.branches_on() else P.NO_BRANCH)
            extra += P.FOCUS_ACTION_SPACE if focus == "action_space" else ""
        if self.parts() != set(H.PARTS):
            extra += PR.fill(P.ONLY_PARTS, parts=", ".join(sorted(self.parts())))
        return extra

    def propose(self, champion, champ_ev, focus=None, branch=None, wire=False):
        """(args with the valid items, proposer call metas, dropped items) or (None, metas, dropped) if nothing valid.
        With branch (the branch view), champion/champ_ev are the branch's harness and episodes."""
        P = self.P
        system = self.system_proposer(self.mode_text(focus, branch, wire))
        user = {"harness": {"action_space": champion["action_space"], "playbook": champion["playbook"], "executor.py": champion["executor"]},
                ("branch_result" if branch else "champion_result"): V.summary(champ_ev, self.obj),
                "episodes": V.episodes_view(champ_ev, self.obj), "edit_log": self.state["log"][-int(self.L["edit_log"]):]}
        if branch:
            user["branch"] = branch
        self.ev.smoke(champion)
        user["options_without_effect"] = V.dead_text(self.ev.dead.get(H.digest(champion), []), champion["action_space"], self.lang) or "(none)"
        if self.state.get("incidents"):
            user["recent_incidents"] = self.state["incidents"]
        msg = json.dumps(user, ensure_ascii=False)
        calls, dropped_all, repair = [], [], ""
        t = self.L["tools"]
        for attempt in range(int(self.L["repairs"]) + 1):
            trials, point_trials = [], []
            lookups = self.lookups(champion, champ_ev, trials, point_trials)
            try:
                args, metas, looked = self.proposer.call(system, msg + repair, self.propose_tool(), lookups,
                                                          t["trace"] + t["trial_episode"] + t["point_trial"], P.ASK_FOR_TOOL)
                calls.append({"calls": metas, "lookups": looked, "trials": trials, "point_trials": point_trials})
            except RuntimeError as e:
                calls.append({"error": str(e)[:300], "trials": trials, "point_trials": point_trials})
                dropped_all.append({"id": "(call)", "reason": str(e)[:300], "attempt": attempt})
                repair = PR.fill(P.REPAIR_JSON, error=str(e)[:200])
                continue
            raw = [it for it in args.get("items") or [] if isinstance(it, dict)]
            allowed = self.parts()
            gone = closure(raw, {it.get("id") for it in raw if it.get("part") not in allowed})
            off = [{"id": it.get("id"), "reason": PR.fill(P.ONLY_PARTS_DROP, parts=", ".join(sorted(allowed)))} for it in raw if it.get("id") in gone]
            if not self.branches_on() and not branch:
                args = dict(args, open_branch=False)
            items, dropped = sanitize(champion, [it for it in raw if it.get("id") not in gone], self.schema)
            dropped += off
            dropped_all += [d | {"attempt": attempt} for d in dropped]
            ids = {it["id"] for it in items}
            if branch or args.get("open_branch"):   # a branch is judged as a whole: only the whole must run
                if not items:
                    pass
                elif branch and self.ev.smoke(champion) is not None:
                    return args | {"items": items, "open_branch": False}, calls, dropped_all
                else:
                    err = self.ev.smoke(self.build(champion, items, ids))
                    if not err or not branch:
                        return args | {"items": items, "open_branch": not branch}, calls, dropped_all
                    dropped_all.append({"id": "(runtime)", "reason": json.dumps(err, ensure_ascii=False)[:600], "attempt": attempt})
                    repair = PR.fill(P.REPAIR_BRANCH_RUNTIME, detail=json.dumps({"previous_items": args.get("items"), "runtime_problem": err}, ensure_ascii=False)[:24000])
                    continue
            if focus == "action_space" and items and not any(it["part"] == "action_space" for it in items):
                dropped_all.append({"id": "(focus)", "reason": P.FOCUS_DROP, "attempt": attempt})
                repair = P.REPAIR_FOCUS
                continue
            crash = self.runtime_problems(champion, items) if items else None
            nd = self.new_dead(champion, self.build(champion, items, ids)) if items and not crash else []
            if nd:
                txt = V.dead_text(nd, self.build(champion, items, ids)["action_space"], self.lang)
                dropped_all.append({"id": "(no effect)", "reason": "; ".join(txt)[:600], "attempt": attempt})
                repair = PR.fill(P.REPAIR_DEAD, detail=json.dumps({"previous_items": args.get("items"), "options_without_effect": txt}, ensure_ascii=False)[:24000])
                continue
            if items and not crash:
                return args | {"items": items}, calls, dropped_all
            if crash:
                dropped_all.append({"id": "(runtime)", "reason": json.dumps(crash, ensure_ascii=False)[:600], "attempt": attempt})
                repair = PR.fill(P.REPAIR_RUNTIME, detail=json.dumps({"previous_items": args.get("items"), "runtime_problems": crash}, ensure_ascii=False)[:24000])
                continue
            repair = PR.fill(P.REPAIR_INVALID, detail=json.dumps({"previous_items": args.get("items"), "problems": dropped}, ensure_ascii=False)[:20000])
        return None, calls, dropped_all

    # ---- verifier ---------------------------------------------------------------------------------------------
    def rule_verdicts(self, items, champ_ev, alone):
        """Without the LLM verifier: admit an item when, added alone, it gains at least the gate's min_gain, runs
        without errors, and no scenario loses as much as the total gain."""
        out = {}
        need = self.gate.required_gain(champ_ev)
        for it in items:
            a = alone[it["id"]]
            gain = self.obj.gain(champ_ev["value"], a["value"])
            worst = max([-g for g in self.obj.gains_per_scenario(champ_ev, a).values()] + [0.0])
            ok = gain >= max(need, 1e-9) and not a["errors"] and worst < gain
            out[it["id"]] = {"id": it["id"], "admit": ok, "reason": PR.fill(self.P.RULE_VERDICT, gain=round(gain, 4), worst=round(worst, 4)) + ("" if ok else self.P.RULE_REJECT)}
        return out

    def measured(self, champ_ev, cand_ev, without, alone):
        return {"improves_by": self.obj.gain(without["value"], cand_ev["value"]),
                "improves_by_per_scenario": self.obj.gains_per_scenario(without, cand_ev), "without_errors": without["errors"],
                "alone_improves_by": self.obj.gain(champ_ev["value"], alone["value"]),
                "alone_improves_by_per_scenario": self.obj.gains_per_scenario(champ_ev, alone), "alone_errors": alone["errors"]}

    def review(self, user, ids):
        try:
            args, meta, _ = self.verifier.call(self.system_verifier(), json.dumps(user, ensure_ascii=False), self.verify_tool(), ask_for_tool=self.P.ASK_FOR_TOOL)
            return {v["id"]: v for v in args.get("verdicts", []) if isinstance(v, dict) and "id" in v}, meta
        except RuntimeError as e:
            return {i: {"id": i, "admit": False, "reason": PR.fill(self.P.VERIFIER_FAILED, error=str(e)[:200])} for i in ids}, {"error": str(e)[:300]}

    def verify(self, champion, items, champ_ev, cand_ev, without, alone):
        if not self.verifier_on():
            return self.rule_verdicts(items, champ_ev, alone), {"rule": True}
        user = {"items": [{"id": it["id"], "part": it["part"], "op": it["op"], "rationale": it.get("rationale"), "depends_on": it.get("depends_on") or [],
                           "change": V.item_view(self.build(champion, items, ancestors(items, it["id"])), it),
                           "measured": self.measured(champ_ev, cand_ev, without[it["id"]], alone[it["id"]])} for it in items],
                "champion": V.summary(champ_ev, self.obj), "candidate": V.summary(cand_ev, self.obj),
                "candidate_errors": {s: (e.get("traceback") or "")[-1200:] for s, e in cand_ev["episodes"].items() if e.get("error")},
                "point_time_changes_candidate_vs_champion_s": V.point_changes(self.obj, champ_ev, cand_ev, 0.05)}
        return self.review(user, [it["id"] for it in items])

    # ---- one round --------------------------------------------------------------------------------------------
    def add_incidents(self, n, champ_ev, versions):
        new = []
        for name, ev in versions:
            for s, e in ev["episodes"].items():
                new += [x | {"round": n, "version": name} for x in V.incidents(e, champ_ev["episodes"].get(s))][:1]
        if new:
            self.state["incidents"] = ((self.state.get("incidents") or []) + new[:4])[-KEEP_INCIDENTS:]

    def log_invalid(self, n, args, calls, dropped):
        for d in dropped:
            if d.get("attempt") == len(calls) - 1 or args is None:
                self.state["log"].append({"round": n, "id": d.get("id"), "part": "-", "op": "-", "rationale": "",
                                          "verifier": PR.fill(self.P.INVALID_UNTESTED, reason=str(d.get("reason"))[:300]), "admitted": False, "round_outcome": "invalid_item"})

    def round(self, n):
        champion = self.state["champion"]
        champ_ev = self.evaluate(champion, "champion")
        rec = {"round": n, "champion": {"digest": champ_ev["digest"]} | V.summary(champ_ev, self.obj), "started": time.time()}
        every = int(self.L.get("action_space_every") or 0)
        focus = "action_space" if every and n % every == 0 and "action_space" in self.parts() else None
        rec["focus"] = focus
        args, calls, dropped = self.propose(champion, champ_ev, focus)
        rec["proposer_calls"], rec["dropped"] = calls, dropped
        self.log_invalid(n, args, calls, dropped)
        if args is None:
            rec["outcome"] = "no_valid_proposal"
            return self.finish(rec)
        items = args["items"]
        rec["hypothesis"], rec["items"] = args.get("hypothesis"), items
        ids = [it["id"] for it in items]
        cand = self.build(champion, items, set(ids))
        cand_ev = self.evaluate(cand, "candidate")
        rec["candidate"] = {"digest": cand_ev["digest"]} | V.summary(cand_ev, self.obj)
        if args.get("open_branch"):
            b = {"harness": cand, "origin_round": n, "hypothesis": args.get("hypothesis"),
                 "items": [{"id": it["id"], "part": it["part"], "rationale": (it.get("rationale") or "")[:300]} for it in items],
                 "left": self.L["branches"]["rounds"], "used": 0,
                 "history": [{"round": n, "what": "opened"} | V.summary(cand_ev, self.obj)]}
            self.add_incidents(n, champ_ev, [("new branch", cand_ev)])
            rec["outcome"] = self.try_merge(n, rec, champion, champ_ev, b, cand_ev) or "branch_opened"
            if rec["outcome"] in ("branch_opened", "branch_merge_blocked"):
                self.state["branch"] = b
            self.state["log"].append({"round": n, "id": "branch", "part": "branch", "op": "open", "rationale": (args.get("hypothesis") or "")[:400],
                                      "branch_value": cand_ev["value"], "champion_value": champ_ev["value"], "round_outcome": rec["outcome"],
                                      "admitted": rec["outcome"] == "branch_merged"})
            return self.finish(rec)
        if len(items) > 1:
            without = {it["id"]: self.evaluate(self.build(champion, items, set(ids) - closure(items, {it["id"]})), "without-" + it["id"]) for it in items}
            alone = {it["id"]: self.evaluate(self.build(champion, items, ancestors(items, it["id"]) | {it["id"]}), "alone-" + it["id"]) for it in items}
        else:
            without, alone = {ids[0]: champ_ev}, {ids[0]: cand_ev}
        rec["without"] = {i: {k: w[k] for k in ("value", "per_scenario", "errors")} for i, w in without.items()}
        rec["alone"] = {i: {k: w[k] for k in ("value", "per_scenario", "errors")} for i, w in alone.items()}
        self.add_incidents(n, champ_ev, [("candidate", cand_ev)] + [("alone " + i, w) for i, w in alone.items()])
        verdicts, vmeta = self.verify(champion, items, champ_ev, cand_ev, without, alone)
        rec["verifier_call"], rec["verdicts"] = vmeta, verdicts
        admitted = {i for i in ids if (verdicts.get(i) or {}).get("admit")}
        admitted -= closure(items, set(ids) - admitted)   # nothing whose dependency was rejected
        rec["admitted"] = sorted(admitted)
        nd = self.new_dead(champion, self.build(champion, items, admitted)) if admitted else []
        if nd:
            rec["dead_options"], rec["outcome"], rec["admitted"] = nd, "confirm_failed", []
            txt = "; ".join(V.dead_text(nd, self.build(champion, items, admitted)["action_space"], self.lang))[:300]
            for v in verdicts.values():
                if v.get("id") in admitted:
                    v["reason"] = (v.get("reason") or "") + PR.fill(self.P.DEAD_IN_ADMITTED, detail=txt)
            admitted = set()
        elif admitted:
            new = self.build(champion, items, admitted)
            new_ev = self.evaluate(new, "admitted")
            ok, why, gate_ev = self.passes(champion, champ_ev, new, new_ev)
            rec["confirm"] = {"digest": new_ev["digest"]} | V.summary(new_ev, self.obj) | {"gate": V.summary(gate_ev, self.obj), "gate_failures": why}
            rec["outcome"] = "adopted" if ok else "confirm_failed"
            if ok:
                self.state["champion"] = new
        else:
            rec["outcome"] = "nothing_admitted"
        for it in items:
            v = verdicts.get(it["id"]) or {}
            m = self.measured(champ_ev, cand_ev, without[it["id"]], alone[it["id"]])
            self.state["log"].append({"round": n, "id": it["id"], "part": it["part"], "op": it["op"], "path": it.get("path"),
                                      "rationale": (it.get("rationale") or "")[:400], "improves_by": m["improves_by"], "alone_improves_by": m["alone_improves_by"],
                                      "admitted": it["id"] in admitted, "verifier": (v.get("reason") or "")[:400], "round_outcome": rec["outcome"]})
        return self.finish(rec)

    def finish(self, rec):
        rec["finished"] = time.time()
        self.state["rounds"].append(rec)
        self.save()
        return rec

    # ---- branches ---------------------------------------------------------------------------------------------
    def try_merge(self, n, rec, champion, champ_ev, b, br_ev):
        """If the branch passes the gate against the champion, the verifier reviews it as one change. Returns the
        outcome or None."""
        ok, why, _ = self.passes(champion, champ_ev, b["harness"], br_ev)
        if not ok:
            return None
        nd = self.new_dead(champion, b["harness"])
        if nd:
            wired = self.wire(n, rec, champion, champ_ev, b, br_ev, nd)
            if wired is None:
                return "branch_merge_blocked"
            b["harness"], br_ev = wired
        br = b["harness"]
        rationale = PR.fill(self.P.BRANCH_RATIONALE, hypothesis=(b.get("hypothesis") or "")[:600])
        measured = {"alone_improves_by": self.obj.gain(champ_ev["value"], br_ev["value"]), "alone_improves_by_per_scenario": self.obj.gains_per_scenario(champ_ev, br_ev),
                    "improves_by": self.obj.gain(champ_ev["value"], br_ev["value"]), "alone_errors": br_ev["errors"]}
        user = {"note": self.P.BRANCH_MERGE_NOTE, "items": [{"id": "branch", "rationale": rationale, "branch_history": b["history"],
                                                              "change": V.harness_diff(champion, br), "measured": measured}],
                "champion": V.summary(champ_ev, self.obj), "candidate": V.summary(br_ev, self.obj),
                "point_time_changes_candidate_vs_champion_s": V.point_changes(self.obj, champ_ev, br_ev, 0.05)}
        if not self.verifier_on():
            verdicts, meta = self.rule_verdicts([{"id": "branch"}], champ_ev, {"branch": br_ev}), {"rule": True}
        else:
            verdicts, meta = self.review(user, ["branch"])
        v = verdicts.get("branch") or {"admit": False, "reason": "no verdict"}
        rec["merge_review"] = {"verdict": v, "call": meta}
        b["history"].append({"round": n, "what": "merge review", "admit": bool(v.get("admit")), "reason": (v.get("reason") or "")[:400]})
        if v.get("admit"):
            self.state["champion"] = br
            rec["confirm"] = {"digest": br_ev["digest"]} | V.summary(br_ev, self.obj)
            return "branch_merged"
        return "branch_merge_rejected"

    def wire(self, n, rec, champion, champ_ev, b, br_ev, nd):
        """A passing branch with options without effect goes back to the proposer to wire or remove them. Returns
        (harness, episodes) of a wired version that still passes, else None."""
        h, ev = b["harness"], br_ev
        log = rec.setdefault("wiring", [])
        for k in range(int(self.L["branches"]["wire_attempts"])):
            view = {"hypothesis": b["hypothesis"], "history": b["history"], "branch_vs_champion_gain": self.obj.gain(champ_ev["value"], ev["value"]),
                    "champion_result": V.summary(champ_ev, self.obj), "options_without_effect_to_wire": V.dead_text(nd, h["action_space"], self.lang), "previous_attempts": log}
            args, calls, dropped = self.propose(h, ev, branch=view, wire=True)
            entry = {"attempt": k, "options_without_effect": V.dead_text(nd, h["action_space"], self.lang), "proposer_calls": calls, "dropped": dropped}
            log.append(entry)
            if args is None:
                entry["result"] = "no valid proposal"
                continue
            items = args["items"]
            entry["items"] = items
            new = self.build(h, items, {it["id"] for it in items})
            left = self.new_dead(champion, new)
            if left:
                entry["result"] = "still without effect: " + "; ".join(V.dead_text(left, new["action_space"], self.lang))[:400]
                nd = left
                continue
            new_ev = self.evaluate(new, "branch")
            entry["value"] = new_ev["value"]
            self.add_incidents(n, champ_ev, [("branch wiring", new_ev)])
            ok, why, _ = self.passes(champion, champ_ev, new, new_ev)
            if ok:
                entry["result"] = "wired"
                b["history"].append({"round": n, "what": "wired options without effect", "value": new_ev["value"]})
                return new, new_ev
            entry["result"] = f"wired but no longer passes the gate: {why}"
        b["history"].append({"round": n, "what": "merge blocked", "reason": PR.fill(self.P.BRANCH_MERGE_BLOCKED, detail="; ".join(V.dead_text(nd, h["action_space"], self.lang))[:400])})
        return None

    def branch_round(self, n):
        b = self.state["branch"]
        champion = self.state["champion"]
        champ_ev = self.evaluate(champion, "champion")
        br_ev = self.evaluate(b["harness"], "branch")
        rec = {"round": n, "branch_of": b["origin_round"], "champion": {"digest": champ_ev["digest"]} | V.summary(champ_ev, self.obj),
               "branch_before": {"digest": br_ev["digest"]} | V.summary(br_ev, self.obj), "started": time.time()}
        view = {"hypothesis": b["hypothesis"], "opened_with": b["items"], "rounds_left_including_this": b["left"], "history": b["history"],
                "champion_result": V.summary(champ_ev, self.obj), "branch_vs_champion_gain_per_scenario": self.obj.gains_per_scenario(champ_ev, br_ev),
                "points_where_branch_differs_s": V.point_changes(self.obj, champ_ev, br_ev, 0.1),
                "errors": {s: (e.get("traceback") or "")[-1500:] for s, e in br_ev["episodes"].items() if e.get("error")},
                "branch_incidents": [x for s, e in br_ev["episodes"].items() for x in V.incidents(e, champ_ev["episodes"].get(s))][:3]}
        args, calls, dropped = self.propose(b["harness"], br_ev, branch=view)
        rec["proposer_calls"], rec["dropped"] = calls, dropped
        b["left"] -= 1
        b["used"] = b.get("used", 0) + 1
        if args is None:
            rec["outcome"] = "branch_no_valid_proposal"
            b["history"].append({"round": n, "what": "no valid proposal"})
        else:
            items = args["items"]
            rec["hypothesis"], rec["items"] = args.get("hypothesis"), items
            new = self.build(b["harness"], items, {it["id"] for it in items})
            new_ev = self.evaluate(new, "branch")
            self.add_incidents(n, champ_ev, [("branch repair", new_ev)])
            kept = self.obj.gain(br_ev["value"], new_ev["value"]) > 1e-6 or (new_ev["unfinished"] < br_ev["unfinished"] and not new_ev["errors"])
            rec["candidate"] = {"digest": new_ev["digest"]} | V.summary(new_ev, self.obj)
            b["history"].append({"round": n, "what": "repair", "hypothesis": (args.get("hypothesis") or "")[:300],
                                 "items": [{"id": it["id"], "part": it["part"], "rationale": (it.get("rationale") or "")[:200]} for it in items],
                                 "kept": kept} | V.summary(new_ev, self.obj))
            if kept:
                b["harness"], br_ev = new, new_ev
                bc = self.L["branches"]
                b["left"] = min(max(b["left"], bc["extend"]), bc["max"] - b["used"])   # still improving: keep going
            rec["outcome"] = "branch_improved" if kept else "branch_not_improved"
        merged = self.try_merge(n, rec, champion, champ_ev, b, br_ev)
        if merged:
            rec["outcome"] = merged
        if merged == "branch_merged" or b["left"] <= 0:
            if merged != "branch_merged":
                rec["outcome"] += "+branch_dropped"
            self.state["branch"] = None
        self.state["log"].append({"round": n, "id": "branch", "part": "branch", "op": "repair", "rationale": (rec.get("hypothesis") or "")[:300],
                                  "branch_value": br_ev["value"], "champion_value": champ_ev["value"], "round_outcome": rec["outcome"],
                                  "admitted": rec["outcome"] == "branch_merged", "verifier": ((rec.get("merge_review") or {}).get("verdict") or {}).get("reason", "")[:400]})
        return self.finish(rec)

    # ---- the run ----------------------------------------------------------------------------------------------
    def run(self, rounds=None, quiet=False):
        rounds = int(rounds if rounds is not None else self.L["rounds"])
        while len(self.state["rounds"]) < rounds:
            n = len(self.state["rounds"]) + 1
            rec = self.branch_round(n) if self.state.get("branch") else self.round(n)
            if not quiet:
                print(json.dumps({"round": n, "outcome": rec["outcome"], "champion": rec["champion"]["value"],
                                  "candidate": (rec.get("candidate") or {}).get("value"), "admitted": rec.get("admitted"),
                                  "new": (rec.get("confirm") or {}).get("value")}, ensure_ascii=False), flush=True)
        self.state["branch"] = None
        self.save()
        H.save(self.state["champion"], self.out / "champion")
        final = self.evaluate(self.state["champion"], "final")
        if not quiet:
            print(json.dumps({"final": final["value"], "per_scenario": final["per_scenario"]}), flush=True)
        return final
