import json

from evoharness import harness as H
from evoharness.loop import Evolution

import scripted
from conftest import drone_config

FASTER = {"id": "i1", "part": "executor", "op": "replace", "old": "BASE_SPEED = 6.0", "new": "BASE_SPEED = 9.0",
          "rationale": "every leg is flown well below the top speed; a higher base speed should shorten every leg"}
RULE = {"id": "i2", "part": "playbook", "op": "set", "path": ["rules"],
        "value": [{"when": [{"field": "length_m", "op": ">", "value": 25}], "then": {"speed": "FAST"}, "note": "long legs"}],
        "rationale": "long legs have room to accelerate, so fly them fast"}
USELESS = {"id": "i3", "part": "action_space", "op": "set", "path": ["questions", "mood"],
           "value": {"instructions": "mood of the pilot", "default": "CALM", "options": {"CALM": "calm", "ANGRY": "angry"}},
           "rationale": "a mood question that the executor never reads"}


def test_round_adopts_measured_improvements(tmp_path):
    scripted.reset([{"hypothesis": "fly faster", "items": [FASTER, RULE]}])
    evo = Evolution(drone_config(), tmp_path)
    seed_value = evo.evaluate(evo.state["champion"], "seed")["value"]
    final = evo.run(quiet=True)
    rec = evo.state["rounds"][0]
    assert rec["outcome"] == "adopted" and rec["admitted"] == ["i1", "i2"]
    assert final["value"] < seed_value - 1
    assert set(rec["without"]) == {"i1", "i2"} and set(rec["alone"]) == {"i1", "i2"}
    log = {e["id"]: e for e in evo.state["log"]}
    assert log["i1"]["alone_improves_by"] > 0 and log["i1"]["admitted"]
    assert "BASE_SPEED = 9.0" in (tmp_path / "champion" / "executor.py").read_text()
    roles = [r for r, *_ in scripted.SEEN]
    assert roles == ["proposer", "verifier"]
    system = scripted.SEEN[0][1]
    assert "total flight time" in system and "length_m" in system and "wind" in system   # objective, features, domain brief


def test_rule_verifier_rejects_what_does_not_help(tmp_path):
    slower = dict(FASTER, id="s1", new="BASE_SPEED = 3.0", rationale="slower legs, a deliberately bad change for the test")
    scripted.reset([{"hypothesis": "one good, one bad", "items": [FASTER, slower | {"old": "GAIN = 2.0", "new": "GAIN = 0.5"}]}])
    evo = Evolution(drone_config(verifier={"enabled": False}), tmp_path)
    evo.run(quiet=True)
    rec = evo.state["rounds"][0]
    assert rec["verifier_call"] == {"rule": True}
    assert rec["admitted"] == ["i1"] and rec["outcome"] == "adopted"


def test_gate_blocks_too_small_gains(tmp_path):
    scripted.reset([{"hypothesis": "fly faster", "items": [FASTER]}])
    evo = Evolution(drone_config(gate={"min_gain": 1000}), tmp_path)
    evo.run(quiet=True)
    rec = evo.state["rounds"][0]
    assert rec["outcome"] == "confirm_failed" and rec["confirm"]["gate_failures"]
    assert H.digest(evo.state["champion"]) == rec["champion"]["digest"]


def test_options_without_effect_go_back_to_the_proposer(tmp_path):
    scripted.reset([{"hypothesis": "mood", "items": [USELESS]}, {"hypothesis": "fly faster", "items": [FASTER]}])
    evo = Evolution(drone_config(), tmp_path)
    evo.run(quiet=True)
    rec = evo.state["rounds"][0]
    assert any(d["id"] == "(no effect)" for d in rec["dropped"])
    assert "mood" in scripted.SEEN[1][2] and rec["admitted"] == ["i1"]


def test_branch_opens_repairs_and_merges(tmp_path):
    slow_branch = {"hypothesis": "rewrite the speed logic", "open_branch": True,
                   "items": [dict(FASTER, id="b1", new="BASE_SPEED = 5.0", rationale="first step of a new speed model, slower for now")]}
    repair = {"hypothesis": "finish it", "items": [dict(FASTER, id="b2", old="BASE_SPEED = 5.0", new="BASE_SPEED = 10.0", rationale="the new speed model, now faster")]}
    scripted.reset([slow_branch, repair])
    evo = Evolution(drone_config(loop={"rounds": 2}), tmp_path)
    evo.run(quiet=True)
    r1, r2 = evo.state["rounds"]
    assert r1["outcome"] == "branch_opened"
    assert r2["outcome"] == "branch_merged" and evo.state["branch"] is None
    assert "BASE_SPEED = 10.0" in evo.state["champion"]["executor"]
    assert "development branch" in scripted.SEEN[1][1]   # the repair round runs in branch mode


def test_resume_continues_at_the_next_round(tmp_path):
    scripted.reset([{"hypothesis": "fly faster", "items": [FASTER]}])
    Evolution(drone_config(), tmp_path).run(quiet=True)
    scripted.reset([{"hypothesis": "rule", "items": [RULE]}])
    evo = Evolution(drone_config(loop={"rounds": 2}), tmp_path)
    evo.run(quiet=True)
    assert len(evo.state["rounds"]) == 2 and evo.state["rounds"][1]["round"] == 2
    assert [r for r, *_ in scripted.SEEN].count("proposer") == 1
    assert json.loads((tmp_path / "state.json").read_text())["rounds"][1]["outcome"] in ("adopted", "confirm_failed", "nothing_admitted")


def test_chinese_prompts(tmp_path):
    scripted.reset([{"hypothesis": "fly faster", "items": [FASTER]}])
    Evolution(drone_config(prompt_language="zh"), tmp_path).run(quiet=True)
    assert "你在进化" in scripted.SEEN[0][1] and "审查员" in scripted.SEEN[1][1]


def test_lookup_tools(tmp_path):
    results = {}

    def grab(system, messages):
        results["trace"], results["trial"], results["point"] = [json.loads(json.dumps(m["result"])) for r, m in messages if r == "tool"]
        return {"hypothesis": "fly faster", "items": [FASTER]}

    scripted.reset([("get_point_trace", {"scenario": "a1", "points": ["wp2"]}),
                    ("trial_episode", {"scenario": "a2", "items": [FASTER]}),
                    ("point_trial", {"scenario": "a1", "point": "wp3", "items": [FASTER], "with_trace": True}),
                    grab])
    evo = Evolution(drone_config(loop={"tools": {"trace": 2, "trial_episode": 1, "point_trial": 1}}), tmp_path)
    evo.run(quiet=True)
    assert results["trace"]["points"]["wp2"]["rows"]
    assert results["trial"]["improvement"] > 0 and results["trial"]["items_applied"] == ["i1"]
    assert results["point"]["point_gain_s"] > 0 and results["point"]["candidate_trace"]["rows"]
    rec = evo.state["rounds"][0]
    assert [x["tool"] for x in rec["proposer_calls"][0]["lookups"]] == ["get_point_trace", "trial_episode", "point_trial"]
    assert rec["outcome"] == "adopted"
