import pytest

from evoharness import harness as H
from evoharness.evaluator import Evaluator, RunStopped
from evoharness.interfaces import DecisionError, DecisionUnavailable

import scripted
from conftest import DRONE, drone_config


def evaluator(cfg, tmp_path):
    E = cfg.env_class()
    return Evaluator(cfg, tmp_path, H.Schema(features=dict(E.FEATURES), limits=H.Limits.from_config(cfg["limits"], cfg["executor_imports"])))


def test_episode_in_a_subprocess(cfg, tmp_path):
    ev = evaluator(cfg, tmp_path)
    r = ev.episode(H.load(DRONE / "seed"), "a1")
    assert r["finished"] and r["time_s"] > 10 and not r.get("error")
    assert [p["id"] for p in r["points"]] == [f"wp{k}" for k in range(1, 7)]
    assert abs(sum(p["time_s"] for p in r["points"]) - r["time_s"]) < 0.1
    assert r["decisions"]["calls"] == 6 and r["decisions"]["default"] == 6
    assert r["trace_fields"][:2] == ["t", "point"] and r["trace"]


def test_evaluate_is_memoised_and_totals(cfg, tmp_path):
    ev = evaluator(cfg, tmp_path)
    h = H.load(DRONE / "seed")
    a = ev.evaluate(h, ["a1", "a2"])
    assert a["unfinished"] == 0 and a["value"] == round(sum(a["per_scenario"].values()), 4)
    assert (tmp_path / "evals" / H.digest(h) / "a1-r0.json").exists()
    assert ev.evaluate(h, ["a1"])["per_scenario"]["a1"] == a["per_scenario"]["a1"]


def test_option_probe_finds_options_without_effect(cfg, tmp_path):
    ev = evaluator(cfg, tmp_path)
    h = H.apply_item(H.load(DRONE / "seed"), {"part": "action_space", "op": "set", "path": ["questions", "mood"],
                                             "value": {"instructions": "mood", "default": "CALM", "options": {"CALM": "calm", "ANGRY": "angry"}}})
    h = H.apply_item(h, {"part": "action_space", "op": "set", "path": ["questions", "speed", "options", "SAME"], "value": "like NORMAL"})
    h = H.apply_item(h, {"part": "executor", "op": "replace", "old": '"FAST": 1.4}', "new": '"FAST": 1.4, "SAME": 1.0}'})
    assert ev.smoke(h) is None
    assert sorted(ev.dead[H.digest(h)]) == [["mood", "*"], ["speed", "SAME"]]


def test_executor_crash_is_reported(cfg, tmp_path):
    ev = evaluator(cfg, tmp_path)
    h = H.apply_item(H.load(DRONE / "seed"), {"part": "executor", "op": "replace", "old": "k = obs[\"target\"]", "new": "k = obs[\"target\"] + 1 / 0"})
    r = ev.episode(h, "a1")
    assert r["error"] == "executor_act" and "ZeroDivisionError" in r["traceback"]


def test_point_trial(cfg, tmp_path):
    ev = evaluator(cfg, tmp_path)
    r = ev.episode(H.load(DRONE / "seed"), "a1", point="wp3")
    assert r["finished"] and len(r["points"]) == 1 and r["points"][0]["id"] == "wp3" and r["points"][0]["time_s"] > 0


def test_python_decision_model_and_cache(tmp_path):
    cfg = drone_config(decision={"backend": "python", "callable": "scripted:decide"}, isolation="inline")
    ev = evaluator(cfg, tmp_path)
    scripted.DECISIONS["fn"] = lambda s, q: {"speed": "FAST", "aim": "CUT"}
    try:
        r = ev.episode(H.load(DRONE / "seed"), "a1")
        assert r["decisions"]["model"] == 6
        assert all(p["choice"] == {"speed": "FAST", "aim": "CUT"} for p in r["points"])
        r2 = ev.episode(H.load(DRONE / "seed"), "a1")
        assert r2["decisions"]["cached"] == 6 and r2["time_s"] == r["time_s"]
    finally:
        scripted.DECISIONS["fn"] = None


def test_unavailable_decision_model_stops_the_run(tmp_path):
    cfg = drone_config(decision={"backend": "python", "callable": "scripted:decide"}, isolation="inline")
    ev = evaluator(cfg, tmp_path)

    def refuse(s, q):
        raise DecisionUnavailable(451)
    scripted.DECISIONS["fn"] = refuse
    try:
        with pytest.raises(RunStopped, match="refused"):
            ev.episode(H.load(DRONE / "seed"), "a1")
    finally:
        scripted.DECISIONS["fn"] = None


def test_too_many_fallbacks_stop_the_run(tmp_path):
    cfg = drone_config(decision={"backend": "python", "callable": "scripted:decide"}, isolation="inline")
    ev = evaluator(cfg, tmp_path)

    def fail(s, q):
        raise DecisionError("timeout")
    scripted.DECISIONS["fn"] = fail
    try:
        with pytest.raises(RunStopped, match="answered only 0 of 6"):
            ev.episode(H.load(DRONE / "seed"), "a1")
    finally:
        scripted.DECISIONS["fn"] = None


def test_deadline_timing_falls_back_when_late(tmp_path):
    import time
    cfg = drone_config(decision={"backend": "python", "callable": "scripted:decide", "timing": "deadline", "deadline_ms": 50, "max_fallback": 1.0},
                       isolation="inline")
    ev = evaluator(cfg, tmp_path)

    def slow(s, q):
        time.sleep(0.3)
        return {"speed": "FAST", "aim": "CUT"}
    scripted.DECISIONS["fn"] = slow
    try:
        r = ev.episode(H.load(DRONE / "seed"), "a1", max_seconds=1.0)
        assert r["decisions"]["fallback"] >= 1 and r["decisions"]["model"] == 0
    finally:
        scripted.DECISIONS["fn"] = None
