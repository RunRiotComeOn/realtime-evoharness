import pytest

from evoharness import harness as H
from evoharness.admission import closure, sanitize

from conftest import DRONE

SCHEMA = H.Schema(features={"length_m": "", "speed_mps": ""})


def seed():
    return H.load(DRONE / "seed")


def test_seed_is_valid():
    H.validate(seed(), SCHEMA)


def test_set_and_delete_items():
    h = seed()
    h2 = H.apply_item(h, {"part": "playbook", "op": "set", "path": ["rules", 0],
                          "value": {"when": [{"field": "length_m", "op": ">", "value": 30}], "then": {"speed": "FAST"}, "note": "long leg"}})
    H.validate(h2, SCHEMA)
    assert H.resolve(h2["playbook"], {"length_m": 40})[0] == {"speed": "FAST"}
    assert H.resolve(h2["playbook"], {"length_m": 10})[0] == {}
    h3 = H.apply_item(h2, {"part": "playbook", "op": "delete", "path": ["rules", 0]})
    assert h3["playbook"]["rules"] == []
    assert H.digest(h3) == H.digest(h)


def test_playbook_fields_come_from_the_schema():
    h = H.apply_item(seed(), {"part": "playbook", "op": "set", "path": ["rules"],
                              "value": [{"when": [{"field": "altitude", "op": ">", "value": 1}], "then": {"speed": "FAST"}}]})
    with pytest.raises(H.HarnessError, match="fields"):
        H.validate(h, SCHEMA)


def test_executor_replace_must_match_once():
    with pytest.raises(H.HarnessError, match="occurs 0 times"):
        H.apply_item(seed(), {"part": "executor", "op": "replace", "old": "NOT THERE", "new": "x"})


@pytest.mark.parametrize("src,msg", [
    ("import os\nclass Executor: pass\n", "imports 'os'"),
    ("class Executor:\n    def act(self, obs):\n        return open('x')\n", "open"),
    ("x = 1\n", "class Executor"),
    ("class Executor:\n  def f(self): return ().__class__\n", "__class__"),
])
def test_executor_static_checks(src, msg):
    with pytest.raises(H.HarnessError, match=msg):
        H.validate_executor(src, SCHEMA)


def test_sanitize_drops_invalid_and_adds_undeclared_dependency():
    h = seed()
    items = [
        {"id": "q", "part": "action_space", "op": "set", "path": ["questions", "aim", "options", "WIDE"], "value": "aim wide of the waypoint",
         "rationale": "a third aiming option for wide turns"},
        {"id": "r", "part": "playbook", "op": "set", "path": ["default"], "value": {"aim": "WIDE"},
         "rationale": "use the new option by default everywhere"},
        {"id": "bad", "part": "executor", "op": "replace", "old": "nope", "new": "x", "rationale": "this text does not occur at all"},
        {"id": "short", "part": "playbook", "op": "set", "path": ["default"], "value": {}, "rationale": "x"},
    ]
    kept, dropped = sanitize(h, items, SCHEMA)
    assert [it["id"] for it in kept] == ["q", "r"]
    assert kept[1]["auto_depends_on"] == ["q"]
    assert {d["id"] for d in dropped} == {"bad", "short"}
    assert closure(kept, {"q"}) == {"q", "r"}
