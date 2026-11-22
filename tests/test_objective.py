from evoharness.objective import Gate, Objective


def totals(obj, values, **metrics):
    eps = [{"scenario": f"s{i}", "finished": v is not None, "time_s": v, **{k: m[i] for k, m in metrics.items()}} for i, v in enumerate(values)]
    return obj.totals(eps)


def test_min_objective_and_failure_value():
    o = Objective({"metric": "time_s", "direction": "min", "aggregate": "sum", "failure_value": 100})
    a = totals(o, [10, 20])
    b = totals(o, [9, None])
    assert a["value"] == 30 and b["value"] == 109 and b["unfinished"] == 1
    assert o.gain(a["value"], b["value"]) == -79
    assert o.gains_per_scenario(a, b) == {"s0": 1, "s1": -80}


def test_max_objective_gain_sign():
    o = Objective({"metric": "time_s", "direction": "max", "aggregate": "mean", "failure_value": 0})
    assert o.gain(10, 12) == 2


def test_gate_checks_gain_and_constraints():
    o = Objective({"metric": "time_s", "direction": "min", "aggregate": "sum", "failure_value": 100})
    g = Gate({"min_gain": 0.5, "constraints": [{"metric": "unfinished", "max_increase": 0}, {"metric": "crashes", "max_increase": 0}],
              "per_scenario_max_loss": 2}, o)
    champ = totals(o, [10, 20], crashes=[0, 0])
    assert g.check(champ, totals(o, [9, 20], crashes=[0, 0]))[0]
    assert not g.check(champ, totals(o, [9.8, 20], crashes=[0, 0]))[0]            # gain 0.2 < 0.5
    assert not g.check(champ, totals(o, [5, 20], crashes=[1, 0]))[0]              # a crash more
    ok, why = g.check(champ, totals(o, [1, 23], crashes=[0, 0]))                   # s1 loses 3 > 2
    assert not ok and "s1" in why[0]


def test_gate_relative_gain():
    o = Objective({"metric": "time_s", "direction": "min", "aggregate": "sum", "failure_value": 100})
    g = Gate({"min_gain": 0, "min_gain_relative": 0.1, "constraints": []}, o)
    champ = totals(o, [50, 50])
    assert not g.check(champ, totals(o, [45, 50]))[0]
    assert g.check(champ, totals(o, [40, 50]))[0]
