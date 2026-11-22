"""The objective (one comparable number per set of episodes) and the gate (when a candidate may replace the champion).

Inside the framework every comparison is a *gain*: how much better b is than a, in the objective's own unit, positive =
better, whatever the direction. The prompts use the same sign.
"""
from __future__ import annotations


class Objective:
    def __init__(self, cfg: dict):
        self.metric = cfg["metric"]
        self.direction = cfg["direction"]
        self.aggregate = cfg["aggregate"]
        self.failure_value = float(cfg["failure_value"])
        self.description = cfg.get("description") or f"{cfg['aggregate']} of {cfg['metric']} ({'lower' if cfg['direction'] == 'min' else 'higher'} is better)"
        self.unit = cfg.get("unit") or ""

    def episode_value(self, ep: dict) -> float:
        """The value of one episode; unfinished or failed episodes count as failure_value."""
        if not ep or ep.get("error") or not ep.get("finished") or ep.get(self.metric) is None:
            return self.failure_value
        return float(ep[self.metric])

    def totals(self, episodes: list[dict]) -> dict:
        vals = [self.episode_value(e) for e in episodes]
        value = sum(vals) if self.aggregate == "sum" else (sum(vals) / len(vals) if vals else self.failure_value)
        metrics = {}
        for e in episodes:
            for k, v in e.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("seed",):
                    metrics[k] = round(metrics.get(k, 0) + v, 4)
        return {
            "value": round(value, 4),
            "unfinished": sum(not e.get("finished") for e in episodes),
            "errors": [e.get("scenario") for e in episodes if e.get("error")],
            "per_scenario": {e.get("scenario"): (round(float(e[self.metric]), 4) if e.get("finished") and not e.get("error") and e.get(self.metric) is not None else None)
                             for e in episodes},
            "metrics": metrics,
        }

    def gain(self, a: float, b: float) -> float:
        """How much better b is than a (positive = b better)."""
        return round((a - b) if self.direction == "min" else (b - a), 4)

    def scenario_gain(self, a_totals: dict, b_totals: dict, scenario: str) -> float:
        fv = self.failure_value
        va = a_totals["per_scenario"].get(scenario)
        vb = b_totals["per_scenario"].get(scenario)
        return self.gain(fv if va is None else va, fv if vb is None else vb)

    def gains_per_scenario(self, a_totals: dict, b_totals: dict) -> dict:
        return {s: self.scenario_gain(a_totals, b_totals, s) for s in a_totals["per_scenario"] if s in b_totals["per_scenario"]}

    def phrase(self) -> str:
        return self.description + (f" [{self.unit}]" if self.unit else "")


def constraint_value(totals: dict, metric: str):
    if metric == "unfinished":
        return totals["unfinished"]
    if metric == "errors":
        return len(totals["errors"])
    return totals["metrics"].get(metric, 0)


class Gate:
    """Admission: the champion and the candidate are both evaluated on the gate scenarios; the candidate passes when
    it gains at least min_gain (or min_gain_relative x |champion value|) and breaks no constraint."""

    def __init__(self, cfg: dict, objective: Objective):
        self.cfg = cfg
        self.obj = objective

    def required_gain(self, champ: dict) -> float:
        g = float(self.cfg.get("min_gain") or 0)
        rel = self.cfg.get("min_gain_relative")
        if rel:
            g = max(g, float(rel) * abs(champ["value"]))
        return g

    def check(self, champ: dict, cand: dict) -> tuple[bool, list[str]]:
        """(passes, reasons it does not)."""
        why = []
        gain = self.obj.gain(champ["value"], cand["value"])
        need = self.required_gain(champ)
        if need > 0 and gain < need - 1e-9:
            why.append(f"gain {gain} < required {need}")
        elif need <= 0 and gain <= 1e-9:
            why.append(f"gain {gain} is not positive")
        for c in self.cfg.get("constraints") or []:
            m = c["metric"]
            a, b = constraint_value(champ, m), constraint_value(cand, m)
            if "max_increase" in c and b - a > c["max_increase"] + 1e-9:
                why.append(f"{m} rose from {a} to {b} (max increase {c['max_increase']})")
            if "max" in c and b > c["max"] + 1e-9:
                why.append(f"{m} is {b} (max {c['max']})")
        lim = self.cfg.get("per_scenario_max_loss")
        if lim is not None:
            for s, g in self.obj.gains_per_scenario(champ, cand).items():
                if -g > lim + 1e-9:
                    why.append(f"{s} got worse by {-g} (max {lim})")
        return not why, why
