"""Executor (crude seed): fly straight at the target waypoint at a constant cruise speed.

It never brakes before a waypoint, ignores the wind and does not plan the turn to the next waypoint.
"""
import math

BASE_SPEED = 6.0
GAIN = 2.0
SPEED = {"SLOW": 0.7, "NORMAL": 1.0, "FAST": 1.4}


class Executor:
    def __init__(self, ctx, action_space):
        self.ctx = ctx
        self.wp = ctx["waypoints"]
        self.defaults = {q: s["default"] for q, s in action_space["questions"].items()}
        self.choices = {}

    def commit(self, point_id, choice):
        self.choices[point_id] = dict(self.defaults) | dict(choice)

    def aim_point(self, k, choice):
        p = self.wp[k]
        if choice["aim"] == "CUT" and k + 1 < len(self.wp):
            n = self.wp[k + 1]
            d = math.dist(p, n) or 1.0
            r = self.ctx["radius"] * 0.8
            return [p[0] + (n[0] - p[0]) / d * r, p[1] + (n[1] - p[1]) / d * r]
        return p

    def act(self, obs):
        k = obs["target"]
        choice = self.choices.get(f"wp{k + 1}", self.defaults)
        tx, ty = self.aim_point(k, choice)
        x, y = obs["pos"]
        d = math.hypot(tx - x, ty - y) or 1.0
        v = BASE_SPEED * SPEED[choice["speed"]]
        want = [(tx - x) / d * v, (ty - y) / d * v]
        return {"ax": GAIN * (want[0] - obs["vel"][0]), "ay": GAIN * (want[1] - obs["vel"][1])}
