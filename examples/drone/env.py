"""A small pure-Python scenario: a drone flies through a course of waypoints in a 100 m x 100 m arena with wind zones.

The drone is a point mass in the plane: every control tick (50 Hz) it commands an acceleration (ax, ay) of at most
A_MAX m/s^2; drag and the wind zones act on it. A waypoint counts when the drone comes within RADIUS m of it; leaving
the arena is a crash. The goal is the shortest time through all waypoints.

Decision points are the legs of the course: leg "wpK" is the flight to waypoint K. The decision for a leg is requested
when the drone is within `lead` metres of the previous waypoint.

Scenarios are generated from their names, so any name is a course: "a1", "a2", ... are as good as "test-7".
"""
from __future__ import annotations

import hashlib
import math
import random

ARENA = 100.0
RADIUS = 2.5
A_MAX = 10.0
DRAG = 0.3
DT = 0.02
N_WAYPOINTS = 6


def course(name: str) -> dict:
    rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:12], 16))
    pts = [[rng.uniform(15, 85), rng.uniform(8, 15)]]
    while len(pts) < N_WAYPOINTS + 1:
        p = [rng.uniform(10, 90), rng.uniform(10, 90)]
        if all(math.dist(p, q) > 18 for q in pts) and math.dist(p, pts[-1]) < 60:
            pts.append(p)
    zones = []
    for _ in range(rng.randint(1, 3)):
        ang = rng.uniform(0, 2 * math.pi)
        f = rng.uniform(1.0, 3.5)
        zones.append({"center": [rng.uniform(20, 80), rng.uniform(20, 80)], "radius": rng.uniform(10, 20),
                      "force": [round(f * math.cos(ang), 3), round(f * math.sin(ang), 3)]})
    return {"start": pts[0], "waypoints": pts[1:], "wind_zones": zones}


def wind_at(zones, p):
    wx = wy = 0.0
    for z in zones:
        if math.dist(p, z["center"]) < z["radius"]:
            wx += z["force"][0]
            wy += z["force"][1]
    return wx, wy


def angle_deg(a, b, c):
    """Turn angle at b on the path a -> b -> c (0 = straight on, 180 = turn back)."""
    u = (b[0] - a[0], b[1] - a[1])
    v = (c[0] - b[0], c[1] - b[1])
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cosang = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(cosang))


class DroneEnv:
    CONTROL_DT = DT
    MAX_SECONDS = 90.0
    FEATURES = {
        "length_m": "length of this leg (previous waypoint to this one), m",
        "turn_deg": "turn at this leg's waypoint towards the next one, degrees (0 = straight on; 0 for the last waypoint)",
        "is_last": "this leg ends at the last waypoint (1/0)",
        "next_length_m": "length of the next leg, m (0 for the last)",
        "max_wind_mps2": "strongest wind along this leg, m/s^2",
        "speed_mps": "drone speed when the decision is requested, m/s",
        "dist_to_waypoint_m": "distance to this leg's waypoint when the decision is requested, m",
    }
    TRACE_FIELDS = ["x", "y", "vx", "vy", "speed", "ax", "ay", "wind_x", "wind_y", "target"]
    EXECUTOR_DOC = """
Executor(ctx, decision_interface)    ctx is a dict: "waypoints" [[x, y], ...] in order, "start" [x, y], "radius" (a waypoint
                               counts within this distance, m), "arena" (side of the square arena, m; leaving it is a
                               crash), "a_max" (largest acceleration command, m/s^2), "drag" (velocity damping, 1/s),
                               "dt" (s per control tick), "wind_zones" [{"center", "radius", "force": [fx, fy] m/s^2}]
                               (the wind is constant inside a zone and zero outside; zones add up)
commit(point_id, choice)       the choice {question: LABEL} for leg point_id ("wp1" = the flight to waypoints[0], ...),
                               given before the drone starts that leg; may be repeated for the same leg
act(obs) -> {"ax", "ay"}        every control tick (50 Hz). obs: "pos" [x, y], "vel" [vx, vy], "target" (index of the
                               next waypoint), "t" (s). The command is clipped to a_max.
Physics per tick: vel += (command + wind(pos) - drag * vel) * dt; pos += vel * dt.
"""

    def __init__(self, **options):
        self.options = options

    # ---- episode ------------------------------------------------------------------------------------------------
    def reset(self, scenario, seed=0):
        self.c = course(scenario)
        self.wp = self.c["waypoints"]
        self.pos = list(self.c["start"])
        self.vel = [0.0, 0.0]
        self.target = 0
        self.t = 0.0
        self.crashed = False
        self.finished_at = None
        self.stop_after = None
        self.misses = []
        self._closing = True
        self._last_d = math.dist(self.pos, self.wp[0])
        self._wind = (0.0, 0.0)
        self.path_m = 0.0
        return self.obs()

    def start_at(self, scenario, point_id, seed=0):
        self.reset(scenario, seed)
        k = int(point_id[2:]) - 1
        self.pos = list(self.c["start"] if k == 0 else self.wp[k - 1])
        self.target = k
        self.stop_after = k
        self._last_d = math.dist(self.pos, self.wp[k])
        return self.obs()

    def obs(self):
        return {"pos": [self.pos[0], self.pos[1]], "vel": [self.vel[0], self.vel[1]], "target": self.target, "t": round(self.t, 4)}

    def decision_points(self):
        pts = [self.c["start"]] + self.wp
        out = []
        for k in range(len(self.wp)):
            a, b = pts[k], pts[k + 1]
            nxt = pts[k + 2] if k + 2 < len(pts) else None
            wind = max((math.hypot(*wind_at(self.c["wind_zones"], [a[0] + (b[0] - a[0]) * s / 20, a[1] + (b[1] - a[1]) * s / 20])) for s in range(21)), default=0.0)
            out.append({"id": f"wp{k + 1}", "features": {
                "length_m": round(math.dist(a, b), 1), "turn_deg": round(angle_deg(a, b, nxt), 1) if nxt else 0.0,
                "is_last": int(nxt is None), "next_length_m": round(math.dist(b, nxt), 1) if nxt else 0.0, "max_wind_mps2": round(wind, 2)}})
        return out

    def due(self, obs, point, lead):
        k = int(point["id"][2:]) - 1
        if obs["target"] >= k:
            return True
        return obs["target"] == k - 1 and math.dist(obs["pos"], self.wp[k - 1]) <= lead

    def current_point(self, obs):
        return f"wp{obs['target'] + 1}" if obs["target"] < len(self.wp) else None

    def step(self, control):
        ax, ay = float(control.get("ax", 0.0)), float(control.get("ay", 0.0))
        n = math.hypot(ax, ay)
        if not math.isfinite(n):
            raise ValueError(f"control is not finite: {control}")
        if n > A_MAX:
            ax, ay = ax * A_MAX / n, ay * A_MAX / n
        wx, wy = self._wind = wind_at(self.c["wind_zones"], self.pos)
        self.vel[0] += (ax + wx - DRAG * self.vel[0]) * DT
        self.vel[1] += (ay + wy - DRAG * self.vel[1]) * DT
        old = list(self.pos)
        self.pos[0] += self.vel[0] * DT
        self.pos[1] += self.vel[1] * DT
        self.path_m += math.dist(old, self.pos)
        self.t += DT
        if not (0 <= self.pos[0] <= ARENA and 0 <= self.pos[1] <= ARENA):
            self.crashed = True
            return self.obs(), True
        d = math.dist(self.pos, self.wp[self.target])
        if d <= RADIUS:
            if self.stop_after is not None and self.target == self.stop_after:
                self.finished_at = self.t
                self.target += 1
                return self.obs(), True
            self.target += 1
            if self.target == len(self.wp):
                self.finished_at = self.t
                return self.obs(), True
            self._last_d = math.dist(self.pos, self.wp[self.target])
            self._closing = True
            return self.obs(), False
        if self._closing and d > self._last_d and d < 4 * RADIUS:   # passed the waypoint without touching it
            self.misses.append(round(self.t, 2))
        self._closing = d < self._last_d
        self._last_d = d
        return self.obs(), False

    def result(self):
        return {"finished": self.finished_at is not None and not self.crashed, "time_s": round(self.finished_at, 3) if self.finished_at else None,
                "crashed": int(self.crashed), "misses": len(self.misses), "path_m": round(self.path_m, 1), "waypoints_reached": min(self.target, len(self.wp))}

    def executor_context(self):
        return {"waypoints": [list(p) for p in self.wp], "start": list(self.c["start"]), "radius": RADIUS, "arena": ARENA, "a_max": A_MAX,
                "drag": DRAG, "dt": DT, "wind_zones": [dict(z) for z in self.c["wind_zones"]]}

    def describe_point(self, point):
        return {"leg": point["id"], "of_legs": len(self.wp), **point["features"]}

    def live_features(self, obs, point):
        k = int(point["id"][2:]) - 1
        return {"speed_mps": round(math.hypot(*obs["vel"]), 2), "dist_to_waypoint_m": round(math.dist(obs["pos"], self.wp[k]), 1)}

    def trace_row(self, obs, control):
        return [round(obs["pos"][0], 2), round(obs["pos"][1], 2), round(obs["vel"][0], 2), round(obs["vel"][1], 2),
                round(math.hypot(*obs["vel"]), 2), round(float(control.get("ax", 0.0)), 2), round(float(control.get("ay", 0.0)), 2),
                round(self._wind[0], 2), round(self._wind[1], 2), obs["target"]]

    def close(self):
        pass

    # ---- optional capabilities --------------------------------------------------------------------------------
    def probe_observations(self):
        pts = [self.c["start"]] + self.wp
        out = []
        for k in range(len(self.wp)):
            a, b = pts[k], pts[k + 1]
            d = math.dist(a, b)
            ux, uy = (b[0] - a[0]) / d, (b[1] - a[1]) / d
            for back in (12.0, 5.0):
                p = [b[0] - ux * back, b[1] - uy * back]
                for v in (3.0, 9.0, 16.0):
                    out.append({"pos": p, "vel": [ux * v, uy * v], "target": k, "t": 0.0})
        return out

    def control_repr(self, control):
        return [float(control.get("ax", 0.0)), float(control.get("ay", 0.0))]

    def summarize_point(self, rows):
        if not rows:
            return {"time_s": 0.0}
        sp = [r["speed"] for r in rows]
        return {"time_s": round(len(rows) * DT, 3), "v_entry": sp[0], "v_max": max(sp), "v_exit": sp[-1], "mean_speed": round(sum(sp) / len(sp), 2)}

    def incidents(self, rows, result):
        out = [{"t": t, "kind": "missed_waypoint"} for t in self.misses]
        if result.get("crashed"):
            out.append({"t": round(self.t, 2), "kind": "crash"})
        return out
