Facts about the scenario:
- The drone is a point mass; with full thrust and no wind its top speed is a_max / drag (about 33 m/s), but it cannot stop quickly: it slows down only by thrusting backwards (at most a_max) plus drag.
- A waypoint counts only within `radius` m of its centre; flying past it means turning back, which costs a lot of time. Missed waypoints are reported as incidents.
- The wind zones are known in advance (ctx["wind_zones"]); inside a zone a constant force pushes the drone.
- Leaving the arena is a crash: the episode ends unfinished.
