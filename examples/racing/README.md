# racing (placeholder)

How a kart time-trial scenario maps onto evoharness. There is no code here. The letters stand for values you choose.

| evoharness | racing |
|---|---|
| scenario | a track |
| decision point | a track section (straight or corner), with geometry features such as corner or not, smallest radius, total turn |
| lead | *d* m before the section starts |
| control tick | one physics frame (1/*f* s): steering, throttle, brake and the game's other inputs |
| Env rules | the game's own rules, e.g. a rescue after *a* s off the road or *b* s stuck |
| objective | sum of lap times over the training tracks; an unfinished lap counts as *T* s |
| gate | at least *g* s faster in total, no more unfinished laps, no executor errors |
| incidents | rescues and run-offs, with the trace from *m* s before to *n* s after |

To build it, write an Env around the game's time-trial mode; see `docs/writing-an-env.md` and `examples/drone/env.py`.
