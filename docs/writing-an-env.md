# Writing an Env

An Env connects evoharness to your scenario. It is one Python class. The framework starts episodes with it, asks it
when decisions are due, steps it once per control tick and reads the result at the end. Everything about your
domain lives in the Env, its config file and the seed harness. The framework code stays unchanged.

This guide uses `examples/drone/env.py` as the running example. Keep it open next to this page.

## 1. Decide the three time scales

Answer these before writing any code:

| question | drone example |
|---|---|
| What is one **episode**, and what is a **scenario** (the thing an episode is run on)? | one flight through a course; a course generated from its name |
| What is a **control tick**, and what does the executor output each tick? | 1/50 s; an acceleration command `{"ax", "ay"}` |
| Where are the **decision points**, the moments the decision model answers questions? | one per leg, i.e. per waypoint |
| How early must a decision be requested (**lead**), in a unit that makes sense for you? | when the drone is within `lead` metres of the previous waypoint |
| What is the **objective**, and what counts as **unfinished**? | total flight time; leaving the arena or running out of time |

Decision points should be places where the right choice depends on the situation. They should also be far enough
apart that a model taking 0.1–1 s per answer is not a bottleneck. If you find yourself wanting a decision every
tick, that logic belongs in the executor.

## 2. The class attributes

```python
class MyEnv:
    CONTROL_DT = 0.02          # seconds of scenario time per control tick
    MAX_SECONDS = 90.0         # an episode stops after this much scenario time and counts as unfinished
    FEATURES = {...}           # the numbers a playbook rule may test, with a description each
    TRACE_FIELDS = [...]       # the columns of trace_row()
    EXECUTOR_DOC = """..."""   # the executor's contract, shown to the proposer
```

**`FEATURES`** lists every number a playbook rule may use in a condition (`{"field": ..., "op": "<" | ">", "value": ...}`).
It holds two kinds of numbers under one name space:
- the decision point's own features (from `decision_points()`, e.g. `length_m`, `turn_deg`);
- the live features at decision time (from `live_features()`, e.g. `speed_mps`).

The descriptions are shown to the proposer, so include units and the meaning of 1/0 flags. A rule that uses a field
not listed here is rejected.

**`EXECUTOR_DOC`** is the proposer's only description of the executor's world. Cover four things:
- what `ctx` contains, which is whatever `executor_context()` returns;
- the fields of `obs` and their units and sign conventions;
- the control the executor must return, with its fields and ranges;
- any physics or rules the executor cannot see but should know.

Write it as precisely as you would for a new colleague. Anything missing here the proposer has to guess.

**`TRACE_FIELDS`** names the values of `trace_row()`. The framework prepends `t` (seconds) and `point` (the current
decision point), so do not use those two names.

## 3. The required methods

```python
def __init__(self, **options)                 # options = env_options from the config
def reset(self, scenario, seed=0) -> obs      # start an episode
def decision_points(self) -> list[dict]       # [{"id": str, "features": {name: number}}], in order
def due(self, obs, point, lead) -> bool       # request the decision for this point now?
def current_point(self, obs) -> str | None    # the point the episode is in now
def step(self, control) -> (obs, done)        # one control tick
def result(self) -> dict                      # {"finished": bool, <metric>: number, ...}
def executor_context(self)                    # what Executor(ctx, decision_interface) receives
def describe_point(self, point) -> dict       # what the decision model is told about a point
def live_features(self, obs, point) -> dict   # live numbers at decision time
def trace_row(self, obs, control) -> list     # one row, in TRACE_FIELDS order
def close(self)
```

Each call is made by the framework at a specific time:

```
obs = env.reset(scenario, seed)
points = env.decision_points()
ex = Executor(env.executor_context(), decision_interface)
for every tick:
    for each point not decided yet: if env.due(obs, point, lead): ask the decision model, ex.commit(point id, choice)
    control = ex.act(obs)
    row = env.trace_row(obs, control)        # attributed to env.current_point(obs)
    obs, done = env.step(control)
    stop when done or after MAX_SECONDS
res = env.result()
```

Notes on each method:

- **`reset(scenario, seed)`**: `scenario` is a name from the config's `scenarios` lists. `seed` is the repeat index
  (0, 1, ...) when the gate or `evaluate` runs repeats. Make an episode a deterministic function of
  `(scenario, seed)` if you can. Reproducible episodes make every measured difference mean something.
- **`decision_points()`**: called once after `reset`. Ids must be stable strings that are unique within the episode
  (`"wp3"`, `"section_12"`). Features are plain numbers.
- **`due(obs, point, lead)`**: it is called every tick for every point not yet decided, so keep it cheap. Return
  `True` for points that should already have been decided, for example the first point at the start.
- **`current_point(obs)`**: used to add up time and statistics per decision point. The proposer and the reviewer see
  these per-point tables.
- **`step(control)`**: enforce your scenario's own rules here (resets, penalties, collisions). Those rules are not part
  of the harness, and the proposer cannot change them. Raise an exception for an invalid control. The framework
  records it as an executor error.
- **`result()`**: must contain `finished` and the objective's metric. Every other number you return can be used by
  the gate's constraints (summed over the gate scenarios), e.g. `crashed`, `misses`. When `finished` is false, the
  objective uses `failure_value` instead of the metric.
- **`describe_point(point)`**: written for the decision model, so keep it short and concrete.
- **Return types**: everything you return must be JSON-serialisable (dicts, lists, numbers, strings), except
  `executor_context()` and the control.
- **`obs` must survive `copy.deepcopy`**: the option probe copies it.

## 4. Optional methods

Each optional method switches on a proposer tool or a check. Add them once the required part works.

| method | what it enables |
|---|---|
| `start_at(scenario, point_id, seed=0) -> obs` | the `point_trial` tool: a short episode that starts just before a point and is done right after it, so the proposer can tune one place in seconds |
| `probe_observations() -> list[obs]` | the option probe: every option is committed alone, and the executor's state and controls on these synthetic observations are compared with the default option. Options that change nothing are reported to the proposer, and new ones send a proposal back. Without it, useless options go unnoticed. |
| `control_repr(control) -> list[float]` | how the probe compares controls (default: the control's JSON) |
| `summarize_point(rows) -> dict` | the per-point statistics the proposer sees (default: time plus the mean of every numeric trace field). `rows` are dicts with `t`, `point` and your `TRACE_FIELDS`. |
| `incidents(rows, result) -> [{"t", "kind"}]` | failure moments (crashes, misses, ...). The framework cuts the trace around each one and shows it to the proposer as `recent_incidents`. |

`probe_observations()` is called after `reset(scenario)`, so it can depend on the scenario. Cover the situations
where options should matter: different speeds, different distances to the next decision point.

## 5. The seed harness

Write a deliberately simple starting harness, three files in one directory (see `examples/drone/seed/`):

- `decision_interface.json`: a `goal` sentence for the decision model and 1–2 questions with 2–3 options each. Every
  option must change the executor's behaviour.
- `executor.py`: a `class Executor` with `__init__(self, ctx, decision_interface)`, `commit(self, point_id, choice)` and
  `act(self, obs)`. Handle every option of every question. Store the choices per point id and fall back to the
  defaults for points that have not been committed. Imports are limited to `executor_imports` in the config
  (default `math`, `numpy`).
- `playbook.json`: `{"default": {}, "rules": []}` is fine.

Start simple, not tuned. The point of the system is that the proposer improves it, and a crude seed leaves room to
see that happen.

Optionally the executor may define `sample()` returning `{name: number}` of internal reference values (a target
speed, a planned heading). They are added to the trace, so the proposer can compare intent with outcome.

## 6. The domain brief

`brief.md` (`domain_brief` in the config) is pasted into the proposer's instructions. Put facts there that the data
will not reveal quickly: physics limits, how the scenario's rules work, measurement quirks (for example "the first
section includes the standing start"). Keep it factual. Strategy is the proposer's job.

## 7. The config

```yaml
env: env:MyEnv                  # module:Class, importable from python_path
python_path: ["."]              # relative to this file
seed: seed
domain_brief: brief.md
env_options: {}                 # passed to MyEnv(**env_options)

scenarios:
  train: [...]                  # the loop runs these every round
  test: [...]                   # only `evoharness evaluate --split test` uses these

objective:
  metric: time_s                # a key of result()
  direction: min                # or max
  aggregate: sum                # or mean
  failure_value: 90             # the value of an unfinished episode
  description: "total flight time over the scenarios"
  unit: s

gate:                           # when a candidate may replace the champion
  scenarios: train              # or a list, or {train_plus: [...]}
  repeats: 1
  min_gain: 0.05                # in objective units; or min_gain_relative
  constraints:
    - {metric: unfinished, max_increase: 0}
    - {metric: errors, max: 0}
  per_scenario_max_loss: null

decision:
  backend: default              # start without a model; then openai_compatible, jev or python
  timing: blocking
  lead: 15                      # in your Env's unit, passed to due()
```

All values are yours to choose. The full list of options and their defaults is in `evoharness/config.py`.

## 8. Check it

```bash
evoharness check path/to/config.yaml
```

This verifies that the class has every required member and that the seed harness is valid. It then runs a few
seconds of the first training scenario with the option probe and lists the optional capabilities it found. Next,
measure the seed:

```bash
evoharness evaluate path/to/config.yaml --harness seed=path/to/seed --split train --out runs/seed-eval
```

Look at `runs/seed-eval/evals/<digest>/<scenario>-r0.json`. It holds the per-point table, the trace and the
decision statistics that the proposer will see. If these do not explain why an episode went the way it did, the
proposer cannot explain it either. Improve `summarize_point`, `trace_row` or `EXECUTOR_DOC` first.

Before spending money on a real proposer, run a few rounds with a scripted one. `tests/scripted.py` and
`tests/test_loop.py` show how.

## Common mistakes

- **Non-deterministic episodes.** Measured differences become noise and the gate admits luck. Seed every random
  source from `(scenario, seed)`.
- **Decision points too dense.** The model's latency starts to matter, and every point becomes a weak signal.
- **Features the playbook cannot use well.** Booleans as 1/0 and numbers with units work; ids and strings do not.
- **An `EXECUTOR_DOC` that leaves out a convention** (sign of steering, units, coordinate frame). The proposer will
  get it wrong in a way that looks like a bad idea rather than a misunderstanding.
- **Rules hidden in the executor's world.** If the scenario resets or penalises something, do it in `step()` and
  report it in `result()`, so that the gate can constrain it.
