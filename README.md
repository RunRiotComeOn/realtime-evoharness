# realtime-evoharness

Evolve the harness of a real-time LLM decision agent. An LLM cannot act at control rate (an answer takes 0.1–1 s), so
the agent is split in two:

- a fast **decision model** answers a few multiple-choice questions at every **decision point** (a track section, a
  waypoint, an encounter), and
- an **executor** (code) turns the committed answers into controls at every control tick.

The questions (`action_space.json`), the executor (`executor.py`) and advice rules for the decision model
(`playbook.json`) form the **harness**. Between episodes an offline LLM (the **proposer**) rewrites the harness; every
change is measured (each item left out, each item alone), reviewed by a **verifier**, and admitted only when the result
passes your **gate**. Structural changes can grow in development branches.

You bring the scenario: an `Env` class, the scenarios to train, gate and test on, the objective and the gate thresholds.

```
control tick    (Hz)       executor.act(obs) -> control          never waits
decision point  (seconds)  decision model answers the questions  requested `lead` ahead
round           (offline)  proposer edits, verifier reviews, gate admits
```

## Quick start (no network)

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
pytest                                                    # the whole loop on the drone example, scripted LLM
evoharness check examples/drone/config.yaml               # validate the Env and the seed, run a short episode
evoharness evaluate examples/drone/config.yaml --harness seed=examples/drone/seed --split train --out runs/drone-eval
```

To evolve, set a proposer in the config (any OpenAI-compatible endpoint; the key comes from the environment variable
named by `api_key_env`, or from a `.env` file) and run:

```bash
evoharness run examples/drone/config.yaml --out runs/drone/s1
```

A stopped run resumes at its next round with the same command. Everything (state, every evaluated harness, episodes,
the decision cache) is in the output directory.

## Writing an Env

The step-by-step guide is `docs/writing-an-env.md`; the contracts are in `evoharness/interfaces.py`, the example in
`examples/drone/env.py`. Required:

| member | meaning |
|---|---|
| `CONTROL_DT`, `MAX_SECONDS` | seconds per control tick; an episode stops (unfinished) after this |
| `FEATURES` | `{name: description}` of the numbers a playbook rule may test |
| `EXECUTOR_DOC` | what the executor gets (`ctx`), sees (`obs`) and returns (the control); shown to the proposer |
| `TRACE_FIELDS` | the columns of `trace_row` |
| `reset(scenario, seed)` | start an episode |
| `decision_points()` | `[{"id", "features"}]` in order |
| `due(obs, point, lead)`, `current_point(obs)` | when to request a decision; where the episode is |
| `step(control) -> (obs, done)` | one control tick; the Env enforces its own rules |
| `result()` | `{"finished": bool, <metrics>}` |
| `executor_context()`, `describe_point()`, `live_features()`, `trace_row()`, `close()` | |

Optional, each switches on a tool or a check: `start_at` (the `point_trial` tool), `probe_observations` (detects options
that change nothing), `summarize_point`, `incidents` (failure moments shown to the proposer), `control_repr`.

## Objective and gate

```yaml
objective: {metric: time_s, direction: min, aggregate: sum, failure_value: 90, description: "total flight time", unit: s}
gate:
  scenarios: train            # or a list, or {train_plus: [...]}
  repeats: 1
  min_gain: 0.05              # or min_gain_relative: 0.002
  constraints: [{metric: unfinished, max_increase: 0}, {metric: errors, max: 0}]
  per_scenario_max_loss: null
```

## Backends

- Decision model: `default` (defaults plus playbook advice, a baseline), `random`, `openai_compatible`, `jev`, `python`.
  `decision.timing: blocking` (the scenario waits for answers: reproducible, cached; for evolving) or `deadline`
  (real-time pacing, late answers fall back; for measuring latency). A refused key or too many fallbacks stop the run.
- Proposer / verifier: `openai_compatible` (optional `fallback` after quota errors), `codex` (`codex exec` with your
  own login), `python` (your function; the tests use a scripted one).

Prompts exist in English (`prompt_language: en`, default) and Chinese (`zh`).

## Safety note

`executor.py` is LLM-written code. It runs in a subprocess and is checked statically (allowed imports, no file, eval
or dunder access), but that is a guard against accidents, not a sandbox: run evolution in a container or VM if that
matters to you.

## License

MIT
