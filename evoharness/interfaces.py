"""The contracts between the framework and a scenario.

You write an Env (one class) for your scenario; the framework only calls the methods below. Everything an Env returns
must be JSON-serialisable (dicts, lists, numbers, strings), except the executor context and the control.

The executor (part of the harness, evolved by the proposer) follows the contract described by Env.EXECUTOR_DOC:

    class Executor:
        def __init__(self, ctx, action_space): ...      # ctx = Env.executor_context()
        def commit(self, point_id, choice): ...         # a decision point's answers {question: LABEL}; may be repeated
        def act(self, obs): ...                         # every control tick; returns what Env.step accepts
        def sample(self): ...                           # optional: {name: number} of internal reference values, traced
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Env(Protocol):
    # ---- class attributes ------------------------------------------------------------------------------------
    CONTROL_DT: float        # seconds of scenario time per control tick
    MAX_SECONDS: float       # an episode stops (unfinished) after this much scenario time
    FEATURES: dict           # {name: description}: the numbers a playbook rule may test (point and live features)
    EXECUTOR_DOC: str        # what ctx, obs and the control are; shown to the proposer
    TRACE_FIELDS: list       # names of the values trace_row returns, in order (must include nothing named "t")

    # ---- required ------------------------------------------------------------------------------------------
    def __init__(self, **options): ...

    def reset(self, scenario: str, seed: int = 0) -> Any:
        """Start an episode; returns the first observation."""

    def decision_points(self) -> list[dict]:
        """All decision points of the episode in the order they occur: [{"id": str, "features": {name: number}}]."""

    def due(self, obs: Any, point: dict, lead: float) -> bool:
        """True once the decision for `point` should be requested (lead is in the Env's own unit)."""

    def current_point(self, obs: Any) -> str | None:
        """The id of the decision point the episode is in now (used to attribute time and statistics)."""

    def step(self, control: Any) -> tuple[Any, bool]:
        """Advance one control tick; returns (observation, done). The Env enforces its own rules here."""

    def result(self) -> dict:
        """At the end: {"finished": bool, <metric>: number, ...}. The objective and the gate read these metrics."""

    def executor_context(self) -> Any:
        """What the Executor gets at construction (geometry, constants, ...)."""

    def describe_point(self, point: dict) -> dict:
        """What the decision model is told about a decision point."""

    def live_features(self, obs: Any, point: dict) -> dict:
        """The live state at decision time: {name: number}; also available to playbook rules."""

    def trace_row(self, obs: Any, control: Any) -> list:
        """One row of the episode trace (values in TRACE_FIELDS order), recorded every few ticks."""

    def close(self) -> None: ...


class OptionalEnv(Protocol):
    """Capabilities an Env may add; each one switches on a proposer tool or a check."""

    def start_at(self, scenario: str, point_id: str, seed: int = 0) -> Any:
        """A short episode that starts just before `point_id` and is done shortly after it (point_trial tool)."""

    def probe_observations(self) -> list:
        """Synthetic observations of the current scenario, used to find options that change nothing."""

    def control_repr(self, control: Any) -> list[float]:
        """Numbers that identify a control (for the option probe); default: the control's JSON."""

    def summarize_point(self, rows: list[dict]) -> dict:
        """Statistics of one decision point from its trace rows (dicts with "t" and TRACE_FIELDS);
        default: time and the mean of every numeric field."""

    def incidents(self, rows: list[dict], result: dict) -> list[dict]:
        """Moments worth showing to the proposer: [{"t": seconds, "kind": str}] (crashes, near misses, ...)."""


class DecisionModel(Protocol):
    """The fast model that answers the questions at every decision point."""

    def decide(self, state: dict, questions: dict) -> dict:
        """questions: {name: {"instructions", "options": {LABEL: description}}}.
        Returns {name: LABEL} for every question, or raises DecisionError / DecisionUnavailable."""


class DecisionError(Exception):
    """One answer failed (timeout, malformed reply): the runner falls back to defaults plus playbook advice."""


class DecisionUnavailable(BaseException):
    """No answer will come (key refused, unpaid, region): the run must stop rather than silently measure defaults.
    A BaseException so that the runner's error handling for executor crashes cannot swallow it."""
