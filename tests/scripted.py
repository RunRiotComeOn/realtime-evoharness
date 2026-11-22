"""A scripted proposer / verifier / decision model for the tests (the `python` backends)."""
import json

PROPOSALS = []      # each: propose_harness_changes arguments, a (lookup tool, arguments) tuple, or a callable(system, messages) -> either
SEEN = []           # (role, system, last user message) of every call
DECISIONS = {"fn": None}


def reset(proposals):
    PROPOSALS[:] = list(proposals)
    SEEN.clear()


def chat(system, messages, tools):
    names = [t["name"] for t in tools]
    last = next((m for r, m in reversed(messages) if r == "user"), "")
    if "verify_items" in names:
        SEEN.append(("verifier", system, last))
        items = json.loads(messages[0][1])["items"]
        return "verify_items", {"verdicts": [{"id": it["id"], "admit": True, "reason": "scripted: admit"} for it in items]}
    SEEN.append(("proposer", system, last))
    p = PROPOSALS.pop(0) if PROPOSALS else {"hypothesis": "nothing", "items": []}
    if callable(p):
        p = p(system, messages)
    if isinstance(p, tuple):   # a lookup call (tool name, arguments)
        return p
    return "propose_harness_changes", p


def decide(state, questions):
    if DECISIONS["fn"]:
        return DECISIONS["fn"](state, questions)
    return {q: sorted(s["options"])[0] for q, s in questions.items()}
