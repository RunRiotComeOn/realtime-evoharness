"""English prompt templates. Placeholders in {braces} are filled by loop.py with str.replace."""

PROPOSER = """You are evolving the whole harness of a real-time decision agent. The only goal: {objective}, on the training scenarios, and the result must generalise to scenarios you have not seen. An unfinished episode counts as {failure_value}.

How the agent works:
- At every decision point (the request goes out {lead} before it, in the scenario's own unit), a fast decision model reads the goal of action_space.json, a description of the decision point, the live state, and the playbook's advice (instruction, may be absent), and answers every multiple-choice question of action_space.json. It can only pick one option per question; it sees no history and does not learn, so option descriptions must state concrete meanings it can judge from. Answers take about 0.1-1 s and get slower with more questions and options: keep options few and clear (2-{max_options} options per question, at most {max_questions} questions).
- executor.py turns the committed choices into controls every control tick ({hz} Hz).
- playbook.json holds advice rules for the decision model: a default plus rules that apply by the decision point's features and the live state; later rules override earlier ones.

The executor's interface and what it can read:
{executor_doc}

You may change any of the three parts and submit many changes in one round (at most {max_items}); each change is an item:
- executor.py: {"part":"executor","op":"replace","old":"original text (must occur exactly once in the current file)","new":"new text"}; or {"part":"executor","op":"rewrite","source":"the whole file"} (only when necessary).
- action_space.json or playbook.json: {"part":"action_space"|"playbook","op":"set","path":[keys...],"value":new value} or {"op":"delete","path":[...]}. Examples: a new question: path ["questions","mode"]; an option's description: path ["questions","speed","options","FAST"]; the whole rule list: path ["rules"].
Every item also has an id (short, e.g. "i1"), a rationale (which scenarios and decision points, what data, and what you expect to change where) and depends_on (ids of the items it needs, e.g. executor code that uses a new question depends on the item adding the question).

Constraints:
- executor.py must keep the Executor(ctx, action_space), commit(point_id, choice) and act(obs) interface, and handle every option of every question (when options change, the executor must change too). It may import only {imports} and must not read or write files or use the network.
- The playbook format (a rule has only when, then and note): {"default":{"q":"LABEL"},"rules":[{"when":[{"field":"f","op":">","value":0.5}],"then":{"q":"LABEL"},"note":"why"}]}. when is 1-{max_conditions} conditions that must all hold, then is {question: option}. Conditions may only use these fields: {features}. The operator is > or <.
- Never write special cases for one scenario or one decision point id: rules and code must be general.
- Every item is tested on its own: the system runs "all items", "without this item (and what depends on it)" and "the current harness plus only this item (and what it depends on)", and a verifier decides item by item; the admitted items together must pass the admission gate ({gate}). Split items well: independent ideas separately, ideas that only work together as dependencies.
{domain_brief}
{mode}
Tools you may call before proposing:
{tools}
- recent_incidents: where candidates of recent rounds failed (crashes, run-offs, whatever the scenario reports), with the trace around the moment (fields are the column names).
- options_without_effect: questions and options of the current harness that change nothing in the executor (each option is committed alone and the executor's state and controls are compared with the default option). The decision model's answers to them are wasted. Your items must not add such questions or options (a new question needs executor code that makes every option change the behaviour), or the round is sent back; existing ones may be removed or wired up.
Earlier rounds' items and verdicts are in edit_log (improves_by: what removing it from all items loses; alone_improves_by: what it gains added alone to the harness of that round; positive is good). Read the reasons for rejections and do not resubmit rejected items unchanged. Reason only from the measured data you are given. Call propose_harness_changes once."""

TOOL_TRACE = "- get_point_trace (at most {n} calls): the trace of 1-3 decision points of one training scenario, every few ticks, so that you can see what the agent did moment by moment."
TOOL_TRIAL = "- trial_episode (at most {n} calls per round): apply candidate items to the current harness and run one episode on one training scenario; returns the comparison with the current harness (value, metrics, per-point differences, incident traces). One episode is one measurement; small differences may be noise."
TOOL_POINT = "- point_trial (at most {n} calls per round, a few seconds each): apply candidate items and run a short episode that starts just before one decision point of one training scenario and stops just after it, next to the current harness from the same start. Use it to tune one place quickly; a point getting better does not mean the episode does, so confirm with trial_episode or the real submission."

BRANCH_HINT = """
Structural changes (rewriting a whole block of executor logic, redesigning the action space with the matching executor) rarely work in one go and are usually rejected item by item. For those set open_branch=true: this round's items open a development branch as one unit, without per-item tests; you then keep repairing the branch ({branch_rounds} rounds at first, extended while repairs keep improving it, at most {branch_max}) seeing its results, errors and per-point comparison with the champion, and as soon as it passes the gate against the champion it goes to the verifier as one change. Do not open a branch for small changes (constants, one or two rules). Only one branch at a time.
"""
BRANCH_MODE = """
This round works on a development branch: the harness below is the branch's current version (not the champion), and `branch` holds its origin, every round's changes and results, the per-point comparison with the champion and the errors. Your items change the branch directly: they are kept if the branch improves (fixing a crash or an unfinished episode counts), otherwise undone. The aim is to make the branch beat the champion soon: first the errors and unfinished scenarios, then the points where it loses most. Do not set open_branch.
"""
WIRE_MODE = """
This round wires up a development branch: the branch already beats the champion and could be merged, but the questions or options listed below have no effect in its executor (whatever the decision model answers, the behaviour is the same), so the merge was sent back to you.
Only wire: make the executor really change its behaviour by these options, or delete them from action_space (and from the playbook where it refers to them). Do not change the rest of the branch's behaviour: with the default options it must behave as before, or the branch may no longer beat the champion. Do not set open_branch.
"""
NO_BRANCH = """
This experiment has no development branches: every item must be tested and pay off on its own; do not set open_branch.
"""
ONLY_PARTS = """
This experiment only allows changes to {parts}; items that change other parts are dropped.
"""
FOCUS_ACTION_SPACE = """
This round redesigns the action space: submit at least one substantial change to action_space, with the executor changes that make the new questions or options really work. Prefer options that name concrete ways of acting the decision model can judge from the situation over bare multipliers; merge or delete useless questions. Update the playbook where it refers to renamed or removed questions and options.
"""

VERIFIER = """You are the reviewer of a system that evolves the harness of a real-time decision agent. The proposer submitted change items and the system measured them on the training scenarios:
- champion: the harness before the changes; candidate: all items; without: all items except one (and what depends on it).
- Every item has two measured effects, **positive means the item makes the result better, negative worse**, also per scenario:
  - improves_by = (without) vs (candidate): what removing it from all items loses;
  - alone_improves_by = (champion) vs (champion + this item and what it depends on): what it gains added alone to the current harness.
  Admitting means adding it to the current harness, so alone is the most direct number; when the two differ a lot, its effect depends on other items (check whether those are admitted too).
- The admitted items together run once more and must pass the admission gate ({gate}) to replace the current harness.
The goal: {objective}, generalising to unseen scenarios; an unfinished episode counts as {failure_value}.

Decide item by item whether to admit it. Look at:
1. Correctness: bugs in the code, failures on other scenarios (division by zero, index errors, an option not handled, a broken interface); whether the JSON matches what the executor really does; special cases for one scenario or decision point.
2. Evidence: whether alone_improves_by (and improves_by) is positive in total and matches the stated reason. Small differences may come from the decision model answering a new question differently. Admit a correct, general item with a clear total gain even if one scenario gets a little worse (say so in the reason); do not admit when the loss on one scenario is close to or larger than the total gain. Items with zero effect are usually not admitted unless they fix a clear error or another admitted item needs them.
3. Dependencies: an item whose dependency is rejected cannot be admitted.
Reject what is risky or contradicts its own evidence, and write the reason clearly (it goes back to the proposer). Call verify_items once."""

BRANCH_MERGE_NOTE = "This is the merge of a development branch built from many items over several rounds; it can only be admitted or rejected as a whole."

REPAIR_JSON = "\n\nThe arguments of your last propose_harness_changes call were not valid JSON ({error}). Submit again; escape newlines and quotes inside strings; submit fewer items if they are very long."
REPAIR_BRANCH_RUNTIME = "\n\nYour items made the branch fail at run time (it ran before):\n{detail}\nFix it and submit again."
REPAIR_FOCUS = "\n\nThis round redesigns the action space, but your submission has no action_space change. Submit again with at least one substantial action_space change (and the matching executor changes)."
REPAIR_DEAD = "\n\nYour last items added questions or options that have no effect in the executor (whatever the decision model answers, the behaviour is the same):\n{detail}\nMake the executor really act on them (and declare depends_on), or remove them, then submit all items again."
REPAIR_RUNTIME = ("\n\nAfter your last items were applied the executor failed at run time (the static checks passed). Common causes: an item uses a name "
                  "(constant, method, option) defined by another item without declaring depends_on, or removing an item breaks what others use. "
                  "Fix it and submit all items again:\n{detail}")
REPAIR_INVALID = ("\n\nNone of your last items was valid. Your submission and the problem with each item are below; fix them and submit again "
                  "(old text must be copied verbatim from the current executor.py and occur exactly once; JSON changes must follow the format):\n{detail}")
ASK_FOR_TOOL = "Call {name} to submit your changes (or call a lookup tool first)."

NOTE_TRIAL = "positive improvement = better than the current harness; one episode is one measurement, small differences may be noise"
NOTE_POINT = "short episode from just before the point; point_gain_s > 0 = the candidate is faster through the point"

PROPOSE_DESC = "Submit this round's changes to the harness"
OPEN_BRANCH_DESC = "These items are one structural change: open a development branch with them as a unit (no per-item tests)"
VERIFY_DESC = "Decide for every item whether to admit it"
TRACE_DESC = "Trace of 1-3 decision points of one training scenario (current harness, every few ticks)."
TRIAL_DESC = "Apply candidate items (same format as propose_harness_changes) to the current harness and run one episode on one training scenario; returns the comparison with the current harness. Does not submit anything."
POINT_DESC = "Apply candidate items to the current harness and run a short episode through one decision point of one training scenario, compared with the current harness from the same start. items may be empty (only the current harness)."

INVALID_UNTESTED = "invalid, not tested: {reason}"
RULE_VERDICT = "rule (no reviewer): alone improves by {gain}, the worst scenario loses {worst}"
RULE_REJECT = ", not admitted"
VERIFIER_FAILED = "the review call failed, nothing admitted this round: {error}"
DEAD_IN_ADMITTED = " (but the admitted set has options without effect in the executor, so none is admitted: {detail})"
ONLY_PARTS_DROP = "this experiment only allows changes to {parts} (or the item depends on a forbidden one)"
FOCUS_DROP = "this round must include an action_space change"
BRANCH_MERGE_BLOCKED = "the branch beats the champion but has questions or options without effect, and wiring did not succeed: {detail}"
BRANCH_RATIONALE = "Merge of a development branch. Original idea: {hypothesis}"
