"""中文提示模板。{花括号} 里的占位符由 loop.py 用 str.replace 填入。"""

PROPOSER = """你在进化一个实时决策智能体的整套 harness。目标只有一个：{objective}（在训练场景上），并且要能泛化到没见过的场景。没完成的一局按 {failure_value} 计。

智能体的工作方式：
- 每个决策点之前（提前 {lead}，单位由场景决定），快速决策模型读到 decision_interface.json 的目标说明（goal）、这个决策点的描述、实时状态，以及 playbook 给出的建议（instruction，可能没有），为这个决策点回答 decision_interface 里的每个选择题。它只能从每题的选项里选一个；它看不到历史、不会学习，所以选项说明要写成它能据以判断的具体含义。它每次大约 0.1–1 秒答完，问题和选项越多越慢，选项必须少而明确（每题 2–{max_options} 个选项，最多 {max_questions} 题）。
- executor.py 每个控制步（{hz} Hz）把已提交的选择翻译成控制量。
- playbook.json 是写给决策模型的建议规则：default 加按决策点特征/实时状态生效的 rules，后面的规则覆盖前面的。

executor 的接口和它能读到的东西：
{executor_doc}

你可以修改这三部分中的任何一部分，一轮可以提交很多条改动（最多 {max_items} 条），每条改动叫一个 item：
- 改 executor.py：{"part":"executor","op":"replace","old":"原文（必须在当前文件里恰好出现一次）","new":"新文本"}；或 {"part":"executor","op":"rewrite","source":"整份文件"}（只在必要时用）。
- 改 decision_interface.json 或 playbook.json：{"part":"decision_interface"|"playbook","op":"set","path":[键...],"value":新值} 或 {"op":"delete","path":[...]}。例如新增一个问题：path ["questions","mode"]；改一个选项说明：path ["questions","speed","options","FAST"]；整体替换规则表：path ["rules"]。
每个 item 还要写 id（短名，如 "i1"）、rationale（依据哪些场景、哪些决策点的什么数据，改了之后预期哪里怎样变化）、depends_on（它依赖的其他 item 的 id 列表，比如 executor 用到新增问题时，要依赖新增问题的那条）。

约束：
- executor.py 必须保留 Executor(ctx, decision_interface)、commit(point_id, choice)、act(obs) 接口；对 decision_interface 里每个问题的每个选项都要能处理（选项变了，executor 也要跟着改）。代码只能 import {imports}，不能读写文件、不能联网。
- playbook 的格式（规则里只能有 when、then、note 三个键）：{"default":{"q":"LABEL"},"rules":[{"when":[{"field":"f","op":">","value":0.5}],"then":{"q":"LABEL"},"note":"理由"}]}。when 是 1–{max_conditions} 个同时成立的条件，then 是 {问题: 选项}。条件只能用这些字段：{features}。比较符 > 或 <。
- 不要针对某一个场景或某个决策点编号写特例：规则和代码必须是通用的规律。
- 每条改动会被单独检验：系统会跑"全部改动"、"去掉这一条（及依赖它的）"和"当前 harness 只加这一条（及它依赖的）"三种版本，由审查员逐条决定是否采用；采用的改动合起来必须通过准入（{gate}）。所以 item 要拆分得当：互相独立的想法分开写，必须一起才成立的写成依赖关系。
{domain_brief}
{mode}
提交前你可以调用的工具：
{tools}
- recent_incidents：最近几轮的候选版本在哪个场景出了事（场景自己报告的失败，比如撞毁、冲出），附出事前后的逐步记录（fields 是列名）。
- options_without_effect：当前 harness 里在 executor 中没有任何作用的问题和选项（系统把每个选项单独提交给 executor，比较它的状态和控制输出，和默认选项完全一样就算没有作用）。决策模型在这些问题上的选择全是白答。你的改动不能新增这样的问题或选项（新增问题必须同时改好 executor 让每个选项真的改变行为），否则整轮会被退回；已有的可以删掉或接到 executor 上。
以往各轮的改动和审查结果在 edit_log 里（improves_by：在全部改动里去掉它损失多少；alone_improves_by：单独加到当时的 harness 上改善多少；正数为好），被拒的理由要认真看，不要原样重复被拒的改动。只依据给出的实测数据推理。调用一次 propose_harness_changes。"""

TOOL_TRACE = "- get_point_trace（最多 {n} 次）：查看某个训练场景 1–3 个决策点的逐步记录，看清楚每一刻智能体怎么做的。"
TOOL_TRIAL = "- trial_episode（每轮最多 {n} 次）：把一组候选 items 应用到当前 harness，在一个训练场景上实跑一局，返回与当前 harness 的对比（结果、指标、逐决策点的差异、出事片段）。一局只是一次实测，小的差别可能是噪声。"
TOOL_POINT = "- point_trial（每轮最多 {n} 次，每次几秒）：把一组候选 items 应用到当前 harness，从某个训练场景某个决策点之前开始跑一小段，过了这个点就停，同时用当前 harness 从同一起点跑一次做对比。用来在具体位置上快速调参；一个点变好不代表整局变好，最后用 trial_episode 或正式提交确认。"

BRANCH_HINT = """
结构性大改（重写 executor 的一整块逻辑、重新设计动作空间和配套的 executor）往往一次跑不好，逐条检验时会被直接拒掉。这种改动请设置 open_branch=true：本轮的 items 作为一个整体开一个开发分支，不做逐条检验；之后由你继续修这个分支（先给 {branch_rounds} 轮；只要修完还在变好就继续延长，最多 {branch_max} 轮）（你会看到分支的实测、出错信息、和冠军逐决策点的对比），分支一旦对冠军通过准入就作为整体交给审查员合并。小改动（调常数、改一两条规则）不要开分支。同一时间只有一个分支。
"""
BRANCH_MODE = """
本轮在开发分支上工作：下面的 harness 是分支的当前版本（不是冠军），branch 里有它的来历、每轮的修改与结果、和冠军的逐决策点对比、以及出错信息。你的 items 会直接改这个分支：改完更好（修好崩溃或跑不完也算）就保留，否则撤销。目标是让分支尽快超过冠军——先修出错和跑不完的场景，再修比冠军差最多的决策点。不要设置 open_branch。
"""
WIRE_MODE = """
本轮是开发分支的接线修复：这个分支已经超过冠军，可以合并，但它让下面列出的问题或选项在 executor 里没有任何作用（决策模型选哪个行为都一样），所以合并被退回给你。
请只做接线：让 executor 真正按这些选项改变行为，或者把它们从 decision_interface 删掉（playbook 里引用它们的地方一起改）。不要改变分支其余的行为：默认选项下的行为要保持不变，否则分支可能不再超过冠军。不要设置 open_branch。
"""
NO_BRANCH = """
本次实验没有开发分支：每条改动都要在本轮单独检验、单独见效，不能设置 open_branch。
"""
ONLY_PARTS = """
本次实验只允许修改 {parts}，其余部分固定不变；改其他部分的 item 会被直接丢弃。
"""
FOCUS_DECISION_INTERFACE = """
本轮是动作空间重设计轮：至少提交一条对 decision_interface 的实质改动，并同时改好 executor 让新的问题/选项真正起作用。选项最好是决策模型能按局面判断的具体做法，而不是单纯的倍数；合并或删掉没有用的问题。删掉或改名的问题和选项，playbook 里引用它们的地方也要一起改。
"""

REVIEWER = """你是实时决策智能体 harness 进化系统的审查员。提议者提交了一组改动（items），系统已经在训练场景上实测：
- champion：改动前的 harness；candidate：全部改动；without：去掉某一条（及依赖它的）之后的版本。
- 每条改动有两个实测效果，**正数都表示这一条让结果更好，负数表示更差**，也都给了逐场景的数字：
  - improves_by：without 对比 candidate，即在全部改动里去掉它损失多少；
  - alone_improves_by：champion 对比 (champion + 这一条及它依赖的条目)，即单独加到当前 harness 上的收益。
  采用意味着把它加到当前 harness 上，所以 alone 的数字最直接；两者差得很多时，说明它的效果依赖别的改动（要看那些改动是否也被采用）。
- 采用的改动合在一起还会再跑一次，必须通过准入（{gate}）才真正替换当前 harness。
目标：{objective}，并能泛化到没见过的场景；没完成的一局按 {failure_value} 计。

请逐条判断是否采用（admit）。判断时要看：
1. 改动是否正确：代码有没有 bug、会不会在别的场景上出错（除零、索引越界、漏处理某个选项、接口被破坏）；JSON 是否与 executor 的实际行为一致；是否针对某个场景或某个决策点写了特例。
2. 实测证据是否支持：alone_improves_by（以及 improves_by）的总数是否为正，理由和实测是否一致。小的差别可能只是决策模型对新问题的回答变了。改动正确且通用、总收益明显为正时可以采用，即使某个场景略差（在理由里说明取舍）；某个场景的损失接近或超过总收益的，不采用。效果为 0 的改动一般不采用，除非它修正明显错误或为其他已采用改动所必需。
3. 依赖关系：依赖的条目被拒时，依赖它的条目也不能采用。
有明显风险或证据与理由矛盾的就拒绝，并写清楚理由（会反馈给提议者）。调用一次 review_items。"""

BRANCH_MERGE_NOTE = "这是一个开发分支的整体合并：它由多条改动在几轮里逐步修成，只能整体采用或整体拒绝。"

REPAIR_JSON = "\n\n上一次调用 propose_harness_changes 时你输出的参数不是合法的 JSON（{error}）。请重新提交；字符串里的换行、引号要正确转义；改动太长时可以少提交几条。"
REPAIR_BRANCH_RUNTIME = "\n\n你的改动让分支在运行时出错（分支本来能跑）：\n{detail}\n请修正后重新提交。"
REPAIR_FOCUS = "\n\n本轮是动作空间重设计轮，你的提交里没有任何 decision_interface 改动。请重新提交，至少包含一条有实质意义的 decision_interface 改动（以及 executor 的配套改动）。"
REPAIR_DEAD = "\n\n你上一次的改动新增了在 executor 里没有任何作用的问题或选项（决策模型选哪个行为都一样）：\n{detail}\n请让 executor 真正按这些选项改变行为（并在 depends_on 里标明依赖），或者去掉它们，然后重新提交全部改动。"
REPAIR_RUNTIME = ("\n\n你上一次的改动应用后，executor 在运行时出错（静态检查通过了，但真跑起来会崩）。常见原因：一条改动用到了另一条改动才定义的名字"
                  "（常数、方法、选项）却没有在 depends_on 里声明；或者去掉某一条之后别的改动用到的东西没了。请修正后重新提交全部改动：\n{detail}")
REPAIR_INVALID = ("\n\n你上一次提交的改动全部不合法，没有一条能用。上一次的提交和每条的问题如下，请修正后重新提交"
                  "（old 文本必须逐字取自当前 executor.py，且在文件里只出现一次；JSON 改动要满足格式）：\n{detail}")
ASK_FOR_TOOL = "请调用 {name} 提交改动（或先调用查询工具）。"

NOTE_TRIAL = "improvement 为正 = 比当前 harness 好；一局只是一次实测，小的差别不一定可信"
NOTE_POINT = "从决策点之前开始的一小段；point_gain_s 为正 = 候选通过这个点更快"

PROPOSE_DESC = "提交这一轮对 harness 的改动"
OPEN_BRANCH_DESC = "这些 items 是一个结构性大改，作为整体开开发分支（不逐条检验）"
REVIEW_DESC = "逐条决定是否采用"
TRACE_DESC = "查看某个训练场景 1–3 个决策点的逐步记录（当前 harness）。"
TRIAL_DESC = "把一组候选 items（格式同 propose_harness_changes）应用到当前 harness，在一个训练场景上实跑一局，返回与当前 harness 的对比。不会提交改动。"
POINT_DESC = "把一组候选 items 应用到当前 harness，在某个训练场景的某个决策点上单独试跑一小段，并与当前 harness 同一起点的结果对比。items 可以为空数组（只看当前 harness）。"

INVALID_UNTESTED = "不合法，没有测试：{reason}"
RULE_VERDICT = "规则判定（无审查员）：单独加入改善 {gain}，最差的场景损失 {worst}"
RULE_REJECT = "，不采用"
REVIEWER_FAILED = "审查调用失败，本轮不采用：{error}"
DEAD_IN_ADMITTED = "（但采用的这组改动里有在 executor 中没有作用的选项，整组不采用：{detail}）"
ONLY_PARTS_DROP = "本次实验只允许改 {parts}（或依赖了不允许的改动）"
FOCUS_DROP = "本轮必须包含对 decision_interface 的改动"
BRANCH_MERGE_BLOCKED = "分支超过冠军，但有在 executor 里没有作用的问题或选项，接线修复没有成功：{detail}"
BRANCH_RATIONALE = "开发分支整体合并。最初的设想：{hypothesis}"
