"""A2 离线测试用：no-SGLang 的 AgentFlow stub generate。

同 V4/V5/A1 的 stub —— rollout 在 RolloutManager actor 的独立进程里，主进程 monkeypatch 不到，
故离线跑闭环时把 `custom_generate_function_path` 指到这里（不连 SGLang）。它复用 rollout.py 里
**同一个** `_build_solver` + `Solver.solve`（agent loop 只有一份），只注入两个"造假答案"的 chat_fn：

  - **planner chat_fn**（训练引擎位）：plan 轮产一句规划；next_step 轮产可被 formatters 解析的
    `Context / Sub-Goal / Tool Name: calculator`（Context 里放算术表达式）。
  - **fixed chat_fn**（固定引擎位）：executor 轮产 `tool.execute(query="<expr>")`（让**真** calculator
    执行）；verifier 轮首步即 `Conclusion: STOP`；final_output 轮产 `\\boxed{label}`（让 boxed reward 命中）。

这样离线就能验证 A2 核心不变量（多轮结构 + **只有 planner 进 turns / executor token 不训练** + boxed
reward），不需要 GPU。reward_func 直接复用 rollout.py 的（re-export 供 hook 加载）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mini_slime.args import Args
from toy_rl.agent.agentflow.rollout import _build_solver, reward_func  # noqa: F401  (reward re-export)
from toy_rl.sample import Sample


def _extract_expr(question: str) -> str:
    """从 "What is 37 * 24?" 抽出算术表达式 "37 * 24"（stub 专用；真 SGLang 由模型自己形成）。"""
    m = re.search(r"[-+*/().\d\s]+", question.replace("What is", "").rstrip("?"))
    return m.group(0).strip() if m else ""


async def generate(args: Args, sample: Sample) -> Sample:
    """stub：注入两个造假 chat_fn 后调共享的 _build_solver/Solver.solve（不连 SGLang）。"""
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["original_question"] = sample.prompt

    label = str(sample.label) if sample.label is not None else ""
    expr = _extract_expr(sample.prompt)

    async def planner_chat_fn(prompt_text: str) -> str:
        if "Determine the optimal next step" in prompt_text:  # next_step 轮
            return (
                "Justification: The query is an arithmetic computation; the calculator tool solves it.\n"
                f"Context: {expr}\n"
                "Sub-Goal: Compute the arithmetic expression.\n"
                "Tool Name: calculator\n"
            )
        # plan 轮（或其它）——产一句规划即可（不影响解析）
        return "Analysis: this is an arithmetic query; plan to use the calculator tool."

    async def fixed_chat_fn(prompt_text: str) -> str:
        if "Generate a precise command" in prompt_text:       # executor 轮
            return f'Generated Command:\n```python\nexecution = tool.execute(query="""{expr}""")\n```'
        if "Evaluate if the current memory" in prompt_text:   # verifier 轮
            return "The memory now contains the computed result. Conclusion: STOP"
        if "Generate the final output" in prompt_text:        # final_output 轮
            return f"The computation is complete. \\boxed{{{label}}}"
        # judge 轮（reward 回退）——离线 boxed 命中时用不到，兜底给 True
        return "VERDICT: True"

    solver = _build_solver(args, planner_chat_fn, fixed_chat_fn)
    return await solver.solve(args, sample)
