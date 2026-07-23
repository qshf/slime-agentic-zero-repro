"""V6.0 GSM8K 离线 rollout：复用真实 solver，注入确定性 orchestrator/expert。

与 A3 stub 的差别：orchestrator 走 `calculator -> answer` 两轮（而非 search -> answer），
让离线也能验证 calculator 工具真求值 + 结果回灌下一轮 prompt。expert 直接吐 label 的 \\boxed{}。

这只是确定性回放（固定路由），不把它当模型能力断言——真实模型行为在服务器 V6.1+ 验证。
"""

from __future__ import annotations

import re

from mini_slime.args import Args
from toy_rl.sample import Sample

from .rollout import _build_solver, reward_func  # noqa: F401


def _expression_for(question: str) -> str:
    """从题面里抓两个数字拼个表达式，仅用于让离线 calculator 有真东西可算。"""
    numbers = re.findall(r"\d+", question)
    if len(numbers) >= 2:
        return f"{numbers[0]}+{numbers[1]}"
    return "1+1"


async def generate(args: Args, sample: Sample) -> Sample:
    label = str(sample.label) if sample.label is not None else ""
    expression = _expression_for(str(sample.prompt))

    async def orchestrator_chat_fn(prompt_text: str) -> str:
        if "[TOOL name=calculator]" in prompt_text:
            return '<tool_call>{"name":"answer","arguments":{"expert":"expert_fast"}}</tool_call>'
        return (
            '<tool_call>{"name":"calculator","arguments":'
            f'{{"expression":"{expression}"}}}}</tool_call>'
        )

    async def expert_chat_fn(prompt_text: str) -> str:
        return f"Using the computed intermediate result. \\boxed{{{label}}}"

    return await _build_solver(args, orchestrator_chat_fn, expert_chat_fn).solve(args, sample)
