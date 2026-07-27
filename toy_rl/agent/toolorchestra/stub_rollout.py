"""A3 离线 rollout：复用真实 solver，仅注入确定性 orchestrator/expert。"""

from __future__ import annotations

from mini_slime.args import Args
from toy_rl.sample import Sample

from .rollout import _build_solver, reward_func  # noqa: F401


async def generate(args: Args, sample: Sample) -> Sample:
    label = str(sample.label) if sample.label is not None else ""

    async def orchestrator_chat_fn(prompt_text: str) -> str:
        if "[TOOL name=search]" in prompt_text:
            return '<tool_call>{"name":"call_expert","arguments":{"expert":"expert_fast"}}</tool_call>'
        return '<tool_call>{"name":"search","arguments":{"query":"find the supplied evidence"}}</tool_call>'

    async def expert_chat_fn(prompt_text: str) -> str:
        return f"The evidence gives the requested value. \\boxed{{{label}}}"

    return await _build_solver(args, orchestrator_chat_fn, expert_chat_fn).solve(args, sample)
