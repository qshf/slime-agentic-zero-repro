"""V2: 对齐 slime 契约的 calculator generate / reward hook。

从 V1 的 calculator_agent.py 重构而来，关键变化是**签名对齐源项目**:
  V1: generate(prompt, label) -> Sample                   # 便捷入口
  V2: async def generate(args, sample) -> Sample          # 对齐 rollout.py:104
      async def reward_func(args, sample) -> dict          # 对齐 rollout.py:209

这样主循环通过 load_function("toy_rl.agent.calculator_hooks.generate") 动态加载，
不再 import 具体函数 —— 换 agent 只改配置路径，零改代码。

范式对齐（铁律）: agent loop 只在 calculator_agent.run_agent_loop 一处，
本文件的 generate 只做"框架契约 → loop → 回填 sample"的薄适配，
正如源项目 rollout.py:104 只是 solver.solve() 的薄适配器，绝不重抄一遍 loop。

args 用一个极简 dataclass 承载配置(源项目是庞大的 argparse Namespace，这里只取需要的)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAI

from toy_rl.agent.calculator_agent import run_agent_loop
from toy_rl.reward import extract_answer
from toy_rl.sample import Sample


@dataclass
class Args:
    """极简配置对象，对齐源项目 args 的角色(但只保留必要字段)。"""
    sglang_base_url: str = "http://localhost:30000/v1"
    model_name: str = "Qwen/Qwen3-0.6B"
    max_turns: int = 5
    max_tokens: int = 128
    temperature: float = 0.0
    # hook 路径(对齐 --custom-generate-function-path / --custom-rm-path)
    custom_generate_function_path: str = "toy_rl.agent.calculator_hooks.generate"
    custom_rm_path: str = "toy_rl.agent.calculator_hooks.reward_func"


async def generate(args: Args, sample: Sample) -> Sample:
    """对齐 agentic/agentflow/rollout.py:104 的 generate 签名（薄适配器）。

    入参 sample 已带 prompt/label；本函数只负责调 loop、把 loop 结果回填进 sample。
    loop 本身在 calculator_agent.run_agent_loop（唯一实现处）。
    """
    client = OpenAI(base_url=args.sglang_base_url, api_key="EMPTY")
    traj, final_answer = await run_agent_loop(
        client,
        args.model_name,
        sample.prompt,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    sample.response = traj.response
    sample.tokens = traj.tokens
    sample.loss_mask = traj.loss_mask
    sample.metadata = {"final_output": final_answer, "turns": traj.turns}
    return sample


async def reward_func(args: Args, sample: Sample) -> dict:
    """对齐 agentic/agentflow/rollout.py:209 的 reward_func 签名。

    返回 dict(源项目 reward 可以是 dict 承载多组件)；V2 只放 reward 一项。
    """
    predicted = extract_answer(sample.response)
    if predicted is None:
        score = 0.0
    else:
        try:
            score = 1.0 if float(predicted) == float(sample.label) else 0.0
        except (ValueError, TypeError):
            score = 1.0 if predicted.strip() == (sample.label or "").strip() else 0.0
    return {"reward": score}

