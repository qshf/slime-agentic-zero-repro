"""V2: 对齐 slime 契约的 calculator generate / reward hook。

从 V1 的 calculator_agent.py 重构而来，关键变化是**签名对齐源项目**:
  V1: generate(prompt, label) -> (Sample, final_answer)   # 自定义签名
  V2: async def generate(args, sample) -> Sample          # 对齐 rollout.py:104
      async def reward_func(args, sample) -> dict          # 对齐 rollout.py:209

这样主循环通过 load_function("toy_rl.agent.calculator_hooks.generate") 动态加载，
不再 import 具体函数 —— 换 agent 只改配置路径，零改代码。

args 用一个极简 dataclass 承载配置(源项目是庞大的 argparse Namespace，这里只取需要的)。
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAI

from toy_rl.agent.calculator_agent import MASK_AGENT, MASK_PROMPT, MASK_TOOL, calculator, _tokenize
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
    """对齐 agentic/agentflow/rollout.py:104 的 generate 签名。

    入参 sample 已带 prompt/label；本函数负责把 response/tokens/loss_mask 填好后返回。
    """
    client = OpenAI(base_url=args.sglang_base_url, api_key="EMPTY")
    response_parts: list[tuple[str, int]] = []
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "你是一个数学助手。如果需要计算，用 <tool>calculator: <表达式></tool> 调用工具。"
                "得到工具结果后，给出最终答案: <answer>数字</answer>。"
                "注意: 只输出工具调用或最终答案，不要多余解释。"
            ),
        },
        {"role": "user", "content": sample.prompt},
    ]

    for turn in range(args.max_turns):
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=["</tool>", "</answer>"],
            ),
        )
        agent_text = resp.choices[0].message.content or ""
        tool_match = re.search(r"<tool>calculator:\s*(.+?)(?:</tool>|$)", agent_text, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.+?)(?:</answer>|$)", agent_text, re.DOTALL)

        if tool_match:
            agent_call = agent_text + "</tool>"
            tool_result = calculator(tool_match.group(1).strip())
            tool_text = f"\n[calculator结果: {tool_result}]\n"
            response_parts.append((agent_call, MASK_AGENT))
            response_parts.append((tool_text, MASK_TOOL))
            messages.append({"role": "assistant", "content": agent_call})
            messages.append({"role": "tool", "content": tool_text, "tool_call_id": f"t{turn}"})
        elif answer_match:
            response_parts.append((agent_text + "</answer>", MASK_AGENT))
            break
        else:
            response_parts.append((agent_text, MASK_AGENT))
            messages.append({"role": "assistant", "content": agent_text})

    # 拼 response / tokens / loss_mask
    full_response = "".join(t for t, _ in response_parts)
    vocab: dict[str, int] = {}
    prompt_tokens = _tokenize(sample.prompt, vocab)
    tokens = prompt_tokens[:]
    loss_mask = [MASK_PROMPT] * len(prompt_tokens)
    for text, mask_type in response_parts:
        toks = _tokenize(text, vocab)
        tokens.extend(toks)
        loss_mask.extend([mask_type] * len(toks))

    sample.response = full_response
    sample.tokens = tokens
    sample.loss_mask = loss_mask
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
