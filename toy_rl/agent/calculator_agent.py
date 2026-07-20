"""V1: 极简 calculator agent + Qwen3-0.6B 真推理。

从 V0 的"写死单样本"升级到"真实多轮交互生成样本"。
Qwen3-0.6B 通过 SGLang 的 OpenAI 兼容接口运行，这和源项目的架构形态一致:
  源项目: agentic/agentflow/rollout.py:104 generate() 调 SGLangEngine 生成 agent 轨迹
  V1:    calculator_agent.py         调 http://localhost:30000/v1 生成 agent 轨迹

V1 的核心教学点: 看清 tokens/loss_mask 如何从真实多轮对话里程序化生成——
  不再是手工填写，而是每轮交互结束后，按"谁生成的这段文字"来打 loss_mask。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from toy_rl.sample import Sample

# SGLang 的 OpenAI 兼容接口，对齐源项目 agentic/agentflow/rollout.py 里的默认端口
SGLANG_BASE_URL = "http://localhost:30000/v1"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# loss_mask 的三种段类型（与 V0 保持概念对齐）
MASK_PROMPT = 0   # 题目/context，不训练
MASK_AGENT  = 1   # 模型生成，训练
MASK_TOOL   = 0   # 工具返回，不训练


def calculator(expr: str) -> str:
    """安全计算数学表达式，返回结果字符串。

    只允许数字和基本运算符，不 eval 任意代码。
    """
    expr = expr.strip()
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        return "Error: 不支持的表达式"
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — 已通过正则限制
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _tokenize(text: str, existing_vocab: dict[str, int]) -> list[int]:
    """简单的字符级 tokenizer（V1 教学用）。

    源项目用 SGLang 的真 tokenizer；V1 用字符级假 tokenizer，
    目的是让 tokens/loss_mask 的对应关系在 print 里肉眼可读。
    真 tokenizer 在 V6 接 SGLang RolloutManager 时自然引入。
    """
    tokens = []
    for ch in text:
        tokens.append(existing_vocab.setdefault(ch, len(existing_vocab)))
    return tokens


async def generate(prompt: str, label: str, max_turns: int = 5) -> Sample:
    """多轮 calculator agent rollout，返回完整 Sample。

    对齐源项目签名: agentic/agentflow/rollout.py:104
      async def generate(args, sample, sampling_params, evaluation=False) -> Sample

    这里简化了 args/sampling_params，把核心流程讲清楚就够了。

    agent 流程:
      模型判断是否需要调用 calculator
        -> 是: 生成 <tool>calculator: <expr></tool>，执行，把结果拼回上下文
        -> 否: 生成最终答案 <answer>...</answer>
    """
    import asyncio

    client = OpenAI(base_url=SGLANG_BASE_URL, api_key="EMPTY")

    # ---------- 累积轨迹 ----------
    # response_parts: 每段 (text, mask_type)
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
        {"role": "user", "content": prompt},
    ]

    final_answer: str | None = None

    for turn in range(max_turns):
        # 调 SGLang 生成下一段
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=128,
                temperature=0.0,
                stop=["</tool>", "</answer>"],
            ),
        )
        agent_text = response.choices[0].message.content or ""

        # 检查是否含工具调用
        tool_match = re.search(r"<tool>calculator:\s*(.+?)(?:</tool>|$)", agent_text, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.+?)(?:</answer>|$)", agent_text, re.DOTALL)

        if tool_match:
            # --- 工具调用轮 ---
            agent_call_text = agent_text + "</tool>"  # 补完闭合 tag
            tool_result = calculator(tool_match.group(1).strip())
            tool_text = f"\n[calculator结果: {tool_result}]\n"

            response_parts.append((agent_call_text, MASK_AGENT))   # 模型生成 -> 训练
            response_parts.append((tool_text,        MASK_TOOL))    # 工具返回 -> 不训练

            # 把工具结果拼回对话，模型继续看
            messages.append({"role": "assistant", "content": agent_call_text})
            messages.append({"role": "tool",      "content": tool_text, "tool_call_id": f"t{turn}"})

        elif answer_match:
            # --- 最终答案轮 ---
            final_text = agent_text + "</answer>"
            response_parts.append((final_text, MASK_AGENT))   # 最终答案 -> 训练
            # 模型可能吐出 <answer>5</</answer> 这类脏尾（0.6B 小模型格式不稳），
            # 答案本身不含 '<'，截到第一个 '<' 之前最鲁棒。
            final_answer = answer_match.group(1).split("<", 1)[0].strip()
            break

        else:
            # 模型输出了别的，当作普通 agent 文本继续
            response_parts.append((agent_text, MASK_AGENT))
            messages.append({"role": "assistant", "content": agent_text})

    # ---------- 拼 response / tokens / loss_mask ----------
    full_response = "".join(text for text, _ in response_parts)
    vocab: dict[str, int] = {}

    # prompt 段 token（loss_mask 全 0）
    prompt_tokens = _tokenize(prompt, vocab)
    prompt_mask   = [MASK_PROMPT] * len(prompt_tokens)

    # response 段 token（按段类型打 mask）
    resp_tokens: list[int] = []
    resp_mask:   list[int] = []
    for text, mask_type in response_parts:
        toks = _tokenize(text, vocab)
        resp_tokens.extend(toks)
        resp_mask.extend([mask_type] * len(toks))

    return Sample(
        prompt=prompt,
        label=label,
        response=full_response,
        tokens=prompt_tokens + resp_tokens,
        loss_mask=prompt_mask + resp_mask,
        reward=None,  # reward 由 reward_func 单独计算，V1 先留 None
    ), final_answer
