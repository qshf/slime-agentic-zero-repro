"""A2: AgentFlow LLM-as-judge —— 镜像源 agentic/agentflow/core/rewarder.py:Rewarder。

reward 相比 A1 纯 boxed 精确匹配的**新东西**：boxed miss 时回退到"让固定引擎当裁判"。
判官跑在**固定引擎**上（不是训练 planner），保证 reward 信号在整个 RL 过程稳定（对齐源 rollout.py:213）。

范式对齐：prompt 模板从源 rewarder.py:9-30 搬来；解析照搬源——抓 `VERDICT: True/False`，
回退 `<true_false>`，再回退看末行。离线 stub 注入一个假 judge chat_fn。
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

ChatFn = Callable[[str], Awaitable[str]]


class Rewarder:
    def __init__(self, chat_fn: ChatFn) -> None:
        self.chat_fn = chat_fn          # 固定引擎（判官，权重不变）

    def build_judge_prompt(self, question: str, model_response: str, groundtruth: str) -> str:
        return f"""You are a strict answer evaluator.

**Task:** Read the Model Response, extract its final answer, and determine if it matches the Ground Truth.

**Steps:**
1. Read the Model Response carefully. Find the final answer — look for \\boxed{{...}}, "the answer is ...", or the last conclusion.
2. Compare the extracted answer to the Ground Truth.
3. They are equivalent ONLY if they represent the same value.
4. If the Model Response has no clear final answer, or it does not match, output False.
5. Do NOT be lenient. When in doubt, output False.

**Inputs:**
Question: {question}

Model Response:
{model_response}

Ground Truth: {groundtruth}

**You MUST end your response with exactly one of these two lines (no extra text after it):**
VERDICT: True
VERDICT: False"""

    async def compute_reward(self, question: str, model_response: str, groundtruth: str) -> float:
        prompt_text = self.build_judge_prompt(question, model_response, groundtruth)
        response = (await self.chat_fn(prompt_text)).strip()

        match = re.search(r"VERDICT\s*:\s*(True|False)", response, re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).lower() == "true" else 0.0

        match = re.search(r"<true_false>\s*:?\s*(true|false)", response, re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).lower() == "true" else 0.0

        last_line = response.splitlines()[-1].strip().lower() if response else ""
        if last_line in ("true", "true.", "verdict: true"):
            return 1.0
        return 0.0
