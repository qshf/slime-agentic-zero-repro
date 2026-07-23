"""A2: AgentFlow 校验器 —— 镜像源 agentic/agentflow/core/verifier.py:Verifier。

**Verifier 不被训练**：跑在**固定引擎**上，判断"当前 memory 是否够回答 query"→ 产 STOP / CONTINUE，
只用来决定循环是否继续，生成 token 不进 training turns。

范式对齐：prompt 模板从源 verifier.py:19-42 搬来；`parse_conclusion` 照搬源 verifier.py:50-65
（先正则抓 `Conclusion: STOP/CONTINUE`，回退看末尾三行，再回退默认 CONTINUE）。
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from .memory import Memory

ChatFn = Callable[[str], Awaitable[str]]


class Verifier:
    def __init__(self, chat_fn: ChatFn, available_tools: list[str], tools_metadata: dict) -> None:
        self.chat_fn = chat_fn          # 固定引擎（不训练）
        self.available_tools = available_tools
        self.tools_metadata = tools_metadata

    def build_verification_prompt(self, question: str, query_analysis: str, memory: Memory) -> str:
        return f"""
Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.tools_metadata}
- **Initial Analysis:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess whether the memory fully addresses all parts of the query.
3.  Check for inconsistencies, ambiguity, or missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""

    async def verificate_context(self, question: str, query_analysis: str, memory: Memory) -> str:
        prompt_text = self.build_verification_prompt(question, query_analysis, memory)
        response = await self.chat_fn(prompt_text)
        return self.parse_conclusion(response)

    @staticmethod
    def parse_conclusion(response: str) -> str:
        """照搬源 parse_conclusion：抓 `Conclusion: STOP/CONTINUE`，回退看末尾三行，再默认 CONTINUE。"""
        pattern = r"conclusion\**:?\s*\**\s*(STOP|CONTINUE)\b"
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

        tail = "\n".join(response.strip().splitlines()[-3:]).lower()
        if re.search(r"\bstop\b", tail):
            return "STOP"
        if re.search(r"\bcontinue\b", tail):
            return "CONTINUE"
        return "CONTINUE"
