"""A2: base_generator 工具 —— 镜像源 agentic/agentflow/tools/base_generator/tool.py。

源 `Base_Generator_Tool.execute(query)` 把 query 当一条 user message 直接交 `llm_engine.generate`
返回文本——**没有** subprocess、没有代码翻译，就是「拿固定 base 模型直接答」。它和 python_coder 是
executor 可分发的两个工具：planner 吐 `Tool Name` 决定走哪个（见 executor.py:Executor._resolve_tool）。

三模型分工里它属于**固定 base 引擎**那一档（对齐源 engine_map，rollout.py:141 `base_generator` →
`generate_engine`，与 executor/verifier/final_output 同引擎）——纯环境、不训练。故 nano 里它收的是
Solver 的 `fixed_chat_fn`（同 executor/verifier），而 python_coder 收独立 coder 模型。

范式对齐（铁律）：
  - `execute(query)` 直接 chat_fn(query) 返文本，1:1 对齐源（源 `messages=[{user: query}]` → generate）。
  - **偏离**：源持真 `SGLangEngine`（`generate` 返 GenerationOutput）；nano 沿用 V1-V5/A1 的 chat_fn
    抽象（`chat_fn(prompt)->text`）。语义等价（都是固定引擎对 query 出一段文本）。
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .base import BaseTool

TOOL_NAME = "Generalist_Solution_Generator_Tool"   # 对齐源外部工具名
TOOL_DESCRIPTION = (
    "A generalized tool that takes a query and answers the question step by step "
    "to the best of its ability. Returns the model's textual answer."
)

# chat_fn 契约同 executor/verifier：给定 prompt 文本返回模型文本（固定 base 引擎）。
ChatFn = Callable[[str], Awaitable[str]]


class BaseGeneratorTool(BaseTool):
    """镜像源 Base_Generator_Tool：固定 base 引擎直接答 query（无 subprocess、纯环境、不训练）。"""

    tool_name = TOOL_NAME
    tool_description = TOOL_DESCRIPTION

    def __init__(self, chat_fn: ChatFn) -> None:
        self.chat_fn = chat_fn   # 固定 base 引擎（与 executor/verifier 同一档，纯环境）

    async def execute(self, query: str) -> str:
        """把 query 直接交固定引擎返文本（对齐源 execute：messages=[{user: query}] → generate）。"""
        try:
            return await self.chat_fn(query)
        except Exception as exc:  # 对齐源风格：永不抛异常，失败以字符串返回
            return f"Base generator error: {exc}"
