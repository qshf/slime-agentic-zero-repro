"""A2: AgentFlow 执行器 —— 镜像源 agentic/agentflow/core/executor.py:Executor。

**Executor 不被训练**：它跑在**固定引擎**上，生成"工具命令"这一步的 token 不进 training turns。
它做两件事（对齐源）：
  1. `generate_tool_command`：固定引擎产 `execution = tool.execute(query="…")`，正则抽出 query。
  2. `execute_command`：把 query 分发给 `ToolRegistry`（复用 V1 的 calculator）执行，返回结果串。

范式对齐（铁律）：
  - prompt 模板从源 executor.py:85-124 搬来（保持文本可比）。
  - **偏离①（命令解析只留正则）**：源 `_parse_command_kwargs` 有三级（exec 执行 / 正则 / 贪婪）+
    `_extract_command` 剥 ```python 代码块；nano 命令格式固定简单，只保留源 Level-2 的正则回退
    （三引号/单双引号），砍掉 exec 执行（安全 + 淹没重点）与代码块剥离。语义等价（都从命令里抽出
    query 字符串）。
  - **偏离②（单玩具工具 + 复用 ToolRegistry）**：源 `_load_tool` 按 tool_name→dir_name importlib
    动态加载 base_generator(LLM)/python_coder(子进程)；nano 只有一个 calculator，直接复用 V1 的
    `ToolRegistry.execute_command` 分发，砍掉目录扫描/多工具映射/子进程。角色等价（executor 生成命令→
    分发执行），多工具留后续。
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from toy_rl.agent.tools import ToolRegistry

ChatFn = Callable[[str], Awaitable[str]]


class Executor:
    def __init__(self, chat_fn: ChatFn, registry: ToolRegistry) -> None:
        self.chat_fn = chat_fn          # 固定引擎（不训练）
        self.registry = registry

    def build_tool_command_prompt(self, query: str, context: str, sub_goal: str, tool_name: str) -> str:
        return f"""
Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {query}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {self.registry.toolbox_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Construct valid Python code that addresses the sub-goal using the provided context and data.
2.  The command must include exactly one call to `tool.execute()`.
3.  `tool.execute()` only accepts a single keyword argument: `query`.
4.  The `query` value MUST be a plain-language / arithmetic description of what to compute.
5.  Always wrap the `query` value in triple double-quotes (\"\"\"...\"\"\").

Output Format (no extra text):
Generated Command:
```python
execution = tool.execute(query=\"\"\"<what to compute>\"\"\")
```
"""

    async def generate_tool_command(self, query: str, context: str, sub_goal: str, tool_name: str) -> str:
        """固定引擎产 `tool.execute(query="…")`，正则抽出 query（不进 training turns）。"""
        prompt_text = self.build_tool_command_prompt(query, context, sub_goal, tool_name)
        response = await self.chat_fn(prompt_text)
        return self._extract_query(response)

    @staticmethod
    def _extract_query(command: str) -> str:
        """从 `tool.execute(query=<delim>...<delim>)` 抽出 query（源 Level-2 正则回退的 nano 版）。"""
        for delim in ['"""', "'''", '"', "'"]:
            esc = re.escape(delim)
            m = re.search(rf"tool\.execute\(\s*query\s*=\s*{esc}(.*?){esc}\s*\)", command, re.DOTALL)
            if m:
                return m.group(1).strip()
        return ""

    def execute_command(self, tool_name: str, query: str) -> str:
        """分发给 ToolRegistry 执行（复用 V1 calculator）。永不抛异常（对齐源 execute_command）。

        源把 query 作为一段自然语言喂给 LLM 工具；nano 的 calculator 直接把 query 当算术表达式，
        registry.execute_command 内部有正则守卫 + eval（V1 已定，不 eval 任意代码）。
        """
        return self.registry.execute_command(tool_name, query)
