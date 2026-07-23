"""A2: AgentFlow 执行器 —— 镜像源 agentic/agentflow/core/executor.py:Executor。

**Executor 不被训练**：它跑在**固定 base 引擎**上，生成"工具命令"这一步的 token 不进 training turns。
它做两件事（对齐源）：
  1. `generate_tool_command`：固定引擎产 `execution = tool.execute(query="…")`，正则抽出**自然语言** query。
  2. `execute_command`：把 query 交给 `python_coder` 工具执行——**工具内部再调独立 coder 模型（DeepSeek）**
     把 query 翻成 Python、subprocess 执行取 stdout（见 tools/python_coder.py）。

这就是源"两层结构 + 三模型分工"：executor（base，不训练）出自然语言命令 → python_coder（纯环境）内部
coder 模型（独立）出代码 → 执行。写代码那次 LLM 调用**不在** executor、不进 trajectory，故"规划"（planner，
训练）与"写代码"（coder，不训练）解耦——对齐源可分离性，不违铁律。

范式对齐（铁律）：
  - prompt 模板从源 executor.py:85-124 搬来（保持文本可比）。
  - **偏离①（命令解析只留正则）**：源 `_parse_command_kwargs` 有三级（exec 执行 / 正则 / 贪婪）+
    `_extract_command` 剥代码块；nano 命令格式固定简单，只保留源 Level-2 的正则回退（三引号/单双引号），
    砍掉 exec 执行（安全 + 淹没重点）。语义等价（都从命令抽出 query 字符串）。
  - **偏离②（单工具 python_coder，直接持有工具实例）**：源 `_load_tool` 按 tool_name→dir_name importlib
    动态加载 base_generator/python_coder；nano 只有 python_coder 一个工具，直接持有实例，砍掉目录扫描/
    多工具映射。角色等价（executor 生成命令 → 分发给工具执行）；多工具留后续。
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from .tools.python_coder import PythonCoderTool

ChatFn = Callable[[str], Awaitable[str]]


class Executor:
    def __init__(self, chat_fn: ChatFn, coder_tool: PythonCoderTool) -> None:
        self.chat_fn = chat_fn          # 固定 base 引擎（不训练）
        self.coder_tool = coder_tool    # python_coder（内部 coder 模型独立、纯环境）

    @property
    def tool_name(self) -> str:
        return self.coder_tool.tool_name

    def build_tool_command_prompt(self, query: str, context: str, sub_goal: str, tool_name: str) -> str:
        return f"""
Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {query}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {self.coder_tool.tool_description}
- **Relevant Data:** {context}

Instructions:
1.  The command must include exactly one call to `tool.execute()`.
2.  `tool.execute()` only accepts a single keyword argument: `query`.
3.  The `query` value MUST be a **plain-language** description of what to compute. Do NOT put Python code inside the query.
4.  Always wrap the `query` value in triple double-quotes (\"\"\"...\"\"\").

Output Format (no extra text):
Generated Command:
```python
execution = tool.execute(query=\"\"\"<plain-language description of what to compute>\"\"\")
```
"""

    async def generate_tool_command(self, query: str, context: str, sub_goal: str, tool_name: str) -> str:
        """固定引擎产 `tool.execute(query="…")`，正则抽出自然语言 query（不进 training turns）。"""
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

    async def execute_command(self, tool_name: str, query: str) -> str:
        """交给 python_coder 工具执行（内部 coder 模型写代码 → subprocess）。永不抛异常（对齐源）。"""
        return await self.coder_tool.execute(query)
