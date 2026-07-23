"""A2: AgentFlow 执行器 —— 镜像源 agentic/agentflow/core/executor.py:Executor。

**Executor 不被训练**：它跑在**固定 base 引擎**上，生成"工具命令"这一步的 token 不进 training turns。
它做两件事（对齐源）：
  1. `generate_tool_command`：固定引擎产 `execution = tool.execute(query="…")`，正则抽出**自然语言** query。
  2. `execute_command`：按 planner 吐出的 `tool_name` **在注册表里真分发**到对应工具执行（对齐源
     `_resolve_tool_mapping` → `_load_tool` → `tool.execute`）。nano 注册两个工具：
       · `python_coder`（Python_Code_Generator_Tool）：内部独立 coder 模型写代码 → subprocess 取 stdout；
       · `base_generator`（Generalist_Solution_Generator_Tool）：固定 base 引擎直接答。
     两者统一 `execute(query=...)` 契约（对齐源两工具同签名），故分发只按名字选实例。

这就是源"两层结构 + 三模型分工"：executor（base，不训练）出自然语言命令 → 分发到工具执行。写代码那次
LLM 调用（python_coder 内部 coder 模型）**不在** executor、不进 trajectory，故"规划"（planner，训练）与
"写代码"（coder，不训练）解耦——对齐源可分离性，不违铁律。

范式对齐（铁律）：
  - prompt 模板从源 executor.py:85-124 搬来（保持文本可比）；命令里注入 resolve 出的那个工具的
    description（对齐源 build prompt 用 tool_metadata）。
  - **偏离①（命令解析只留正则）**：源 `_parse_command_kwargs` 有三级（exec 执行 / 正则 / 贪婪）+
    `_extract_command` 剥代码块；nano 命令格式固定简单，只保留源 Level-2 的正则回退（三引号/单双引号），
    砍掉 exec 执行（安全 + 淹没重点）。语义等价（都从命令抽出 query 字符串）。
  - **偏离②（dict 注册表替代 importlib 目录扫描）**：源 `_load_tool` 按 tool_name→dir_name 用 importlib
    从 `tools_dir` 动态加载 tool.py；nano 沿用仓内既定注册表范式（见 toy_rl/agent/tools.py:ToolRegistry），
    构造时直接传入工具实例 dict。**分发本身语义等价**（tool_name→tool→execute），只是发现方式从"文件系统
    扫描"简化成"直接注册"。多工具真分发已恢复（本版修正：早先只持单工具 + 假 tool_name 参数）。
"""

from __future__ import annotations

import logging
import re
from typing import Awaitable, Callable

from .tools.base import BaseTool

ChatFn = Callable[[str], Awaitable[str]]

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, chat_fn: ChatFn, toolbox: dict[str, BaseTool]) -> None:
        if not toolbox:
            raise ValueError("Executor toolbox 不能为空")
        self.chat_fn = chat_fn        # 固定 base 引擎（不训练）
        self.toolbox = toolbox        # {tool_name: BaseTool}，execute_command 按名分发

    def _resolve_tool(self, tool_name: str) -> BaseTool:
        """按 tool_name 查注册表（nano 版 `_resolve_tool_mapping`）：清洗名字→命中即返；
        未命中回退到第一个注册工具 + warning（对齐源"未知名回退 Base_Generator"的兜底）。"""
        cleaned = tool_name.strip().strip("`").strip()
        tool = self.toolbox.get(cleaned)
        if tool is not None:
            return tool
        fallback_name, fallback = next(iter(self.toolbox.items()))
        logger.warning("Unknown tool_name: %r, falling back to %s.", tool_name, fallback_name)
        return fallback

    def build_tool_command_prompt(self, query: str, context: str, sub_goal: str, tool_name: str) -> str:
        tool_description = self._resolve_tool(tool_name).tool_description
        return f"""
Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {query}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_description}
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
        """固定引擎产 `tool.execute(query="…")`，正则抽出自然语言 query（不进 training turns）。

        保留这步是**源保真** + A2 教材：它是一次固定引擎 LLM 调用，token **不进 turns**，正是"executor
        token 不训练"的活样本。功能上因两工具统一 `execute(query=...)`、无 per-tool 参数适配，这步价值偏弱
        （源亦如此），但删了会抹掉训练边界的演示，故保留（详见 docs/decisions/a2.md）。
        """
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
        """按 tool_name 在注册表里**真分发**到对应工具执行（对齐源 execute_command）。永不抛异常。"""
        return await self._resolve_tool(tool_name).execute(query)
