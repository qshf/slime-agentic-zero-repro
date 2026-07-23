"""A2: AgentFlow 工具基类 —— 镜像源 agentic/agentflow/tools/base.py:BaseTool。

源 BaseTool 只持 tool_name / tool_description / demo_commands 三个元数据字段，具体工具子类实现
`execute`。nano 这里把 `execute` 的抽象签名写清楚（async、收 query 返 str），让「executor 分发 →
tool.execute(query=...)」的契约在类型上一目了然（比源"更清晰非更差"）。

两个工具都实现这个契约、都只收单个 `query`（对齐源两工具统一 `execute(query=...)`，见
executor.py:98-101 的 prompt 硬约束）：
  - PythonCoderTool：内部独立 coder 模型写代码 → subprocess 执行（见 python_coder.py）；
  - BaseGeneratorTool：固定 base 引擎直接答（见 base_generator.py）。
executor 按 planner 吐出的 tool_name 在两者间**真分发**（见 executor.py:Executor._resolve_tool）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    tool_name: str
    tool_description: str

    @abstractmethod
    async def execute(self, query: str) -> str:
        """把自然语言 query 执行成结果字符串。永不抛异常（错误以字符串返回，对齐源）。"""
        raise NotImplementedError
