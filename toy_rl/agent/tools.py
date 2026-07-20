"""V1 工具层：对齐 slime-agentic 的工具定义 + 注册 + 分发。

把"工具"从 agent loop 里分出来，正如源项目 `tools/` 与 `core/` 分离：
  源项目 tools/base.py        -> BaseTool（本文件 BaseTool）
  源项目 tools/<x>/tool.py     -> 具体工具类（本文件 CalculatorTool）
  源项目 planner._discover_tools -> available_tools + toolbox_metadata（本文件 ToolRegistry 的两个属性）
  源项目 planner prompt 注入    -> render_for_prompt()（把元数据拼进 system prompt，不进 tools=）
  源项目 executor.execute_command -> ToolRegistry.execute_command（分发，永不抛异常，返回纯串）

范式对齐（铁律）——唯一偏离，写明三要素：
  ① 源怎么做：源项目 Planner._discover_tools 扫描 `tools_dir`，对每个子目录的 tool.py
     用 importlib 动态加载，读 TOOL_NAME / TOOL_DESCRIPTION 组装 available_tools / toolbox_metadata。
  ② 为何偏离：nano V1 只有 1 个工具，目录扫描 + importlib 是仪式，会淹没"工具系统"这个概念。
     故改为单文件内注册——构造 ToolRegistry 时直接传入 tool 实例列表。
  ③ 语义是否等价 / 何时补齐：等价——两者都产出 available_tools + toolbox_metadata + 一个
     execute_command 分发口。`tools_dir` 文件系统发现暂缓到"多工具 / 多 agent"阶段再引入。

次级借鉴（nano_hermes_agent）：只借"单 registry 对象 + dispatch 接缝"的结构（比手写 dict 规整）。
  不借它的 native tool-calling（get_definitions → tools=，破坏 RL 的 token/log_prob 保真度）；
  不借它 tool_result()/tool_error() 的 JSON 信封——源项目 execute_command 返回纯字符串/Any，
  套 JSON 就偏离源了，故这里工具返回一律纯字符串。
"""

from __future__ import annotations

import re


class BaseTool:
    """对齐源项目 agentic/agentflow/tools/base.py:BaseTool。

    源项目 BaseTool 只持有 tool_name / tool_description / demo_commands 三个元数据字段，
    具体工具子类实现 execute。这里加上 execute 的抽象签名让契约更清晰（比源"更清晰非更差"）。
    """

    def __init__(self, tool_name=None, tool_description=None, demo_commands=None):
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.demo_commands = demo_commands or []

    def execute(self, command: str) -> str:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    """安全计算数学表达式的工具。V1 的 calculator() 逻辑收进这个工具类。"""

    def __init__(self) -> None:
        super().__init__(
            tool_name="calculator",
            tool_description="计算数学表达式，只支持数字和 + - * / ( ) . ，不执行任意代码。",
            demo_commands=["<tool>calculator: 2 + 3</tool>"],
        )

    def execute(self, command: str) -> str:
        """只允许数字和基本运算符，正则守卫后再 eval，不 eval 任意代码。"""
        expr = command.strip()
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
            return "Error: 不支持的表达式"
        try:
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — 已通过正则限制
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class ToolRegistry:
    """工具注册表：对齐源项目 planner._discover_tools 的产物 + executor.execute_command 的分发。

    构造时传入工具实例列表（偏离源的 tools_dir 目录扫描，理由见文件头）。
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {t.tool_name: t for t in tools}

    @property
    def available_tools(self) -> list[str]:
        """对齐源项目 available_tools：可用工具名列表。"""
        return sorted(self._tools)

    @property
    def toolbox_metadata(self) -> dict:
        """对齐源项目 toolbox_metadata：{工具名: {description, demo_commands}}。"""
        return {
            name: {
                "description": tool.tool_description or "",
                "demo_commands": tool.demo_commands,
            }
            for name, tool in self._tools.items()
        }

    def render_for_prompt(self) -> str:
        """把工具元数据拼成可读文本，供 system prompt 使用。

        对齐源项目"把 available_tools / toolbox_metadata 注入 planner prompt 文本"的做法
        （planner.py:85-86 / 116-117），而不是喂给 OpenAI 的 tools= 参数——
        RL 要模型自己吐出 <tool>...</tool> 标签、作为可训练 token，不能让服务端替它格式化。
        """
        lines = []
        for name in self.available_tools:
            meta = self.toolbox_metadata[name]
            lines.append(f"- {name}: {meta['description']}")
            for demo in meta["demo_commands"]:
                lines.append(f"  示例: {demo}")
        return "\n".join(lines)

    def execute_command(self, tool_name: str, command: str) -> str:
        """按工具名分发执行，返回结果字符串。

        对齐源项目 executor.execute_command:225——**永不抛异常**，失败返回错误字符串，
        让上层 loop 能继续下一步而不是崩掉整条 rollout。返回纯字符串（不套 JSON 信封）。
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Error: 未知工具 {tool_name}"
        try:
            return tool.execute(command)
        except Exception as exc:
            return f"Tool execution error ({tool_name}): {exc}"
