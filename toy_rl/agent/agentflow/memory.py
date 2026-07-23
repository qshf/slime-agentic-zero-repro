"""A2: AgentFlow 记忆 —— 几乎照搬源 agentic/agentflow/core/memory.py:Memory。

跨步累积"用了哪个工具、子目标、命令、结果"的动作字典，供 planner/verifier 的 prompt 读取
（`memory.get_actions()`）。它是 ReAct 循环里流转的"环境状态"，不进 training turns。
"""

from __future__ import annotations

from typing import Any


class Memory:
    MAX_RESULT_CHARS = 4096

    def __init__(self) -> None:
        self.actions: dict[str, dict[str, Any]] = {}

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        result_str = str(result) if result is not None else ""
        if len(result_str) > self.MAX_RESULT_CHARS:
            half = self.MAX_RESULT_CHARS // 2
            result_str = (
                result_str[:half]
                + f"\n... [truncated {len(result_str) - self.MAX_RESULT_CHARS} chars] ...\n"
                + result_str[-half:]
            )
        self.actions[f"Action Step {step_count}"] = {
            "tool_name": tool_name,
            "sub_goal": sub_goal,
            "command": command,
            "result": result_str,
        }

    def get_actions(self) -> dict[str, dict[str, Any]]:
        return self.actions
