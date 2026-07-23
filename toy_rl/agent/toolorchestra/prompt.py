"""A3 的 messages 构造器，镜像源 ToolOrchestra PromptBuilder QA 分支。"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are an orchestrator. Always call exactly one available tool; never solve the "
    "problem directly. Use search before answer when evidence would help."
)


def initial_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def append_tool_result(messages: list[dict[str, Any]], event: dict[str, Any]) -> None:
    """将结构化事件作为 tool observation 追加。错误绝不降级为模糊字符串。"""
    messages.append({
        "role": "tool",
        "name": event["tool_name"],
        "content": json.dumps({
            "status": event["status"],
            "output": event.get("output", ""),
            "error": event.get("error", ""),
        }, ensure_ascii=False),
    })


def render_orchestrator_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """把 messages/tool schema 渲染为兼容普通 chat completion 的单段 prompt。"""
    history: list[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        name = f" name={message['name']}" if message.get("name") else ""
        history.append(f"[{role}{name}]\n{message.get('content', '')}")

    return "\n\n".join([
        "Available tools:\n" + json.dumps(tools, ensure_ascii=False),
        "Conversation:\n" + "\n\n".join(history),
        (
            "Return exactly one tool call and no surrounding explanation:\n"
            '<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>\n'
            "or\n"
            '<tool_call>{"name":"answer","arguments":{"expert":"expert_fast"}}</tool_call>'
        ),
    ])
