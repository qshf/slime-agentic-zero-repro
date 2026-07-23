"""A3 ToolOrchestra QA solver —— messages/tool observation 的最小源对齐版。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from mini_slime.args import Args
from toy_rl.agent.tools import CalculatorTool
from toy_rl.sample import Sample

from .prompt import append_tool_result, initial_messages, render_orchestrator_prompt

ChatFn = Callable[[str], Awaitable[str]]


@dataclass
class GenOutput:
    """orchestrator 单轮生成的完整产物（对齐源 GenerationOutput 的 nano 子集）。

    源 `_generate_with_tools` 返回 prompt_token_ids/token_ids/log_probs（来自 SGLang /generate）。
    V6.1 起 nano 也让 orchestrator 生成一次性带回真 token + 真 log_probs（真 RL 保真度）。
    """

    response: str
    prompt_token_ids: list[int]
    token_ids: list[int]         # response 段的真 token_ids
    log_probs: list[float]       # response 段每个 token 的真 log_prob（与 token_ids 等长）


GenFn = Callable[[str], Awaitable[GenOutput]]

# V6 确定性 calculator 工具（V1 已建，execute(expression)->str，正则守卫后 eval）。
# 模块级单例：无状态、纯函数式求值，solver 各样本共享。
_CALCULATOR = CalculatorTool()


def _tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.setdefault(ch, len(vocab)) for ch in text]


def _estimate_tokens(text: str) -> int:
    """A3 沿用 nano 的字符级 token 占位，成本仅用于相对比较。"""
    return len(text)


def parse_tool_call(text: str) -> tuple[str | None, dict[str, Any], str | None]:
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None, {}, "no <tool_call> found"
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return None, {}, f"invalid tool-call JSON: {exc.msg}"
    name = raw.get("name")
    arguments = raw.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None, {}, "tool call has no name"
    if not isinstance(arguments, dict):
        return None, {}, "tool call arguments must be an object"
    return name, arguments, None


class OrchestraSolver:
    """源 `OrchestraSolver._solve_qa` 的精简实现。

    每个 orchestrator 输出是一条独立训练序列。工具事件只进入后续 prompt，因此工具 token 永远不训练。
    """

    def __init__(
        self,
        orchestrator_chat_fn: ChatFn,
        expert_chat_fn: ChatFn,
        max_steps: int,
        orchestrator_gen_fn: Optional[GenFn] = None,
    ) -> None:
        # orchestrator_gen_fn（V6.1+）：给了就用真 token+log_probs 生成；没给则回退
        # orchestrator_chat_fn + 字符级假 token（A3/离线路径，行为不变）。
        self.orchestrator_chat_fn = orchestrator_chat_fn
        self.expert_chat_fn = expert_chat_fn
        self.max_steps = max_steps
        self.orchestrator_gen_fn = orchestrator_gen_fn

    async def _generate_orchestrator(
        self, prompt_text: str, vocab: dict[str, int]
    ) -> tuple[str, list[int], int, list[float]]:
        """产 orchestrator 一轮：(tokens, response_length, log_probs)。

        真路径（gen_fn）：token_ids/log_probs 来自 SGLang /generate（真 RL 保真度）。
        假路径（chat_fn）：字符级假 token、log_probs 置 0 占位（离线/A3，无真 tokenizer）。
        """
        if self.orchestrator_gen_fn is not None:
            out = await self.orchestrator_gen_fn(prompt_text)
            tokens = list(out.prompt_token_ids) + list(out.token_ids)
            return out.response, tokens, len(out.token_ids), list(out.log_probs)

        response = await self.orchestrator_chat_fn(prompt_text)
        ptoks = _tokenize(prompt_text, vocab)
        rtoks = _tokenize(response, vocab)
        return response, ptoks + rtoks, len(rtoks), [0.0] * len(rtoks)

    async def solve(self, args: Args, sample: Sample) -> Sample:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        tools = list(metadata.get("tools", []))
        messages: list[dict[str, Any]] = initial_messages(sample.prompt)
        events: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        response_parts: list[str] = []
        vocab: dict[str, int] = {}
        final_output = ""

        for step in range(self.max_steps):
            prompt_text = render_orchestrator_prompt(messages, tools)
            response, tokens, response_length, log_probs = await self._generate_orchestrator(
                prompt_text, vocab
            )
            turns.append({
                "kind": "orchestrator",
                "tokens": tokens,
                "response_length": response_length,
                "loss_mask": [1] * response_length,
                "rollout_log_probs": log_probs,
                "prompt_text": prompt_text,
                "response": response,
            })
            response_parts.append(f"===== [orchestrator {step}] =====\n{response}")
            messages.append({"role": "assistant", "content": response})

            tool_name, tool_args, parse_error = parse_tool_call(response)
            if parse_error:
                events.append(self._event("invalid", "", {}, "", parse_error, 0, 0, 0.0))
                break

            event, done = await self._execute_tool(tool_name or "", tool_args, sample.prompt, messages, metadata)
            events.append(event)
            response_parts.append(
                f"===== [tool] name={event['tool_name']} role={event['role_name'] or '-'} "
                f"status={event['status']} =====\n{event['output'] or event['error']}"
            )

            if done:
                final_output = event["output"]
                break
            append_tool_result(messages, event)

        cat_tokens: list[int] = []
        cat_loss_mask: list[int] = []
        cat_log_probs: list[float] = []
        for turn in turns:
            prompt_len = len(turn["tokens"]) - turn["response_length"]
            cat_tokens += turn["tokens"]
            cat_loss_mask += [0] * prompt_len + turn["loss_mask"]
            # log_probs 只覆盖 response 段；prompt 段补 0（对齐源 _build_output 的 cat_log_probs）。
            cat_log_probs += [0.0] * prompt_len + turn["rollout_log_probs"]

        sample.response = "\n\n".join(response_parts)
        sample.tokens = cat_tokens
        sample.loss_mask = cat_loss_mask
        sample.rollout_log_probs = cat_log_probs
        metadata["original_question"] = sample.prompt
        metadata["final_output"] = final_output
        metadata["messages"] = messages
        metadata["events"] = events
        metadata["turns"] = turns
        return sample

    @staticmethod
    def _event(
        tool_name: str,
        role_name: str,
        tool_input: dict[str, Any],
        output: str,
        error: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "role_name": role_name,
            "input": tool_input,
            "status": "error" if error else "ok",
            "output": output,
            "error": error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 2),
        }

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        question: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if tool_name == "search":
            query = str(tool_args.get("query", question))
            output = str(metadata.get("search_context", "No evidence was supplied for this task."))
            return self._event("search", "", {"query": query}, output, "", _estimate_tokens(query), _estimate_tokens(output), 0.0), False

        if tool_name == "calculator":
            # V6 确定性工具（非 LLM）：真求值 orchestrator 给的表达式，结果回灌下一轮 prompt。
            # 对齐源 SearchRetrievalTool 的"非终止工具、结果作为 tool observation 回流"结构语义。
            expression = str(tool_args.get("expression", ""))
            output = _CALCULATOR.execute(expression)
            return self._event(
                "calculator", "", {"expression": expression}, output, "",
                _estimate_tokens(expression), _estimate_tokens(output), 0.0,
            ), False

        if tool_name != "answer":
            return self._event(tool_name, "", tool_args, "", f"unknown tool: {tool_name}", 0, 0, 0.0), False

        role_name = str(tool_args.get("expert", ""))
        if role_name not in metadata.get("model_mapping", {}):
            return self._event("answer", role_name, tool_args, "", f"unknown expert role: {role_name}", 0, 0, 0.0), False

        observations = "\n".join(
            message["content"] for message in messages if message.get("role") == "tool"
        )
        expert_prompt = "\n\n".join([
            "You are the selected expert. Solve the user question accurately.",
            f"Question: {question}",
            f"Tool observations: {observations or '(none)'}",
            "Return a concise answer and end with the answer in \\boxed{}.",
        ])
        start = time.perf_counter()
        try:
            output = await self.expert_chat_fn(f"[expert={role_name}]\n{expert_prompt}")
            if not output.strip():
                raise RuntimeError("expert returned an empty response")
            error = ""
        except Exception as exc:
            output = ""
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000
        event = self._event(
            "answer", role_name, tool_args, output, error,
            _estimate_tokens(expert_prompt), _estimate_tokens(output), latency_ms,
        )
        return event, not error
