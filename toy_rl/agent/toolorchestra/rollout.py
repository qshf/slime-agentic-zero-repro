"""A3 ToolOrchestra QA hooks：多专家路由 + 多组件 per-sample reward。"""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

from openai import OpenAI

from mini_slime.args import Args
from toy_rl.agent.memagent.rollout import _is_equiv, _last_boxed_only_string, _remove_boxed, _strip_think
from toy_rl.sample import Sample

from .solver import OrchestraSolver

ChatFn = Callable[[str], Awaitable[str]]


def _sglang_chat_fn(base_url: str, model: str, args: Args) -> ChatFn:
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    loop = asyncio.get_event_loop()

    def _call(prompt_text: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=args.orchestra_max_tokens,
            temperature=args.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""

    async def chat_fn(prompt_text: str) -> str:
        return _strip_think(await loop.run_in_executor(None, _call, prompt_text))

    return chat_fn


def _build_solver(args: Args, orchestrator_chat_fn: ChatFn, expert_chat_fn: ChatFn) -> OrchestraSolver:
    return OrchestraSolver(orchestrator_chat_fn, expert_chat_fn, args.orchestra_max_steps)


async def generate(args: Args, sample: Sample) -> Sample:
    solver = _build_solver(
        args,
        _sglang_chat_fn(args.orchestra_orchestrator_base_url, args.orchestra_orchestrator_model, args),
        _sglang_chat_fn(args.orchestra_expert_base_url, args.orchestra_expert_model, args),
    )
    return await solver.solve(args, sample)


def _prediction(final_output: str) -> str:
    boxed = _last_boxed_only_string(final_output)
    if boxed is not None:
        return _remove_boxed(boxed).strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", final_output)
    return numbers[-1] if numbers else final_output.strip()


def extract_features(sample: Sample) -> dict:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    events = metadata.get("events", [])
    total_cost = 0.0
    total_latency = 0.0
    tool_counts: dict[str, int] = {}
    mapping = metadata.get("model_mapping", {})
    pricing = metadata.get("tool_pricing", {})

    for event in events:
        total_latency += float(event.get("latency_ms", 0.0))
        role = event.get("role_name", "")
        if not role:
            continue
        tool_counts[role] = tool_counts.get(role, 0) + 1
        price = pricing.get(mapping.get(role, ""), {})
        total_cost += (
            event.get("input_tokens", 0) * float(price.get("input", 0.0))
            + event.get("output_tokens", 0) * float(price.get("output", 0.0))
        )

    final_output = str(metadata.get("final_output", ""))
    pred = _prediction(final_output)
    label = str(sample.label) if sample.label is not None else ""
    return {
        "correctness": 1.0 if pred and label and _is_equiv(pred, label) else 0.0,
        "pred": pred,
        "total_cost": total_cost,
        "total_latency": total_latency,
        "tool_counts": tool_counts,
    }


def preference_utility(features: dict, metadata: dict) -> float:
    """单样本版 preference reward；源的同题 min-max/GRPO 放在 custom_convert，nano 无该接口。"""
    if features["correctness"] < 0.5:
        return 0.0
    pref = metadata.get("pref_vec", {})
    cost_budget = max(float(metadata.get("cost_budget", 1.0)), 1e-12)
    latency_budget = max(float(metadata.get("latency_budget_ms", 1.0)), 1e-12)
    components = {
        "accuracy": 1.0,
        "cost": max(0.0, 1.0 - features["total_cost"] / cost_budget),
        "latency": max(0.0, 1.0 - features["total_latency"] / latency_budget),
    }
    used_roles = features.get("tool_counts", {})
    for role in used_roles:
        components[role] = 1.0

    weights = {key: max(0.0, float(pref.get(key, 0.0))) for key in components}
    if sum(weights.values()) == 0:
        return 1.0
    return sum(components[key] * weights[key] for key in components) / sum(weights.values())


async def reward_func(args: Args, sample: Sample) -> dict:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    features = extract_features(sample)
    reward = preference_utility(features, metadata)
    metadata["reward_features"] = features
    return {"reward": reward, **features}
