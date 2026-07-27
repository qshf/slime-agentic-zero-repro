"""A3 ToolOrchestra QA hooks：多专家路由 + 多组件 per-sample reward。"""

from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from typing import Awaitable, Callable

import requests
from openai import OpenAI

from mini_slime.args import Args
from toy_rl.agent.memagent.rollout import _is_equiv, _last_boxed_only_string, _remove_boxed, _strip_think
from toy_rl.sample import Sample

from .solver import GenFn, GenOutput, OrchestraSolver

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


@lru_cache(maxsize=4)
def _tokenizer(model_path: str):
    """加载训练模型的 tokenizer（只读，用于 apply_chat_template + 真 token 化）。

    V6.1 用真 tokenizer 补齐 A3 登记的"手工渲染 prompt"偏离——把 prompt_text 包成单条
    user message 过 chat_template，再 tokenize 得真 prompt_token_ids。缓存避免每轮重载。
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _sglang_gen_fn(args: Args) -> GenFn:
    """orchestrator 走 SGLang 原生 /generate，返回真 token_ids + 真 log_probs。

    对齐源 orchestra_solver._generate_with_tools（:351-403）：
      payload={"text": prompt, "sampling_params":..., "return_logprob": True}
      从 meta_info.output_token_logprobs 取 (log_prob, token_id) 对。
    这是真 RL 保真度的关键——log_probs 必须来自产生该 response 的同一次前向。
    """
    tokenizer = _tokenizer(args.train_model_path)
    loop = asyncio.get_event_loop()

    def _call(prompt_text: str) -> GenOutput:
        # 把单段 prompt 包成 chat message 过真 chat_template（补齐"手工渲染"偏离）。
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": args.orchestra_max_tokens,
                "temperature": args.rollout_temperature,
            },
            "return_logprob": True,
        }
        raw = requests.post(args.sglang_generate_url, json=payload, timeout=360).json()
        meta = raw["meta_info"]
        pairs = meta["output_token_logprobs"]  # [(log_prob, token_id, ...), ...]
        log_probs = [float(p[0]) for p in pairs]
        token_ids = [int(p[1]) for p in pairs]
        return GenOutput(
            response=_strip_think(raw["text"]),
            prompt_token_ids=list(prompt_token_ids),
            token_ids=token_ids,
            log_probs=log_probs,
        )

    async def gen_fn(prompt_text: str) -> GenOutput:
        return await loop.run_in_executor(None, _call, prompt_text)

    return gen_fn


def _build_solver(
    args: Args,
    orchestrator_chat_fn: ChatFn,
    expert_chat_fn: ChatFn,
    orchestrator_gen_fn: GenFn | None = None,
) -> OrchestraSolver:
    return OrchestraSolver(
        orchestrator_chat_fn, expert_chat_fn, args.orchestra_max_steps, orchestrator_gen_fn
    )


async def generate(args: Args, sample: Sample) -> Sample:
    # orchestrator 走真 /generate（真 token+log_probs）；expert 是纯环境，仍走 chat 端点。
    solver = _build_solver(
        args,
        _sglang_chat_fn(args.orchestra_orchestrator_base_url, args.orchestra_orchestrator_model, args),
        _sglang_chat_fn(args.orchestra_expert_base_url, args.orchestra_expert_model, args),
        orchestrator_gen_fn=_sglang_gen_fn(args),
    )
    return await solver.solve(args, sample)


def _prediction(final_output: str) -> str:
    boxed = _last_boxed_only_string(final_output)
    if boxed is not None:
        return _remove_boxed(boxed).strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", final_output)
    return numbers[-1] if numbers else final_output.strip()


async def reward_func(args: Args, sample: Sample) -> dict:
    """计算 reward：只看正确性，0.0 或 1.0。

    简化版：删除了成本/延迟/工具偏好等多组件计算。A3 的教学核心是"多专家路由"
    （orchestrator 能选 search/answer/不同 expert），不是复杂 reward 函数。
    单一正确性 reward 足够验证路由机制，且与 A1/A2 保持一致。

    args 参数保留用于接口统一，但本函数不使用。
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    final_output = str(metadata.get("final_output", ""))
    pred = _prediction(final_output)
    label = str(sample.label) if sample.label is not None else ""
    correctness = 1.0 if pred and label and _is_equiv(pred, label) else 0.0
    return {"reward": correctness, "pred": pred}
