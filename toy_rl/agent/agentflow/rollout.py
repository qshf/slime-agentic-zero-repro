"""A2: AgentFlow rollout —— 薄适配器，镜像源 agentic/agentflow/rollout.py。

对齐源 rollout.py 的两个职责：
  - `generate(args, sample)`：造**三模型**引擎（对齐源 engine_map，rollout.py:136-144）→ 组装 Solver →
    `solver.solve` → 回填 Sample。三分：
      · planner（plan/next_step）= 训练引擎（policy，权重每轮更新）—— **唯一训练目标**；
      · executor/verifier/final_output = 固定 base 引擎（纯环境，不训练）；
      · python_coder 内部 coder 模型 = **外部 DeepSeek API**（独立模型，纯环境，不训练）。
  - `reward_func(args, sample)`：boxed 精确匹配（命中给 1.0）→ miss 回退 `Rewarder`（固定引擎判官），
    对齐源 rollout.py:209/237-244。

范式对齐（铁律）：agent loop 全在 solver.py（对齐源 rollout 是薄适配器、loop 在 core/solver.py），
本文件只做"造引擎 + 回填 sample + reward"。boxed/is_equiv/strip_think **复用 A1 memagent.rollout**。

偏离（详见 docs/decisions/a2.md 偏离表）：
  - **base 角色共享 planner 端点**：源 executor/verifier/final 用独立 base 引擎（30000），与 planner
    policy 分开；nano 默认两字段同端点（0.6B 跑 judge/final 太弱），**角色仍分两 chat_fn**，loss_mask
    边界与端点无关；服务器把 af_fixed_base_url 指 30000 即成两真实引擎。
  - **coder = 外部 DeepSeek API**：源 coder 挂独立 SGLang（30001，coder_engine）；nano 不起第二个
    SGLang，coder 是独立于 policy 的纯环境模型，用外部 API 当它最贴源三分。密钥走环境变量、离线用 stub。
  - 字符级假 token / 无真 log_probs（沿用 V1-V5/A1）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from openai import OpenAI

from mini_slime.args import Args  # noqa: F401  (re-export，保持 `from ...rollout import Args` 可用)
from toy_rl.agent.agentflow.executor import Executor
from toy_rl.agent.agentflow.planner import Planner
from toy_rl.agent.agentflow.rewarder import Rewarder
from toy_rl.agent.agentflow.solver import Solver
from toy_rl.agent.agentflow.tools.base_generator import BaseGeneratorTool
from toy_rl.agent.agentflow.tools.python_coder import (
    PythonCoderTool,
    deepseek_coder_chat_fn,
)
from toy_rl.agent.agentflow.verifier import Verifier
# 复用 A1 的 boxed/归一化/strip_think（同一套，不重复造）
from toy_rl.agent.memagent.rollout import (
    _is_equiv,
    _last_boxed_only_string,
    _remove_boxed,
    _strip_think,
)
from toy_rl.sample import Sample

ChatFn = Callable[[str], Awaitable[str]]


def _sglang_chat_fn(base_url: str, model: str, args: Args) -> ChatFn:
    """造一个打 SGLang chat completion 的 chat_fn（关 Qwen3 思考段 + 剥残留 <think>，同 A1）。"""
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    loop = asyncio.get_event_loop()

    def _call(prompt_text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=args.af_max_tokens,
            temperature=args.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""

    async def chat_fn(prompt_text: str) -> str:
        text = await loop.run_in_executor(None, _call, prompt_text)
        return _strip_think(text)

    return chat_fn


def _build_solver(args: Args, planner_chat_fn: ChatFn, fixed_chat_fn: ChatFn, coder_chat_fn) -> Solver:
    """组装三模型 Solver：planner→训练引擎，executor/verifier/final_output→固定引擎，
    python_coder 内部→独立 coder 模型。对齐源 rollout.py:136-144 的 engine_map（nano 三分版）。
    real / stub 都调它，只换三个 chat_fn。

    工具注册两个（对齐源 engine_map 的两工具 + executor 真分发）：
      · python_coder → 内部独立 coder 模型（coder_chat_fn，纯环境）；
      · base_generator → 固定 base 引擎（fixed_chat_fn，与 executor/verifier 同档，对齐源 rollout.py:141
        `base_generator`→`generate_engine`）。
    executor 按 planner 吐出的 tool_name 在两者间真分发（见 executor.py:Executor._resolve_tool）。
    """
    coder_tool = PythonCoderTool(coder_chat_fn)
    base_generator_tool = BaseGeneratorTool(fixed_chat_fn)
    toolbox = {t.tool_name: t for t in (coder_tool, base_generator_tool)}
    available_tools = sorted(toolbox)
    toolbox_metadata = {name: {"description": t.tool_description} for name, t in toolbox.items()}

    planner = Planner(planner_chat_fn, available_tools, toolbox_metadata)
    executor = Executor(fixed_chat_fn, toolbox)
    verifier = Verifier(fixed_chat_fn, available_tools, toolbox_metadata)
    return Solver(planner, executor, verifier, final_output_chat_fn=fixed_chat_fn, max_steps=args.af_max_steps)


async def generate(args: Args, sample: Sample) -> Sample:
    """薄适配器：造三模型 chat_fn → 组装 Solver → solve（对齐源 rollout.py:104）。"""
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["original_question"] = sample.prompt  # reward 用原问题（solver 不覆盖 prompt）

    planner_chat_fn = _sglang_chat_fn(args.af_planner_base_url, args.af_planner_model, args)
    fixed_chat_fn = _sglang_chat_fn(args.af_fixed_base_url, args.af_fixed_model, args)
    coder_chat_fn = deepseek_coder_chat_fn()   # 独立 coder 模型 = 外部 DeepSeek API
    solver = _build_solver(args, planner_chat_fn, fixed_chat_fn, coder_chat_fn)
    return await solver.solve(args, sample)


async def reward_func(args: Args, sample: Sample) -> dict:
    """boxed 精确匹配 → miss 回退固定引擎判官（对齐源 rollout.py:209/237-244）。

    返回 {"reward": score, ...}（nano reward dict 约定，RolloutManager 读 ["reward"]；源返回 {"score"}）。
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = metadata.get("original_question", "") or sample.prompt
    final_output = metadata.get("final_output", "") or sample.response or ""
    label = str(sample.label) if sample.label is not None else ""

    boxed = _last_boxed_only_string(final_output)
    pred = _remove_boxed(boxed).strip() if boxed is not None else ""

    if pred and label and _is_equiv(pred, label):
        score = 1.0
    else:
        # 回退 LLM-as-judge：固定引擎判官（对齐源——用固定引擎、不用训练 planner，reward 稳定）
        fixed_chat_fn = _sglang_chat_fn(args.af_fixed_base_url, args.af_fixed_model, args)
        rewarder = Rewarder(fixed_chat_fn)
        score = await rewarder.compute_reward(question=question, model_response=final_output, groundtruth=label)

    return {"reward": score, "pred": pred}
