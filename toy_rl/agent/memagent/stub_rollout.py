"""A1 离线测试用：no-SGLang 的 MemAgent stub generate。

同 V4/V5 的 stub_hooks —— rollout 在 RolloutManager actor 的独立进程里，主进程 monkeypatch 不到，
故离线跑闭环时把 `custom_generate_function_path` 指到这里（不连 SGLang）。它复用 rollout.py 里
**同一个** `_run_memory_loop`（循环只有一份），只注入一个"造假答案"的 chat_fn：
  - 记忆更新轮：返回一句非空假记忆（让 memory 逐轮更新）；
  - 最终回答轮：返回 \\boxed{label}（让 boxed reward 命中，验证闭环打分）。

这样离线就能验证 MemAgent 的核心不变量（多轮结构 + loss_mask 全 1 + boxed reward），不需要 GPU。
reward_func 直接复用 rollout.py 的（re-export 供 hook 加载）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mini_slime.args import Args
from toy_rl.agent.memagent.rollout import _run_memory_loop, reward_func  # noqa: F401  (reward re-export)
from toy_rl.sample import Sample


async def generate(args: Args, sample: Sample) -> Sample:
    """stub：注入造假 chat_fn 后调共享的 _run_memory_loop（不连 SGLang）。"""
    label = str(sample.label) if sample.label is not None else ""

    async def fake_chat(prompt_text: str, is_final: bool) -> str:
        if is_final:
            return f"Based on the memory, the answer is \\boxed{{{label}}}."
        return "Noted relevant facts from this section."

    return await _run_memory_loop(args, sample, fake_chat)
