"""V4 离线测试用：no-SGLang 的 stub generate hook。

为什么需要它：V4 的 rollout 跑在 **RolloutManager actor 的独立进程**里，主进程 monkeypatch
`_chat` 打不到那个进程。所以离线跑 ray 装配/闭环时，改用"换 hook 路径"——把
`args.custom_generate_function_path` 指到这里，actor 进程里 load_function 加载的就是这个不连
SGLang 的 stub。这恰好演示了 V2 hook 可插拔的价值（换 agent/rollout 只改配置路径）。

对齐源项目：源有 `debug_rollout` 等不跑真引擎的模式；stub hook 是同类角色的 nano 版。
签名与 calculator_hooks 完全一致（`generate(args, sample)->Sample` / `reward_func(args, sample)->dict`）。
reward 直接复用 calculator 的（规则匹配），只有 generate 换成"不连 SGLang、直接算答案"。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from toy_rl.agent.calculator_hooks import reward_func  # noqa: F401 — 复用规则打分，re-export 供 hook 加载
from toy_rl.agent.tools import CalculatorTool
from toy_rl.sample import Sample

_CALC = CalculatorTool()


async def generate(args, sample: Sample) -> Sample:
    """不连 SGLang 的 stub rollout：直接算出答案、程序化构造一段假 trajectory。

    trajectory = prompt 段(loss_mask=0) + agent 答案段(loss_mask=1)，与 V1/V2 的段语义一致，
    只是把"多轮 SGLang 交互"替换成"一步算出"。tokens 用字符级假 token（够验证契约/闭环）。

    V5 加：`fake_gen_seconds` sleep **代表**真 SGLang 多轮推理的 wall-clock（离线无引擎故无天然
    耗时）——它给 train 提供一个可被 overlap 藏起来的时间窗。服务器路径走真 SGLang（此值恒为 0，
    不 sleep）。默认 0.0 时 no-op，V4 离线测试行为不变。偏离登记见 docs/decisions/v5.md。
    """
    if getattr(args, "fake_gen_seconds", 0) > 0:
        await asyncio.sleep(args.fake_gen_seconds)

    expr = sample.prompt.replace("=", "").replace("?", "").strip()  # "2 + 3 = ?" -> "2 + 3"
    ans = _CALC.execute(expr)
    resp = f"<answer>{ans}</answer>"

    vocab: dict[str, int] = {}
    def tok(s: str) -> list[int]:
        return [vocab.setdefault(c, len(vocab)) for c in s]

    ptoks, rtoks = tok(sample.prompt), tok(resp)
    sample.response = resp
    sample.tokens = ptoks + rtoks
    sample.loss_mask = [0] * len(ptoks) + [1] * len(rtoks)  # prompt 不训练 / agent 答案训练
    sample.metadata = {"final_output": ans}
    return sample
