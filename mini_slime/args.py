"""V3: mini_slime 框架的统一配置对象（single source of truth）。

对齐源项目 —— slime 的 args 是一个巨大的 argparse Namespace，从命令行构造后**一路透传**
给 RolloutManager / actor / 各 hook（generate/reward）。框架和 agent 都只**读** args 字段，
不各自 new 一份配置。

V2 时 Args 临时定义在 toy_rl/agent/calculator_hooks.py（agent 侧），这在角色上是反的：
配置的归属应是**框架**，agent 只是被透传进来读它。V3 把 Args 收归 mini_slime（框架层），
calculator_hooks 改为 re-export，`from toy_rl.agent.calculator_hooks import Args` 仍可用
（V2 测试零改）。这样只有一个 Args，避免"扩 loop 字段要改两处"。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Args:
    """极简配置对象，对齐源项目 args 的角色（庞大 Namespace 里只保留 nano 用到的字段）。"""

    # --- rollout / SGLang 推理配置（generate hook 读）---
    sglang_base_url: str = "http://localhost:30000/v1"
    model_name: str = "Qwen/Qwen3-0.6B"
    max_turns: int = 5
    # Qwen3-0.6B 是混合推理模型，每轮先吐一段 <think>。128 装不下"think + 闭合 + 动作"，
    # 会在 think 中途撞 max_tokens 被截断 → 该轮无 tool/answer，白耗一轮。提到 512 让整段
    # 思考+动作在一轮内跑完（512*max_turns=2560 < context-length 4096，不溢出）。
    max_tokens: int = 512
    temperature: float = 0.0

    # --- hook 路径（对齐 --custom-generate-function-path / --custom-rm-path）---
    #     RolloutManager 用 load_function 按这两个路径动态加载 generate / reward。
    custom_generate_function_path: str = "toy_rl.agent.calculator_hooks.generate"
    custom_rm_path: str = "toy_rl.agent.calculator_hooks.reward_func"

    # --- 主循环编排配置（V3 新增，对齐 train.py 的 args.num_rollout / rollout_batch_size）---
    num_rollout: int = 2              # 主循环轮数，对齐 args.num_rollout（train.py:65）
    batch_size: int = 2              # 每轮取几条 prompt，对齐 args.rollout_batch_size
    update_weights_interval: int = 1  # 每几轮同步一次权重；V3 恒为 1，字段先留给 V5 异步
