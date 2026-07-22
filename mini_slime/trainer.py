"""V3: Trainer —— "谁负责训" 的角色（fake，不算真梯度）。

对齐源项目 actor（slime/backends/fsdp_utils/actor.py）的**角色与签名**：
  - train(rollout_id, rollout_data) : actor.py:437，消费 RolloutManager 产的 train_data dict
  - update_weights()                : actor.py:725，训练后把权重同步回推理引擎

偏离说明（详见 docs/decisions/v3.md 偏离表）：
  - **fake train（不算真梯度）**：主线一聚焦系统骨架/编排，非训练数学。train() 不跑
    forward/backward，只**消费 train_data 的结构** + 产出可断言的指标（reward_mean、
    trainable_tokens）。真 FSDP/Megatron 训练一步留主线三 V6-V9。角色/签名/数据契约等价。
  - update_weights 委托给独立的 WeightUpdater（见 weight_sync.py），对齐源 actor 只是调用方。
"""

from __future__ import annotations

import time

from mini_slime.args import Args
from mini_slime.weight_sync import WeightUpdater


class Trainer:
    """训练器角色。对齐源 actor（train / update_weights 签名），V3 内部打桩不算真梯度。"""

    def __init__(self, args: Args) -> None:
        self.args = args
        # 权重同步委托给独立角色（对齐源 actor 持有权重同步机制）
        self.weight_updater = WeightUpdater()

    @property
    def weight_version(self) -> int:
        """当前推理引擎上的权重版本号（由 WeightUpdater 维护）。主循环/测试据此断言同步次数。"""
        return self.weight_updater.version

    def train(self, rollout_id: int, rollout_data: dict) -> dict:
        """对齐源 actor.train(rollout_id, rollout_data)：消费 train_data dict。

        V3 fake：不 forward/backward，只消费数据结构、产出可断言指标。
        V5 加：`fake_train_seconds` sleep **代表**真 FSDP/Megatron 一步的计算 wall-clock（fake 不算
        真梯度，故无天然耗时）——这是 V5 异步能 overlap 掉的那段时间。默认 0.0 时此行 no-op，V3/V4
        行为不变。偏离登记见 docs/decisions/v5.md。
        """
        if self.args.fake_train_seconds > 0:
            time.sleep(self.args.fake_train_seconds)
        rewards = rollout_data["rewards"]
        n_trainable = sum(sum(m) for m in rollout_data["loss_masks"])
        n_total = sum(rollout_data["response_lengths"])
        return {
            "rollout_id": rollout_id,
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "trainable_tokens": n_trainable,
            "total_tokens": n_total,
        }

    def update_weights(self) -> None:
        """对齐源 actor.update_weights()：训练后把权重同步回推理引擎（委托 WeightUpdater）。"""
        self.weight_updater.update_weights()
