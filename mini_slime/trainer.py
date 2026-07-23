"""V3/V6: Trainer —— "谁负责训" 的角色。

对齐源项目 actor（slime/backends/fsdp_utils/actor.py）的**角色与签名**：
  - train(rollout_id, rollout_data) : actor.py:437，消费 RolloutManager 产的 train_data dict
  - update_weights()                : actor.py:725，训练后把权重同步回推理引擎

两个后端（args.train_backend）：
  - "fake"（V0-A3/离线，默认）：不 forward/backward，只消费 train_data 结构 + 产可断言指标。
  - "torch"（V6.3）：真 forward/backward/optimizer 一步，loss 公式对齐源 ppo_utils.compute_policy_loss
    + fsdp_utils/actor.py 的 sum_of_sample_mean。单卡纯 torch（不 FSDP/offload，见 v6.md 偏离表）。

偏离说明（docs/decisions/{v3,v6}.md 偏离表）：
  - fake 后端不算真梯度（主线一聚焦系统骨架）；V6.3 torch 后端补真训练一步。
  - torch 后端单卡、无 FSDP 多维并行/CPU offload、关 KL/entropy（nano 取最简，语义是纯 GRPO policy gradient）。
"""

from __future__ import annotations

import time

from mini_slime.args import Args
from mini_slime.weight_sync import WeightUpdater


class Trainer:
    """训练器角色。对齐源 actor（train / update_weights 签名）。"""

    def __init__(self, args: Args) -> None:
        self.args = args
        self._torch_actor = None
        if args.train_backend == "torch":
            # 延迟导入：torch/transformers 只在服务器 V6.3 装（optional [train]）。
            from mini_slime.torch_actor import TorchActor

            self._torch_actor = TorchActor(args)
        # 权重同步委托给独立角色（对齐源 actor 持有权重同步机制）。
        # torch 后端把可训模型交给 WeightUpdater 落盘供 SGLang reload。
        self.weight_updater = WeightUpdater(
            save_path=args.weight_save_path if args.train_backend == "torch" else None,
            torch_actor=self._torch_actor,
        )
        if args.train_backend == "torch":
            # 训练后落盘的权重由 SGLang 从 disk 热重载（disk reload 最小路径，见 weight_sync 注释）。
            self.weight_updater.generate_url = args.sglang_generate_url

    @property
    def weight_version(self) -> int:
        """当前推理引擎上的权重版本号（由 WeightUpdater 维护）。"""
        return self.weight_updater.version

    def train(self, rollout_id: int, rollout_data: dict) -> dict:
        """对齐源 actor.train(rollout_id, rollout_data)：消费 train_data dict。

        fake：不 forward/backward，只消费数据结构、产出可断言指标（+ V5 sleep 旋钮模拟耗时）。
        torch：真 PPO policy-gradient 一步，返回真 loss。
        """
        rewards = rollout_data["rewards"]
        n_trainable = sum(sum(m) for m in rollout_data["loss_masks"])
        n_total = sum(rollout_data["response_lengths"])
        metrics = {
            "rollout_id": rollout_id,
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "trainable_tokens": n_trainable,
            "total_tokens": n_total,
        }

        if self._torch_actor is not None:
            metrics.update(self._torch_actor.train_step(rollout_data))
            return metrics

        if self.args.fake_train_seconds > 0:
            time.sleep(self.args.fake_train_seconds)
        return metrics

    def update_weights(self) -> None:
        """对齐源 actor.update_weights()：训练后把权重同步回推理引擎（委托 WeightUpdater）。"""
        self.weight_updater.update_weights()
