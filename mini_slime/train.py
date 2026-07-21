"""V3: mini_slime 同步主循环 —— 把三个角色串成 rollout→train→update_weights 闭环。

对齐源项目 train.py:65-93 的同步主循环（去掉 critic/offload/eval/save 噪声后的骨架）：
```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))  # 产数据
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))           # 训
    actor_model.update_weights()                                             # 同步权重回引擎
```

V3 偏离（详见 docs/decisions/v3.md 偏离表）：
  - **无 Ray**：角色是普通类，主循环**串行直调**（不用 .remote()/ray.get）。编排调用序列
    与源等价；Ray 化是 V4 的承诺。这版故意全串行——它暴露的痛点（generate 时 trainer 空转、
    train 时引擎空转）正是 V4/V5 的存在理由。

每轮采集的指标（gen_time / train_time / sync_time / reward_mean / tokens_per_rollout）是
V4/V5 的**对照基线**：V4 Ray 化后看 train 是否与 gen overlap，V5 异步看总耗时是否下降。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_slime.args import Args
from mini_slime.rollout_manager import RolloutManager
from mini_slime.trainer import Trainer


async def train(args: Args) -> list[dict]:
    """跑 args.num_rollout 轮同步闭环，返回每轮指标列表。对齐 train.py:65-93 骨架。"""
    rm = RolloutManager(args)
    trainer = Trainer(args)
    metrics_log: list[dict] = []

    for rollout_id in range(args.num_rollout):
        # 1) 产数据（源: ray.get(rollout_manager.generate.remote(rollout_id))）
        t0 = time.time()
        rollout_data = await rm.generate(rollout_id)
        t_gen = time.time() - t0

        # 2) 训（源: ray.get(actor_model.async_train(rollout_id, rollout_data_ref))）
        t0 = time.time()
        m = trainer.train(rollout_id, rollout_data)
        t_train = time.time() - t0

        # 3) 同步权重回推理引擎（源: actor_model.update_weights()）
        #    V3 每轮同步一次（update_weights_interval=1）；字段留给 V5 异步按 interval 同步。
        t0 = time.time()
        if (rollout_id + 1) % args.update_weights_interval == 0:
            trainer.update_weights()
        t_sync = time.time() - t0

        metrics = {
            "rollout_id": rollout_id,
            "gen_time": t_gen,
            "train_time": t_train,
            "sync_time": t_sync,
            "reward_mean": m["reward_mean"],
            "tokens_per_rollout": sum(rollout_data["response_lengths"]),
            "weight_version": trainer.weight_version,
        }
        metrics_log.append(metrics)
        print(
            f"[rollout {rollout_id}] "
            f"gen={t_gen:.3f}s train={t_train:.4f}s sync={t_sync:.4f}s | "
            f"reward_mean={metrics['reward_mean']:.3f} "
            f"tokens={metrics['tokens_per_rollout']} "
            f"weight_v={metrics['weight_version']}"
        )

    return metrics_log


def main() -> None:
    print("=== V3: mini_slime 最小闭环 (rollout→train→update_weights) ===\n")
    args = Args()
    metrics_log = asyncio.run(train(args))
    print(f"\n完成 {len(metrics_log)} 轮闭环，最终权重版本 = {metrics_log[-1]['weight_version']}")


if __name__ == "__main__":
    main()
