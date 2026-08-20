"""V5: Ray 异步主循环 —— 提前发起 rollout N+1，overlap 掉 train N 的耗时。

对齐源 train_async.py:31/39/62 —— 与 V4 [train_ray.py](train_ray.py)（≡ 源同步 train.py）的**唯一**
本质差异是"改 `ray.get` 的位置"（源注释原话：*"one can change the position of the sync operation"*）：

  V4 同步:  ray.get(generate.remote(N)); ray.get(async_train(N))   # 严格串行，gen N+1 等 train N 完
  V5 异步:  先 generate.remote(N+1) 发起下一轮，再 ray.get(async_train(N))  # train N 与 gen N+1 并行

关键前提（已核对）：任意时刻 rollout actor 上**最多一个** generate 在途——gen(N+1) 只在 gen(N) 被
`ray.get` 取走后才发起。故 overlap 是 gen(N+1)@rollout进程 与 train(N)@trainer进程 的**跨进程真并行**。

**为何 nano 能观测到 overlap**（偏离登记见 docs/decisions/v5.md）：源 train 是真 FSDP/Megatron 一步
（秒级），overlap 省真 wall-clock；nano train 是 fake（≈0），故靠 `args.fake_train_seconds` sleep
**代表**这段计算耗时——这正是异步能藏进 gen 时间窗里的那段时间。离线再用 `args.fake_gen_seconds`
给 gen 制造时间窗（服务器上由真 SGLang 提供）。

同步基线复用 train_ray.py（不另建 train_sync.py）：源同步文件就叫 train.py，nano 的 train_ray.py 已
扮演该角色，传入带 fake_train_seconds 的同一份 Args 即成公平对照。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime.args import Args
from mini_slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)


def train(args: Args) -> list[dict]:
    """跑 args.num_rollout 轮 ray **异步** 闭环，返回每轮指标。对齐源 train_async.py 装配 + 异步主循环。"""
    # --- 装配三步（与 V4 完全一致，复用同一批工厂）---
    create_placement_groups(args)
    rollout_manager = create_rollout_manager(args)
    actor_model, _critic = create_training_models(args)
    actor_model.async_init(args)

    # 训练前先同步一次权重（对齐 train_async.py:29；同 V4，故 weight_version 起点一致）。
    actor_model.update_weights()

    metrics_log: list[dict] = []
    traces: list = []

    # 预取 rollout 0（对齐 train_async.py:31 rollout_data_next_future = generate.remote(start_rollout_id)）。
    # 这一步就是异步的起手：主循环还没进，下一轮 rollout 已经在 rollout actor 进程里跑起来了。
    rollout_data_next_future = rollout_manager.generate.remote(0, actor_model.weight_version())

    for rollout_id in range(args.num_rollout):
        # 1) sync 上一次已发起的 generation（对齐 train_async.py:34-36）
        t0 = time.time()
        if rollout_data_next_future is not None:
            rollout_data_curr = ray.get(rollout_data_next_future)
        t_wait_gen = time.time() - t0

        # 2) **提前发起下一轮 rollout**（对齐 train_async.py:38-40）——overlap 的核心。
        #    先把 gen(N+1) 丢到 rollout actor 进程跑起来，下面 train(N) 就能与它并行。
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(
                rollout_id + 1, actor_model.weight_version()
            )

        if getattr(args, "learner_contract_validate", False):
            from mini_slime.learner_contract import validate_train_data

            validate_train_data(rollout_data_curr)

        # 3) 训 train(N)：此刻 gen(N+1) 正在另一个进程跑（对齐 train_async.py:42-47 无 critic 分支）
        t0 = time.time()
        train_metrics = ray.get(actor_model.async_train(rollout_id, rollout_data_curr))
        m = train_metrics[0]  # rank0 的指标；Megatron DP 时其 loss 已在 trainer 内跨 DP 平均。
        t_train = time.time() - t0
        tokens = sum(rollout_data_curr["response_lengths"])  # 在 update 块把 curr 改掉之前先取
        model_tokens = sum(len(tokens) for tokens in rollout_data_curr["tokens"])

        # 4) 按 interval 同步权重（对齐 train_async.py:62-66）。换权重前先 sync 掉在途 generation，
        #    防止"在生成中途换权重"（源注释：sync generate before update weights）。
        t_sync = 0.0
        if (rollout_id + 1) % args.update_weights_interval == 0:
            t0 = time.time()
            rollout_data_curr = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()
            t_sync = time.time() - t0

        metrics = {
            "rollout_id": rollout_id,
            "wait_gen_time": t_wait_gen,   # 等当前 gen 的 ray.get 耗时（异步下多半已被上一轮 train 藏掉）
            "train_time": t_train,          # train(N) 的 ray.get 耗时（≈ fake_train_seconds + IPC）
            "sync_time": t_sync,
            "reward_mean": m["reward_mean"],
            "raw_reward_mean": m.get("raw_reward_mean", m["reward_mean"]),
            "trainable_tokens": m.get("trainable_tokens", 0),
            "model_tokens": m.get("_trace", {}).get("model_tokens", model_tokens),
            "grpo_group_count": m.get("grpo_group_count", 0),
            "grpo_active_group_count": m.get("grpo_active_group_count", 0),
            "trained_samples": m.get("trained_samples", 0),
            "tokens_per_rollout": tokens,
            "weight_version": actor_model.weight_version(),
        }
        if "loss" in m:
            metrics["loss"] = m["loss"]
        if "tflops" in m:
            metrics["tflops"] = m["tflops"]
        if getattr(args, "learner_trace", False) and m.get("_trace") is not None:
            from mini_slime.learner_metrics import LearnerTrace

            tr = m["_trace"]
            rollout_pv = tr["rollout_policy_version"]
            if rollout_pv is not None:
                trace = LearnerTrace(
                    rollout_id=rollout_id,
                    rollout_policy_version=rollout_pv,
                    trainer_policy_version=metrics["weight_version"],
                    samples=m.get("trained_samples", 0),
                    model_tokens=tr["model_tokens"],
                    trainable_tokens=m["trainable_tokens"],
                    batch_preparation_seconds=tr["batch_preparation_seconds"],
                    log_prob_seconds=0.0,
                    forward_backward_seconds=tr["forward_backward_seconds"],
                    optimizer_seconds=0.0,
                    weight_publish_seconds=t_sync,
                )
                traces.append(trace)
                metrics["policy_version_gap"] = trace.to_dict()["policy_version_gap"]
        metrics_log.append(metrics)
        print(
            f"[rollout {rollout_id}] "
            f"wait_gen={t_wait_gen:.3f}s train={t_train:.3f}s | "
            f"reward_mean={metrics['reward_mean']:.3f} "
            f"tokens={metrics['tokens_per_rollout']} "
            f"weight_v={metrics['weight_version']}"
        )

    if getattr(args, "learner_trace", False) and traces:
        from mini_slime.learner_metrics import summarize_learner_traces

        print(f"[learner_trace] {summarize_learner_traces(traces)}")
    return metrics_log


def main() -> None:
    print("=== V5: Ray 异步闭环 (提前发起 rollout N+1 overlap train N) ===\n")
    # 打开 fake_train_seconds 让 overlap 可观测（服务器真 SGLang 提供 gen 时间窗，fake_gen 保持 0）。
    args = Args(num_rollout=3, fake_train_seconds=5.0)
    metrics_log = train(args)
    print(f"\n完成 {len(metrics_log)} 轮闭环，最终权重版本 = {metrics_log[-1]['weight_version']}")
    ray.shutdown()


if __name__ == "__main__":
    main()
