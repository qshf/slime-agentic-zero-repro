"""V4: Ray 同步主循环 —— rollout / train 拆进程，`ray.get` 标注同步点。

对齐源 train.py:9-95：装配三步 + 同步主循环。**这一版仍是串行的**——`ray.get(generate.remote())`
和 `ray.get(async_train())` 严格顺序执行，rollout N+1 必须等 train N 完全结束。V4 不解决串行，
只把"角色分进程"立起来；`ray.get` 强制等待正是 V5 要 overlap 的靶子。

与 V3 [train.py](train.py) 的差异只有一处本质：V3 是单进程 `await rm.generate(id)`，
V4 是跨进程 `ray.get(rollout_manager.generate.remote(id))`。指标同名，便于 V3↔V4↔V5 对照。
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
    """跑 args.num_rollout 轮 ray 同步闭环，返回每轮指标。对齐源 train.py 装配 + 主循环。"""
    # --- 装配三步（对齐 train.py:9-24）---
    create_placement_groups(args)                       # ray.init（nano 无 GPU bundle）
    rollout_manager = create_rollout_manager(args)      # 单 @ray.remote actor（独立进程）
    actor_model, _critic = create_training_models(args)  # RayTrainGroup（本地包装 + worker actor）
    actor_model.async_init(args)                        # 各 rank 建 Trainer

    # 训练前先同步一次权重（对齐 train.py:26 "always update weight first so sglang has loaded weights"）。
    # fake 语义：把训练侧初始权重版本推给推理引擎，让引擎从训练模型的权重起步。
    actor_model.update_weights()

    metrics_log: list[dict] = []
    traces: list = []  # V7.4：opt-in learner trace（分相位计时）
    for rollout_id in range(args.num_rollout):
        step_t0 = time.perf_counter()
        # 1) 产数据：另一进程的 RolloutManager actor（对齐 rollout_data_ref = ray.get(...generate.remote)）
        #    V7.4（缺口 1）：把当时的权重版本传给 generate，戳到每条 Sample（None-safe，默认路径不启用校验）。
        print(f"[rollout {rollout_id}] generating...", flush=True)
        t0 = time.time()
        cur_version = actor_model.weight_version()
        rollout_data = ray.get(rollout_manager.generate.remote(rollout_id, cur_version))
        t_gen = time.time() - t0

        # V7.4 learner_contract（opt-in）：把 train_data 过一遍 LearnerSample 校验（不改数据，失败即抛）。
        if getattr(args, "learner_contract_validate", False):
            from mini_slime.learner_contract import validate_train_data

            validate_train_data(rollout_data)

        # 2) 训：fan-out 到 worker actor 们，ray.get 是同步点（对齐 ray.get(actor_model.async_train(...))）
        token_lengths = [len(tokens) for tokens in rollout_data["tokens"]]
        response_lengths = rollout_data["response_lengths"]
        print(
            f"[rollout {rollout_id}] samples={len(token_lengths)} "
            f"token_lengths={token_lengths} response_lengths={response_lengths}",
            flush=True,
        )
        print(f"[rollout {rollout_id}] training...", flush=True)
        t0 = time.time()
        train_metrics = ray.get(actor_model.async_train(rollout_id, rollout_data))
        m = train_metrics[0]  # rank0 的指标；Megatron DP 时其 loss 已在 trainer 内跨 DP 平均。
        t_train = time.time() - t0
        # Staleness is defined at learner consumption, before this round
        # publishes a new rollout-engine version. Synchronous rollout/train is
        # therefore gap=0; using the post-publish version would falsely show 1.
        trainer_policy_version = actor_model.weight_version()

        # 3) 同步权重回推理引擎（对齐 actor_model.update_weights()）
        print(f"[rollout {rollout_id}] updating weights...", flush=True)
        t0 = time.time()
        weight_published = False
        if (rollout_id + 1) % args.update_weights_interval == 0 and m["trainable_tokens"] > 0:
            actor_model.update_weights()
            weight_published = True
        t_sync = time.time() - t0
        e2e_time = time.perf_counter() - step_t0

        metrics = {
            "rollout_id": rollout_id,
            "gen_time": t_gen,
            "train_time": t_train,
            "sync_time": t_sync,
            "e2e_time": e2e_time,
            "reward_mean": m["reward_mean"],
            "raw_reward_mean": m.get("raw_reward_mean", m["reward_mean"]),
            "trainable_tokens": m.get("trainable_tokens", 0),
            "model_tokens": m.get("_trace", {}).get("model_tokens", sum(len(t) for t in rollout_data["tokens"])),
            "grpo_group_count": m.get("grpo_group_count", 0),
            "grpo_active_group_count": m.get("grpo_active_group_count", 0),
            "trained_samples": m.get("trained_samples", 0),
            "weight_published": weight_published,
            "tokens_per_rollout": sum(rollout_data["response_lengths"]),
            "weight_version": actor_model.weight_version(),
        }
        if "loss" in m:  # V6.3 torch 后端：透传真训练 loss（fake 后端无此键）
            metrics["loss"] = m["loss"]
        if "tflops" in m:
            metrics["tflops"] = m["tflops"]
        # V7.4 learner_trace（opt-in）：用 trainer 回传的可测相位 + 主循环的 publish 计时组装 LearnerTrace。
        if getattr(args, "learner_trace", False) and m.get("_trace") is not None:
            from mini_slime.learner_metrics import LearnerTrace

            tr = m["_trace"]
            rollout_pv = tr["rollout_policy_version"]
            if rollout_pv is not None:  # 未戳版本则跳过 trace（int 契约要求非空）
                trace = LearnerTrace(
                    rollout_id=rollout_id,
                    rollout_policy_version=rollout_pv,
                    trainer_policy_version=trainer_policy_version,
                    samples=m.get("trained_samples", 0),
                    model_tokens=tr["model_tokens"],
                    trainable_tokens=m["trainable_tokens"],
                    batch_preparation_seconds=tr["batch_preparation_seconds"],
                    log_prob_seconds=0.0,          # repo 融合 log-prob 进 forward（偏离，见 learner_metrics）
                    forward_backward_seconds=tr["forward_backward_seconds"],
                    optimizer_seconds=0.0,         # 未与 fwd/bwd 拆分（偏离）
                    weight_publish_seconds=t_sync,
                )
                traces.append(trace)
                metrics["policy_version_gap"] = trace.to_dict()["policy_version_gap"]
        metrics_log.append(metrics)
        print(
            f"[rollout {rollout_id}] "
            f"gen={t_gen:.3f}s train={t_train:.4f}s sync={t_sync:.4f}s | "
            f"reward_mean={metrics['reward_mean']:.3f} "
            f"tokens={metrics['tokens_per_rollout']} "
            f"weight_v={metrics['weight_version']}"
        )

    if getattr(args, "learner_trace", False) and traces:
        from mini_slime.learner_metrics import summarize_learner_traces

        print(f"[learner_trace] {summarize_learner_traces(traces)}")

    return metrics_log


def main() -> None:
    print("=== V4: Ray 化最小闭环 (actor 拆进程 + ray.get 同步点) ===\n")
    args = Args()
    metrics_log = train(args)
    print(f"\n完成 {len(metrics_log)} 轮闭环，最终权重版本 = {metrics_log[-1]['weight_version']}")
    ray.shutdown()


if __name__ == "__main__":
    main()
