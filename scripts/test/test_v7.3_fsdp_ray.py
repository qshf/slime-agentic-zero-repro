"""V7.3 验证：Ray + FSDP + 权重同步（把 FSDPTrainer 搬进 Ray actor 闭合训练闭环）。

  python scripts/test_v7.3_fsdp_ray.py              # 单卡：world_size=1 编排闭环
  python scripts/test_v7.3_fsdp_ray.py --world 2    # 双卡：DP-split + 跨 rank 权重同步（收官）

前置（服务器 5090）：
  - 有 SGLang 未占用的空闲卡；driver 侧 CUDA_VISIBLE_DEVICES 限定到空闲卡（否则 Ray 可能把
    FSDP actor 调度到 SGLang 已占的卡 → OOM）。例：CUDA_VISIBLE_DEVICES=2,3 跑 --world 2。
  - SGLang 0.6B 服务在 30000（与 train_model_path 一致，rollout 采样==训练重算 log_prob 的模型）。

契约（对齐 v7-onward-plan.md V7.3 + 源 train_actor.py + actor.py:744-746）：
  ✅ Ray actor 起在独立进程、形成真 torch.distributed 进程组（world_size 个 rank）
  ✅ FSDPTrainer 在 Ray actor 内初始化成功（真分片）
  ✅ 闭环通过：rollout → train → update_weights（weight_version 每轮递增）
  ✅ 至少一轮真 loss（真 backward 发生）
  ✅ rollout engine disk reload 成功（rank0 落盘 + POST 一次）
  ✅ reward_mean ∈ [0,1]

注意（DP 均衡）：world_size 须整除全局样本数（batch_size*n_samples_per_prompt），
否则某 rank 分不到样本、跳过 FSDP 反向的集体 reduce → 与其他 rank 死锁。
默认 batch_size=1, n_samples_per_prompt=4 → 4 样本，world 1/2/4 均整除。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime import train_ray
from mini_slime.args import Args


def _server_args(world_size: int) -> Args:
    """服务器真训练配置：FSDP 分布式后端 + 真 log_probs + GRPO + disk reload 权重同步。

    与 V6 _server_args 一致的关键一致性：rollout 采样模型 == 训练重算 log_prob 的模型（0.6B@30000）。
    差别只在 train_backend="fsdp" + fsdp_world_size（V6 是单卡 "torch"）。
    """
    return Args(
        num_rollout=3,
        batch_size=1,
        n_samples_per_prompt=4,   # 4 样本/轮，world 1/2/4 均整除（DP 均衡，见模块 docstring）
        gsm8k_num_train=4,
        gsm8k_num_eval=10,
        orchestra_orchestrator_base_url="http://localhost:30000/v1",
        orchestra_orchestrator_model="Qwen/Qwen3-0.6B",
        orchestra_expert_base_url="http://localhost:30000/v1",
        orchestra_expert_model="Qwen/Qwen3-0.6B",
        sglang_generate_url="http://localhost:30000/generate",
        train_model_path=os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B"),
        data_source_path="toy_rl.agent.toolorchestra.gsm8k_data.load_data_source",
        custom_generate_function_path="toy_rl.agent.toolorchestra.rollout.generate",
        custom_rm_path="toy_rl.agent.toolorchestra.rollout.reward_func",
        custom_convert_path="mini_slime.custom_convert.custom_convert",
        train_backend="fsdp",
        fsdp_world_size=world_size,
    )


def test_fsdp_ray_closed_loop(args: Args) -> None:
    """V7.3 核心断言：Ray+FSDP 训练闭环全链路发生（真分布式 + 真权重同步 disk reload）。

    机制断言（硬）：weight_version 每轮递增（真同步）、至少一轮真 loss（真 backward）、reward 合法。
    """
    metrics = train_ray.train(args)  # 真 FSDP 训练 num_rollout 轮 + 每轮权重同步回 SGLang

    assert len(metrics) == args.num_rollout, f"应跑 {args.num_rollout} 轮，得到 {len(metrics)}"
    # weight_version：训练前初始同步(=1) + 每轮同步一次 → 末轮 >= num_rollout（对齐 V4 语义）。
    assert metrics[-1]["weight_version"] >= len(metrics), "每轮应真同步一次权重（version 递增）"
    trained = any(m.get("loss") is not None for m in metrics)
    assert trained, "至少一轮应真算 loss（FSDP 真 backward 发生）"
    for m in metrics:
        assert 0.0 <= m["reward_mean"] <= 1.0, f"reward_mean 须 ∈[0,1]，得到 {m['reward_mean']}"
        assert m["tokens_per_rollout"] > 0, "每轮应产出可训练 token"

    print(
        f"  fsdp_ray_closed_loop OK "
        f"(world={args.fsdp_world_size}, rounds={len(metrics)}, "
        f"weight_v={metrics[-1]['weight_version']}, "
        f"loss={[round(m.get('loss', 0.0), 4) for m in metrics]})"
    )


def main() -> int:
    world_size = 1
    for i, a in enumerate(sys.argv):
        if a == "--world" and i + 1 < len(sys.argv):
            world_size = int(sys.argv[i + 1])

    failed = 0
    try:
        test_fsdp_ray_closed_loop(_server_args(world_size))
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"  FAIL test_fsdp_ray_closed_loop: {type(exc).__name__}: {exc}")
    ray.shutdown()
    print(f"[V7.3] {'PASSED' if failed == 0 else f'FAILED ({failed})'}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
