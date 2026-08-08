"""V7.0 验证：FSDP2 单样本真训练一步（2 卡真分片）。

用 torchrun 起 world_size 个进程，每进程绑一张卡：
  torchrun --nproc_per_node=2 scripts/test_v7.0_fsdp_single.py

契约（对齐 v7-plan.md §5 V7.0）：
  ✅ FSDP model 初始化成功（参数是 DTensor → 真分片；tie 模型不炸）
  ✅ Forward 返回 logits、loss 是标量 tensor
  ✅ backward 不报错、optimizer.step 执行
  ✅ 权重真被更新（step 前后某参数 L2 距离 > 0）
  ✅ 分片验证：DTensor 的 local_shard 元素数 < 全量（world_size>1 时）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from toy_rl.trainer.fsdp_trainer import FSDPTrainer

import os
MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")


def _make_sample(trainer: FSDPTrainer):
    """构造一条最小训练样本：prompt(loss_mask=0) + response(loss_mask=1)。

    用真 tokenizer 编码，token 合法、长度真实。reward=+1（正 advantage → 该样本被"鼓励"）。
    """
    prompt = "Question: What is 12 * 12?\nAnswer:"
    response = " The answer is 144."
    p_ids = trainer.tokenizer.encode(prompt)
    r_ids = trainer.tokenizer.encode(response, add_special_tokens=False)
    tokens = p_ids + r_ids
    loss_mask = [0] * len(p_ids) + [1] * len(r_ids)
    return tokens, loss_mask, 1.0


def main() -> int:
    trainer = FSDPTrainer(
        model_path=MODEL_PATH,
        lr=1e-4,  # 放大 lr 让"权重被改"这一步可观测（V7.0 只验机制非调参）
        global_batch_size=1,
    )
    rank = trainer.rank
    failed = 0

    def check(cond, msg):
        nonlocal failed
        if not cond:
            failed += 1
            if rank == 0:
                print(f"  FAIL: {msg}")
        elif rank == 0:
            print(f"  OK: {msg}")

    # --- 契约 1：FSDP 分片 ---
    # 找一个被 shard 的 decoder 层参数，确认它是 DTensor 且 local shard 比全量小。
    sharded_param = None
    for name, p in trainer.model.named_parameters():
        if isinstance(p, DTensor) and "layers.0" in name:
            sharded_param = (name, p)
            break
    check(sharded_param is not None, "decoder 层参数是 DTensor（FSDP 分片生效）")
    if sharded_param is not None:
        name, p = sharded_param
        full_numel = p.shape.numel()
        local_numel = p.to_local().numel()
        if trainer.world_size > 1:
            check(local_numel < full_numel, f"{name} local shard({local_numel}) < 全量({full_numel})")
        if rank == 0:
            print(f"  [shard] {name}: full={full_numel} local={local_numel} world={trainer.world_size}")

    # --- 契约 2-4：forward + loss + backward + step + 权重变化 ---
    tokens, loss_mask, reward = _make_sample(trainer)

    # 记录 step 前的一个参数快照（用 full_tensor 聚合，跨 rank 一致）。
    def _snapshot():
        _, p = sharded_param
        return p.full_tensor().detach().clone() if isinstance(p, DTensor) else p.detach().clone()

    before = _snapshot()
    metrics = trainer.train_step(tokens, loss_mask, reward, old_log_probs=None)
    after = _snapshot()

    check(metrics["trained_samples"] == 1, "train_step 训练了 1 条样本")
    check(isinstance(metrics["loss"], float), f"loss 是标量 float = {metrics['loss']:.6f}")
    check(metrics["grad_norm"] > 0, f"grad_norm > 0（真算了梯度）= {metrics['grad_norm']:.6f}")
    delta = (after - before).norm().item()
    check(delta > 0, f"step 前后权重 L2 距离 > 0（权重真被更新）= {delta:.6e}")

    if rank == 0:
        print(f"\n[V7.0] metrics: {metrics}")
        print(f"[V7.0] {'PASSED' if failed == 0 else f'FAILED ({failed})'}")

    dist.barrier()
    dist.destroy_process_group()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
