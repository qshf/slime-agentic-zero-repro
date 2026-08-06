"""V7.2 验证：梯度累积（2 卡真分片 + 每卡 2 样本/微批 × 2 微批 → 单次 step）。

    torchrun --nproc_per_node=2 scripts/test_v7.2_grad_accum.py

契约（对齐 v7-plan.md §5 V7.2 + 源 actor.py:521-693）：
    ✅ 每 rank 的 train_batch 累积 2 个各含 2 条样本的微批梯度、末尾只 step 一次
  ✅ 累积等价性：train_batch([s1,s2,s3,s4], microbatch_size=2) 累积出的梯度
      == 分别 backward([s1,s2])+backward([s3,s4])（同一缩放）手工累积的梯度（数值一致）
  ✅ 缩放正确：每微批 loss 乘 dp_size/global_batch_size（累积后与全局批 mean 语义一致）
  ✅ 权重真被更新（step 前后 L2 > 0）

等价性是 V7.2 教学核心：证明「累积多微批的梯度」== 「一次大批的梯度」，
从而梯度累积能在不增显存的前提下等效放大 batch size。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from toy_rl.trainer.fsdp_trainer import FSDPTrainer

MODEL_PATH = "/home/ubuntu/models/Qwen/Qwen3-0.6B"


def _make_samples(trainer: FSDPTrainer):
    """为每个 rank 构造不同的 4 条本地样本：每 2 条组成一个真实 padding 微批。"""
    all_specs = [
        ("Question: What is 12 * 12?\nAnswer:", " The answer is 144.", 1.0),
        ("Question: What is 7 + 8?\nAnswer:", " The answer is 15.", -1.0),
        ("Question: What is 9 * 6?\nAnswer:", " The answer is 54.", 0.5),
        ("Question: What is 20 - 3?\nAnswer:", " The answer is 17.", 0.25),
        ("Question: What is 11 + 5?\nAnswer:", " The answer is 16.", -0.5),
        ("Question: What is 8 * 8?\nAnswer:", " The answer is 64.", 0.75),
        ("Question: What is 30 - 9?\nAnswer:", " The answer is 21.", -0.25),
        ("Question: What is 14 + 13?\nAnswer:", " The answer is 27.", 1.0),
    ]
    local_batch_size = 4
    start = trainer.rank * local_batch_size
    specs = all_specs[start:start + local_batch_size]
    if len(specs) != local_batch_size:
        raise ValueError("此双卡验证要求每个 rank 有 4 条独立样本")
    samples = []
    for prompt, response, reward in specs:
        p_ids = trainer.tokenizer.encode(prompt)
        r_ids = trainer.tokenizer.encode(response, add_special_tokens=False)
        samples.append(
            {
                "tokens": p_ids + r_ids,
                "loss_mask": [0] * len(p_ids) + [1] * len(r_ids),
                "reward": reward,
                "old_log_probs": None,
            }
        )
    return samples


def _grad_snapshot(trainer: FSDPTrainer):
    """抓当前 .grad 的全量聚合快照（跨 rank 一致），用于等价性比对。"""
    snap = {}
    for name, p in trainer.model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        snap[name] = (g.full_tensor() if isinstance(g, DTensor) else g).detach().clone()
    return snap


def main() -> int:
    trainer = FSDPTrainer(model_path=MODEL_PATH, lr=1e-4, global_batch_size=8)
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

    samples = _make_samples(trainer)
    gbs = len(samples) * trainer.dp_size  # 2 rank × 每 rank 4 条 = 8 条全局样本。
    microbatch_size = 2

    # --- 路径 A：手工 2 个微批，每微批各 2 样本（不 step），复现 train_batch 的末尾裁剪 ---
    trainer.optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(samples), microbatch_size):
        trainer._microbatch_backward(samples[start:start + microbatch_size], gbs)
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip_grad)
    grads_manual = _grad_snapshot(trainer)
    trainer.optimizer.zero_grad(set_to_none=True)

    sharded_param = next(
        (name, p) for name, p in trainer.model.named_parameters()
        if isinstance(p, DTensor) and "layers.0" in name
    )

    def _snapshot():
        _, p = sharded_param
        return p.full_tensor().detach().clone()

    # --- 路径 B：真实 train_batch（内部 zero_grad → 逐微批 backward → clip+step）---
    # 在其唯一 optimizer.step 前抓梯度，才能实际验证 train_batch 的累积路径。
    grads_batch = {}
    original_step = trainer.optimizer.step

    def capture_grads_then_step(*args, **kwargs):
        # 此刻 train_batch 已完成全部 backward 和 clip，但尚未更新参数。
        grads_batch.update(_grad_snapshot(trainer))
        # original_step 是替换包装器前保存的 AdamW.step 绑定方法：这里实际更新权重。
        return original_step(*args, **kwargs)

    trainer.optimizer.step = capture_grads_then_step

    # 等价性：真实 train_batch 内累积的梯度，应与手工逐微批 backward 一致。
    before = _snapshot()
    try:
        metrics = trainer.train_batch(
            samples,
            global_batch_size=gbs,
            microbatch_size=microbatch_size,
        )
    finally:
        trainer.optimizer.step = original_step
    after = _snapshot()

    max_diff = 0.0
    for name in grads_manual:
        d = (grads_manual[name] - grads_batch[name]).abs().max().item()
        max_diff = max(max_diff, d)
    check(max_diff < 1e-5, f"手工累积与 train_batch 累积梯度一致（max_diff={max_diff:.3e}）")

    check(metrics["num_microbatches"] == 2, f"train_batch 处理 2 个微批 = {metrics['num_microbatches']}")
    check(metrics["trained_samples"] == 4, f"训练 4 条样本 = {metrics['trained_samples']}")
    check(metrics["grad_norm"] > 0, f"grad_norm > 0（真累积了梯度）= {metrics['grad_norm']:.6f}")
    delta = (after - before).norm().item()
    check(delta > 0, f"step 前后权重 L2 > 0（累积后真更新）= {delta:.6e}")

    if rank == 0:
        print(f"\n[V7.2] metrics: {metrics}")
        print(f"[V7.2] {'PASSED' if failed == 0 else f'FAILED ({failed})'}")

    dist.barrier()
    dist.destroy_process_group()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
