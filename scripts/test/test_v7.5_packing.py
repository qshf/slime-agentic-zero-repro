"""V7.5 验证：sequence packing 与 padding 路径 loss/grad 等价（退休偏离 #1）。

    torchrun --nproc_per_node=1 scripts/test_v7.5_packing.py          # bf16 主门（loss + 方向）
    torchrun --nproc_per_node=1 scripts/test_v7.5_packing.py --fp32   # fp32 副证（数学精确）

契约（对齐 v7.5 计划 + infra spike 04_megatron_te_thd_spike 的验收范式）：
  part (a) 结构：cu_seqlens[-1]==sum(len)、position_ids 每段 reset、块对角 mask（跨段/未来位=blocked、同段过去位=0）。
  part (b) 等价（本版核心）：同一批真样本、合成 old_log_probs（ratio≠1 → loss 也测前向）、同权重，
      分别过 _microbatch_backward（padding）与 _packed_backward（packing），断言：
        |loss_P - loss_K| < tol
        grad 全局相对 L2 = ‖g_P-g_K‖/‖g_P‖，cosine = <g_P,g_K>/(‖g_P‖‖g_K‖)
        trained_P == trained_K

  度量分层（诚实）：
    - **fp32 是数学精确证明**：padding 逐行 [B,L] 前向 vs packing 单条 [1,T] 前向，fp32 下两路 loss/grad
      应逐值一致（rel_L2<5e-3、cosine>0.9999）。真跨样本注意力泄漏会在 fp32 也留 O(1) 痕迹，故 fp32
      过 = 隔离算法正确、无泄漏。这是本版的**硬门**。
    - **bf16 只能匹配到舍入**：28 层前向的两种规约顺序差 ~bf16 epsilon 累积，magnitude rel_L2 可达十几 %
      （**非 bug**，fp32 已证精确）。故 bf16 只 gate loss + cosine 方向一致（>0.98），rel_L2 仅报 info。
  world=1（去 sharding-reduce 噪声，隔离 packing 正确性）。

前置（服务器 5090）：单卡空闲；模型 /home/ubuntu/models/Qwen/Qwen3-0.6B。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "toy_rl").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError(f"Could not locate project root from {_HERE}")

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from toy_rl.trainer.data_packing import build_block_diagonal_causal_mask, pack_sequences
from toy_rl.trainer.fsdp_trainer import FSDPTrainer

import os
MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")


def _make_samples(trainer: FSDPTrainer) -> list[dict]:
    """4 条不同长度的真样本（长度不齐才能暴露 padding 与 packing 的差异）。"""
    specs = [
        ("Question: What is 12 * 12?\nAnswer:", " The answer is 144.", 1.0),
        ("Question: What is 7 + 8?\nAnswer:", " The answer is 15.", -1.0),
        ("Question: What is 9 * 6 minus 3?\nAnswer:", " It is 51 exactly.", 0.5),
        ("Question: What is 20 - 3?\nAnswer:", " 17.", 0.25),
    ]
    samples = []
    for prompt, response, reward in specs:
        p_ids = trainer.tokenizer.encode(prompt)
        r_ids = trainer.tokenizer.encode(response, add_special_tokens=False)
        tokens = p_ids + r_ids
        # 给合成 old_log_probs（非 None）→ ratio=exp(cur_lp-old_lp)≠1 → **loss 值也依赖 logits**
        # （on-policy None 时 ratio≡1、loss 变成常数、只有 grad 测前向；这里让 loss 也真正测前向）。
        # 两路仍必须逐值匹配：old_lp 按段/行切片方式不同，正是要验的对齐点。
        old_lp = [-0.5 * (i % 3) for i in range(len(tokens))]
        samples.append(
            {
                "tokens": tokens,
                "loss_mask": [0] * len(p_ids) + [1] * len(r_ids),
                "reward": reward,
                "old_log_probs": old_lp,
            }
        )
    return samples


def _grad_snapshot(trainer: FSDPTrainer) -> dict:
    snap = {}
    for name, p in trainer.model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        snap[name] = (g.full_tensor() if isinstance(g, DTensor) else g).detach().float().clone()
    return snap


def _test_structure(rank: int) -> int:
    """part (a)：pack_sequences 索引 + 块对角 mask 结构（纯 CPU 逻辑）。"""
    if rank != 0:
        return 0
    failed = 0

    def check(cond, msg):
        nonlocal failed
        if not cond:
            failed += 1
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    samples = [
        {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1], "reward": 1.0, "old_log_probs": None},
        {"tokens": [4, 5], "loss_mask": [0, 1], "reward": -1.0, "old_log_probs": None},
        {"tokens": [6], "loss_mask": [1], "reward": 0.5, "old_log_probs": None},  # 退化(len<2)，应被过滤
    ]
    pack = pack_sequences(samples)
    # len<2 的第 3 条被过滤 → 只剩两段，T=5。
    check(pack["cu_seqlens"] == [0, 3, 5], f"cu_seqlens 边界={pack['cu_seqlens']}")
    check(pack["cu_seqlens"][-1] == len(pack["tokens"]) == 5, "cu_seqlens[-1]==sum(len)==T")
    check(pack["position_ids"] == [0, 1, 2, 0, 1], f"position_ids 每段 reset={pack['position_ids']}")
    check(pack["tokens"] == [1, 2, 3, 4, 5], f"flat tokens={pack['tokens']}")
    check(pack["loss_masks"] == [0, 1, 1, 0, 1], f"flat loss_masks={pack['loss_masks']}")
    check(pack["rewards"] == [1.0, -1.0], f"逐段 rewards={pack['rewards']}")
    check(pack["old_log_probs"] is None, "old_log_probs 缺失时为 None（on-policy）")

    # 块对角 mask：T=5，段 [0:3) 和 [3:5)。
    mask = build_block_diagonal_causal_mask(pack["cu_seqlens"], 5, torch.device("cpu"), torch.float32)
    check(tuple(mask.shape) == (1, 1, 5, 5), f"mask shape={tuple(mask.shape)}")
    m = mask[0, 0]
    BLOCK = torch.finfo(torch.float32).min
    # 段内因果：(2,0),(2,1),(2,2) 可注意（0）；未来 (0,1) blocked；跨段 (3,0)/(0,3) blocked。
    check(m[2, 0].item() == 0.0 and m[2, 2].item() == 0.0, "段内过去/自身位=attend(0)")
    check(m[0, 1].item() == BLOCK, "同段未来位=blocked")
    check(m[3, 0].item() == BLOCK and m[0, 3].item() == BLOCK, "跨段位=blocked（隔离）")
    check(m[4, 3].item() == 0.0, "第二段内过去位=attend(0)")

    # 无可训练样本 → None（与 _packed_backward 空返回对称）。
    check(pack_sequences([{"tokens": [1], "loss_mask": [0], "reward": 0.0, "old_log_probs": None}]) is None,
          "全退化样本 → pack_sequences 返回 None")
    return failed


def _test_equivalence(
    trainer: FSDPTrainer, tol: float, rel_tol: float, cos_min: float, gate_rel_l2: bool, rank: int
) -> int:
    """part (b)：padding vs packing 的 loss/grad 等价（本版核心）。

    loss 用绝对阈值 tol；grad 用全局相对 L2（rel_tol，magnitude-weighted）+ cosine（cos_min，方向）。
    gate_rel_l2=True（fp32）才把 rel_L2 当硬门——bf16 的 rel_L2 是纯舍入（fp32 已证精确），只 gate cosine。
    """
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
    gbs = len(samples) * trainer.dp_size

    # --- 路径 P：padding（_microbatch_backward，逐样本一行）---
    trainer.optimizer.zero_grad(set_to_none=True)
    loss_p, trained_p = trainer._microbatch_backward(samples, gbs)
    grads_p = _grad_snapshot(trainer)
    trainer.optimizer.zero_grad(set_to_none=True)

    # --- 路径 K：packing（_packed_backward，flat 单微批 + 块对角 mask）---
    loss_k, trained_k = trainer._packed_backward(samples, gbs)
    grads_k = _grad_snapshot(trainer)
    trainer.optimizer.zero_grad(set_to_none=True)

    loss_diff = abs(loss_p - loss_k)
    check(loss_diff < tol, f"loss 等价：|{loss_p:.6f} - {loss_k:.6f}| = {loss_diff:.3e} < {tol}")
    check(trained_p == trained_k == len(samples), f"trained_samples 一致={trained_p}/{trained_k}")

    # 梯度等价用**全局相对 L2 范数**（magnitude-weighted）：sqrt(Σ‖g_P-g_K‖²)/sqrt(Σ‖g_P‖²)。
    # 这是比较两个梯度**向量**的标准度量——逐参数 max-relative 会被小信号参数的 bf16 roundoff 放大
    # （如 k_proj 梯度量级小、噪声≈信号 → 相对误差虚高），全局 L2 让大信号参数主导、真隔离泄漏才留痕。
    # 再报 cosine 相似度（方向一致性）作旁证。
    diff_sq, ref_sq, dot, pnorm_sq, knorm_sq = 0.0, 0.0, 0.0, 0.0, 0.0
    for name in grads_p:
        if name not in grads_k:
            failed += 1
            if rank == 0:
                print(f"  FAIL: 参数 {name} 在 packing 路径无梯度")
            continue
        gp, gk = grads_p[name], grads_k[name]
        diff_sq += ((gp - gk) ** 2).sum().item()
        ref_sq += (gp ** 2).sum().item()
        dot += (gp * gk).sum().item()
        pnorm_sq += (gp ** 2).sum().item()
        knorm_sq += (gk ** 2).sum().item()
    rel_l2 = (diff_sq ** 0.5) / (ref_sq ** 0.5 + 1e-12)
    cosine = dot / ((pnorm_sq ** 0.5) * (knorm_sq ** 0.5) + 1e-12)
    if rank == 0:
        print(f"  [info] grad 全局 rel_L2={rel_l2:.3e}，cosine={cosine:.8f}")
    # 度量策略（诚实分层）：
    #   - fp32（gate_rel_l2=True）：数学精确证据。padding 逐行 [B,L] 前向 vs packing 单条 [1,T] 前向，
    #     fp32 下两路 loss/grad 应逐值一致（rel_L2、cosine 都收紧）。**这是隔离正确性的硬证明**——
    #     真跨样本注意力泄漏会在 fp32 也留 O(1) 痕迹，不会随 dtype 收缩。
    #   - bf16（gate_rel_l2=False）：只能匹配到舍入。28 层前向的两种规约顺序差 ~bf16 epsilon 累积，
    #     rel_L2 量级可达几到十几 %（**非 bug**，fp32 已证精确）。故 bf16 只 gate loss + cosine
    #     （方向一致），magnitude rel_L2 仅报 info 不 gate。
    if gate_rel_l2:
        check(rel_l2 < rel_tol, f"grad 全局相对等价：rel_L2={rel_l2:.3e} < {rel_tol}")
    check(cosine > cos_min, f"grad 方向一致：cosine={cosine:.6f} > {cos_min}")
    return failed


def main() -> int:
    fp32 = "--fp32" in sys.argv
    # fp32 副证：compute_dtype=float32 关掉 FSDP 混合精度（去 bf16 舍入噪声，证明算法数学精确）。
    trainer = FSDPTrainer(
        model_path=MODEL_PATH,
        lr=1e-4,
        global_batch_size=4,
        fp16=False,
        compute_dtype=torch.float32 if fp32 else None,
    )
    rank = trainer.rank
    # loss 绝对阈值；grad 全局 rel_L2 + cosine。fp32 是数学精确证明（rel_L2 硬门）；
    # bf16 只能匹配到舍入 → 只 gate loss + cosine，rel_L2 仅报 info（fp32 已证精确，非 bug）。
    if fp32:
        tol, rel_tol, cos_min, gate_rel_l2 = 1e-4, 5e-3, 0.9999, True
    else:
        tol, rel_tol, cos_min, gate_rel_l2 = 1e-2, 0.0, 0.98, False

    if rank == 0:
        mode = "fp32 副证（数学精确）" if fp32 else "bf16 主门（loss+方向）"
        print(f"=== V7.5 packing 等价性（{mode}，loss_tol={tol} cos_min={cos_min}）===")

    failed = 0
    failed += _test_structure(rank)
    failed += _test_equivalence(trainer, tol, rel_tol, cos_min, gate_rel_l2, rank)

    if rank == 0:
        print(f"[V7.5] {'PASSED' if failed == 0 else f'FAILED ({failed})'}")

    dist.barrier()
    dist.destroy_process_group()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
