"""V8.0 验证：Megatron TP=2 与 TP=1 的 loss/grad 等价（TP 是数学恒等切分）。

用法（两次 torchrun，第二次比对第一次）：
    torchrun --nproc_per_node=1 scripts/test_v8.0_megatron_tp.py --dump /tmp/v8_tp1.pt
    torchrun --nproc_per_node=2 scripts/test_v8.0_megatron_tp.py --dump /tmp/v8_tp2.pt \
                                                                 --compare /tmp/v8_tp1.pt
    # fp32 精确证明（两次都加 --fp32）：
    torchrun --nproc_per_node=1 scripts/test_v8.0_megatron_tp.py --fp32 --dump /tmp/v8_tp1_fp32.pt
    torchrun --nproc_per_node=2 scripts/test_v8.0_megatron_tp.py --fp32 --dump /tmp/v8_tp2_fp32.pt \
                                                                 --compare /tmp/v8_tp1_fp32.pt

契约：
  part (a) 结构：TP=2 时参数**真被切开**（qkv/fc1 列切、o_proj/fc2 行切、vocab 按 TP 切），
      且 local 参数量 ≈ 全量/tp_size。TP=1 时不切（对照）。
  part (b) 等价（本版硬门，**分层**——沿用 v7.5.md 的度量教训）：
      同一批样本、同一份 HF 预训练权重，TP=1 与 TP=2 的
        loss:  |Δ| < 1e-4                       （两档共用）
        grad:  fp32 档 → rel_L2 < 5e-3 且 cosine > 0.9999   ← **精确证明**
               bf16 档 → cosine > 0.999，rel_L2 仅报 info    ← 训练实际 dtype，只能到舍入
      **为什么分层**：TP 是数学恒等切分（col-parallel 切输出维、row-parallel 切输入维
      + all-reduce），理论上应精确等价，fp32 下误差只剩 all-reduce 的规约顺序；而 bf16
      在 28 层前向里舍入会累积到 rel_L2 几个百分点——那**不是 TP 切错**。
      **门仍有判别力**：真切错（qkv 按错误维度切、忘了 all-reduce、fc1 的 GLU 重排搞反）
      误差是 **O(1) 级**、cosine 会掉到 0.x——实测过一次 dropout 未关的假阳性，cosine=0.25。

  **梯度怎么跨 TP 比**：先 TP-gather 回全量、再转成 **HF 布局**（`megatron_to_hf`），
  这样 TP=1 与 TP=2 的张量**名字与形状都一致**才能逐值比。这顺带复用并验证了 V8.1 的转换路径。

  两次运行必须从**同一份 HF 预训练权重**载入（随机初始化在两个独立进程里不可能一致）——
  这正是 `MegatronTrainer(load_hf_weights=True)` 存在的理由。

  **单卡多 rank 必须 gloo**（NCCL 实测硬拒 `Duplicate GPU detected`），见 megatron_trainer 偏离说明。

前置：5090 容器内（megatron-core 已装）；`NANO_MODEL_PATH` 指向 Qwen3-0.6B。
默认 `--layers 4`（等价性与层数无关，缩减层数快且避开单卡显存风险；v8-plan §7 已拍板）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

from toy_rl.trainer.megatron_to_hf import (
    all_gather_param,
    convert_qwen3_to_hf,
    remove_padding,
    strip_param_name_prefix,
)
from toy_rl.trainer.megatron_trainer import MegatronTrainer

MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")


def _make_samples(trainer: MegatronTrainer) -> list[dict]:
    """4 条不同长度的真样本 + 合成 old_log_probs（ratio≠1 → loss 本身也测前向）。

    与 V7.5/V7.6 同一套样本构造：on-policy（old=None）时 ratio≡1、loss 退化成常数，
    只有 grad 能测前向；给合成 old_lp 才能让 loss 值本身也成为前向的函数。
    """
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
        loss_mask = [0] * len(p_ids) + [1] * len(r_ids)
        old_lp = [-0.5 * (i % 3) for i in range(len(tokens))]
        samples.append(
            {
                "tokens": tokens,
                "loss_mask": loss_mask,
                "reward": reward,
                "old_log_probs": old_lp,
            }
        )
    return samples


def _grads_in_hf_layout(trainer: MegatronTrainer) -> dict[str, torch.Tensor]:
    """本 rank 的分片梯度 → TP-gather → HF 布局的全量梯度 dict。

    梯度与权重的分片布局**完全相同**（每个 param 的 grad 与 param 同形），故可直接复用
    `all_gather_param`（含 linear_fc1 的 GLU 重排）与 `convert_qwen3_to_hf`。
    这里构造一个带 TP 元数据的轻量 shim，把 grad 冒充成 param 喂进去——比给
    `all_gather_param` 加一个 `tensor=` 参数更干净（那会偏离源签名）。
    """
    out: dict[str, torch.Tensor] = {}
    for name, param in trainer.model.named_parameters():
        if param.grad is None:
            continue
        clean = strip_param_name_prefix(name)
        if trainer.tie and clean == "output_layer.weight":
            continue
        shim = SimpleNamespace(
            data=param.grad.detach(),
            tensor_model_parallel=getattr(param, "tensor_model_parallel", False),
            partition_dim=getattr(param, "partition_dim", -1),
            partition_stride=getattr(param, "partition_stride", 1),
            parallel_mode=getattr(param, "parallel_mode", None),
        )
        full = all_gather_param(clean, shim)
        full = remove_padding(clean, full, trainer.hf_config.vocab_size)
        for hf_name, hf_grad in convert_qwen3_to_hf(trainer.hf_config, clean, full):
            out[hf_name] = hf_grad.float().cpu().clone()
    return out


def _shard_stats(trainer: MegatronTrainer) -> dict:
    """part (a)：抓几个代表性参数的 local 形状，证明 TP 真切开了。"""
    stats = {}
    for name, param in trainer.model.named_parameters():
        clean = strip_param_name_prefix(name)
        for key in (
            "decoder.layers.0.self_attention.linear_qkv.weight",   # column-parallel
            "decoder.layers.0.self_attention.linear_proj.weight",  # row-parallel
            "decoder.layers.0.mlp.linear_fc1.weight",              # column-parallel (GLU)
            "decoder.layers.0.mlp.linear_fc2.weight",              # row-parallel
            "embedding.word_embeddings.weight",                    # vocab-parallel
        ):
            if clean == key:
                stats[key] = tuple(param.shape)
    stats["_local_numel"] = sum(p.numel() for p in trainer.model.parameters())
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4, help="缩减层数（等价性与层数无关）")
    ap.add_argument("--dump", type=str, default=None, help="把 loss/grad 存到此路径")
    ap.add_argument("--compare", type=str, default=None, help="与此路径的 dump 比对")
    ap.add_argument("--fp32", action="store_true", help="fp32 副证（数学精确证明）")
    args = ap.parse_args()

    # 这一步完成整个 Megatron 侧初始化：建立 torch.distributed / TP 并行组，
    # 按 tp_size 构造当前 rank 的模型分片，并把 HF checkpoint 的对应 slice 写入该分片。
    # 因而构造器返回后，trainer.model.named_parameters() 已不是完整 HF 模型参数表。
    trainer = MegatronTrainer(
        MODEL_PATH,
        num_layers=args.layers,
        global_batch_size=1,
        backend="gloo",  # 单卡多 rank 唯一通路 / 多卡 ncll
        params_dtype=torch.float32 if args.fp32 else torch.bfloat16,
    )
    # rank 是 torchrun 分配的全局进程编号；本脚本只开 TP，故 tp_size == world_size。
    # TP=2 时每个 rank 各持一半可切分参数，数值不同但大多数本地张量形状相同。
    rank, tp_size = trainer.rank, trainer.tp_size
    failed = 0

    def check(cond, msg):
        nonlocal failed
        if not cond:
            failed += 1
            if rank == 0:
                print(f"  FAIL: {msg}")
        elif rank == 0:
            print(f"  OK: {msg}")

    if rank == 0:
        print(f"=== V8.0 Megatron TP 等价（tp_size={tp_size}, layers={trainer.num_layers}）===")
        print("[a] TP 分片结构")

    # 这里只读取当前 rank 的 local shard，绝不触发 all-gather；因此它能证明模型真的被切开。
    # 对照：TP=1 时 local shape 是全量；TP=2 时 column/vocab 层的 dim0、row 层的 dim1 缩为一半。
    stats = _shard_stats(trainer)
    # 以下是完整 HF 配置中的逻辑维度，用来计算每个 local shard 应有的形状，
    # 而不是从 stats 反推；这样错误的切分也会被断言捕获。
    hidden = trainer.hf_config.hidden_size
    ffn = trainer.hf_config.intermediate_size
    head_dim = getattr(trainer.hf_config, "head_dim", None) or (
        hidden // trainer.hf_config.num_attention_heads
    )
    # 注意：o_proj 的输入维是 num_heads*head_dim（Qwen3-0.6B = 16*128 = 2048），**不是 hidden_size**
    # （0.6B 的 head_dim 128 > hidden/heads = 64，两者不相等）。
    q_dim = trainer.hf_config.num_attention_heads * head_dim
    if rank == 0:
        for k, v in stats.items():
            print(f"  [info] {k}: {v}")

    # column-parallel 切 dim0，row-parallel 切 dim1，vocab 切 dim0。
    check(
        stats["decoder.layers.0.self_attention.linear_proj.weight"][1] == q_dim // tp_size,
        f"o_proj 行切：dim1={stats['decoder.layers.0.self_attention.linear_proj.weight'][1]} "
        f"== heads*head_dim({q_dim})/tp({tp_size})",
    )
    check(
        stats["decoder.layers.0.mlp.linear_fc1.weight"][0] == 2 * ffn // tp_size,
        f"fc1 列切（GLU 两倍）：dim0={stats['decoder.layers.0.mlp.linear_fc1.weight'][0]} "
        f"== 2*ffn({ffn})/tp({tp_size})",
    )
    check(
        stats["decoder.layers.0.mlp.linear_fc2.weight"][1] == ffn // tp_size,
        f"fc2 行切：dim1={stats['decoder.layers.0.mlp.linear_fc2.weight'][1]} == ffn/tp",
    )
    check(
        stats["embedding.word_embeddings.weight"][0]
        == trainer.hf_config.vocab_size // tp_size,
        f"vocab 切：dim0={stats['embedding.word_embeddings.weight'][0]} "
        f"== vocab({trainer.hf_config.vocab_size})/tp({tp_size})",
    )

    # --- 前向 + 反向 ---
    if rank == 0:
        print("[b] forward + GRPO loss + backward")
    samples = _make_samples(trainer)
    gbs = len(samples)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss, trained = trainer._microbatch_backward(samples, gbs)
    # **必须与 train_batch 同序**：TP 特有的跨 rank 梯度规约（qk-layernorm 的部分和）。
    # 漏掉它就是在比「未规约的部分梯度 vs 全量梯度」——门会红，且红得对。
    trainer._finalize_model_grads()
    check(trained == len(samples), f"全部 {len(samples)} 条样本参与训练（trained={trained}）")
    check(torch.isfinite(torch.tensor(loss)).item(), f"loss 有限：{loss:.6f}")

    grads = _grads_in_hf_layout(trainer)
    check(len(grads) > 0, f"梯度非空（HF 布局 {len(grads)} 个张量）")
    all_finite = all(torch.isfinite(g).all().item() for g in grads.values())
    check(all_finite, "全部梯度 finite")
    gnorm = torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values())).item()
    check(gnorm > 0, f"梯度非零：global grad norm={gnorm:.4f}")

    if rank == 0:
        print(f"  [info] loss={loss:.8f}  grad_norm={gnorm:.6f}  tensors={len(grads)}")

    # --- dump / compare（跨两次 torchrun 运行）---
    if args.dump and rank == 0:
        torch.save(
            {
                "tp_size": tp_size,
                "loss": loss,
                "grad_norm": gnorm,
                "grads": grads,
                "fp32": bool(args.fp32),
            },
            args.dump,
        )
        print(f"  [info] dumped → {args.dump}")

    if args.compare and rank == 0:
        print("[c] TP 等价（本版硬门）")
        ref = torch.load(args.compare, weights_only=False)
        print(f"  [info] 参照 tp_size={ref['tp_size']} vs 本次 tp_size={tp_size}")
        check(ref["tp_size"] != tp_size, f"两次 tp_size 不同才有意义（{ref['tp_size']} vs {tp_size}）")
        check(
            bool(ref.get("fp32", False)) == bool(args.fp32),
            f"两次 dtype 一致（参照 fp32={bool(ref.get('fp32', False))} vs 本次 fp32={bool(args.fp32)}）",
        )

        # **分层 gate（V7.5 度量教训：fp32 才是精确证明、bf16 只能到舍入）**：
        #   fp32 路径 = 「TP 是数学恒等切分」的精确证明，误差只剩 all-reduce 规约顺序 → 严门；
        #   bf16 路径 = 训练实际用的 dtype，但 28 层前向的 bf16 舍入会累积到 rel_L2 几个百分点，
        #     那不是 TP 切错（真切错是 O(1) 级、cosine 会掉到 0.x —— 见 dropout 那次实测 cosine=0.25）。
        #     故 bf16 只硬门 loss + cosine，rel_L2 报 info。
        if args.fp32:
            rel_l2_bar, cos_bar, tier = 5e-3, 0.9999, "fp32 精确证明"
        else:
            rel_l2_bar, cos_bar, tier = None, 0.999, "bf16 舍入级"
        print(f"  [info] gate 分层：{tier}")

        dloss = abs(ref["loss"] - loss)
        check(dloss < 1e-4, f"loss 等价：|{ref['loss']:.8f} - {loss:.8f}| = {dloss:.3e} < 1e-4")

        # 名字集合必须完全一致（TP 不该改变参数集合）
        check(
            set(ref["grads"]) == set(grads),
            f"梯度名字集合一致（{len(ref['grads'])} vs {len(grads)}）",
        )

        common = sorted(set(ref["grads"]) & set(grads))
        shape_ok = all(ref["grads"][n].shape == grads[n].shape for n in common)
        check(shape_ok, "gather 回全量后各梯度形状一致（TP 无关）")

        # 全局 rel_L2 + cosine（V7.5 度量教训：向量级、大信号主导，不用逐参数 max）。
        # **逐张量累加而不是 cat 成一条大向量**：全 28 层时 cat 两条 float64 要 ~10GB，
        # 本机只有 15G RAM 会被 OOM kill（实测 full-28 在此静默挂掉）。累加数学等价——
        # 全局 L2/cosine 本就只依赖 Σ|a-b|²、Σa·b、Σa²、Σb² 这四个标量。
        sq_diff = dot = norm_a = norm_b = 0.0
        for n in common:
            x = ref["grads"][n].flatten().double()
            y = grads[n].flatten().double()
            sq_diff += ((x - y) ** 2).sum().item()
            dot += (x * y).sum().item()
            norm_a += (x * x).sum().item()
            norm_b += (y * y).sum().item()
        rel_l2 = (sq_diff**0.5) / max(norm_a**0.5, 1e-12)
        cos = dot / max((norm_a**0.5) * (norm_b**0.5), 1e-12)
        print(f"  [info] grad rel_L2={rel_l2:.3e}  cosine={cos:.8f}")
        if rel_l2_bar is not None:
            check(rel_l2 < rel_l2_bar, f"grad 全局 rel_L2={rel_l2:.3e} < {rel_l2_bar:g}")
        check(cos > cos_bar, f"grad cosine={cos:.8f} > {cos_bar:g}")

        # 逐参数最大偏差仅报 info（V7.5 教训：小信号参数的 bf16 噪声会虚高）
        worst = max(common, key=lambda n: (ref["grads"][n] - grads[n]).abs().max().item())
        print(
            f"  [info] 逐参数最大绝对偏差：{worst} "
            f"{(ref['grads'][worst] - grads[worst]).abs().max().item():.3e}"
        )

    dist.barrier()
    if rank == 0:
        print("[V8.0] " + ("PASSED" if failed == 0 else f"FAILED ({failed})"))
    dist.barrier()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
