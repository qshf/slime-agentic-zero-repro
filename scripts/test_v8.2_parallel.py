"""V8.2 验证：Megatron 多维并行（TP × PP × CP）的 loss/grad 等价 + 拓扑 + PP 下的权重转换。

沿用 V8.0 的 **dump / compare** 范式：两次独立 torchrun，第二次与第一次的 dump 逐值比。
参照系永远是 `TP=1 × PP=1 × CP=1`（单卡、无任何模型并行），因为那是"没有可切错的东西"的配置。

用法（容器内，`NANO_MODEL_PATH` 指向 Qwen3-0.6B）：

    # 基线（1 卡）——G1/G2/G3/G4 都跟它比
    torchrun --nproc_per_node=1 scripts/test_v8.2_parallel.py --fp32 --dump /tmp/v82_base.pt

    # G1 NCCL 基线：换后端不改数学（2 卡，TP=2）
    torchrun --nproc_per_node=2 scripts/test_v8.2_parallel.py --fp32 --tp 2 --backend nccl \
        --dump /tmp/v82_tp2.pt --compare /tmp/v82_base.pt

    # G2 PP：**本版最硬的门**（2 卡，PP=2）——抓 tie 的 embedding 漏规约
    torchrun --nproc_per_node=2 scripts/test_v8.2_parallel.py --fp32 --pp 2 --backend nccl \
        --dump /tmp/v82_pp2.pt --compare /tmp/v82_base.pt

    # G3 CP：**只能 bf16**（见下方"G3 为什么没有 fp32 档"），故拆成三步：
    #   G3a 先证 THD(packing) + TE spec 这条前向路径本身与 bshd 等价（CP=1，隔离变量）
    torchrun --nproc_per_node=1 scripts/test_v8.2_parallel.py --backend nccl \
        --qkv-format thd --te-spec --dump /tmp/v82_thd1.pt --compare /tmp/v82_base_bf16.pt
    #   G3b 再证 CP=2 与 CP=1 等价（唯一变量就是 CP）
    torchrun --nproc_per_node=2 scripts/test_v8.2_parallel.py --cp 2 --backend nccl \
        --qkv-format thd --dump /tmp/v82_cp2.pt --compare /tmp/v82_thd1.pt
    #   G3c negative control：把 2-chunk 对称切分换成朴素连续切分，**必须判红**
    torchrun --nproc_per_node=2 scripts/test_v8.2_parallel.py --cp 2 --backend nccl \
        --qkv-format thd --negative-control --compare /tmp/v82_thd1.pt

    # G4/G5/G6 组合（4 卡，TP=2 × PP=2）+ 拓扑断言 + PP 下的 HF 往返
    torchrun --nproc_per_node=4 scripts/test_v8.2_parallel.py --fp32 --tp 2 --pp 2 \
        --backend nccl --topology --convert --dump /tmp/v82_combo.pt --compare /tmp/v82_base.pt

判据（沿用 V7.5 的度量教训 + V8 的分层 gate）：
  - **fp32 才是精确证明**：并行切分理论上是数学恒等，fp32 下误差只剩规约顺序 →
    loss |Δ| < 1e-4、grad 全局 rel_L2 < 5e-3、cosine > 0.9999。
  - **bf16 只到舍入**：28 层前向的 bf16 舍入会累积到 rel_L2 几个百分点，那**不是切错**，
    故 bf16 档只硬门 loss + cosine > 0.999，rel_L2 报 info。
  - 度量用**全局 rel_L2 + cosine**（向量级、大信号主导），不用逐参数 max
    （小信号参数的 bf16 噪声会把 max 虚高，V7.5 实测过 0.83 的假红）。

**为什么 fp32 是必须的而不是锦上添花**：本版三处跨 rank 梯度规约（qk-layernorm 跨 TP /
tie 的 embedding 跨 PP / 全部梯度跨 DP·CP）漏掉任何一处，指纹都是
**「loss Δ 恰为 0，而梯度系统性偏、最差参数指向漏掉的那类」**。V8 实测：漏 qk-layernorm 时
bf16 的 5e-2 会被当舍入放过，fp32 才把它从 1.8e-2 打到 1.9e-5（930×）。

**G3 为什么没有 fp32 档（本版实测出来的证据边界）**：CP 只有 TE 后端支持，而 CP>1 必须走
flash(FA2) 后端（fused 在 sm_120 上静默算错）—— **FA2 内核只收 fp16/bf16**，
fp32 下 TE 直接报 `No dot product attention backend is available`（实测）。
这与 V7.6 偏离 #8 是同一条物理限制。故 G3 改用 **bf16 + negative control** 的验收范式：
先证明这套度量能把"故意写错的切分"判红，"写对时判绿"才有意义。
**证据边界要诚实**：G1/G2/G4 有 fp32 逐值精确证明，G3 只到 bf16 + 反例判别力。
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
    _local_params_by_global_name,
    all_gather_param,
    convert_qwen3_to_hf,
    load_hf_weights,
    remove_padding,
)
from toy_rl.trainer.megatron_trainer import MegatronTrainer, observed_attention_backend

MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")


def _make_samples(trainer: MegatronTrainer) -> list[dict]:
    """4 条不同长度的真样本 + 合成 old_log_probs（ratio≠1 → loss 本身也测前向）。

    与 V7.5/V7.6/V8.0 同一套样本构造：on-policy（old=None）时 ratio≡1、loss 退化成常数，
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
            {"tokens": tokens, "loss_mask": loss_mask, "reward": reward, "old_log_probs": old_lp}
        )
    return samples


def _grads_in_hf_layout(trainer: MegatronTrainer) -> dict[str, torch.Tensor]:
    """本 rank 的分片梯度 → TP-gather →（PP>1 时）跨 stage 汇总 → HF 布局的全量梯度 dict。

    梯度与权重的分片布局**完全相同**（每个 param 的 grad 与 param 同形），故可直接复用
    `all_gather_param`（含 linear_fc1 的 GLU 重排）与 `convert_qwen3_to_hf`。
    这里构造一个带 TP 元数据的轻量 shim，把 grad 冒充成 param 喂进去——比给
    `all_gather_param` 加一个 `tensor=` 参数更干净（那会偏离源签名）。

    PP>1 时每个 stage 只有自己那几层的梯度，故最后用 `all_gather_object` 在 PP 组里
    把各 stage 的结果并起来（**只是验收用的汇总**，不是训练路径的一部分——训练路径的
    PP broadcast 在 `megatron_to_hf_state_dict` 里，由 `--convert` 单独验）。
    """
    out: dict[str, torch.Tensor] = {}
    for name, param in _local_params_by_global_name(trainer.model, trainer.tie).items():
        if param.grad is None:
            continue
        shim = SimpleNamespace(
            data=param.grad.detach(),
            tensor_model_parallel=getattr(param, "tensor_model_parallel", False),
            partition_dim=getattr(param, "partition_dim", -1),
            partition_stride=getattr(param, "partition_stride", 1),
            parallel_mode=getattr(param, "parallel_mode", None),
        )
        full = all_gather_param(name, shim)
        full = remove_padding(name, full, trainer.hf_config.vocab_size)
        for hf_name, hf_grad in convert_qwen3_to_hf(trainer.hf_config, name, full):
            out[hf_name] = hf_grad.float().cpu().clone()

    if trainer.pp_size > 1:
        gathered: list = [None] * trainer.pp_size
        dist.all_gather_object(gathered, out, group=trainer.pp_group)
        merged: dict[str, torch.Tensor] = {}
        for part in gathered:
            merged.update(part)
        out = merged
    return out


def _compare(ref: dict, loss: float, grads: dict, fp32: bool, check, negative: bool = False) -> None:
    """与参照 dump 逐值比对（分层 gate）。

    `negative=True` 时判据**反转**：这是 negative control，正确的度量必须把
    "故意写错的切分"判红。反例若"通过"，说明这套度量没有判别力、整轮验收作废
    （V7.6 踩出来的教训：bf16 档的绝对阈值很容易变成一条谁都能过的线）。
    """
    if fp32:
        rel_l2_bar, cos_bar, tier = 5e-3, 0.9999, "fp32 精确证明"
    else:
        rel_l2_bar, cos_bar, tier = None, 0.999, "bf16 舍入级"
    print(f"  [info] gate 分层：{tier}{'（negative control：期望判红）' if negative else ''}")

    dloss = abs(ref["loss"] - loss)
    check(set(ref["grads"]) == set(grads), f"梯度名字集合一致（{len(ref['grads'])} vs {len(grads)}）")

    common = sorted(set(ref["grads"]) & set(grads))
    check(
        all(ref["grads"][n].shape == grads[n].shape for n in common),
        "gather 回全量后各梯度形状一致（并行配置无关）",
    )

    # **逐张量累加而不是 cat 成一条大向量**：全 28 层时 cat 两条 float64 要 ~10GB，
    # 本机只有 15G RAM 会被 OOM kill。累加数学等价——全局 L2/cosine 本就只依赖
    # Σ|a-b|²、Σa·b、Σa²、Σb² 这四个标量。
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
    print(f"  [info] loss |Δ|={dloss:.3e}  grad rel_L2={rel_l2:.3e}  cosine={cos:.8f}")

    if negative:
        check(cos < 0.99, f"negative control 被判红：cosine={cos:.8f} < 0.99（度量有判别力）")
        return

    check(dloss < 1e-4, f"loss 等价：|{ref['loss']:.8f} - {loss:.8f}| = {dloss:.3e} < 1e-4")
    if rel_l2_bar is not None:
        check(rel_l2 < rel_l2_bar, f"grad 全局 rel_L2={rel_l2:.3e} < {rel_l2_bar:g}")
    check(cos > cos_bar, f"grad cosine={cos:.8f} > {cos_bar:g}")

    # 逐参数最大偏差仅报 info，但**最差参数的名字**是漏规约的指纹：
    #   q_norm.weight → 漏 qk-layernorm 跨 TP；embed_tokens.weight → 漏 tie 跨 PP。
    worst = max(common, key=lambda n: (ref["grads"][n] - grads[n]).abs().max().item())
    print(
        f"  [info] 逐参数最大绝对偏差：{worst} "
        f"{(ref['grads'][worst] - grads[worst]).abs().max().item():.3e}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4, help="缩减层数（等价性与层数无关）")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--qkv-format", type=str, default="bshd", choices=["bshd", "thd"])
    ap.add_argument("--te-spec", action="store_true", help="强制 TE spec（CP>1 时自动开）")
    ap.add_argument(
        "--attn-backend",
        type=str,
        default="flash",
        choices=["flash", "fused", "unfused", "local", "auto"],
        help="≡ megatron --attention-backend。CP>1 必须 flash；"
        "**fp32 只能走 unfused**（FA2 内核拒收 fp32）",
    )
    ap.add_argument("--fp32", action="store_true", help="fp32 精确证明档")
    ap.add_argument("--dump", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None)
    ap.add_argument("--topology", action="store_true", help="G5：打印并断言各并行组 ranks")
    ap.add_argument("--convert", action="store_true", help="G6：PP 下的 Megatron→HF 往返")
    ap.add_argument(
        "--negative-control",
        action="store_true",
        help="把 CP 的 2-chunk 对称切分换成朴素连续切分，**期望等价门变红**（证明度量有判别力）",
    )
    args = ap.parse_args()

    if args.negative_control:
        # 必须在建 trainer 之前设（cp_utils 在 import 时读它）。
        os.environ["NANO_CP_NAIVE_SPLIT"] = "1"

    trainer = MegatronTrainer(
        MODEL_PATH,
        num_layers=args.layers,
        global_batch_size=1,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        backend=args.backend,
        qkv_format=args.qkv_format,
        use_te_spec=True if args.te_spec else None,
        attention_backend=args.attn_backend,
        params_dtype=torch.float32 if args.fp32 else torch.bfloat16,
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

    if rank == 0:
        print(
            f"=== V8.2 多维并行（tp={trainer.tp_size} pp={trainer.pp_size} cp={trainer.cp_size} "
            f"dp={trainer.dp_size} backend={args.backend} qkv={args.qkv_format} "
            f"spec={'te' if trainer.use_te_spec else 'local'} layers={trainer.num_layers}）==="
        )

    # --- G5 拓扑：TP 组必须落在同 NUMA node 内 ---
    if args.topology:
        groups = trainer.parallel_group_ranks()
        if rank == 0:
            print("[G5] 拓扑感知的 rank 映射")
            for k, v in groups.items():
                print(f"  [info] {k} group ranks = {v}")
        # 本机 `nvidia-smi topo -m`：0-1 同 NUMA、2-3 同 NUMA、跨组 SYS（无 NVLink）。
        # TP 每层两次 all-reduce 激活（通信量最大）必须放同 NODE 内；
        # PP 每 stage 边界只一次 p2p（通信量最小），跨组可接受。
        numa = {0: 0, 1: 0, 2: 1, 3: 1}
        tp_ranks = groups["tp"]
        check(
            len({numa.get(r, -1) for r in tp_ranks}) == 1,
            f"TP 组 {tp_ranks} 落在同一 NUMA node 内（避开 SYS 跨 root complex）",
        )

    # --- G6 转换：PP>1 时 Megatron→HF 必须先跨 stage broadcast ---
    # **必须在训练一步之前做**：`train_batch` 末尾有 `optimizer.step()`，跑完权重就被
    # AdamW 改动了 ~lr 的量级 —— 那时再跟原始 HF ckpt 比，max_abs_diff 恰好 ≈ 1e-6(=lr)，
    # 看起来像"转换有微小误差"，实际是"比错了对象"。（这条是踩出来的：初版把 G6 放在
    # 训练之后，报 1.013e-06 于 `input_layernorm.weight`。）
    if args.convert:
        if rank == 0:
            print("[G6] Megatron→HF 往返（PP>1 时任何单 rank 都拿不到完整模型）")
        converted = trainer.to_hf_state_dict()
        original = load_hf_weights(MODEL_PATH)
        names = [n for n in converted if n in original]
        check(
            len(names) == len(converted),
            f"转出的名字都在原始 HF ckpt 里（{len(names)}/{len(converted)}）",
        )
        # 期望每层 11 个 + embed + norm（tie 故无 lm_head）
        expect = trainer.num_layers * 11 + 2
        check(len(converted) == expect, f"参数个数 {len(converted)} == {expect}（含全部 stage 的层）")
        worst_name, worst = None, 0.0
        for n in names:
            d = (converted[n].cpu().float() - original[n].cpu().float()).abs().max().item()
            if d > worst:
                worst, worst_name = d, n
        # **无容差硬门**：转换全程只有 reshape/split/cat/rename，无浮点运算。
        check(worst == 0.0, f"往返逐值精确：max_abs_diff={worst:.3e} == 0（最差 {worst_name}）")
        del converted, original

    # --- 前向 + 反向（走 train_batch，即 mcore 的 forward_backward_func 调度）---
    if rank == 0:
        print("[a] forward + GRPO loss + backward（1F1B 调度）")
    samples = _make_samples(trainer)
    gbs = len(samples)
    # PP 要求各微批形状一致，故 microbatch_size 取能整除的最大值 = 全批一个微批时
    # 退化成无流水；这里用 2 个微批，让 1F1B 真的有 warmup/steady/cooldown 三段。
    micro = 1 if trainer.pp_size > 1 else gbs
    metrics = trainer.train_batch(samples, global_batch_size=gbs, microbatch_size=micro)
    loss = metrics["loss"]

    if trainer.use_te_spec:
        # **后端要断言不要假设**：NVTE_* 是请求，TE 可能否掉它；实选后端只有跑过真前向才有值。
        # CP>1 时落到 fused(cuDNN) 会**静默算错**（sm_120 上 dq cosine 0.356，TE#3333）。
        want = {"flash": "FlashAttention", "fused": "FusedAttention",
                "unfused": "UnfusedDotProductAttention"}.get(args.attn_backend)
        backend = observed_attention_backend()
        if rank == 0:
            print(f"  [info] TE 实选 attention 后端 = {backend}")
        if want is not None:
            check(backend.startswith(want), f"attention 后端 == 请求的 {args.attn_backend}（实选 {backend}）")
        if trainer.cp_size > 1:
            check(backend.startswith("FlashAttention"), "CP>1 时后端是 flash —— CP 的正确性前提")

    check(metrics["trained_samples"] == gbs,
          f"全部 {gbs} 条样本参与训练（trained={metrics['trained_samples']}）")
    check(torch.isfinite(torch.tensor(loss)).item(), f"loss 有限：{loss:.6f}")

    grads = _grads_in_hf_layout(trainer)
    check(len(grads) > 0, f"梯度非空（HF 布局 {len(grads)} 个张量）")
    check(all(torch.isfinite(g).all().item() for g in grads.values()), "全部梯度 finite")
    gnorm = torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values())).item()
    check(gnorm > 0, f"梯度非零：global grad norm={gnorm:.4f}")
    if rank == 0:
        print(f"  [info] loss={loss:.8f}  grad_norm={gnorm:.6f}  tensors={len(grads)}")

    # --- dump / compare ---
    cfg = {
        "tp": trainer.tp_size, "pp": trainer.pp_size, "cp": trainer.cp_size,
        "backend": args.backend, "qkv_format": args.qkv_format,
        "spec": "te" if trainer.use_te_spec else "local",
        "attn": args.attn_backend if trainer.use_te_spec else "-",
    }
    if args.dump and rank == 0:
        torch.save(
            {"config": cfg, "loss": loss, "grad_norm": gnorm, "grads": grads,
             "fp32": bool(args.fp32)},
            args.dump,
        )
        print(f"  [info] dumped → {args.dump}")

    if args.compare and rank == 0:
        print("[b] 与参照配置的等价（本版硬门）")
        ref = torch.load(args.compare, weights_only=False)
        print(f"  [info] 参照 {ref['config']}\n         vs 本次 {cfg}")
        # 「两次配置必须不同」——否则就是在跟自己比，门恒绿而毫无信息。
        # 比的是**整份配置**而不只是 tp/pp/cp：G3a 那种「同并行度、换布局/后端」也算不同。
        check(ref["config"] != cfg, "两次配置不同才有意义")
        check(
            bool(ref.get("fp32", False)) == bool(args.fp32),
            f"两次 dtype 一致（参照 fp32={bool(ref.get('fp32', False))} vs 本次 fp32={bool(args.fp32)}）",
        )
        _compare(ref, loss, grads, args.fp32, check, negative=args.negative_control)

    dist.barrier()
    if rank == 0:
        print("[V8.2] " + ("PASSED" if failed == 0 else f"FAILED ({failed})"))
    dist.barrier()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
