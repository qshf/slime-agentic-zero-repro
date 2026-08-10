"""V8.2 前置实测：CP × packing 组合可行性（sm_120 / RTX 5090）。

**为什么有这个脚本**：v8.2-plan.md 初稿把「CP=2 × FA2 varlen 是否可用」列为待实测风险 R1，
并写了「若不通就降级 padding」。本脚本把它测清楚——结论是**降级预案成为唯一路径**，
细节与证据见 docs/decisions/v8.2-plan.md §1.1。

**四个检查**（每个都是独立判据，前一个失败不影响后一个的诊断价值）：

  C1  THD vs BSHD **forward** 逐值等价
      单段（cu_seqlens=[0,S]）时两种布局数学上完全等价，故 forward 必须 max_abs_diff==0。
      这一步建立「同一算法」的基线——没有它，C2 的差异可以被辩解成「布局不同本就该不同」。

  C2  THD vs BSHD **backward** 逐值等价  ← 决定性判据
      C1 既然证明了是同一算法，backward 也必须相等。实测 fused 后端差 40×
      （上游 bug TE#3333，见 docs/decisions/v8-5090-followup.md §3）。
      **这是「forward 相等 ≠ backward 相等」的活教材**——与 V8 坑③（漏 qk-layernorm 规约：
      loss Δ 恰为 0 而梯度系统性偏）同源。

  C3  fused 的 **BSHD** backward 是否可信（fused vs unfused 对照）
      C2 若失败，需要知道「坏的是整个 fused 内核还是只有 THD 路径」——
      这直接决定「CP + padding」这条退路是否成立。

  C4  CP=2 在 BSHD / THD 下各自能否跑
      CP 只有 TE 后端支持（mcore dot_product_attention.py:57-59 assert cp==1）；
      而 Unfused（TE#3333 的 workaround）不支持 CP，故 CP 只能走 fused。

**注意「上游放行 ≠ 算得对」**：TE `utils.py:992-999` 对 sm_120 的 THD 有一条
`cudnn_version < (9,18,1)` 的 gate，本容器 cuDNN 9.25.0 **高于门槛故不触发**——
TE 自报 `FusedAttention=True (sub-backend 1)` 并选中它，即上游认为这条路已修好，
而 C2 实测 backward 仍错。想看后端选择过程加 `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`。
（另：本 bug 与 TE#2186「THD+CP tail-padding NaN」不是同一条——那条依赖 CP，
本条单卡无 CP、单段即复现。）

**怎么跑**（容器 nano-mcore，镜像 agentic-rl-infra-lab:te-cudnn-system-spike）：

    # C1-C3：单卡，跑两遍（fused / unfused）后比对
    TAG=fused   torchrun --nproc_per_node=1 --master_port 29791 scripts/probe_v8.2_cp_packing.py
    NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 TAG=unfused \
                torchrun --nproc_per_node=1 --master_port 29792 scripts/probe_v8.2_cp_packing.py
    PROBE_MODE=compare python scripts/probe_v8.2_cp_packing.py

    # C4：双卡
    PROBE_CP=2 torchrun --nproc_per_node=2 --master_port 29793 scripts/probe_v8.2_cp_packing.py

**偏离登记**：本脚本用 TEDotProductAttention 直接测单层 attention，不搭完整 MegatronTrainer——
源项目没有这类后端可行性探针（它假定 TE/FA 内核是对的）。nano 加这一层是因为
sm_120 上实测内核有 bug，且这个 bug 是**静默**的（forward 正确、backward 错），
不主动验就会在 CP 验收时得到「通过但错」的结果。语义上不影响主线代码，纯诊断工具。
"""

import os
import sys

import torch
import torch.distributed as dist

OUT_DIR = os.environ.get("PROBE_OUT", "/out")
TAG = os.environ.get("TAG", "fused")
CP = int(os.environ.get("PROBE_CP", "1"))
MODE = os.environ.get("PROBE_MODE", "run")

# Qwen3-0.6B 的真实 attention 形状（GQA：16 query heads / 8 kv groups）
NH, NKV, HD = 16, 8, 128


def _compare() -> int:
    """PROBE_MODE=compare：比对两次 run 落盘的结果（不需要 GPU / dist）。"""
    f = torch.load(f"{OUT_DIR}/probe_fused.pt")
    u = torch.load(f"{OUT_DIR}/probe_unfused.pt")
    print(f"\n=== C3 · fused vs unfused（unfused 为可信参考）===")
    print(f"{'layout':<8}{'tensor':<6}{'fused_max':>13}{'unfused_max':>13}"
          f"{'max_abs_diff':>14}{'cosine':>13}")
    bad = 0
    for layout in ("bshd", "thd"):
        for nm in ("o", "dq", "dk", "dv"):
            key = f"{layout}_{nm}"
            if key not in f or key not in u:
                continue
            a, b = f[key].view(-1), u[key].view(-1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            diff = (a - b).abs().max().item()
            flag = ""
            # bf16 舍入量级是 1e-1；差到 1e0 以上就是真错，不是精度
            if nm != "o" and cos < 0.99:
                flag = "  <-- WRONG"
                bad += 1
            print(f"{layout:<8}{nm:<6}{a.abs().max():>13.4e}{b.abs().max():>13.4e}"
                  f"{diff:>14.4e}{cos:>13.8f}{flag}")
    print(f"\n结论：{'fused 内核在某路径上给出错误梯度' if bad else 'fused 与 unfused 一致'}")
    return 0


if MODE == "compare":
    sys.exit(_compare())


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group(backend="nccl", world_size=WORLD, rank=RANK)

from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.extensions.transformer_engine import TEDotProductAttention  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

mpu.initialize_model_parallel(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=CP
)

# dropout 显式设 0：V8 坑①——TransformerConfig 默认 0.1，而各 TP/CP rank 持不同
# dropout RNG，会让等价性判据出现假阳性（V8 实测 cosine 掉到 0.25）。
cfg = TransformerConfig(
    num_layers=1, hidden_size=1024, num_attention_heads=NH, num_query_groups=NKV,
    kv_channels=HD, ffn_hidden_size=3072,
    context_parallel_size=CP, tensor_model_parallel_size=1,
    hidden_dropout=0.0, attention_dropout=0.0,
    bf16=True, params_dtype=torch.bfloat16,
)


def p(*a):
    if RANK == 0:
        print(*a, flush=True)


def mk():
    return TEDotProductAttention(
        config=cfg, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
    ).cuda()


def run_bshd(S: int, seed: int):
    """[S, B=1, H, D] 布局。CP>1 时每 rank 只持 S/CP。"""
    torch.manual_seed(seed)
    S_local = S // CP
    q = torch.randn(S_local, 1, NH, HD, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(S_local, 1, NKV, HD, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(S_local, 1, NKV, HD, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = mk()(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.causal)
    out.sum().backward()
    return out, q.grad, k.grad, v.grad


def run_thd(seqlens, seed: int):
    """flat [T, H, D] + cu_seqlens 定段边界（varlen/packed 布局）。"""
    torch.manual_seed(seed)
    T = sum(seqlens) // CP
    cu = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    ) // CP
    cu = cu.to(torch.int32)
    q = torch.randn(T, NH, HD, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, NKV, HD, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, NKV, HD, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    psp = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(seqlens) // CP, max_seqlen_kv=max(seqlens) // CP,
    )
    out = mk()(q, k, v, attention_mask=None,
               attn_mask_type=AttnMaskType.causal, packed_seq_params=psp)
    out.sum().backward()
    return out, q.grad, k.grad, v.grad


p(f"=== TAG={TAG} CP={CP} world={WORLD} device={torch.cuda.get_device_name(0)} "
  f"cap={torch.cuda.get_device_capability(0)}")
p(f"    NVTE_FUSED_ATTN={os.environ.get('NVTE_FUSED_ATTN')} "
  f"NVTE_FLASH_ATTN={os.environ.get('NVTE_FLASH_ATTN')}")

S, SEED = 128, 1234

if CP == 1:
    # C1/C2：单段 THD vs BSHD —— 数学上必须逐值相等
    ob, dqb, dkb, dvb = run_bshd(S, SEED)
    ot, dqt, dkt, dvt = run_thd([S], SEED)
    fwd_d = (ot.view(-1).float() - ob.view(-1).float()).abs().max().item()
    p(f"\n=== C1 · THD vs BSHD forward ===\n    max_abs_diff = {fwd_d:.3e}"
      f"  {'✅ 同一算法' if fwd_d == 0 else '⚠️ 不等'}")
    p(f"\n=== C2 · THD vs BSHD backward（C1 已证同一算法，故必须也相等）===")
    for nm, a, b in (("dq", dqt, dqb), ("dk", dkt, dkb), ("dv", dvt, dvb)):
        x, y = a.float().view(-1), b.float().view(-1)
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        ratio = (x.abs().max() / y.abs().max().clamp(min=1e-12)).item()
        p(f"    {nm}: THD_max={x.abs().max():.3e} BSHD_max={y.abs().max():.3e} "
          f"ratio={ratio:.2f}× cosine={cos:.8f}{'   <-- WRONG' if cos < 0.99 else ''}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({
        "bshd_o": ob.detach().float().cpu(), "bshd_dq": dqb.float().cpu(),
        "bshd_dk": dkb.float().cpu(), "bshd_dv": dvb.float().cpu(),
        "thd_o": ot.detach().float().cpu(), "thd_dq": dqt.float().cpu(),
        "thd_dk": dkt.float().cpu(), "thd_dv": dvt.float().cpu(),
    }, f"{OUT_DIR}/probe_{TAG}.pt")
    p(f"\n  saved -> {OUT_DIR}/probe_{TAG}.pt  （两个 TAG 都跑完后用 PROBE_MODE=compare 比对 = C3）")
else:
    # C4：CP=2 下两种布局各自能否跑
    p(f"\n=== C4 · CP={CP} ===")
    for name, fn in (("BSHD(padding)", lambda: run_bshd(256, SEED)),
                     ("THD(packing)", lambda: run_thd([64, 96, 96], SEED))):
        try:
            out, dq, _, _ = fn()
            p(f"    {name:<16} PASS  out={tuple(out.shape)} "
              f"|dq|={dq.float().norm().item():.4f} finite={torch.isfinite(dq).all().item()}")
        except Exception as e:  # noqa: BLE001 — 探针要报告失败类型而非崩掉
            p(f"    {name:<16} FAIL  {type(e).__name__}: {str(e)[:160]}")

dist.barrier()
dist.destroy_process_group()
