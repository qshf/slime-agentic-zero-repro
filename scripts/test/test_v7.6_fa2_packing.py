"""V7.6 验证：FA2 varlen packing 与 padding/mask 路径等价（退休偏离 #7）。

    # 容器内需先 pip uninstall -y flash-attn-4 && pip install --no-deps <FA2 wheel> && pip install accelerate
    torchrun --nproc_per_node=1 scripts/test_v7.6_fa2_packing.py

契约（三部分）：
  part (a) dispatch：确认 transformers 真把 pack 判成 packed 并调到 flash_attn_varlen_func，
      且反推出的 cu_seqlens/max_seqlen 与 pack_sequences 的手工值逐值一致。
      —— 这是"varlen 真的生效"的直接证据（而非"没报错所以大概生效了"）。
  part (b) negative control：**本版验收的判别力来源**。同一度量下，正例（reset position_ids）
      与反例（单调 position_ids → varlen 不启用 → 段间串扰）必须相差数量级。
      **反例若"通过"等价断言，说明度量无判别力，本轮验收作废。**
  part (c) 等价：fa2 packing vs padding、fa2 packing vs mask packing 的 loss/grad 一致。

为什么不像 V7.5 那样用 fp32 硬门（重要）：
  **FA2 内核只接受 fp16/bf16**（实测 `RuntimeError: FlashAttention only support fp16 and bf16
  data type`），故 V7.5 那套"fp32 逐值精确"证明在 fa2 路径**物理不可达**。bf16 下 28 层前向的
  舍入累积会让 grad magnitude rel_L2 到十几 %（V7.5 已确认这是舍入非 bug），单看误差大小无法
  区分"隔离正确"与"隔离失效"。故改用 negative control：不证明误差足够小，而证明**度量能把
  正确与泄漏分开**（实测差 ~140×）。

证据边界（诚实声明，勿写过头）：
  - **fp32 逐值精确证明只属于 "mask" 路径**（V7.5：loss 2.4e-7 / cosine 0.99999999）。
  - **fa2 路径自身的证据 = negative control + 与 padding/mask 路径在 bf16 下一致**，
    不宣称继承 mask 路径的 fp32 精确性——两条路径走不同内核，bf16 一致不等于 fa2 也被 fp32 验证过。

world=1（去 sharding-reduce 噪声，隔离 packing 正确性本身）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from toy_rl.trainer.data_packing import pack_sequences
from toy_rl.trainer.fsdp_trainer import FSDPTrainer

MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")

# 正例门（bf16）：与 padding/mask 路径的一致性。
LOSS_TOL = 5e-2
COS_MIN = 0.99
# 反例门：negative control 必须**显著超出**正例门才算度量有判别力。
LEAK_REL_L2_MIN = 0.5
LEAK_COS_MAX = 0.9


def _make_samples(trainer: FSDPTrainer) -> list[dict]:
    """4 条不同长度的真样本（长度不齐才能暴露 padding 与 packing 的差异）。与 V7.5 同一批。"""
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
        # 合成 old_log_probs → ratio≠1 → loss 值也依赖 logits（否则 loss 是常数、只有 grad 测前向）。
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


def _grad_compare(a: dict, b: dict) -> tuple[float, float]:
    """两份梯度的全局相对 L2 + cosine（度量理由见 V7.5：逐参数 max-rel 会被小信号参数放大）。"""
    diff_sq = ref_sq = dot = an_sq = bn_sq = 0.0
    for name, ga in a.items():
        gb = b.get(name)
        if gb is None:
            continue
        diff_sq += ((ga - gb) ** 2).sum().item()
        ref_sq += (ga ** 2).sum().item()
        dot += (ga * gb).sum().item()
        an_sq += (ga ** 2).sum().item()
        bn_sq += (gb ** 2).sum().item()
    rel_l2 = (diff_sq ** 0.5) / (ref_sq ** 0.5 + 1e-12)
    cosine = dot / ((an_sq ** 0.5) * (bn_sq ** 0.5) + 1e-12)
    return rel_l2, cosine


def _run(trainer: FSDPTrainer, mode, samples: list[dict]) -> tuple[float, dict]:
    """按指定 packing 模式跑一次 backward，返回 (loss, grad 快照)。"""
    prev = trainer.packing_mode
    trainer.packing_mode = mode
    trainer.optimizer.zero_grad(set_to_none=True)
    gbs = len(samples) * trainer.dp_size
    try:
        if mode is None:
            loss, _ = trainer._microbatch_backward(samples, gbs)
        else:
            loss, _ = trainer._packed_backward(samples, gbs)
        grads = _grad_snapshot(trainer)
    finally:
        trainer.packing_mode = prev
        trainer.optimizer.zero_grad(set_to_none=True)
    return loss, grads


def _test_dispatch(trainer: FSDPTrainer, samples: list[dict], rank: int) -> int:
    """part (a)：varlen 真被调用，且 cu_seqlens 与手工值一致。"""
    failed = 0

    def check(cond, msg):
        nonlocal failed
        if not cond:
            failed += 1
            if rank == 0:
                print(f"  FAIL: {msg}")
        elif rank == 0:
            print(f"  OK: {msg}")

    import transformers.modeling_flash_attention_utils as mfu

    pack = pack_sequences(samples)
    expected_cu = pack["cu_seqlens"]
    expected_max = max(
        expected_cu[i + 1] - expected_cu[i] for i in range(len(expected_cu) - 1)
    )

    # 先跑一次触发 transformers 的 lazy import（_flash_varlen_fn 在首次前向后才绑定）。
    _run(trainer, "fa2", samples)

    seen: list[dict] = []
    orig = mfu._flash_varlen_fn

    def spy(*args, **kwargs):
        seen.append(
            {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in kwargs.items()
             if "cu_seqlens" in k or "max_seqlen" in k}
        )
        return orig(*args, **kwargs)

    mfu._flash_varlen_fn = spy
    try:
        _run(trainer, "fa2", samples)
    finally:
        mfu._flash_varlen_fn = orig

    check(len(seen) > 0, f"flash_attn_varlen_func 被调用（{len(seen)} 次 = 每层一次）")
    if seen:
        got = seen[0]
        check(got.get("cu_seqlens_q") == expected_cu,
              f"cu_seqlens 与 pack_sequences 手工值一致：{got.get('cu_seqlens_q')} == {expected_cu}")
        check(got.get("max_seqlen_q") == expected_max,
              f"max_seqlen 一致：{got.get('max_seqlen_q')} == {expected_max}")
    return failed


def _test_negative_control(trainer: FSDPTrainer, samples: list[dict], rank: int) -> int:
    """part (b)：度量判别力——正例 vs 反例必须差数量级。**本版验收的根本。**

    反例构造：把每段 reset 的 position_ids 换成全局单调递增。transformers 的
    `_is_packed_sequence` 因此返回 False → varlen 不启用 → attention_mask=None 变成
    满因果注意力 → 段间真串扰。这正是 fa2 路径最危险的静默失败模式。
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

    ref_loss, ref_grads = _run(trainer, None, samples)          # padding = 参照
    ok_loss, ok_grads = _run(trainer, "fa2", samples)           # 正例

    # 反例：monkey-patch pack_sequences 的产物，让 position_ids 单调（其余一切不变）。
    import toy_rl.trainer.data_packing as dp

    orig_pack = dp.pack_sequences

    def leaky_pack(s):
        pk = orig_pack(s)
        if pk is not None:
            pk["position_ids"] = list(range(len(pk["tokens"])))  # 全局单调 → 不再被判成 packed
        return pk

    dp.pack_sequences = leaky_pack
    try:
        # 反例会让 _assert_varlen_will_engage 触发（这本身就是好事——生产路径拦住了泄漏）。
        # 为了**测量**泄漏的数值幅度，这里临时绕过该断言，直接验证度量的判别力。
        import toy_rl.trainer.fsdp_trainer as ft

        orig_assert = ft._assert_varlen_will_engage
        ft._assert_varlen_will_engage = lambda *a, **k: None
        try:
            leak_loss, leak_grads = _run(trainer, "fa2", samples)
        finally:
            ft._assert_varlen_will_engage = orig_assert
    finally:
        dp.pack_sequences = orig_pack

    ok_rel, ok_cos = _grad_compare(ref_grads, ok_grads)
    leak_rel, leak_cos = _grad_compare(ref_grads, leak_grads)

    if rank == 0:
        print(f"  [info] 正例 vs padding：loss|Δ|={abs(ref_loss - ok_loss):.3e} "
              f"rel_L2={ok_rel:.3e} cosine={ok_cos:.8f}")
        print(f"  [info] 反例 vs padding：loss|Δ|={abs(ref_loss - leak_loss):.3e} "
              f"rel_L2={leak_rel:.3e} cosine={leak_cos:.8f}")

    # 反例必须被度量判为「不等价」——否则度量无判别力，整轮验收作废。
    check(leak_rel > LEAK_REL_L2_MIN and leak_cos < LEAK_COS_MAX,
          f"negative control 被正确判为泄漏（rel_L2={leak_rel:.3e}>{LEAK_REL_L2_MIN} "
          f"且 cosine={leak_cos:.4f}<{LEAK_COS_MAX}）—— 度量有判别力")
    # 判别力 = 反例与正例的差距倍数（报出来，供 v7.6.md 记录）。
    if rank == 0 and ok_rel > 0:
        print(f"  [info] 判别力：反例/正例 rel_L2 = {leak_rel / ok_rel:.1f}×")
    return failed


def _test_equivalence(trainer: FSDPTrainer, samples: list[dict], rank: int) -> int:
    """part (c)：fa2 vs padding、fa2 vs mask 的 loss/grad 一致（bf16 门 + cosine）。"""
    failed = 0

    def check(cond, msg):
        nonlocal failed
        if not cond:
            failed += 1
            if rank == 0:
                print(f"  FAIL: {msg}")
        elif rank == 0:
            print(f"  OK: {msg}")

    pad_loss, pad_grads = _run(trainer, None, samples)
    fa2_loss, fa2_grads = _run(trainer, "fa2", samples)

    rel, cos = _grad_compare(pad_grads, fa2_grads)
    check(abs(pad_loss - fa2_loss) < LOSS_TOL,
          f"fa2 vs padding loss：|{pad_loss:.6f}-{fa2_loss:.6f}|={abs(pad_loss - fa2_loss):.3e} < {LOSS_TOL}")
    check(cos > COS_MIN, f"fa2 vs padding grad cosine={cos:.8f} > {COS_MIN}（rel_L2={rel:.3e} 仅 info）")

    # fa2 vs mask：与"已被 fp32 精确验证过"的 V7.5 路径对照。注意 mask 路径要求 eager/sdpa 后端，
    # 而本 trainer 是 FA2 后端 → 该对照需另起 trainer，成本高；此处只在同后端可跑时比较。
    attn = getattr(trainer.model.config, "_attn_implementation", "")
    if attn in ("eager", "sdpa"):
        mask_loss, mask_grads = _run(trainer, "mask", samples)
        m_rel, m_cos = _grad_compare(mask_grads, fa2_grads)
        check(abs(mask_loss - fa2_loss) < LOSS_TOL, f"fa2 vs mask loss diff={abs(mask_loss - fa2_loss):.3e}")
        check(m_cos > COS_MIN, f"fa2 vs mask grad cosine={m_cos:.8f} > {COS_MIN}")
    elif rank == 0:
        print("  [skip] fa2 vs mask 对照：当前后端为 FA2，mask 路径需 eager/sdpa "
              "（用 scripts/test_v7.5_packing.py 单独验 mask 路径的 fp32 精确性）")
    return failed


def main() -> int:
    trainer = FSDPTrainer(
        model_path=MODEL_PATH,
        lr=1e-4,
        global_batch_size=4,
        fp16=False,
        attn_implementation="flash_attention_2",  # fa2 路径的前提（_packed_backward 有硬断言）
    )
    rank = trainer.rank
    if rank == 0:
        try:
            import flash_attn
            ver = getattr(flash_attn, "__version__", "<无 __version__：可能是 FA4，需卸载>")
        except ImportError:
            ver = "<未安装>"
        print(f"=== V7.6 FA2 varlen packing（flash_attn={ver}，bf16 + negative control）===")

    samples = _make_samples(trainer)
    failed = 0
    if rank == 0:
        print("[a] varlen dispatch")
    failed += _test_dispatch(trainer, samples, rank)
    if rank == 0:
        print("[b] negative control（度量判别力）")
    failed += _test_negative_control(trainer, samples, rank)
    if rank == 0:
        print("[c] 等价性")
    failed += _test_equivalence(trainer, samples, rank)

    if rank == 0:
        print(f"[V7.6] {'PASSED' if failed == 0 else f'FAILED ({failed})'}")

    dist.barrier()
    dist.destroy_process_group()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
