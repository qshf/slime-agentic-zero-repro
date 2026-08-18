"""V8.1 验证：Megatron ↔ HF 权重转换的**逐值精确**往返等价 + 名字集合双向比对。

用法：
    torchrun --nproc_per_node=1 scripts/test_v8.1_megatron_to_hf.py   # TP=1
    torchrun --nproc_per_node=2 scripts/test_v8.1_megatron_to_hf.py   # TP=2（含 TP-gather）

契约：
  part (a) 往返逐值精确（本版硬门）：
        HF ckpt → load_hf_into_megatron → megatron_to_hf_state_dict → 与原始 HF 逐值比
        **max_abs_diff == 0**（全程只有 reshape/split/cat/rename，无浮点运算，故必须精确为 0）
      这比 V7 系列的等价门更强——V7 那些是数值等价（有容差），这里是**结构等价**（无容差）。
  part (b) 名字集合双向比对：转换出的名字集合 == HF ckpt 的名字集合（去掉 tie 的 lm_head）。
      双向（缺失 + 多余）能抓住"漏了某层""多吐一个名字"这类静默错误——
      源那 10 个转换文件最常见的 bug 类型正是这个。
  part (c) TP-gather 正确性（tp_size>1 才有意义）：TP=2 gather 回的全量张量与 TP=1 的一致，
      特别是 `linear_fc1` 的 **GLU 重排**（朴素 cat 会拼成 [gate0,up0,gate1,up1]，
      正确是 [gate0,gate1,up0,up1]）——这是 TP-gather 最易错的一处，故单独断言。
  part (d) 语义验证：转换出的 HF 权重能被 HF 模型正常载入并前向（不只是形状对）。

前置：5090 容器内；`NANO_MODEL_PATH` 指向 Qwen3-0.6B。默认 `--layers 4`。
"""

from __future__ import annotations

import argparse
import os
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

from toy_rl.trainer.megatron_to_hf import (
    hf_to_megatron_state_dict,
    load_hf_weights,
    megatron_to_hf_state_dict,
)
from toy_rl.trainer.megatron_trainer import MegatronTrainer

MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")


def _expected_hf_names(hf_config, num_layers: int, tie: bool) -> set[str]:
    """本次（缩减层数）配置下应当产出的 HF 名字集合。"""
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    if not tie:
        names.add("lm_head.weight")
    for i in range(num_layers):
        p = f"model.layers.{i}"
        names |= {
            f"{p}.self_attn.q_proj.weight",
            f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.self_attn.q_norm.weight",
            f"{p}.self_attn.k_norm.weight",
            f"{p}.mlp.gate_proj.weight",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.down_proj.weight",
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
        }
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    trainer = MegatronTrainer(
        MODEL_PATH,
        num_layers=args.layers,
        backend="gloo",
        load_hf_weights=True,  # part (a) 的前半程就是它
    )
    rank, tp_size = trainer.rank, trainer.tp_size
    hf_config, tie, num_layers = trainer.hf_config, trainer.tie, trainer.num_layers
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
        print(f"=== V8.1 Megatron↔HF 转换（tp={tp_size}, layers={num_layers}, tie={tie}）===")

    original = load_hf_weights(MODEL_PATH)
    # 缩减层数时只比对存在的层（等价性与层数无关）。
    expected = _expected_hf_names(hf_config, num_layers, tie)

    # --- (a) 往返：Megatron（已从 HF 载入）→ HF ---
    if rank == 0:
        print("[a] 往返逐值精确")
    converted = megatron_to_hf_state_dict(trainer.model, hf_config, tie)

    missing = sorted(n for n in expected if n not in converted)
    check(not missing, f"无遗漏参数（缺 {len(missing)}：{missing[:3]}）")

    worst_name, worst_diff = None, 0.0
    mismatched_shape = []
    for name in sorted(expected & set(converted)):
        got = converted[name].cpu()
        ref = original[name].cpu()
        if got.shape != ref.shape:
            mismatched_shape.append((name, tuple(got.shape), tuple(ref.shape)))
            continue
        d = (got.float() - ref.float()).abs().max().item()
        if d > worst_diff:
            worst_diff, worst_name = d, name

    check(not mismatched_shape, f"全部形状匹配（不匹配 {len(mismatched_shape)}：{mismatched_shape[:2]}）")
    # **硬门：精确为 0**。转换全是无损重排，任何非零都说明拆/拼/改名错了。
    check(
        worst_diff == 0.0,
        f"往返逐值精确：max_abs_diff={worst_diff:.3e} == 0（最差参数 {worst_name}）",
    )

    # --- (b) 名字集合双向比对 ---
    if rank == 0:
        print("[b] 名字集合双向")
    extra = sorted(set(converted) - expected)
    check(not extra, f"无多余参数（多 {len(extra)}：{extra[:3]}）")
    check(set(converted) == expected, f"名字集合完全一致（{len(converted)} 个）")
    if tie:
        check("lm_head.weight" not in converted, "tie=True → 不产出独立 lm_head.weight（由 embed_tokens 共享）")

    # --- (c) TP-gather：GLU 重排（fc1）---
    if rank == 0:
        print("[c] TP-gather / GLU 重排")
    ffn = hf_config.intermediate_size
    gate = converted["model.layers.0.mlp.gate_proj.weight"].cpu()
    up = converted["model.layers.0.mlp.up_proj.weight"].cpu()
    check(tuple(gate.shape) == (ffn, hf_config.hidden_size), f"gate_proj 形状 {tuple(gate.shape)}")
    check(tuple(up.shape) == (ffn, hf_config.hidden_size), f"up_proj 形状 {tuple(up.shape)}")
    check(
        torch.equal(gate, original["model.layers.0.mlp.gate_proj.weight"].cpu()),
        "gate_proj 与原始逐值相等（GLU 重排正确）",
    )
    check(
        torch.equal(up, original["model.layers.0.mlp.up_proj.weight"].cpu()),
        "up_proj 与原始逐值相等（GLU 重排正确）",
    )
    if tp_size > 1 and rank == 0:
        # 若 GLU 重排写错（朴素 cat），gate 会拿到 [gate_0, up_0] 的拼接 —— 上面两条会红。
        print("  [info] tp>1：上两条同时绿 ⇒ all_gather_param 的 fc1 chunk 重排生效")

    # --- (c2) qkv 拆分：反向映射与正向映射互逆（结构自洽）---
    own_params = {
        n.removeprefix("module."): p for n, p in trainer.model.named_parameters()
    }
    fused_ln = "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight" in own_params
    mega = hf_to_megatron_state_dict(
        original,
        hf_config,
        num_layers,
        trainer.tp_rank,
        tp_size,
        tie,
        fused_layernorm=fused_ln,
    )
    qkv_name = "decoder.layers.0.self_attention.linear_qkv.weight"
    own_qkv = own_params[qkv_name]
    check(
        tuple(mega[qkv_name].shape) == tuple(own_qkv.shape),
        f"反向映射 qkv 分片形状与模型一致 {tuple(mega[qkv_name].shape)}",
    )
    check(
        set(mega) == set(own_params) - ({"output_layer.weight"} if tie else set()),
        f"反向映射覆盖模型全部参数（转出 {len(mega)} / 模型 {len(own_params)}）",
    )

    # --- (d) 语义：转换出的权重能被 HF 模型载入并前向 ---
    if rank == 0:
        print("[d] HF 模型可载入并前向")
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        cfg.num_hidden_layers = num_layers
        hf_model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        state = {k: v.to(torch.bfloat16) for k, v in converted.items()}
        incompat = hf_model.load_state_dict(state, strict=False)
        unexpected = list(incompat.unexpected_keys)
        # tie 时 HF 的 lm_head.weight 由 embed_tokens 共享，允许出现在 missing 里。
        real_missing = [k for k in incompat.missing_keys if not (tie and k == "lm_head.weight")]
        check(not unexpected, f"无 unexpected keys（{unexpected[:3]}）")
        check(not real_missing, f"无 missing keys（{real_missing[:3]}）")

        hf_model.eval()
        ids = torch.tensor([trainer.tokenizer.encode("What is 12 * 12?")])
        with torch.no_grad():
            out = hf_model(ids).logits
        check(torch.isfinite(out).all().item(), f"HF 前向 logits finite，shape={tuple(out.shape)}")

    dist.barrier()
    if rank == 0:
        print("[V8.1] " + ("PASSED" if failed == 0 else f"FAILED ({failed})"))
    dist.barrier()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
