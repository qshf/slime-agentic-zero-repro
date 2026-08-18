"""V9.1 精度门：MegatronTrainer(TP=1, PP=1) 与 TorchActor 的 loss/grad 等价验证。

固定 5 条样本 + 同一份 HF 预训练权重，两个后端各跑一次 forward+backward（不 step），
在 HF 参数空间逐层比较梯度向量。

判据（沿用 V7.5/V8 分层度量方法论）：
  fp32 精确档：loss |Δ| < 1e-4，grad 全局 rel_L2 < 5e-3，cosine > 0.9999。
  bf16 舍入档：loss |Δ| < 5e-2，grad cosine > 0.999，rel_L2 仅 info。

fp32 档：MegatronTrainer 用 attention_backend="unfused"（FA2 拒收 fp32，同 V8.2 的 G1/G2 路径）。
bf16 档：MegatronTrainer 用 attention_backend="flash"。
TorchActor 侧 dtype 由 --fp32 标志决定；两侧必须一致。

用法（容器内，NANO_MODEL_PATH 指向 Qwen3-0.6B）：
    # fp32 精确证明
    torchrun --nproc_per_node=1 scripts/bench/v9_accuracy_gate.py --fp32 --layers 4
    # bf16 舍入验证
    torchrun --nproc_per_node=1 scripts/bench/v9_accuracy_gate.py --layers 4

为什么 --layers 4（而非全 28 层）：等价性与层数无关；缩减层数减少等待时间，
且在单卡测试中规避 OOM。v8-plan §7 / v9-plan 已拍板。
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

# TE + megatron の native library を transformers より**先に**ロードする。
# _run_torch_actor が AutoModelForCausalLM を引くと cuDNN binding が確定し、
# その後 MegatronTrainer.__init__ で megatron.core → TE の dlopen が
# `libcudnn_engines_runtime_compiled.so.9: undefined symbol` で失敗する。
# V8.0 は MegatronTrainer を先に構築するので自然に回避していた。
# nano はここで明示的に eager import して同じ効果を得る。
import transformer_engine  # noqa: F401 — side-effect: TE native lib loaded
import megatron.core  # noqa: F401 — triggers full TE import chain

import torch
import torch.distributed as dist

from toy_rl.trainer.megatron_to_hf import (
    all_gather_param,
    convert_qwen3_to_hf,
    remove_padding,
    strip_param_name_prefix,
)
from toy_rl.trainer.megatron_trainer import MegatronTrainer, _clear_nvte_backend_env

MODEL_PATH = os.environ.get("NANO_MODEL_PATH", "/home/ubuntu/models/Qwen/Qwen3-0.6B")


# ---------------------------------------------------------------------------
# 固定样本（5 条，带合成 old_log_probs 令 ratio≠1 → loss 本身也是前向函数）
# ---------------------------------------------------------------------------

def _make_samples(model_path: str, dtype: torch.dtype) -> list[dict]:
    """5 条固定 prompt+response 样本，tokenize 并合成 old_log_probs。

    合成 old_lp = cur_lp + noise * 0.1（训练前对齐，保证 ratio 在合理区间）。
    old_lp 计算用 fp32 以保证两侧一致（不受 dtype 影响）。
    与 V8.0 test_v8.0_megatron_tp.py:_make_samples 同一构造思路。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    specs = [
        ("Question: What is 12 * 12?\nAnswer:", " The answer is 144.", 1.0),
        ("Question: What is 7 + 8?\nAnswer:", " The answer is 15.", -1.0),
        ("Question: What is 9 * 6 minus 3?\nAnswer:", " It is 51 exactly.", 0.5),
        ("Question: What is 20 - 3?\nAnswer:", " 17.", 0.25),
        ("Question: What is 100 / 4?\nAnswer:", " The answer is 25.", -0.5),
    ]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 用 fp32 基准模型算一次 log_prob，作为合成 old_log_probs
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_model = ref_model.to(device)

    samples = []
    with torch.no_grad():
        for prompt, response, reward in specs:
            full = prompt + response
            ids = tok.encode(full, add_special_tokens=True)
            p_len = len(tok.encode(prompt, add_special_tokens=True))
            # loss_mask：仅 response 部分
            loss_mask = [0] * p_len + [1] * (len(ids) - p_len)
            # 合成 old_log_probs：用 fp32 参考模型算 cur_lp，再加小扰动
            t = torch.tensor([ids], device=device)
            logits = ref_model(t).logits[0, :-1, :].float()
            tgt = torch.tensor(ids[1:], device=device)
            lp = torch.log_softmax(logits, dim=-1)[range(len(tgt)), tgt]
            noise = torch.randn_like(lp) * 0.1
            old_lp = (lp + noise).tolist()
            # 首位补 0.0（tokens[0] 无前驱，与 loss_mask[0]=0 保持一致）
            old_lp = [0.0] + old_lp
            samples.append({
                "tokens": ids,
                "loss_mask": loss_mask,
                "reward": reward,
                "old_log_probs": old_lp,
            })

    del ref_model
    torch.cuda.empty_cache()
    return samples


# ---------------------------------------------------------------------------
# TorchActor 侧梯度（HF 参数空间，fp32 或 bf16）
# ---------------------------------------------------------------------------

def _run_torch_actor(samples: list[dict], model_path: str, num_layers: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """TorchActor 前向+反向，返回 HF 参数名 → grad（cpu, float32）。

    偏离 TorchActor 原始接口：直接操作模型而不通过 Args，避免构造完整 Args 对象。
    语义等价：同一 HF 模型、同一 GRPO loss 数学。
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_cfg.num_hidden_layers = num_layers  # 缩减层数，与 MegatronTrainer 端对齐

    model = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=dtype).to(device)
    # 用 HF from_pretrained 加载部分权重（只取前 num_layers 层 + embedding + norm + lm_head）
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True, ignore_mismatched_sizes=True
    )
    sd_full = full_model.state_dict()
    sd_partial = {}
    for k, v in sd_full.items():
        if "layers." in k:
            idx = int(k.split("layers.")[1].split(".")[0])
            if idx < num_layers:
                sd_partial[k] = v
        else:
            sd_partial[k] = v
    model.load_state_dict(sd_partial, strict=False)
    del full_model
    torch.cuda.empty_cache()
    model.train()

    pad_id = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)

    # 批量 padding（与 TorchActor._pad_batch 相同逻辑）
    keep = [s for s in samples if len(s["tokens"]) >= 2 and sum(s["loss_mask"]) > 0]
    max_len = max(len(s["tokens"]) for s in keep)

    def _pad(vals, length, fill):
        return vals + [fill] * (length - len(vals))

    input_ids = torch.tensor([_pad(s["tokens"], max_len, pad_id) for s in keep], device=device)
    attn_mask = torch.tensor(
        [_pad([1] * len(s["tokens"]), max_len, 0) for s in keep],
        dtype=torch.long, device=device
    )
    tgt_mask = torch.tensor(
        [_pad(s["loss_mask"][1:], max_len - 1, 0) for s in keep],
        dtype=torch.float32, device=device
    )
    old_lp = torch.tensor(
        [_pad(s["old_log_probs"][1:], max_len - 1, 0.0) for s in keep],
        dtype=dtype, device=device
    )
    adv = torch.tensor([[s["reward"]] for s in keep], dtype=dtype, device=device)

    logits = model(input_ids, attention_mask=attn_mask).logits[:, :-1, :].float()
    targets = input_ids[:, 1:].unsqueeze(-1)
    cur_lp = torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)
    ratio = (cur_lp - old_lp.float()).exp()
    eps = 0.2
    pg1 = ratio * adv.float()
    pg2 = ratio.clamp(1 - eps, 1 + eps) * adv.float()
    pg_loss = torch.min(pg1, pg2)
    per_sample = (pg_loss * tgt_mask).sum(dim=1) / torch.clamp_min(tgt_mask.sum(dim=1), 1.0)
    loss = -per_sample.mean()
    loss.backward()

    grads = {n: p.grad.detach().cpu().float().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return grads, float(loss.detach())


# ---------------------------------------------------------------------------
# MegatronTrainer 侧梯度（Megatron 参数 → gather → HF 名空间）
# ---------------------------------------------------------------------------

def _run_megatron(samples: list[dict], model_path: str, num_layers: int,
                  dtype: torch.dtype, attention_backend: str) -> tuple[dict[str, torch.Tensor], float]:
    """MegatronTrainer 前向+反向，梯度 gather 成全量后转为 HF 名空间。

    对齐 test_v8.0_megatron_tp.py 的 dump 范式：TP-gather + convert_qwen3_to_hf。
    """
    _clear_nvte_backend_env()
    trainer = MegatronTrainer(
        model_path=model_path,
        lr=1e-6,
        eps_clip=0.2,
        eps_clip_high=0.2,
        num_layers=num_layers,
        backend="gloo",          # 单卡场景（V8 精度门 gloo 惯例）
        load_hf_weights=True,
        params_dtype=dtype,
        attention_backend=attention_backend,
    )
    result = trainer.train_batch(samples, global_batch_size=len(samples))

    # 梯度以 Megatron 分片形式存在 → 按 HF 名空间重建全量梯度 dict
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_cfg.num_hidden_layers = num_layers

    grads_hf: dict[str, torch.Tensor] = {}
    for raw_name, param in trainer.model.named_parameters():
        if param.grad is None:
            continue
        full_grad = all_gather_param(raw_name, torch.nn.Parameter(param.grad.detach()))
        full_grad = remove_padding(raw_name, full_grad, hf_cfg.vocab_size)
        hf_pairs = convert_qwen3_to_hf(hf_cfg, raw_name, full_grad.float())
        for hf_name, tensor in hf_pairs:
            grads_hf[hf_name] = tensor.cpu()

    dist.destroy_process_group()
    return grads_hf, float(result["loss"])


# ---------------------------------------------------------------------------
# 度量（rel_L2 + cosine，沿用 V7.5/V8 方法论）
# ---------------------------------------------------------------------------

def _compare_grads(grads_ref: dict[str, torch.Tensor],
                   grads_test: dict[str, torch.Tensor]) -> dict:
    """两个 HF 名空间梯度 dict 的全局 rel_L2 + cosine（向量级，大信号主导）。

    逐参数比对并 flatten 成两个大向量做全局度量，防止小信号参数（k_proj 等）
    把逐参数 max 虚高。与 V7.5 / V8 测试用同一方法。
    """
    common = sorted(set(grads_ref) & set(grads_test))
    missing_ref = sorted(set(grads_test) - set(grads_ref))
    missing_test = sorted(set(grads_ref) - set(grads_test))

    vecs_ref, vecs_test = [], []
    per_param = {}
    for name in common:
        r = grads_ref[name].float().flatten()
        t = grads_test[name].float().flatten()
        if r.shape != t.shape:
            per_param[name] = {"shape_mismatch": (tuple(r.shape), tuple(t.shape))}
            continue
        diff = (r - t).norm().item()
        denom = r.norm().item() + 1e-12
        per_param[name] = {"rel_L2": diff / denom, "cosine": float(torch.nn.functional.cosine_similarity(r.unsqueeze(0), t.unsqueeze(0)).item())}
        vecs_ref.append(r)
        vecs_test.append(t)

    if vecs_ref:
        cat_ref = torch.cat(vecs_ref)
        cat_test = torch.cat(vecs_test)
        global_rel_l2 = float((cat_ref - cat_test).norm() / (cat_ref.norm() + 1e-12))
        global_cosine = float(torch.nn.functional.cosine_similarity(cat_ref.unsqueeze(0), cat_test.unsqueeze(0)).item())
    else:
        global_rel_l2 = float("nan")
        global_cosine = float("nan")

    return {
        "global_rel_L2": global_rel_l2,
        "global_cosine": global_cosine,
        "per_param": per_param,
        "missing_ref": missing_ref,
        "missing_test": missing_test,
    }


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V9.1 accuracy gate: MegatronTrainer vs TorchActor")
    parser.add_argument("--layers", type=int, default=4,
                        help="层数（缩减加速；等价性与层数无关）")
    parser.add_argument("--fp32", action="store_true",
                        help="fp32 精确证明（unfused 后端；FA2 拒收 fp32）")
    args = parser.parse_args()

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    attn_backend = "unfused" if args.fp32 else "flash"
    dtype_label = "fp32" if args.fp32 else "bf16"

    print(f"\n{'='*60}")
    print(f"V9.1 精度门  dtype={dtype_label}  layers={args.layers}  backend={attn_backend}")
    print(f"{'='*60}")

    # 1. 固定样本
    print("\n[1/3] 生成固定 5 条样本（fp32 参考模型计算 old_log_probs）...")
    samples = _make_samples(MODEL_PATH, dtype)
    print(f"      样本长度: {[len(s['tokens']) for s in samples]}")

    # 2. TorchActor 梯度
    print("\n[2/3] TorchActor 前向+反向...")
    grads_torch, loss_torch = _run_torch_actor(samples, MODEL_PATH, args.layers, dtype)
    print(f"      loss = {loss_torch:.6f}  参数数 = {len(grads_torch)}")

    # 3. MegatronTrainer 梯度
    print("\n[3/3] MegatronTrainer(TP=1,PP=1) 前向+反向...")
    grads_megatron, loss_megatron = _run_megatron(samples, MODEL_PATH, args.layers, dtype, attn_backend)
    print(f"      loss = {loss_megatron:.6f}  参数数 = {len(grads_megatron)}")

    # 4. 比对
    print("\n[比对]")
    loss_delta = abs(loss_torch - loss_megatron)
    cmp = _compare_grads(grads_torch, grads_megatron)
    global_rel_l2 = cmp["global_rel_L2"]
    global_cosine = cmp["global_cosine"]

    print(f"  loss |Δ|        = {loss_delta:.2e}  (TorchActor={loss_torch:.6f}, Megatron={loss_megatron:.6f})")
    print(f"  grad global rel_L2 = {global_rel_l2:.2e}")
    print(f"  grad global cosine = {global_cosine:.6f}")

    if cmp["missing_ref"]:
        print(f"  [WARN] Megatron 有但 TorchActor 没有的参数: {cmp['missing_ref'][:5]}")
    if cmp["missing_test"]:
        print(f"  [WARN] TorchActor 有但 Megatron 没有的参数: {cmp['missing_test'][:5]}")

    # 5. 判据（分层：fp32 精确证明 vs bf16 舍入）
    print(f"\n[判据 {dtype_label}]")
    if args.fp32:
        loss_ok   = loss_delta < 1e-4
        cosine_ok = global_cosine > 0.9999
        rl2_ok    = global_rel_l2 < 5e-3
        print(f"  loss |Δ| < 1e-4    : {'✓ PASS' if loss_ok else '✗ FAIL'}  ({loss_delta:.2e})")
        print(f"  grad cosine > 0.9999: {'✓ PASS' if cosine_ok else '✗ FAIL'}  ({global_cosine:.6f})")
        print(f"  grad rel_L2 < 5e-3  : {'✓ PASS' if rl2_ok else '✗ FAIL'}  ({global_rel_l2:.2e})")
        passed = loss_ok and cosine_ok and rl2_ok
    else:
        loss_ok   = loss_delta < 5e-2
        cosine_ok = global_cosine > 0.999
        print(f"  loss |Δ| < 5e-2    : {'✓ PASS' if loss_ok else '✗ FAIL'}  ({loss_delta:.2e})")
        print(f"  grad cosine > 0.999 : {'✓ PASS' if cosine_ok else '✗ FAIL'}  ({global_cosine:.6f})")
        print(f"  grad rel_L2 (info)  : {global_rel_l2:.2e}")
        passed = loss_ok and cosine_ok

    print(f"\n{'='*60}")
    print(f"  总结: {'✓ 全部通过' if passed else '✗ 有失败项'}")
    print(f"{'='*60}\n")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
