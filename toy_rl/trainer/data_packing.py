"""V7.5: sequence packing —— 把多条序列 flat 成 1D 单微批 + 块对角因果 mask 显式隔离。

对齐源 slime/backends/fsdp_utils/data_packing.py（pack_sequences）：
  - flat 1D concat：tokens / loss_masks 拼成一条长序列
  - position_ids **per-sample reset**（每段从 0 重数，RoPE 据此定位）
  - cu_seqlens：各段边界的前缀和（int32），既标边界也供隔离

**与源的关键偏离（本版核心，登记见 docs/decisions/v7.5.md 偏离 #7）**：
  源把 `attention_mask=None` 交给 model，靠 **flash-attn varlen 内核**读 position_ids/cu_seqlens
  在内核里隐式隔离样本（跨段不计算，O(T) 块级稀疏）。5090 是 Blackwell sm_120，
  **既无 flash-attn 也无 TransformerEngine** → 照搬 `attention_mask=None` 会让整个 pack 走满
  因果注意力、**样本间静默串扰、loss 算错**。故 nano 用 `build_block_diagonal_causal_mask`
  **显式物化** T×T 块对角因果 mask（infra 文档承认这是源 varlen 路径的 "conceptual
  equivalent"，只是 varlen 内核不物化它）。语义（隔离）等价，吞吐部分等价——消除 padding 浪费，
  但不得 varlen 的块级 compute 稀疏（付 O(T²) mask+compute），装 flash-attn 后可换 varlen 内核。

nano-minimal（相对源的简化，均为吞吐/无关简化、无正确性影响）：
  - k_partitions=1（一个 train_batch 打成一个 pack）：砍源的 get_seqlen_balanced_partitions
    多 pack 均衡（那是多微批吞吐优化，nano 单 pack 已够演示隔离正确性）。
  - 砍 multimodal / returns / raw_rewards：nano GRPO 只用 tokens/loss_mask/reward/old_log_probs。
"""

from __future__ import annotations

from typing import Optional

import torch


def pack_sequences(samples: list[dict]) -> Optional[dict]:
    """多条样本 → 一个 flat pack dict。对齐源 data_packing.py:63-100 的 flat concat。

    Args:
        samples: [{tokens, loss_mask, reward, old_log_probs?}, ...]（本 rank 本地样本）。

    Returns:
        pack dict（见下），或 None（无可训练样本时——与 _microbatch_backward 的空返回对称）。
        {
          "tokens":       list[int]  长 T（flat concat）
          "position_ids": list[int]  长 T（每段 reset 到 range(len)）
          "cu_seqlens":   list[int]  长 num+1（前缀和边界，int）
          "loss_masks":   list[int]  长 T（flat concat，单数 loss_mask → 复数 flat）
          "rewards":      list[float] 长 num（逐段标量，GRPO 逐序列信用）
          "old_log_probs":list[float]|None 长 T（仅当所有段都有；否则 None 走 on-policy）
        }

    过滤：打包前先按训练器 keep 谓词 `len>=2 and sum(loss_mask)>0` 过滤退化样本
    （对齐 _microbatch_backward:151-153），保证 padding 路径与 packing 路径见**相同样本**。
    """
    trainable = [s for s in samples if len(s["tokens"]) >= 2 and sum(s["loss_mask"]) > 0]
    if not trainable:
        return None

    flat_tokens: list[int] = []
    flat_position_ids: list[int] = []
    flat_masks: list[int] = []
    cu_seqlens: list[int] = [0]
    rewards: list[float] = []
    # old_log_probs 仅在**所有**段都带时才 flat（否则退化 on-policy，与源 use_rollout_logprobs 一致）。
    all_have_lp = all(s.get("old_log_probs") for s in trainable)
    flat_old_lp: list[float] = [] if all_have_lp else None  # type: ignore[assignment]

    for s in trainable:
        seq_tokens = s["tokens"]
        flat_tokens.extend(seq_tokens)
        flat_position_ids.extend(range(len(seq_tokens)))  # 每段从 0 重数（对齐源 seq_positionids）
        flat_masks.extend(s["loss_mask"])
        cu_seqlens.append(cu_seqlens[-1] + len(seq_tokens))
        rewards.append(float(s["reward"]))
        if all_have_lp:
            flat_old_lp.extend(s["old_log_probs"])

    return {
        "tokens": flat_tokens,
        "position_ids": flat_position_ids,
        "cu_seqlens": cu_seqlens,
        "loss_masks": flat_masks,
        "rewards": rewards,
        "old_log_probs": flat_old_lp,
    }


def build_block_diagonal_causal_mask(
    cu_seqlens: list[int],
    total_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """从 cu_seqlens 物化 [1,1,T,T] 块对角因果加性 mask。对齐 infra 文档的 "conceptual equivalent"。

    位置 (i,j) 可注意 ⟺ **同段**（同一 cu_seqlens 桶）**且 j<=i**（因果）。
    可注意 → 0.0；否则 → finfo(dtype).min。

    约定（transformers 5.x 源级确认，见 v7.5.md）：
      - HF `create_causal_mask` → `_preprocess_mask_arguments` 对 4D mask **early-exit 原样返回**，
        不重推因果、不 re-expand 2D → 我们的显式 mask 被完整 honor。
      - eager/sdpa 后端把它当**加性 bias**（attn_weights + mask）。故用**加性 float**：
        0.0=attend / finfo.min=blocked（**不用 -inf**：全屏蔽行 softmax 会 NaN；finfo.min 安全）。
      - dtype 须匹配激活（FSDP2 MixedPrecision param_dtype=bf16 → 激活 bf16 → mask bf16）。

    O(T²) 显存：bf16 下 2·T² 字节（T=3k≈18MB），nano 规模无忧；这是"显式 mask vs varlen O(T)"
    的吞吐代价（偏离 #7）。
    """
    cu = torch.tensor(cu_seqlens[1:], device=device, dtype=torch.long)
    pos = torch.arange(total_tokens, device=device)
    # 每个 flat 位置属于哪一段：searchsorted(边界, pos, right=True)。
    seg = torch.searchsorted(cu, pos, right=True)  # [T]

    same_seg = seg[:, None] == seg[None, :]                      # [T,T] 同段
    causal = pos[None, :] <= pos[:, None]                         # [T,T] j<=i
    attend = same_seg & causal
    mask = torch.where(
        attend,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype),
    )
    return mask[None, None]  # [1,1,T,T]
