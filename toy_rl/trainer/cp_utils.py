"""V8.2: CP（context parallel，上下文并行）的序列切分 —— 对齐源 megatron_utils/cp_utils.py。

**CP 切的是序列长度维**（TP 切层内矩阵、PP 切层的列表、CP 切 token 的序列）。
每个 CP rank 只持有序列的一部分 token，attention 内部用 ring / all-gather 交换 KV，
于是长序列不再受单卡显存限制。

**本模块唯一的精华 = 2-chunk 对称切分（源 cp_utils.py:29-31 / :214-216）**：
每个 CP rank 拿**两块**——一块靠前、一块靠后（关于序列中点对称）。为什么不朴素均分：
因果 attention 下靠后的 token 要 attend 前面所有 token，工作量随位置**线性增长**，
朴素均分会让最后一个 rank 干最多活、其余空等（负载严重倾斜）。
2-chunk 对称让每个 rank 的 (前块轻 + 后块重) 之和相等。

这个切法不是 nano 的自由选择：**mcore 的 RoPE（`get_pos_emb_on_this_cp_rank`）与 TE 的
ring attention 内部都假定了同一套切法**，token 切错就会与位置编码/KV 交换对不上，
且错得**静默**（形状全对、loss 有限、只是数值错）。故这里逐字对齐源。

偏离源项目（登记见 docs/decisions/v8.2.md）：
  - **不实现 `get_logits_and_tokens_offset_with_cp`**：源用它把「response 段的 log_prob」
    从 CP 分片里切出来（因为源的 loss 只算 response 段，需要在全局坐标里定位）。
    nano 改用**等价但更简单的坐标系**——先把 targets / target_mask / old_log_probs 都
    物化成与 token 流**等长**的「full」数组（位置 g 表示 logits[g] 预测 tokens[g+1]），
    再用**同一个** `slice_with_cp` 切，于是三者天然对齐、无需偏移量运算。
    语义等价（源的 offset 就是在算这套对齐），且少一处易错的下标算术。
  - **不实现 `get_sum_of_sample_mean` 的 CP 分支**：它的实质是「分子取本 rank 的部分和、
    分母取**全量** loss_mask.sum()」，nano 直接把 full 分母传进 `_grpo_sample_loss`
    （见 megatron_trainer.py），一行等价。
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# **仅供验收的 negative control 开关**（`NANO_CP_NAIVE_SPLIT=1`）：把 2-chunk 对称切分
# 换成朴素连续切分（形状完全相同、只是取的 token 不同）。
#
# 为什么需要它：CP 的正确性只能在 **bf16** 下验（FA2 内核拒收 fp32，实测
# `No dot product attention backend is available`），而 bf16 的舍入噪声让"数值接近"
# 这件事失去判别力 —— 与 V7.6 遇到的是同一个问题，解法也照搬 V7.6：
# **先证明这套度量能把"故意写错"判红**，"写对时判绿"才有意义。
# 朴素切分是最贴切的反例：mcore 的 RoPE（`get_pos_emb_on_this_cp_rank`）与 TE 的
# ring attention **内部恒按 2-chunk 对称**取，token 换成朴素切分就与它们对不上，
# 而形状、dtype、cu_seqlens 全都合法 —— 正是那种"跑得通但算错"的静默错误。
_NAIVE_SPLIT = os.environ.get("NANO_CP_NAIVE_SPLIT", "0") == "1"


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: float,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """把一条完整序列切成本 CP rank 的那部分（2-chunk 对称）。对齐源 cp_utils.py:175。

    - `cp_size == 1`：thd 原样返回；bshd 右 pad 到 `max_seq_len`（源同样在这里统一 padding）。
    - `cp_size > 1`：先把长度 pad 到 `2 * cp_size * chunk_size`，再取
      `[chunk_size*cp_rank : chunk_size*(cp_rank+1)]` 与
      `[chunk_size*(2*cp_size-cp_rank-1) : chunk_size*(2*cp_size-cp_rank)]` 拼起来。

    chunk_size 的来源随布局不同（源 cp_utils.py:204-207）：
      - thd：按**这条序列自己**的长度分（每条序列独立打包进 flat 流）
      - bshd：按**整批的 max_seq_len** 分（一行一条序列，各行必须等长）
    """
    from megatron.core import mpu

    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    if qkv_format == "bshd":
        assert max_seq_len is not None, "bshd 必须给 max_seq_len（各行要等长）"

    def pad_tokens(x: torch.Tensor, pad: int) -> torch.Tensor:
        return F.pad(x, (0, pad), value=pad_value) if pad > 0 else x

    if cp_size == 1:
        if qkv_format == "bshd":
            return pad_tokens(tokens, max_seq_len - tokens.size(0))
        return tokens

    token_len = tokens.size(0)
    if qkv_format == "thd":
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    tokens = pad_tokens(tokens, 2 * cp_size * chunk_size - token_len)

    if _NAIVE_SPLIT:  # negative control，见模块顶部说明；正常路径永远走不到这里
        return tokens[2 * chunk_size * cp_rank : 2 * chunk_size * (cp_rank + 1)]

    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def cp_padded_length(token_len: int, cp_size: int) -> int:
    """一条长度 `token_len` 的序列在 thd + CP 下被 pad 到的**全局**长度。

    与 `slice_with_cp` 的 thd 分支同一个 chunk_size 公式——单独抽出来是因为
    `PackedSeqParams.cu_seqlens` 要的是**原始（全局）长度**而不是本 rank 的分片长度
    （源 data.py:112 那句 `cu_seqlens = ... * cp_size` 正是在做这件事）。
    """
    if cp_size == 1:
        return token_len
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    return 2 * cp_size * chunk_size
