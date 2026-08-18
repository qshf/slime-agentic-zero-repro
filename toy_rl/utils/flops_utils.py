"""V9.0: FLOPs 计算工具 —— 对齐源 slime/utils/flops_utils.py。

偏离（docs/decisions/v9.md 偏离表）：
  - 源支持 MoE（DeepSeek/Qwen-MoE）和 LoRA（q_lora_rank/kv_lora_rank）；nano 只支持 dense 无
    LoRA（Qwen3-0.6B 是 dense 模型，MoE/LoRA 分支无对象）。逻辑等价：dense 是源的 num_experts=None
    特例，公式逐字对齐。
  - 源接 Megatron args namespace（有 kv_channels / ffn_hidden_size / num_layers 等字段）；nano
    接 HF AutoConfig（字段名不同：head_dim 或 hidden_size÷num_attention_heads、intermediate_size、
    num_hidden_layers）。函数内部统一换算，调用方直接传 HF config 即可。
  - 省略 calculate_embedding_flops（nano V9 不单独计 embedding 前向，embedding 不含乘加）。
"""

from __future__ import annotations


def _head_dim(config) -> int:
    """从 HF config 推导每个注意力头的维度（kv_channels）。

    HF config 有两种写法：显式 head_dim（Qwen3 系列有）或 hidden_size / num_attention_heads。
    """
    if hasattr(config, "head_dim") and config.head_dim:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def calculate_qkv_projection_flops(seqlen: int, config) -> int:
    """QKV 投影的 FLOPs。对齐源 :9-32（dense，无 LoRA）。

    Q 投影：2 × seqlen × hidden × num_heads × kv_channels
    KV 投影：2 × 2 × seqlen × hidden × num_kv_groups × kv_channels
    """
    h = config.hidden_size
    kv_ch = _head_dim(config)
    num_heads = config.num_attention_heads
    num_kv = config.num_key_value_heads
    q_flops = 2 * seqlen * h * num_heads * kv_ch
    kv_flops = 2 * 2 * seqlen * h * num_kv * kv_ch
    return q_flops + kv_flops


def calculate_attention_flops(seqlen: int, config) -> int:
    """Attention 计算的 FLOPs。对齐源 :35-46（无 qk_pos_emb_head_dim / v_head_dim）。

    QK^T（causal，下三角）：2 × num_heads × seqlen² × kv_channels / 2
    A×V：num_heads × seqlen² × kv_channels
    """
    kv_ch = _head_dim(config)
    num_heads = config.num_attention_heads
    flops = 2 * num_heads * seqlen * seqlen * kv_ch // 2
    flops += num_heads * seqlen * seqlen * kv_ch
    return flops


def calculate_output_flops(seqlen: int, config) -> int:
    """Output 投影（O_proj）的 FLOPs。对齐源 :49-50。"""
    h = config.hidden_size
    return 2 * seqlen * h * h


def calculate_mlp_flops(seqlen: int, config) -> int:
    """MLP（SwiGLU）的 FLOPs。对齐源 :53-54。

    SwiGLU 有 gate/up/down 三个投影，每个 2×seqlen×hidden×ffn_hidden，故系数为 3。
    HF config 字段是 intermediate_size（对应源 ffn_hidden_size）。
    """
    h = config.hidden_size
    ffn = config.intermediate_size
    return 2 * seqlen * h * ffn * 3


def calculate_lm_head_flops(seqlen: int, config) -> int:
    """LM head 投影（hidden → vocab）的 FLOPs。对齐源 :5-6。"""
    return 2 * seqlen * config.hidden_size * config.vocab_size


def calculate_fwd_flops(seqlens: list[int], config) -> int:
    """全模型单次前向 FLOPs（逐样本精确算，不含 padding 浪费）。对齐源 :66-127（dense 分支）。

    每条样本的 FLOPs = Σ_层(qkv + attention + output + mlp) + lm_head。
    lm_head 只算一次（在 vocab_size=151936 的 0.6B 上是大头，约占总量 30-40%）。
    不含 embedding lookup（无乘加，对齐源省略 embedding FLOPs）。

    Args:
        seqlens: 各样本序列长度列表（每条样本独立计算，对齐源 for seqlen in seqlens）。
        config:  HF AutoConfig 或任何有以下字段的对象：
                   hidden_size, num_attention_heads, num_key_value_heads,
                   intermediate_size, num_hidden_layers, vocab_size。
                 可选：head_dim（若无则从 hidden_size÷num_attention_heads 推导）。

    Returns:
        整批样本的总前向 FLOPs（int，未乘 3 —— 调用方按 3×fwd/t 算 MFU）。
    """
    num_layers = config.num_hidden_layers
    total = 0
    for seqlen in seqlens:
        layer_flops = (
            calculate_qkv_projection_flops(seqlen, config)
            + calculate_attention_flops(seqlen, config)
            + calculate_output_flops(seqlen, config)
            + calculate_mlp_flops(seqlen, config)
        )
        total += layer_flops * num_layers
        total += calculate_lm_head_flops(seqlen, config)
    return total
