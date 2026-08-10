# Attention 布局、后端与 C1-C4 验证

本文解释 V8.2 实测中出现的 `THD`、`BSHD`、`C1-C4`、`unfused`、`fused`、`cuDNN` 和 `flash`。
对应复现脚本是 [`scripts/probe_v8.2_cp_packing.py`](../scripts/probe_v8.2_cp_packing.py)。

## 1. 两种数据布局

### BSHD

`BSHD` 表示四个维度按以下顺序排列：

```text
[B, S, H, D]
```

| 符号 | 含义 |
|---|---|
| `B` | batch size，批大小 |
| `S` | sequence length，序列长度 |
| `H` | attention heads，头数 |
| `D` | head dimension，每个头的维度 |

例如，单条长度为 128 的序列可以表示成 `[1, 128, H, D]`。BSHD 适合各样本长度相同的普通 batch。

### THD

`THD` 表示：

```text
[T, H, D] + cu_seqlens
```

这里没有单独的 `B` 维，所有样本的 token 沿 `T` 维拼接；`cu_seqlens` 记录每个样本在这条扁平 token 序列中的边界。

例如，两条长度分别为 128、96 的序列：

```text
T = 224
cu_seqlens = [0, 128, 224]
```

这种布局也叫 packed/varlen attention：不需要把短样本补齐到同一个最大长度，因此可以节省 padding 的计算和显存。`THD` 本身只是内存布局，不改变 attention 的数学定义。

### BSHD 与 THD 的关系

当只有一条序列，或者所有序列长度相同且没有 padding 时，BSHD 和 THD 只是同一批 Q/K/V 的两种排列方式。因此，前向和反向结果原则上都应一致。C1、C2 就是在验证这个性质。

## 2. Attention 后端

在 TransformerEngine 的 `TEDotProductAttention` 中，后端是三选一：

```text
FlashAttention / FusedAttention / UnfusedDotProductAttention
```

### unfused

`unfused` 是 TransformerEngine 提供的朴素参考实现，不是本项目手写的。其核心流程是：

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * scale
scores = scores.masked_fill(mask, float("-inf"))
probs = torch.softmax(scores, dim=-1)
probs = dropout(probs)
output = torch.matmul(probs, v)
```

它显式产生 `[B, H, Sq, Sk]` 的 score/probability 矩阵，显存和计算通常是 `O(N^2)`。优点是实现简单、容易作为数值参考；缺点是慢、占显存，并且本版本不支持 Context Parallel（CP）。

#### unfused 的直接调用

下面是直接使用 TransformerEngine 实现的写法。`_alibi_cache` 是该实现内部的缓存参数；没有 ALiBi 时传空字典即可。

```python
from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    UnfusedDotProductAttention,
)

attention = UnfusedDotProductAttention(
    softmax_scale=1.0 / (head_dim ** 0.5),
    attention_dropout=0.0,
)

# BSHD: q/k/v 是 [B, S, H, D]
output = attention(
    {}, q, k, v,
    qkv_layout="bshd",
    attn_mask_type="causal",
)
```

对于 THD，调用接口也能接收 `cu_seqlens`；但它会在内部先转成带 padding 的 BSHD 计算，仍会显式产生 `O(N^2)` 的 score 矩阵，也不能用于 CP：

```python
# THD: q/k/v 是 [T, H, D]
output = attention(
    {}, q, k, v,
    qkv_layout="thd",
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    attn_mask_type="causal",
)
```

### fused

`fused` 在本文中专指 TransformerEngine 调用 cuDNN 的 `FusedAttention` 后端。它把 `QK^T`、mask、softmax、`attention·V` 等步骤交给融合 GPU kernel，减少中间张量读写，通常更快、更省显存。

注意：`fused` 是后端名称，`cuDNN` 是它所使用的底层库；两者不是并列的两种后端。可以理解为：

```text
TransformerEngine FusedAttention
    -> cuDNN FusedAttention kernel
```

本项目在 RTX 5090（sm_120）上发现：`fused` 的 BSHD backward 基本正确，但 THD backward 梯度错误。因此不能因为 forward 相等就认为训练正确。

#### fused 的核心调用

项目不会直接调用 cuDNN。它调用 `TEDotProductAttention`；TransformerEngine 选择 fused 后端后，经过 `FusedAttention` 和 `torch.autograd.Function` 的包装，调用其 C++ 扩展：

```text
项目 TEDotProductAttention
  -> TransformerEngine FusedAttention
  -> FusedAttnFunc.apply(...)
  -> fused_attn_fwd(...) / fused_attn_bwd(...)
  -> cuDNN FusedAttention kernel
```

直接使用 TransformerEngine 的 `FusedAttention` 包装类时，调用形状如下。`FusedAttnBackend` 是 TE 对可用 cuDNN kernel 的枚举；正常训练不应手工选择它，而应让 `TEDotProductAttention` 自动选择。

```python
from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttention
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend

attention = FusedAttention(
    softmax_scale=1.0 / (head_dim ** 0.5),
    attention_dropout=0.0,
)

output = attention(
    q, k, v,
    qkv_layout="thd",                 # 或 "bshd"
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    attn_mask_type="causal",
    fused_attention_backend=FusedAttnBackend["F16_arbitrary_seqlen"],
)
```

省略 FP8、KV cache、窗口注意力等参数后，前向调用的关键形状如下：

```python
# TransformerEngine 内部的等价核心调用，不是项目本地实现
out, aux, *_ = fused_attn_fwd(
    is_training,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q, k, v,
    output_dtype,
    fused_attention_backend,
    attn_bias,
    attn_scale=softmax_scale,
    dropout=dropout_p,
    qkv_layout=qkv_layout,       # "bshd" 或 "thd"
    attn_mask_type="causal",
)
```

反向不是 PyTorch 把前向操作逐个求导，而是对应的 cuDNN 反向入口：

```python
dq, dk, dv, *_ = fused_attn_bwd(
    max_seqlen_q, max_seqlen_kv,
    cu_seqlens_q, cu_seqlens_kv,
    q, k, v, out, dout, aux,
    fused_attention_backend,
    attn_scale=softmax_scale,
    dropout=dropout_p,
    qkv_layout=qkv_layout,
)
```

这也是本项目的 bug 落点：THD 的问题发生在这条 cuDNN fused backward 路径，而不在模型的 QKV projection、THD 打包代码或普通 PyTorch autograd。

### flash

`flash` 是 FlashAttention（本次实测为 FA2）后端。它同样是优化过的融合 kernel，但在 TransformerEngine 的后端选择中，它与 `FusedAttention`、`UnfusedDotProductAttention` 并列。

本项目的主线路径是：

```text
CP > 1  +  THD packing  ->  flash（FA2）
```

它同时通过了 THD/BSHD 的前向、反向一致性检查，并支持 CP。

#### flash 的核心调用

flash 后端同样不是项目手写。TransformerEngine 的 `FlashAttention` 包装 `flash-attn` 包：

```text
项目 TEDotProductAttention
  -> TransformerEngine FlashAttention
  -> flash_attn_func(...) 或 flash_attn_varlen_func(...)
  -> flash-attn CUDA kernel
```

直接使用 TransformerEngine 的包装类：

```python
from transformer_engine.pytorch.attention.dot_product_attention.backends import FlashAttention

attention = FlashAttention(
    softmax_scale=1.0 / (head_dim ** 0.5),
    attention_dropout=0.0,
)

# THD: q/k/v 是 [T, H, D]
output = attention(
    q, k, v,
    qkv_layout="thd",
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    attn_mask_type="causal",
)
```

普通 BSHD/SBHD、没有 padding 的路径调用 `flash_attn_func`：

```python
# TransformerEngine 内部的等价核心调用
output = flash_attn_func(
    q, k, v,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=True,
)
```

THD/packing 走 varlen API，并把真实的段边界传给 kernel：

```python
# q/k/v: [T, H, D]；cu_seqlens 标记每段的起止位置
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=True,
)
```

这里的 `varlen` 不等于“输入中有 padding”。它表示 kernel 通过 `cu_seqlens` 取得每个样本的真实长度，因而能处理长度不等的序列：

```text
THD / packing：token 已连续拼接，没有 padding
  [seq0 的 128 token][seq1 的 96 token]
  cu_seqlens = [0, 128, 224]
  -> 直接调用 flash_attn_varlen_func

BSHD padding batch：短序列为凑齐 shape 而补过 padding
  [seq0 的 128 token]
  [seq1 的  96 token][32 个 padding token]
  -> 包装层先按真实长度取出有效 token，再调用同一个 varlen API
```

所以，THD 的价值正是**不需要 padding**：它不需要先把 packed token 还原成 padding 后的 `[B, S, H, D]`。TransformerEngine 也会在 BSHD/SBHD 的输入带 padding mask 时选用 varlen API；那是为了跳过 padding token，而不是说 THD 内部含有 padding。

### 怎么选中后端

复现脚本以环境变量控制 TransformerEngine 的选择：

```bash
# cuDNN fused
NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=0

# FlashAttention
NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 NVTE_UNFUSED_ATTN=0

# TransformerEngine 的朴素参考实现
NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=1
```

Megatron 的 `--attention-backend flash` 会把选择翻译为第二组。用 `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` 运行时，TransformerEngine 会打印每个后端被启用或禁用的原因，以及最后实际选中的后端。

## 3. C1-C4 是什么

四个检查验证不同层次的问题。每项都覆盖两种布局（BSHD、THD）和三个后端（unfused、fused/cuDNN、flash/FA2）。

```text
布局：BSHD、THD
后端：unfused、fused（cuDNN）、flash（FA2）
```

表中的“基线”表示该行是该项比较的基准，不是额外的失败或成功判据。

### C1：前向是否与布局无关

**任务含义**：同一段序列写成 BSHD 或 THD，只是内存布局不同，前向输出应相同。C1 只验证 forward，不能据此推断 backward 正确。

**判据**：THD 输出相对同后端 BSHD 输出的 `max_abs_diff == 0`（或只有 dtype 舍入误差）。

| 布局 | unfused | fused（cuDNN） | flash（FA2） |
|---|---|---|---|
| BSHD | 基线 | 基线 | 基线 |
| THD，相对 BSHD | `0.000e+00` ✅ | `0.000e+00` ✅ | `0.000e+00` ✅ |

**C1 结论**：三种后端的前向都通过。THD 的数学语义没有问题。

### C2：反向是否与布局无关

**任务含义**：对 C1 的同一组 Q/K/V 和同一 loss 执行 backward，比较 `dq/dk/dv`。如果 C1 通过而 C2 失败，问题在反向 kernel 或布局专用路径，不是 attention 定义不同。

**判据**：梯度 cosine 应接近 `1.0`，梯度范数 ratio 应接近 `1.0x`。

| 布局 | unfused | fused（cuDNN） | flash（FA2） |
|---|---|---|---|
| BSHD | 基线 | 基线 | 基线 |
| THD，相对 BSHD | `dq/dk/dv` cosine >= `0.99999994`、ratio `1.00x` ✅ | **`dq` cosine `0.356`、ratio `24.7x` ❌** | `dq` cosine `1.00000000`、`dk/dv` >= `0.99999988`、ratio `1.00x` ✅ |

**C2 结论**：只有 `fused + THD` 的 backward 错；forward 相等不代表梯度正确。

### C3：后端本身是否与可信参考一致

**任务含义**：用 unfused 作为参考，比较各后端的输出 `o` 和梯度 `dq/dk/dv`。这项区分“整个后端都错”和“只有特定布局路径错”。

**判据**：输出与梯度的 cosine 接近 `1.0`；bf16 的小量级差异视为舍入误差。

| 布局 | unfused | fused（cuDNN），相对 unfused | flash（FA2），相对 unfused |
|---|---|---|---|
| BSHD | 参考 | `dq/dk/dv` cosine >= `0.99996924` ✅ | `dq/dk/dv` cosine >= `0.99996847` ✅ |
| THD | 参考 | 梯度错误 ❌ | `dq/dk/dv` cosine >= `0.99996847` ✅ |

**C3 结论**：fused 的 BSHD 路径正确，只有它的 THD backward 偏离参考；flash 的两种布局都正确。因此故障被锁定为 cuDNN fused 的 THD backward，而不是 THD 布局本身。

### C4：CP=2 下每条路径是否能运行

**任务含义**：在两张卡上启用 Context Parallel（`CP=2`），分别测试 BSHD 与 THD 是否有可用后端，且输出与梯度是否有限值。C4 不以 C2/C3 的数值一致性替代运行检查。

| 布局，CP=2 | unfused | fused（cuDNN） | flash（FA2） |
|---|---|---|---|
| BSHD | 无可用后端 ❌ | PASS，梯度 finite ✅ | PASS，梯度 finite ✅ |
| THD | 无可用后端 ❌ | `thd_second_half_lse_correction` assert ❌ | PASS，梯度 finite ✅ |

**C4 结论**：unfused 不支持 CP；fused 不能跑 CP + THD；flash 是唯一同时支持 CP、THD/packing 且通过 C2/C3 正确性检查的路径。

## 4. 最终结论

```text
BSHD / THD：数据布局
unfused / fused / flash：attention 后端
cuDNN：fused 后端使用的底层库
C1：前向等价
C2：反向等价
C3：与 unfused 参考对比
C4：CP=2 可运行性
```

在本项目的 RTX 5090 配置上，训练应使用 `flash（FA2）`，不要使用 cuDNN `fused` 的 THD backward；`unfused` 仅适合作为慢速数值参考。
