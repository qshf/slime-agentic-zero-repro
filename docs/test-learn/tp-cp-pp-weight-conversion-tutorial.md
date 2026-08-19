# 从函数读懂 TP、PP、CP 与权重转换

本文以项目中的实际代码为主线，解释 Megatron 训练中的张量并行（TP）、流水线并行（PP）、上下文并行（CP）、它们的组合，以及 Hugging Face（HF）权重与 Megatron 分片之间的转换。

阅读目标不是理解 Megatron-Core 或 TransformerEngine 的内部实现，而是能读懂本项目函数的输入、输出、切分坐标和通信意图，并能判断一处改动应该使用哪个并行组。

相关入口：

- 基础章节：[DP、TP 与行并行/列并行](dp-tp-row-column-parallel.md)。
- 前置章节：[`gpu-communication-primer.md`](gpu-communication-primer.md)，先解释 GPU collective、NUMA 拓扑和 `--topology`。
- [`scripts/test_v8.2_parallel.py`](../../scripts/test_v8.2_parallel.py)：构造验证样本、运行训练、收集并比较结果。
- [`toy_rl/trainer/megatron_trainer.py`](../../toy_rl/trainer/megatron_trainer.py)：构造本 rank 模型、batch、loss、训练调度和梯度规约。
- [`toy_rl/trainer/cp_utils.py`](../../toy_rl/trainer/cp_utils.py)：CP 的序列切分。
- [`toy_rl/trainer/megatron_to_hf.py`](../../toy_rl/trainer/megatron_to_hf.py)：HF 与 Megatron 参数布局转换。

## 1. 一张总图：每条轴切的对象不同

设总进程数为 `world_size`：

```text
world_size = TP x PP x CP x DP
```

`DP` 是数据并行大小，即在指定 TP、PP、CP 后剩余的复制份数。比如 4 个进程使用 `TP=2, PP=2, CP=1`，则 `DP=1`。

| 轴 | 切分对象 | 一个 rank 保存/处理的内容 | 典型通信 |
|---|---|---|---|
| TP | 单层内部的权重、attention head、词表 | 同一层的一部分矩阵 | all-reduce、all-gather |
| PP | Transformer 层列表 | 连续的一段层 | stage 间激活/梯度传递 |
| CP | 同一条序列的 token 位置 | 序列的部分 token | attention 的 KV 通信、梯度规约 |
| DP | 样本 | 一份完整的模型并行副本 | 梯度 all-reduce |

不要把它们混为一谈：TP 是横向切一层，PP 是纵向切多层，CP 是沿时间/token 轴切输入。三条轴可正交组合。

例如 `TP=2, PP=2, CP=2, DP=1` 需要 8 个进程。每个 rank 同时只拥有：

```text
本 PP stage 的层
  x 本 TP rank 的参数分片
  x 本 CP rank 的 token 分片
```

### 1.1 本项目在哪里建立这些组

`MegatronTrainer.__init__()` 将 CLI 参数交给 Megatron 的并行组初始化：

```python
self.tp_size = tensor_model_parallel_size
self.pp_size = pipeline_model_parallel_size
self.cp_size = context_parallel_size

mpu.initialize_model_parallel(
    tensor_model_parallel_size=self.tp_size,
    pipeline_model_parallel_size=self.pp_size,
    context_parallel_size=self.cp_size,
)

self.tp_group = mpu.get_tensor_model_parallel_group()
self.pp_group = mpu.get_pipeline_model_parallel_group()
self.cp_group = mpu.get_context_parallel_group()
self.dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
```

见 [`megatron_trainer.py`](../../toy_rl/trainer/megatron_trainer.py)。`dp_cp_group` 不是笔误：CP rank 对同一份参数各自贡献不同 token 的梯度，所以最终梯度同步时，CP 与 DP 必须一起处理。

在 4 卡 `TP=2 x PP=2` 的默认 rank 布局中：

```text
TP groups: [0, 1], [2, 3]
PP groups: [0, 2], [1, 3]
```

项目将高频 TP 通信留在同一 NUMA 节点，把相对低频的 PP 边界通信放在跨节点路径。`parallel_group_ranks()` 和测试脚本的 `--topology` 会把这个假设变成断言。

## 2. TP：把一层内部的权重切开

以线性层为例：

```text
Y = X W^T
```

TP 最常见的两种权重切法如下。

### 2.1 Column parallel：按输出特征切，切 `dim=0`

若 `W` 形状为 `[out_features, in_features]`，沿 `dim=0` 切：

```text
W = [W0; W1]
rank 0: Y0 = X W0^T
rank 1: Y1 = X W1^T
```

每个 rank 得到不同的输出特征，后续操作按需要保留分片或拼接。项目中 embedding、融合 QKV、MLP 的第一层 `fc1` 都属于这一类。

### 2.2 Row parallel：按输入特征切，切 `dim=1`

```text
W = [W0, W1]
X = [X0, X1]
Y = X0 W0^T + X1 W1^T
```

每个 rank 只算一个局部和，必须通过 all-reduce 得到完整 `Y`。注意：row parallel 切 `dim=1`，不是 `dim=0`。项目中的 attention 输出投影 `o_proj` 和 MLP 第二层 `fc2` 属于这一类。

### 2.3 权重载入中的 TP 规则

HF 权重是完整张量；Megatron 的当前 rank 只应得到自己的 shard。辅助函数非常简单：

```python
def _tp_shard(tensor, dim, tp_rank, tp_size):
    if tp_size == 1:
        return tensor
    return tensor.chunk(tp_size, dim=dim)[tp_rank].contiguous()
```

实际映射在 `hf_to_megatron_state_dict()` 中：

```python
# column-parallel：沿 dim=0 切
out["...linear_qkv.weight"] = _tp_shard(qkv, 0, tp_rank, tp_size)

# row-parallel：沿 dim=1 切
out["...linear_proj.weight"] = _tp_shard(o_proj, 1, tp_rank, tp_size)
out["...linear_fc2.weight"] = _tp_shard(down_proj, 1, tp_rank, tp_size)
```

| HF 参数 | Megatron 参数 | TP 类型 | shard 维度 |
|---|---|---|---:|
| `embed_tokens.weight` | `embedding.word_embeddings.weight` | vocab/column | 0 |
| `q_proj/k_proj/v_proj` | `linear_qkv.weight` | column | 0 |
| `o_proj.weight` | `linear_proj.weight` | row | 1 |
| `gate_proj/up_proj` | `linear_fc1.weight` | column | 0 |
| `down_proj.weight` | `linear_fc2.weight` | row | 1 |

### 2.4 TP 下为什么不 gather 完整 logits

模型设置 `parallel_output=True`，所以每个 TP rank 只拿到 `[S, V / TP]` 的词表 logits，而不是 `[S, V]`。这正是省显存的意图。

`_vocab_parallel_log_probs()` 直接对 vocab shard 求目标 token 的 log-prob：

```python
lp = -fused_vocab_parallel_cross_entropy(
    logits_shard.unsqueeze(1).contiguous(),
    targets.unsqueeze(1),
    self.tp_group,
)
```

该函数在 TP 组内完成全局 softmax 所需的 max 和 sum-exp 规约。因此调用方得到正确的每 token `log_prob`，却不必先 all-gather 全词表 logits。

这里的 `clone()` 也不是装饰：fused kernel 可能原地修改传入 logits；当前代码逐样本切 logits，切片会共享底层 storage，所以在 fp32 下必须显式复制，避免污染 autograd 所需的前向值。

### 2.5 TP 特有的 q/k layernorm 梯度规约

Q/K head 被 TP 切开，q/k layernorm 的每个 rank 只见到自己的 head，因此它的梯度是部分和：

```python
for name, p in self.model.named_parameters():
    if p.grad is not None and ("q_layernorm" in name or "k_layernorm" in name):
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.tp_group)
```

这段 `_allreduce_qk_layernorm_grads()` 的含义是“把各 head 分片对同一 layernorm 参数的梯度加起来”。漏掉它通常有一个迷惑性很强的表现：loss 完全相同，但 `q_norm.weight` 的梯度明显偏离。

## 3. PP：把层列表切成多个 stage

设模型有 4 层且 `PP=2`：

```text
stage 0: embedding -> layer 0 -> layer 1
stage 1: layer 2 -> layer 3 -> final norm -> lm head
```

因此建模型时，不是每个进程都拥有 embedding 和 output layer：

```python
model = GPTModel(
    ...,
    pre_process=self.is_first_stage,
    post_process=self.is_last_stage,
    parallel_output=True,
)
```

`pre_process=True` 表示该 stage 接收 token id 并执行 embedding；`post_process=True` 表示该 stage 将 hidden states 转为 logits。

### 3.1 为什么 forward 与 loss 必须分开

PP 中间 stage 的前向输出是 hidden states，并非 logits；只有最后 stage 能计算 loss。项目用调度器要求的函数签名表达这件事：

```python
def forward_step(self, data_iterator, model):
    batch = next(data_iterator)
    output_tensor = self._forward_logits(batch)
    return output_tensor, partial(self.loss_func, batch)
```

返回的第二项是“等拿到 last-stage logits 后再调用”的 loss 函数。随后 `train_batch()` 调用 Megatron 的调度器：

```python
forward_backward_func = get_forward_backward_func()
losses_reduced = forward_backward_func(
    forward_step_func=self.forward_step,
    data_iterator=iter(batches),
    model=[self.model],
    num_microbatches=len(batches),
    seq_length=batches[0]["seq_length"],
    micro_batch_size=microbatch_size,
    forward_only=False,
)
```

PP=1 时这是普通前反向；PP>1 时是 1F1B 流水。业务代码不要自行手写 `send/recv`：微批数量、前反向顺序和激活释放时机必须在所有 stage 严格匹配，任一顺序不一致都可能死锁。

### 3.2 PP 对 batch 的约束

流水线为接收激活预分配缓冲，所以每个微批形状必须相同。脚本在 PP 模式中把 4 条样本拆成每批 1 条：

```python
micro = 1 if trainer.pp_size > 1 else gbs
metrics = trainer.train_batch(samples, global_batch_size=gbs, microbatch_size=micro)
```

而 `train_batch()` 先计算所有微批中的最大长度，再以统一长度构造 batch。这样既能触发实际的 warmup/steady/cooldown，又不会让 stage 间 receive buffer 的形状不一致。

### 3.3 tied embedding 是 PP 的额外梯度问题

Qwen3 将输入 embedding 和输出 lm head 绑定为同一个逻辑权重。PP>1 时它们位于不同 stage：

```text
first stage: embedding.word_embeddings.weight
last stage:  output_layer.weight
```

两处各自产生部分梯度，必须跨 embedding group 求和：

```python
weight = self.model.shared_embedding_or_output_weight()
if weight is not None and weight.grad is not None:
    dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM, group=embd_group)
```

这段 `_allreduce_word_embedding_grads()` 不是普通 DP 同步，而是“同一逻辑权重在 PP 首尾两处的贡献相加”。漏掉时最差梯度通常是 `embed_tokens.weight`。

## 4. CP：把一条序列的 token 位置切开

CP 针对长上下文。每个 CP rank 持有一条序列的一部分 token；attention 实现负责交换必要的 K/V 信息。因此 CP 不切模型权重，切的是输入序列位置。

### 4.1 2-chunk 对称切分

本项目的 `slice_with_cp()` 不使用朴素连续切分。假设：

```text
CP=2, token_len=16, chunk_size=4
```

正确布局是：

```text
rank 0: token [0:4]  + token [12:16]
rank 1: token [4:8]  + token [8:12]
```

对应代码：

```python
start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
start_2 = chunk_size * (2 * cp_size - cp_rank - 1)
end_2 = chunk_size * (2 * cp_size - cp_rank)
return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
```

因果 attention 中越靠后的 token 工作越重。每个 rank 同时取得一块前段和一块后段，可避免最后一个 rank 独占重负载。更重要的是，Megatron 的 RoPE 与 TransformerEngine 的 CP attention 都预期这个布局；连续切分即使形状完全合法，也会让 token、位置编码和 KV 通信逻辑错位。

测试中的 `--negative-control` 就是故意替换成错误的连续切分：

```python
return tokens[2 * chunk_size * cp_rank : 2 * chunk_size * (cp_rank + 1)]
```

正确的 CP 测试不仅要让正例通过，也必须让这个反例失败。

### 4.2 CP 下 target、mask、old log-prob 必须一起切

语言模型的 logits 位置 `g` 预测 `tokens[g+1]`。`_build_batch()` 先将所有监督数组右移并补齐为与 token 等长：

```python
full_t = torch.cat([t[1:], t.new_full((1,), self.pad_id)])
full_m = torch.cat([m[1:], m.new_zeros(1)])
full_o = torch.cat([o[1:], o.new_zeros(1)])
```

随后 `tokens`、`targets`、`tgt_masks`、`old_log_probs` 都调用同一个 `slice_with_cp()`。这样每个局部 logit 和它的 target/mask 天然在相同位置，不必再手写复杂的 offset 计算。

### 4.3 CP 下 loss：局部分子，完整分母

每个 CP rank 只拥有自己的 token，所以 loss 的分子是局部和；但是一条样本的归一化分母必须仍是全序列的有效 token 数：

```python
# 在切 CP 之前计算
denoms.append(torch.clamp_min(full_m.sum(), 1.0))

# 在切 CP 之后计算局部分子
return -(pg * tgt_mask).sum() / denom
```

若错误地使用 `tgt_mask.sum()` 作为局部分母，每个 CP rank 都会独立归一化，合起来不再等于未切分时的样本 loss。

### 4.4 CP 下梯度与 loss 的最终规约

CP rank 对同一份模型权重得到的是梯度部分和。项目的普通 `torch.optim.AdamW` 没有 mcore DDP wrapper，因此显式执行：

```python
for p in self.model.parameters():
    if p.grad is not None:
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.dp_cp_group)
```

loss 也在同一 `dp_cp_group` 上 AVG 规约。由于 `loss_func()` 预先乘了 `dp_cp_size`，该 AVG 的净数学效果是合并所有 DP/CP rank 的贡献。

## 5. BSHD 与 THD：attention 数据布局，不是并行轴

`--qkv-format` 决定 attention Q/K/V 的组织形式；它不改变模型数学，也不等于 TP/PP/CP。

| 布局 | 张量含义 | 样本边界 | 适用情况 |
|---|---|---|---|
| BSHD | `[B, S, H, D]` | 每个 batch row 自带边界 | 等长或已 padding 的 batch |
| THD | `[T, H, D]` | `cu_seqlens` 记录边界 | 变长样本 packing / varlen attention |

### 5.1 BSHD

`B` 是 batch size，`S` 是每行相同的 sequence length，`H` 是 attention heads，`D` 是每头维度。不同长度样本要先右侧 padding 到同一个 `max_len`：

```python
batch["tokens"] = torch.stack(
    [slice_with_cp(t, self.pad_id, "bshd", max_len) for t in toks]
)
```

如果采用 local spec，代码显式构造 `[B, 1, S, S]` 的因果 mask，且其语义是 `True=屏蔽`：

```python
causal = torch.tril(torch.ones(s, s, device=input_ids.device, dtype=torch.bool))
attention_mask = (~causal).view(1, 1, s, s).expand(b, 1, s, s)
```

这与很多 HF 接口中 `True=保留` 的 mask 约定相反。BSHD + CP 必须使用 TE 路径，不能把一个按全局序列构造的显式 `[S, S]` mask 直接拿给局部 token 分片。

### 5.2 THD

THD 把多条变长样本沿 token 维拼成一条 flat 流，省去短样本的大量 padding：

```python
local = [slice_with_cp(t, self.pad_id, "thd") for t in toks]
flat = torch.cat(local)

PackedSeqParams(
    cu_seqlens_q=cu_t,
    cu_seqlens_kv=cu_t,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format="thd",
)
```

`cu_seqlens` 使用全局序列长度，供 attention 知道每条样本的开始和结束；`local_cu_seqlens` 使用当前 CP rank 的局部长度，只供 `loss_func()` 从局部 flat logits 中取回各样本的片段。两者不可混用。

THD 前向路径不传显式 mask 或 position id：

```python
return self.model(
    input_ids=input_ids,
    position_ids=None,
    attention_mask=None,
    packed_seq_params=batch["packed_seq_params"],
)
```

段隔离、位置和 varlen attention 由 `PackedSeqParams` 及后端共同处理。

## 6. HF <-> Megatron 权重转换

转换的难点不在浮点运算，而在布局和名称。正确转换只使用 `reshape`、`split`、`cat`、切片和 rename，所以 HF -> Megatron -> HF 的逐元素最大差应为零。

### 6.1 Megatron 导出为 HF 的总流程

`megatron_to_hf_state_dict()` 的顺序是：

```text
本 rank 的局部参数
  -> TP all-gather 成完整 Megatron 参数
  -> 删除 vocab padding
  -> 拆融合 QKV / gate-up，改为 HF 名称
  -> PP>1 时向各 stage 广播并汇总完整模型
```

TP gather 的入口：

```python
partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
dist.all_gather(partitions, param.data.contiguous(), group=tp_group)
return torch.cat(partitions, dim=param.partition_dim)
```

非 TP 参数或标为 `duplicated` 的参数不应 gather，直接返回本地值。

### 6.2 SwiGLU 的 fc1 是最容易拼错的权重

HF 有两个独立矩阵：

```text
gate_proj.weight
up_proj.weight
```

Megatron 将它们融合为 `[gate; up]`，但 TP=2 后每个 rank 保存：

```text
rank 0: [gate_0; up_0]
rank 1: [gate_1; up_1]
```

因此普通 `torch.cat([rank0, rank1])` 会得到：

```text
[gate_0; up_0; gate_1; up_1]    # 错
```

正确 gather 是每片先拆成两半，再把所有 gate 放前、所有 up 放后：

```python
chunked = [p.chunk(2, dim=0) for p in partitions]
partitions = [c[0] for c in chunked] + [c[1] for c in chunked]
full = torch.cat(partitions, dim=0)
```

导出到 HF 后再拆：

```python
gate, up = param.chunk(2, dim=0)
return [("...gate_proj.weight", gate), ("...up_proj.weight", up)]
```

反向装载必须做严格逆操作：分别对 gate、up 沿 `dim=0` 切分，再在当前 rank 内拼接 `[gate_r; up_r]`。

### 6.3 GQA 的 QKV 不能三等分

HF 保存独立的 `q_proj`、`k_proj`、`v_proj`；Megatron 保存融合 `linear_qkv.weight`。对于 Qwen 的 GQA，布局按 KV group 交错：

```text
group 0: [该 group 的多个 Q head; K0; V0]
group 1: [该 group 的多个 Q head; K1; V1]
...
```

所以导出时不能写 `param.chunk(3, dim=0)`。正确代码先恢复 group 维：

```python
v = param.view(num_query_groups, -1, head_dim, hidden_size)
q, k, val = torch.split(v, [value_num_per_group, 1, 1], dim=1)

return [
    ("...q_proj.weight", q.reshape(-1, hidden_size)),
    ("...k_proj.weight", k.reshape(-1, hidden_size)),
    ("...v_proj.weight", val.reshape(-1, hidden_size)),
]
```

HF -> Megatron 时则反向执行 `view -> cat([q, k, v], dim=1) -> reshape`。

### 6.4 vocab padding 与 tied 权重

某些模型的词表行数不能整除 `TP` 所需粒度，Megatron 会在 vocab 维追加 padding 行。导出 HF 前必须删除：

```python
if name in {"embedding.word_embeddings.weight", "output_layer.weight"}:
    return param[:vocab_size]
```

Qwen3-0.6B 在当前配置通常无需实际删除，但保留这段才能支持不同模型和 TP 设置。

当 `tie_word_embeddings=True` 时，HF 侧 `lm_head` 与 `embed_tokens` 是同一逻辑权重。导出时项目跳过独立 `lm_head`；PP>1 载入时，last stage 的物理 `output_layer.weight` 仍可能存在，必须从 embedding 填入，否则该 stage 的输出投影可能保持初始化的零值。

### 6.5 PP 下的局部层名与全局层名

PP stage 内的层号从零重新开始。四层、PP=2 时：

```text
stage 0 local layer 0, 1 -> HF global layer 0, 1
stage 1 local layer 0, 1 -> HF global layer 2, 3
```

因此必须使用 `get_transformer_layer_offset()`：

```python
return f"decoder.layers.{int(idx) + layer_offset}.{rest}"
```

如果不加 offset，stage 1 的 local `decoder.layers.0` 会被错误导出成 HF 的 `model.layers.0`，覆盖 stage 0 的层；名称和形状都可能合法，因此这是典型的静默错误。

PP>1 导出完整 state dict 还需要两步通信：先 `all_gather_object` 收集“参数名、shape、dtype、归属 stage”的元数据，再让拥有该参数的 stage 在 PP group 内 broadcast 全量参数。所有 rank 必须按相同的 `sorted(owner)` 顺序调用这些 collective，否则会死锁。

## 7. 组合并行的一次训练如何流动

以 `TP=2, PP=2, CP=2` 为例，一条微批的概念流程是：

```text
1. CP: 每个 cp rank 用 2-chunk 规则取本地 token。
2. PP stage 0: 对本地 token 做 embedding 与前半层；层内使用 TP 分片权重。
3. PP: stage 0 将 hidden states 发给 stage 1。
4. PP stage 1: 完成后半层、norm、vocab-parallel logits。
5. TP: 不 gather logits，直接在 tp group 内求 vocab-parallel log-prob。
6. CP: 每个 cp rank 对本地 token 算 loss 分子，使用全局 token 数作分母。
7. 反向: 1F1B 将 activation gradient 沿 PP 反向传回。
8. 收尾: DPxCP 规约全部梯度，TP 规约 q/k norm，PP 规约 tied embedding。
```

代码中最后三类规约集中在：

```python
def _finalize_model_grads(self):
    self._allreduce_dp_cp_grads()
    self._allreduce_qk_layernorm_grads()
    self._allreduce_word_embedding_grads()
```

它们通信组不同、语义也不同，不能用“把所有 grad 再做一次 world all-reduce”替代：那会重复规约、破坏缩放，且不符合参数实际在哪些 rank 上复制或分片的事实。

## 8. 测试脚本如何验证这些规则

`test_v8.2_parallel.py` 使用 `TP=1, PP=1, CP=1` 作为没有模型并行的基线。每次目标并行运行都从同一份 HF 权重载入，使用同一批确定性样本，然后比较 loss 和完整 HF 布局梯度。

训练后，`_grads_in_hf_layout()` 不能直接比较每个 rank 的 `param.grad`，因为它仍是 TP shard 或 PP 局部层。函数的恢复顺序是：

```python
for name, param in _local_params_by_global_name(trainer.model, trainer.tie).items():
    shim = SimpleNamespace(data=param.grad.detach(), ...TP metadata...)
    full = all_gather_param(name, shim)             # 恢复 TP 全量布局
    full = remove_padding(name, full, vocab_size)   # 删除 vocab padding
    for hf_name, grad in convert_qwen3_to_hf(...): # 拆融合参数并改 HF 名
        out[hf_name] = grad.float().cpu().clone()

if trainer.pp_size > 1:
    dist.all_gather_object(gathered, out, group=trainer.pp_group)
```

这里构造 `shim` 是因为“梯度与参数具有相同 TP 分片布局”。所以验收代码可以复用权重导出的 `all_gather_param()`，而不用复制另一套容易漂移的 gather 逻辑。

比较不使用逐参数最大误差作为硬门，而是累加全局相对 L2 和 cosine。bf16 有正常舍入噪声，CP 又受后端限制只能使用 bf16，因此脚本还使用“错误的连续 CP 切分必须被判红”的 negative control，证明度量具备判别力。

## 9. 初学者排错清单

| 现象 | 优先检查 |
|---|---|
| TP loss 正确、`q_norm.weight` 梯度错误 | `_allreduce_qk_layernorm_grads()` 是否在 TP group 上 SUM。 |
| PP loss 正确、`embed_tokens.weight` 梯度错误 | tied embedding 是否跨 first/last stage SUM。 |
| CP 下 loss 或梯度不等价 | token、target、mask、old log-prob 是否使用同一个 `slice_with_cp()`；分母是否为全局有效 token 数。 |
| CP 程序能跑但数值错误 | 是否误用了连续切分；attention 后端是否满足 CP 的 flash 前提。 |
| HF 导出后 gate/up 错乱 | `linear_fc1` gather 是否先对每个 shard `chunk(2)` 再重排。 |
| HF 导出后 Q/K/V 错乱 | 是否错误地把 GQA 融合 QKV 三等分。 |
| PP 导出只有半个模型或层覆盖 | 是否加了 `layer_offset`；是否按 PP 参数 owner broadcast。 |
| 分布式任务挂住 | 同一 process group 的所有 rank 是否以完全相同顺序进入 collective。 |

## 10. 阅读源码的建议顺序

1. 从测试脚本的 `main()` 看参数如何传给 `MegatronTrainer`，再看 `_grads_in_hf_layout()` 为什么要恢复 HF 坐标系。
2. 阅读 `MegatronTrainer._build_batch()`，先弄清 CP 下 token、target 和 mask 的共同坐标系。
3. 阅读 `_forward_logits()`、`loss_func()`、`forward_step()`、`train_batch()`，理解 THD/BSHD、vocab-parallel loss 与 PP 调度如何连接。
4. 阅读三个 `_allreduce_*_grads()`，区分 DPxCP、TP 与 PP 三种“梯度相加”的原因。
5. 最后对照 `hf_to_megatron_state_dict()` 与 `megatron_to_hf_state_dict()`，逐项确认 QKV、SwiGLU、TP shard、PP layer offset 是严格互逆的。

掌握这条顺序后，新增一种参数、修改一种 layout 或增加一个并行轴时，先回答三个问题：它切的是权重、层还是 token？它的局部结果在什么组内合并？导出比较时应如何回到完整 HF 坐标系？
