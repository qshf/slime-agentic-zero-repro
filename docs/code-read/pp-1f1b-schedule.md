# PP>1 是怎么调的：非交错 1F1B 调度细节

本笔记回答「开 `--pp 2` 之后，模型怎么被纵向切成两半、两个 stage 之间怎么 p2p 传激活、1F1B 的三段（warmup/steady/cooldown）各做什么、nano 要补哪些接缝」。聚焦 **PP 调度机制**；TP 切分见 [`tp-sharding-mechanics.md`](./tp-sharding-mechanics.md)，PP=1 的顺序调度见 [`megatron-schedule-internals.md`](./megatron-schedule-internals.md)。

结论先行：**PP 把模型按层纵向切成 P 段，stage 之间用 p2p 传 hidden states（前向）和梯度（反向）；调度走 1F1B——先「只前向」灌满流水线（warmup），再「一前向一反向」交替（steady），最后「只反向」排空（cooldown）。nano 只需补三处接缝：前向/loss 拆开、tie embedding 跨 PP 梯度规约、loss 广播。**

## 1. PP 切什么：模型纵向切两半

PP=2 时，28 层 Qwen3-0.6B 被切成两段，每段跑在一个 rank 上：

```text
Stage 0（first stage，rank 0）:  embedding + layers 0..13
Stage 1（last  stage，rank 1）:  layers 14..27 + final_layernorm + output_layer
```

对应 [`MegatronTrainer.__init__`](../../toy_rl/trainer/megatron_trainer.py#L310-L311) 的角色判定与 [`GPTModel`](../../toy_rl/trainer/megatron_trainer.py#L363) 的构造：

- `pre_process=self.is_first_stage`：只有 first stage 有 embedding。
- `post_process=self.is_last_stage`：只有 last stage 有 final_layernorm + output_layer。

中间 stage（PP>2 时存在）两头都没有，只拿 hidden states 进、hidden states 出。

## 2. 1F1B 的三段（PP=2，4 个微批）

G2 门（[`test_v8.2_parallel.py:16`](../../scripts/test/test_v8.2_parallel.py#L16)）用 4 条样本、`microbatch_size=1` → 4 个微批。两个 stage 的操作序列如下（`Fx` = 第 x 个微批的前向，`Bx` = 反向，`—` = 空闲）：

```text
时间步:   0    1    2    3    4    5    6    7    8
Stage 0:  F0   F1   B0   F2   B1   F3   B2   B3   —
Stage 1:  —    F0   B0   F1   B1   F2   B2   F3   B3
```

- **warmup（灌满）**：Stage 0 先只前向 `F0`（1 个），把激活送进流水线。
- **steady（交替）**：之后每 stage 都是「一前向一反向」交替——`F1,B0` / `F2,B1` / `F3,B2`。
- **cooldown（排空）**：Stage 0 最后只反向 `B3`（1 个），把流水线里的梯度收完。

**bubble（空闲）**就是 Stage 1 开头的 `—` 和 Stage 0 结尾的 `—`：流水线灌满和排空时总有人闲着。bubble 占比 ≈ $\frac{P-1}{m+P-1}$（P=2、m=4 时 = 1/5），**微批越多 bubble 越小、PP 越大 bubble 越大**。nano 未开 VPP（interleaved 1F1B），那是进一步减 bubble 的调度优化，留 V9 当吞吐旋钮（见 `megatron_trainer.py` 模块 docstring 偏离登记）。

**每 stage 的三段长度由 rank 决定**（[`schedules.py:2193`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2193)）：

```text
num_warmup_microbatches = total_stages - current_stage - 1
```

- Stage 0（first）：warmup = 2-0-1 = **1**
- Stage 1（last）：warmup = 2-1-1 = **0**

越靠前的 stage warmup 越长，越靠后的 cooldown 越长。

## 3. p2p 通信原语

stage 之间只传两样东西：**前向的 hidden states**（下游算前向用）和**反向的梯度**（上游算反向用）。mcore 的 `P2PCommunicator` 提供这些原语：

| 原语 | 方向 | 用于 | first/last stage 的行为 |
| --- | --- | --- | --- |
| `recv_forward` | 从上一 stage 收 | warmup / steady | first stage 返回 `None`（从 iterator 读数据） |
| `send_forward` | 给下一 stage 送 | warmup | last stage no-op |
| `recv_backward` | 从下一 stage 收 | cooldown | last stage 返回 `None`（loss 是反向起点） |
| `send_backward` | 给上一 stage 送 | cooldown | first stage no-op |
| `send_forward_recv_backward` | 送前向 + 收反向（合体） | steady | — |
| `send_backward_recv_forward` | 送反向 + 收前向（合体） | steady | — |

**关键点**：前向的 `output_tensor` 在中间 stage 是 hidden states，**直接 p2p 送给下一个 stage**，不是 logits。这就是 `forward_step` 与 `loss_func` 必须拆开的原因（§7）。

## 4. mcore 内部逐段（行号跳转）

入口 [`forward_backward_pipelining_without_interleaving`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2055)。

| 段 | 位置 | 做什么 |
| --- | --- | --- |
| 预热计算 | [`schedules.py:2193-2194`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2193-L2194) | 算 `num_warmup_microbatches`，按 rank 定 |
| recv 缓冲形状 | [`schedules.py:2224-2228`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2224-L2228) | `get_tensor_shapes` 按 `seq_length`/`micro_batch_size` 预分配 p2p 缓冲 |
| **warmup** | [`schedules.py:2250`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2250) | `for i in range(num_warmup_microbatches)`：只前向 |
| 　└ recv + forward | [`schedules.py:2260`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2260) | `recv_forward` → `forward_step` → `send_forward` |
| 进 steady 前 | [`schedules.py:2290`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2290) | 先收第一个前向张量 |
| **steady（1F1B）** | [`schedules.py:2294-2295`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2294-L2295) | `for i in range(num_microbatches_remaining)`：一前向一反向 |
| 　└ 送前向收反向 | [`schedules.py:2332`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2332) | `send_forward_recv_backward`（F 送下游、收上一微批的 grad） |
| 　└ 送反向收前向 | [`schedules.py:2362`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2362) | `send_backward_recv_forward`（grad 送上游、收下一微批的激活） |
| **cooldown** | [`schedules.py:2366-2368`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2366-L2368) | `for i in range(num_warmup_microbatches)`：只反向 |
| 　└ recv + backward | [`schedules.py:2382`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2382) | `recv_backward` → `backward_step` → `send_backward` |
| 梯度收尾 | [`schedules.py:2409`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2409) | `config.finalize_model_grads_func(...)`（nano 挂的钩子） |

**steady 的核心是那两个「合体」原语**：

- `send_forward_recv_backward`：把**当前微批的前向结果**送下游，同时收**上一微批的梯度**——一个微批的前向和更早一个微批的反向在此交错。
- `send_backward_recv_forward`：把**当前微批的梯度**送上游，同时收**下一微批的前向激活**。

这就是 1F1B 名字的由来：**1 个 forward 配 1 个 backward，但配的不是同一个微批**（错开流水线深度）。

## 5. nano 必须补的三处接缝

### 5.1 前向/loss 必须拆开（`forward_step` 只前向）

[`forward_step`](../../toy_rl/trainer/megatron_trainer.py#L692) 只做纯前向、返回 `(logits, partial(loss_func, batch))`。原因正是 1F1B：**前向发生在每个 stage，算 loss 只在 last stage**；中间 stage 的 `output_tensor` 是 hidden states 直接 p2p 送走，根本没有 logits 可算 loss。PP=1 时才能把两者合成一个 `_microbatch_backward`——所以 [`_microbatch_backward`](../../toy_rl/trainer/megatron_trainer.py#L704) 硬断言 `pp_size == 1`（绕过 1F1B 会死锁）。

### 5.2 各微批形状必须一致

PP 的 p2p 缓冲按 `seq_length`/`micro_batch_size` **预分配**（§4 的 `get_tensor_shapes`）。所以 [`train_batch`](../../toy_rl/trainer/megatron_trainer.py#L839) 断言 `len(samples) % microbatch_size == 0`，且所有微批统一 pad 到全局最长；thd(packing) 下还要开 [`variable_seq_lengths`](../../toy_rl/trainer/megatron_trainer.py#L334) 让 recv 形状动态协商。

### 5.3 tie embedding 跨 PP 梯度规约

Qwen3-0.6B `tie_word_embeddings=True`：first stage 持 `embedding.word_embeddings.weight`，last stage 持 `output_layer.weight`，tie 意味着这是**同一份权重**，但物理上在两个进程里，各自梯度只是部分和。所以 [`_allreduce_word_embedding_grads`](../../toy_rl/trainer/megatron_trainer.py#L751) 跨 first/last stage 做 **SUM** all-reduce。

漏掉这一步的指纹与 TP 的 qk-layernorm 完全一样：**loss Δ 恰为 0，而梯度系统性偏、最差参数是 `embed_tokens.weight`**。只有 fp32 档抓得到。

## 6. 为什么 forward 前向/loss 拆开（呼应 §5.1）

对照 PP=1 与 PP=2 的两个调度：

```text
PP=1（no_pipelining）:   每个 stage 就是整个模型
  循环每个微批:  forward → loss → backward

PP=2（1F1B）:            前向在每个 stage，loss 只在 last stage
  Stage 0:  F0  F1  B0  F2  B1  F3  B2  B3     ← 只 forward + backward，不算 loss
  Stage 1:  F0  B0  F1  B1  F2  B2  F3  B3     ← forward 后立刻算 loss 再 backward
```

- **前向**每个 stage 都发生：中间 stage 产出 hidden states 送下游。
- **算 loss** 只在 last stage：只有 last stage 的 `output_tensor` 是 logits。

所以 `forward_step`（纯前向）和 `loss_func`（算 loss）必须拆成两个回调，让 mcore 的 1F1B 调度器在「每个 stage 前向、只有 last stage 算 loss」的节拍下分别调用它们。

## 7. loss 只在 last stage：广播回来

`forward_backward_pipelining_without_interleaving` 的 docstring 明说：**「Returns dictionary with losses if the last stage, empty dict otherwise.」** 所以 PP=2 时 first stage 的 `losses_reduced` 是空列表。

[`train_batch`](../../toy_rl/trainer/megatron_trainer.py#L902) 在结尾跨 PP 组 `broadcast`，把 last stage 的 loss/trained 广播给所有 stage——**纯报告用**，与梯度无关。不广播的话 rank 0 会报 `loss=0`，那不是算错，是「这个数在别的进程里」。

## 8. G2 等价门：PP=2 与 TP=1 基线逐值等价

```bash
# 基线（无任何并行）
torchrun --nproc_per_node=1 scripts/test/test_v8.2_parallel.py --fp32 --dump /tmp/v82_base.pt

# G2：PP=2（本版最硬的门，抓 tie embedding 漏规约）
torchrun --nproc_per_node=2 scripts/test/test_v8.2_parallel.py --fp32 --pp 2 --backend nccl \
    --dump /tmp/v82_pp2.pt --compare /tmp/v82_base.pt
```

测试脚本 [`test_v8.2_parallel.py:327`](../../scripts/test/test_v8.2_parallel.py#L327) 特意让 PP>1 时 `microbatch_size=1`，这样 4 条样本就是 4 个微批，1F1B 真的有三段（warmup/steady/cooldown），而不是退化成无流水。

判据与 TP 门相同（fp32 精确证明）：`loss |Δ| < 1e-4`、grad `rel_L2 < 5e-3`、`cosine > 0.9999`。PP 切分理论上也是数学恒等——只要三处梯度规约（DP/CP、qk 跨 TP、tie embedding 跨 PP）一处不漏。

## 9. 最小心智模型

```python
# PP=2：模型纵向切两半，first stage 有 embedding，last stage 有 output
model = GPTModel(..., pre_process=is_first_stage, post_process=is_last_stage)

# mcore 1F1B 调度替你跑三段：
#   warmup:  只前向，p2p 送 hidden states 灌满流水线
#   steady:  send_forward_recv_backward / send_backward_recv_forward 交错
#   cooldown: 只反向，p2p 送梯度排空流水线
losses = forward_backward_func(forward_step_func=..., data_iterator=..., model=[model], ...)

# 你补的接缝：前向/loss 拆开（forward_step 只前向）、
#             tie embedding 跨 PP 梯度 SUM、
#             loss 从 last stage 广播回所有 stage
```

**一句话总结**：PP 切的是「层」这一维，stage 间 p2p 传 hidden states / 梯度；调度是 1F1B 三段（灌满 → 交替 → 排空），中间有 bubble；nano 负责把前向与 loss 拆开、补 tie embedding 的跨 PP 梯度规约、并把 last stage 的 loss 广播回来。

## 10. 建议阅读顺序

1. 先读 §1-2，建立「纵向切两半 + 三段调度 + bubble」的直觉。
2. 读 §4 的逐行表，对照 [`forward_backward_pipelining_without_interleaving`](../../.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py#L2055) 源码，重点看两个「合体」p2p 原语。
3. 读 §5 的三个接缝，理解「前向/loss 拆开」为什么是 PP 的硬要求。
4. 跑 §8 的 G2 门，验证 PP=2 与基线逐值等价。
