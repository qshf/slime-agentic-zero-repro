# 为什么 CP 的梯度规约用 AVG

这篇笔记解释一个容易困惑的点：CP（context parallel）把一条序列的 token 切到多个 rank 上。每个 CP rank 算到的梯度只是**部分和**，那为什么代码使用 `AVG all-reduce`，而不是 `SUM all-reduce`？

结论先说：

```text
CP 的原始梯度确实是部分和。
但每个 rank 在 backward 前已经把 loss 乘了 DP×CP 的组大小。
因此后面的 AVG 正好抵消这个预乘，最终结果仍是所有部分梯度的和。
```

## 相关代码

本项目在 [`MegatronTrainer.loss_func`](../../toy_rl/trainer/megatron_trainer.py) 中缩放 loss：

```python
loss = loss * batch["num_microbatches"] / batch["global_batch_size"] * self.dp_cp_size
```

对应位置：[`megatron_trainer.py:682-683`](../../toy_rl/trainer/megatron_trainer.py#L682-L683)。

梯度规约在 [`MegatronTrainer._allreduce_dp_cp_grads`](../../toy_rl/trainer/megatron_trainer.py) 中：

```python
dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.dp_cp_group)
```

对应位置：[`megatron_trainer.py:783-802`](../../toy_rl/trainer/megatron_trainer.py#L783-L802)。

`num_microbatches` 的除法不在本项目，而在安装的 Megatron Core 中：

```python
output_tensor /= num_microbatches
```

本机路径是 `.venv/lib/python3.13/site-packages/megatron/core/pipeline_parallel/schedules.py:274`。

## 例子

假设：

```text
DP = 2
CP = 2
global batch size = 32
micro batch size = 8
PP = 1
```

DP×CP 组大小为：

```text
dp_cp_size = 2 × 2 = 4
```

全局 32 条样本先被 DP 切开：

```text
DP 0: 样本 1 到 16
DP 1: 样本 17 到 32
```

CP 不会再把样本分开。它让两个 CP rank 都看到同一份 16 条样本，但每个 rank 只计算每条序列的一部分 token：

```text
(DP 0, CP 0): 样本 1 到 16 的 token 部分 A
(DP 0, CP 1): 样本 1 到 16 的 token 部分 B
(DP 1, CP 0): 样本 17 到 32 的 token 部分 A
(DP 1, CP 1): 样本 17 到 32 的 token 部分 B
```

每个 rank 有 16 条本地样本，所以有两个 microbatch：

```text
16 / 8 = 2
```

## 先看没有缩放时的梯度

为了简单，假设某一个参数的局部梯度贡献已经分别累加为：

```text
(DP 0, CP 0): 120
(DP 0, CP 1): 136
(DP 1, CP 0): 152
(DP 1, CP 1): 168
```

这些数中，CP 的两项来自同一批样本的不同 token；DP 的两组来自不同样本。完整 batch 的正确平均梯度是：

```text
(120 + 136 + 152 + 168) / 32 = 18
```

## 代码实际做了什么

`loss_func` 中的缩放因子是：

```text
num_microbatches / global_batch_size × dp_cp_size
= 2 / 32 × 4
= 1 / 4
```

Megatron 的 schedule 随后除以 `num_microbatches=2`。所以每个 rank 的最终本地梯度系数是：

```text
1 / 4 / 2 = 1 / 8
```

四个 rank 在规约前的梯度变为：

```text
120 / 8 = 15
136 / 8 = 17
152 / 8 = 19
168 / 8 = 21
```

然后执行 AVG all-reduce：

```text
AVG(15, 17, 19, 21)
= (15 + 17 + 19 + 21) / 4
= 18
```

结果正好等于完整 32 条样本的正确平均梯度。

## 用公式看

设 `g = DP × CP`，设 `h_r` 是第 `r` 个 rank 未缩放的局部梯度贡献。

目标是：

```text
完整梯度 = h_0 + h_1 + ... + h_(g-1)
```

代码先在 loss 中把每个 rank 的梯度乘 `g`，再做平均规约：

```text
AVG(g × h_0, g × h_1, ..., g × h_(g-1))
= (g × h_0 + ... + g × h_(g-1)) / g
= h_0 + ... + h_(g-1)
```

因此，`AVG` 不是在把 CP 的部分和错误地取平均；它是在抵消 loss 中预先乘上的 `dp_cp_size`。

## 为什么不直接用 SUM

当前代码已经在 loss 中乘过 `dp_cp_size`。如果这里改成 `SUM`，会得到：

```text
SUM(g × h_0, ..., g × h_(g-1))
= g × (h_0 + ... + h_(g-1))
```

梯度会放大 `DP×CP` 倍。本例中就是放大 4 倍。

另一套同样正确、但不同的设计是：

```text
loss 不乘 dp_cp_size
梯度规约使用 SUM
```

本项目选择的是与 Megatron DDP 一致的组合：

```text
loss 乘 dp_cp_size
梯度使用 AVG all-reduce
```

## 一个重要前提

这里的 `global_batch_size=32` 指全局语义。调用训练器时，每个 DP rank 应传入自己的 16 条样本；同一 DP rank 内的两个 CP rank 应传入相同的 16 条样本，只是由 CP 切分 token。

`train_batch()` 自己只会把传入的本地 samples 按 micro batch 切分，不会替调用方自动完成 DP 数据分片。见 [`megatron_trainer.py:830-831`](../../toy_rl/trainer/megatron_trainer.py#L830-L831)。
