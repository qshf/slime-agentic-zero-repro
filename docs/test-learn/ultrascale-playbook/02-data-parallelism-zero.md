# 02 · 数据并行与 ZeRO-1/2/3（FSDP）

> 对应原文：Data Parallelism / Revisiting global batch size / Our journey up to now / Zero Redundancy Optimizer (ZeRO)

---

## 1. DP 的本质：梯度累积的并行版

模型在 N 张卡上各复制一份（model instance），各自吃不同的 micro-batch 做 fwd/bwd → 梯度不同 → 用 **all-reduce 求平均**保持各副本同步。all-reduce 发生在**反向过程中、optimizer step 之前**。

朴素实现：等反向全部算完 → 触发一次 all-reduce。**这是大忌**——通信期间 GPU 全部空转。

### 三个优化（顺序就是收益顺序）

**① 通信与反向重叠。**某一层的梯度算完就可以立刻开始规约，不必等更浅的层。PyTorch 里给每个参数挂 hook：

```python
def register_backward_hook(self, hook):
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

这是全书第一次出现「**计算/通信重叠**」这个贯穿始终的主题。

**② 梯度分桶（bucketing）。**大张量上的通信比一堆小张量高效得多——把梯度攒进 bucket（PyTorch 默认 25 MB），每桶发一次 all-reduce。比喻：寄几个大箱子比寄一堆小包裹划算。

**③ 与梯度累积的配合。**累积期间的中间步**不需要**同步梯度，最后一步同步一次即可。PyTorch 用 `model.no_sync()` 关掉中间那些。

> 注意：通信要求张量**内存连续**，否则要多余拷贝。所以通常会预分配连续 buffer——这也是训练峰值显存的一部分来源。

---

## 2. 全局 batch size 的完整公式与配方

$$bs = gbs = mbs \times grad\_acc \times dp$$

**优先把 `dp` 拉满，`grad_acc` 只用来补齐剩下的缺口**——因为 DP 是真并行、累积是串行。

**四步配方**：
1. 定目标 gbs（token 数）——查文献或跑收敛实验；
2. 定 seq（一般 2–8k）；
3. 单卡上把 `mbs` 一路加到 OOM 前一格，得到最大 `mbs`；
4. 用可用 GPU 数定 `dp`，剩下的商就是 `grad_acc`。

例子：gbs=4M tokens、seq=4k → 1024 条样本。若单卡只能 `mbs=2`、有 128 卡 → `grad_acc=4`。若突然有 512 卡 → `grad_acc=1`，直接更快。

**若 `grad_acc < 1`（卡多到用不完）**：要么不用满所有卡，要么加大 gbs，要么试更小的 `mbs`（牺牲单卡效率换整体吞吐）。

### DP 的天花板

- **512+ 卡规模**开始受 **ring latency**（信号绕环一圈的时间）限制，DP 通信**没法再被完全重叠**；
- 实测曲线：过了某个点吞吐显著下滑，而**每卡显存占用完全不变**（加 DP 不省显存）；
- 更根本的限制：**DP 要求模型的一层能装进单卡**，且 `mbs=1` 的前向要放得下。大模型开了重算也常常放不下。

---

## 3. ZeRO：沿 DP 轴消除冗余

朴素 DP 把优化器状态/梯度/参数在每个 rank 上**完整复制**了一遍，纯冗余。ZeRO 把它们**沿 DP 轴分片**，需要时再重建。

记 `Ψ` = 参数量，`N_d` = DP degree，`k` = 优化器状态的显存倍数（Adam 混合精度下 **k=12**）。基线（不做 FP32 梯度累积）：

$$2\Psi\;(\text{BF16 params}) + 2\Psi\;(\text{BF16 grads}) + 12\Psi\;(\text{FP32 master} + m + v)$$

| 阶段 | 分片对象 | 显存 |
|---|---|---|
| ZeRO-1 | 优化器状态 | $2\Psi + 2\Psi + \dfrac{k\Psi}{N_d}$ |
| ZeRO-2 | + 梯度 | $2\Psi + \dfrac{2\Psi + k\Psi}{N_d}$ |
| ZeRO-3 | + 参数（= FSDP） | $\dfrac{2\Psi + 2\Psi + k\Psi}{N_d}$ |

> **激活不在列表里，而且分不了**——每个 DP rank 吃的是不同的 micro-batch，激活本来就各不相同、不存在冗余。要压激活得靠重算 / 梯度累积 / TP / CP。

### ZeRO-1：一步的完整序列

1. 各 rank 用**完整** BF16 参数、不同 micro-batch 前向；
2. 反向得到**完整**梯度；
3. 对梯度做 **reduce-scatter**（每 rank 只留自己那 $1/N_d$）；
4. 各 rank 用本地的 $1/N_d$ 优化器状态更新出 $1/N_d$ 的 FP32 参数 → 转成 BF16；
5. 对 BF16 参数做 **all-gather**，把缺的片补回每个 rank。（第 5 步是 ZeRO 新增的，朴素 DP 没有。）

通信上：**把 DP 的 all-reduce 换成了 reduce-scatter，并在 step 后加了一次 all-gather**。（reduce-scatter 的通信量是 all-reduce 的一半。）

新增的 all-gather 也能重叠：①在 optimizer step 中——更新完第一片就开始 gather；②在前向中——逐层 gather。但实现要用到复杂的 hook/bucketing，**实践中直接用 PyTorch 原生 ZeRO-3/FSDP、把 FSDP unit 设成整个模型**即可。

### ZeRO-2：几乎白拿

既然每 rank 只更新 $1/N_d$ 的状态，那它也只需要 $1/N_d$ 的梯度。反向时把 all-reduce 直接换成 **reduce-scatter**，边通信边释放。

> **通信量与 ZeRO-1、与朴素 DP 完全相同**，只是多省一块显存 → **除了实现复杂度，ZeRO-2 相对 ZeRO-1 没有任何额外代价，通常应该直接用 ZeRO-2。**

### ZeRO-3 / FSDP：参数也切

前向逐层**按需 all-gather** 参数、用完立刻释放；反向同理（方向相反），产出梯度分片。

**通信代价**：前向 all-gather 参数 `Ψ` + 反向 all-gather 参数 `Ψ` + 梯度 reduce-scatter `Ψ` = **3Ψ**，对比 ZeRO-2 的 **2Ψ**。
而且多出 `2·num_layers − 1` 次 all-gather，每次都有一点基础延迟。

**解法是 prefetch**：算第 n 层前向时，同时 all-gather 第 n+1 层的权重；反向时 all-gather 第 n−1 层。**这个重叠只在 DP 不太大时成立——经验上 DP 别超过 512。**

---

## 4. 边界与下一步

ZeRO 理论上可以把「模型相关」的显存无限压下去（只要 DP 能继续加），但：

- **它压不了激活**；
- **DP/ZeRO 都要求一层能装进单卡**；
- 大规模下 DP 通信本身成为瓶颈。

于是需要一个**正交的新轴**：TP —— 它把参数、梯度、优化器状态**和激活**一起切开，而且**不需要在 GPU 之间通信模型参数**。

---

## 5. 与本项目的关系

- **nano 的 V7 FSDP 后端 = 这里的 ZeRO-3**。V7.0 那条「tie 时 embedding 不单独 `fully_shard`」的坑，本质就是 ZeRO-3 的「按 unit 分片 + 按需 gather」与权重共享的冲突。
- V7.3 的 DP-split `samples[rank::world_size]` 就是本章「各 rank 吃不同 micro-batch」的最小实现；也解释了为什么**激活不可能被 ZeRO 分片**（各 rank 数据不同）。
- 本章的「①重叠 ②分桶 ③no_sync」三优化，nano 一个都没做（Ray + 每 rank 独立 torch AdamW）。这不是 bug，但在 V9 测吞吐时应意识到：**nano 的 DP 通信是完全暴露的**，与源项目/工业实现不可比。
