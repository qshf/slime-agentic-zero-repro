# 03 · 张量并行 TP 与序列并行 SP

> 对应原文：Tensor Parallelism / Tensor parallelism in a transformer block / Sequence parallelism

---

## 1. 两条恒等式就是 TP 的全部数学

$$A\cdot B = A\begin{bmatrix}B_1 & B_2 & \cdots\end{bmatrix} = \begin{bmatrix}AB_1 & AB_2 & \cdots\end{bmatrix}$$

$$A\cdot B = \begin{bmatrix}A_1 & A_2 & \cdots\end{bmatrix}\begin{bmatrix}B_1\\B_2\\\vdots\end{bmatrix} = \sum_i A_i B_i$$

神经网络里写成 `X × W`（X=激活，W=权重）。按**列**切 W 或按**行**切 W，**需要的通信原语完全不同**：

| | 权重切法 | 输入怎么办 | 输出怎么办 |
|---|---|---|---|
| **column-linear** | 按列切 `W` | 输入**完整复制**到每卡（broadcast） | 各卡得到部分列 → **all-gather** |
| **row-linear** | 按行切 `W` | 输入也必须**切开**（scatter） | 各卡得到部分和 → **all-reduce** |

> TP 相对 ZeRO-3 的根本区别：**ZeRO-3 要把权重 gather 回完整再算；TP 从不 gather 权重，各算各的部分结果**。所以 TP 同时切了参数、梯度、优化器状态**和激活**。

---

## 2. 装进 transformer block

**MLP：column-linear → row-linear。**
一次 broadcast 复制输入 + 前向一次 all-reduce。**顺序不能反**——先 row 后 col 会在中间多一次 all-reduce。（训练中那个 broadcast 其实不需要，因为可以保证输入本来就在 TP 组内同步。）

**MHA：QKV 走 column-parallel，output projection 走 row-linear。**
column 切在这里有非常自然的语义：**每张卡负责一部分注意力头**。MQA/GQA 同理（K/V 在多个 query 之间共享）。

**为什么 TP 对这两块特别有效**：Attention 沿 `num_attention_heads` 天然独立，MLP 沿 `hidden_dim` 天然独立。

### 一条硬约束

**TP degree 不应超过注意力头数**（因为 QKV 是沿头切的）。GQA 下有 `num_attention_heads ≥ num_kv_heads`，理论上 TP 仍可开到 `num_attention_heads`，但**必须自己保证 K/V 头在各 TP rank 间同步**。例：Llama-3 8B 是 32 query heads / 8 kv heads，TP 理论可到 32，但实现要小心。

---

## 3. TP 不是银弹：通信在关键路径上

TP 把通信原语**直接插进了模型的计算路径**，所以**很难像 ZeRO 那样藏到计算后面**。每个 decoder 层的前向里那次 all-reduce 是一个**无法重叠的同步点**——必须先把各 rank 的部分结果合起来，才能做后面的 LayerNorm。它直接加在前向的**关键路径**上。

（Megatron-LM / Nanotron 做了部分重叠：FC1 的一部分矩阵乘结果先发出去、剩下的继续算。Domino 等工作在推进这个方向。）

**实测规律（必须记住）**：
- TP ≤ 8（节点内，走 NVLink）尚可；
- **TP=8 → TP=16 是断崖**（开始跨节点走 EFA/IB）；
- TP=16 → 32 更陡。
- 所以**TP 一律关在节点内**。

**收益**：把参数/梯度/优化器状态/（部分）激活都摊薄，70B 模型能开始塞进单节点 8 卡。**代价**：单卡吞吐随 TP 上升而下降，换来的是能开更大的 batch。

### 两个容易搞错的同步细节

- **纯 TP 的普通 LayerNorm 权重梯度不需要 all-reduce**：各 TP rank 在 row-linear 的 **all-reduce** 后持有相同的完整 residual-stream 激活；反传到这里的上游梯度也相同，所以每张卡已经独立算出了同一份完整 `dγ/dβ`。原文的 "after the all-gather" 若指 ring all-reduce 的末段，则没有错，但它不是图上一个额外的、独立的 TP all-gather；从算法/模型图视角，保证副本一致的是那次 all-reduce。
- **dropout 的 RNG 要按张量布局选**：对各 TP rank 都持有的复制张量，必须使用相同 mask（同 RNG 状态），否则副本会分叉；对 TP 切分的张量，则应使用各 TP rank 不同的 RNG。Megatron 也正是分别维护这两类 RNG，而不是无条件地“TP 内同步种子”。

> 对本项目：V8 那条最实质的坑——`q/k_layernorm` 的梯度**必须**跨 TP 做 SUM all-reduce——正是上面第一条的**例外**：qk-layernorm 作用在**每个头**上，而头被 TP 切开了，所以各 rank 拿到的是**部分和**而不是相同值。playbook 只讲了「LN 权重天然同步」的常规情形，nano 撞上的是它的边界。另外 V8 的等价门必须关掉 dropout：`model_parallel_cuda_manual_seed` 对**TP 切分区**故意给各 rank 不同 RNG；因此 TP=1/TP=2 的随机轨迹不能逐值比较。这是测试可比性问题，不能倒推成“任何 TP dropout 都应使用不同 mask”。

---

## 4. 序列并行 SP：把 TP 管不到的那部分沿序列切

**动机**：LayerNorm、dropout 这些操作**需要完整的 hidden 维**才能算对（LN 要在 h 上求均值方差），所以 TP 区之外它们的激活是 `(b, s, h)` 完整的——虽然计算便宜，**显存却不便宜**。

$$\text{LayerNorm}(x) = \gamma\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

SP 的办法：**这些算子沿 `s` 切**（而不是沿 `h`）。于是全程要么切 s、要么切 h，**最大激活降到 `b·s·h/tp`**。

> ⚠️ **术语坑**：这里的 "sequence parallelism" 是**和 TP 绑定**的、只管 dropout/LayerNorm。而长序列下为 attention 做的沿序列切分（Ring Attention 那套）本书叫 **context parallelism**，**可以独立使用**。两者常被混称。

### f / g 共轭对（本节的核心机制）

TP 区的 `f` / `f*`：
- 前向：`f` = no-op（激活本来就在各 rank 复制），`f*` = **all-reduce**（合并部分和）
- 反向：`f*` = no-op，`f` = **all-reduce**（同步梯度）

→ **一个是 no-op 时另一个就是 all-reduce，反向对调**，故称"共轭对"。

SP 区用 `g` / `g*`：**刻意不用 all-reduce**（那会把完整激活 gather 出来，正好抵消 SP 的意义）：
- `g` = **all-gather**（SP → TP：把 `(b, s/2, h)` 沿 s 拼回 `(b, s, h)`——原文的理由是 column-linear 需要完整的 hidden 维 h）
- `g*` = **reduce-scatter**（TP → SP：既完成 row-linear 需要的规约，又顺手沿 s 切开）

### 一次 MLP 的完整走位

1. **LayerNorm（SP 区）**：输入 `X1*, X2*` 形状 `(b, s/2, h)`，各卡独立算 LN → `Y1*, Y2*`
2. **SP → TP（`g` = all-gather）**：拼回 `Y` 形状 `(b, s, h)`
3. **第一个 linear（TP 区）**：`A1/A2` 是 column-linear，沿 h 切 → GELU 各卡独立 → `Z1*, Z2*` 形状 `(b, s, h/2)`
4. **第二个 linear（TP 区）**：`B1/B2` 是 row-linear，恢复 h → `W1, W2` 形状 `(b, s, h)`，需要相加
5. **TP → SP（`g*` = reduce-scatter）**：规约 + 沿 s 散开 → `W1*, W2*` 形状 `(b, s/2, h)`

### 形状速查表

| 区域 | 仅 TP | TP + SP |
|---|---|---|
| 进 TP（column-linear） | h: 分片 / s: 完整 | h: 分片 / **s: all-gather 成完整** |
| TP 区内 | h: 分片 / s: 完整 | h: 分片 / s: 完整 |
| 出 TP（row-linear） | h: 完整（all-reduce） / s: 完整 | h: 完整（**reduce-scatter**） / **s: 分片** |
| SP 区 | h: 完整 / s: 完整 | h: 完整 / **s: 分片** |

embedding 层（row-linear，沿 vocab 切）同理：仅 TP 是 all-reduce，TP+SP 是 reduce-scatter 并沿 s 切开。

### SP 是不是更贵？——不是

仅 TP： 每个 transformer block, 前向 **2 次 all-reduce**。
TP+SP：每个 transformer block, **2 次 all-gather + 2 次 reduce-scatter**。

操作数翻倍，**但通信量相同**——因为 **all-reduce ≡ reduce-scatter + all-gather**（见附录 Ring AllReduce）。反向同理（取共轭）。

**但有一条真实的额外开销**：SP 区的 LayerNorm 在不同 rank 上处理**不同的序列片段**，所以**它们的梯度确实不同，必须 all-reduce**——和纯 TP 那条「天然同步」正好相反。好在 LN 参数很少，开销可忽略。

### 实测

3B 模型 / seq 4096 扫 TP+SP：仍然是**计算效率（左）与显存容量（右）的取舍**。最大跌幅仍在 **TP=8 → 16**（NVLink → EFA）。SP 带来的激活节省让 batch 能开得比纯 TP 大得多——70B 模型在 TP+SP=16 下能塞下 **16k** 序列。

---

## 5. TP+SP 的两个剩余极限

1. **序列继续加长，TP 区内的激活还是会爆** → 需要 **CP**；
2. **模型大到 TP=8 都装不下** → 跨节点 TP 太慢 → 需要 **PP**。
