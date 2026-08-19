# 09 · 附录：集合通信、尺度速算、重叠数学

> 对应原文：A0 Parallel Programming Crash Course / A2 Typical Scales in LLM Training / A3 Math for Compute/Communication Overlap

---

## 1. 集合通信原语（A0）

| 原语 | 语义 |
|---|---|
| **Broadcast** | 一个 rank 的张量复制给所有 rank |
| **Reduce** | 所有 rank 的张量做规约（sum/max/…），结果只在**一个** rank |
| **AllReduce** | 同上，但**每个** rank 都拿到结果 |
| **Gather** | 各 rank 的分片收集到**一个** rank |
| **AllGather** | 各 rank 的分片收集到**所有** rank |
| **Scatter** | 一个 rank 的张量切开分发给各 rank |
| **ReduceScatter** | 规约 + 每个 rank 只拿属于自己的那一片 |
| **Barrier** | 纯同步，所有 rank 到齐才继续 |
| **NCCL** | NVIDIA 的 GPU 集合通信库实现 |

### Ring AllReduce 与两条必须记住的成本

Ring AllReduce = **ReduceScatter + AllGather** 两个阶段：

- **ReduceScatter**：每卡把数据切成 N 块，环上逐轮发一块 / 收一块并**就地加**，N−1 轮后每卡持有某一块的**全局规约结果**；
- **AllGather**：再转 N−1 轮，把各自那块完整结果传遍全环。

每卡收发各 **N−1** 次、每次 `K/N` 个值 → 总传输量 `2(N−1)K/N`，**N 大时 ≈ 2K**。

> **两条结论：**
> 1. **AllReduce 的通信成本 ≈ 2K**（K = 参数总数）；
> 2. **AllReduce ≡ ReduceScatter + AllGather，且这两者各自的成本只有 AllReduce 的一半（≈ K）。**
>
> 第 2 条是 [03 章](03-tensor-sequence-parallelism.md) 那个「**TP+SP 操作数翻倍但通信量不变**」结论的全部依据。

---

## 2. 尺度速算表（A2）

**元素数**（要换成字节再乘以每个数的字节数：BF16=2、FP32=4）：

| 项 | 元素数 |
|---|---|
| 输入 token | `seq · mbs` |
| 单层 hidden state | `seq · mbs · h` |
| 单个权重矩阵 | 约 `h²`（梯度同样大小） |
| 单个权重矩阵的优化器状态 | 约 **`6h²`**（Adam 混合精度：momentum + variance 各 `2h²`（FP32），master weights `2h²`） |

**每个 transformer block 的参数**：

- Attention：QKV 投影 `3h²` + 输出投影 `h²`
- MLP（**带 GLU**）：gate + up `8h²`（两个 `h×4h`）+ down `4h²`（一个 `4h×h`）
- **合计：带 GLU 是 `16h²`，不带 GLU 是 `12h²`**
- 全模型：`16h² · num_layers` + input embedding `vocab·h` + LM head `vocab·h`（若不 tie）+ 位置编码（若有）`max_seq_len·h`

**FLOPs（这条 V9 直接要用）**：

$$\text{forward} \approx 2\cdot num\_tokens \cdot num\_params, \qquad \text{backward} \approx 2\times\text{forward}$$

$$\Rightarrow \text{fwd+bwd} \approx 6\cdot num\_tokens\cdot num\_params$$

更精确的版本（含 attention 的二次项）：

$$6\cdot seq\_len \cdot num\_params + 12\cdot num\_layers\cdot h\cdot seq\_len^2$$

> **与本项目的对账**：V9 计划里 `tflops = 3 × fwd_flops / train_time` 的那个 **3** 就是这里的「backward = 2× forward，合计 3×」。而**第二项 `12·L·h·seq²` 是 playbook 明说为了简化而丢掉的**——nano 若在长序列上测吞吐，丢掉它会**低估**真实 FLOPs、从而**低估** MFU。源项目 `flops_utils.py:66 calculate_fwd_flops` 逐条 seqlen 算并含 GQA/vocab 项，正是这个更精确版本的实现。**V9 应确认自己复写的是哪一版，并在文档写明。**

---

## 3. 重叠数学：什么时候通信藏得住（A3）

统一思路：算 `t_comm / t_compute`，**≤ 1 就能完全重叠**。

### DP（ZeRO-0）

梯度总量 `≈ num_layers · 16h²`，按 bucket（默认 25MB）通信：

$$t_{comm} = \frac{bucket\_size\cdot 2(DP-1)}{DP\cdot peak\_bw},\qquad t_{compute}=\frac{4\cdot num\_tokens\cdot num\_params}{peak\_flops}$$

$$\boxed{\frac{t_{comm}}{t_{compute}} = \frac{num\_params}{2\cdot num\_tokens}\cdot\frac{DP-1}{DP}\cdot\frac{peak\_flops}{peak\_bw} \le 1}$$

→ **token 数越多越容易藏住**。

### ZeRO-3 / FSDP

每个 block：前向 all-gather 参数 `16h²/DP`、反向 all-gather 参数 `16h²/DP` + reduce-scatter 梯度 `16h²/DP` → **每 block `3·16h²/DP`**，全模型 `3·num_layers·16h²/DP`。

$$t_{comm}=16h^2\cdot\frac{DP-1}{DP\cdot peak\_bw},\qquad t_{compute}=\frac{32\cdot seq\cdot mbs\cdot h^2}{peak\_flops}$$

$$\boxed{\frac{t_{comm}}{t_{compute}}=\frac{1}{2\cdot seq\_len\cdot mbs}\cdot\frac{DP-1}{DP}\cdot\frac{peak\_flops}{peak\_bw}\le 1}$$

→ **`seq_len · mbs` 越大越藏得住** —— 这就是 [06 章](06-expert-parallelism-and-5d.md)「ZeRO-3 偏好大 mbs/seq_len」的出处。

### TP

每个 block 有 2 个 column-linear + 2 个 row-linear，每次通信 `seq·mbs·h/TP` → **每 block `8·seq·mbs·h/TP`**。

$$t_{comm}=\frac{seq\cdot mbs\cdot h(TP-1)}{TP\cdot peak\_bw},\qquad t_{compute}=\frac{2\cdot seq\cdot mbs\cdot h^2}{TP\cdot peak\_flops}$$

$$\boxed{\frac{t_{comm}}{t_{compute}}=\frac{TP-1}{2h}\cdot\frac{peak\_flops}{peak\_bw}\le 1}$$

> **非常反直觉的一条：这个比值只依赖 `h` 和 `TP`，与序列长度、batch size 无关。** 意思是**加大 batch 救不了 TP 的通信开销**——只有更大的 hidden 维或更好的带宽能救。

### PP

每个 micro-batch：前向收发激活 `2·seq·mbs·h`、反向收发梯度 `2·seq·mbs·h` → **每 micro-batch `4·seq·mbs·h`**，`gas` 步累积共 `4·gas·seq·mbs·h`。

$$t_{compute}=\frac{32\cdot seq\cdot mbs\cdot h^2\cdot num\_layers\_in\_next\_pp}{peak\_flops},\qquad t_{comm}=\frac{seq\cdot mbs\cdot h}{peak\_bw}$$

$$\boxed{\frac{t_{comm}}{t_{compute}}=\frac{peak\_flops}{32\cdot h\cdot num\_layers\_in\_next\_pp\cdot peak\_bw}\le 1}$$

> 同样**与 seq/batch 无关**，只取决于 `h`、**下一个 stage 的层数**、以及算力/P2P 带宽之比。
> **实践含义**：PP stage 分得越细（每 stage 层数越少），P2P 通信越难藏住 —— 这是 bubble 之外，PP degree 不能无限加的**第二个**理由。

---

## 4. 四个公式的横向对比（一眼看出该调哪个旋钮）

| 并行 | 比值依赖 | 想改善通信占比，该调什么 |
|---|---|---|
| **DP** | `num_params / num_tokens` | **加大 token 数**（batch × seq） |
| **ZeRO-3** | `1 / (seq · mbs)` | **加大 mbs 或 seq** |
| **TP** | `(TP−1) / h` | **只能减小 TP 或换更大的 h / 更好的带宽**（batch 无用） |
| **PP** | `1 / (h · 下一 stage 层数)` | **减小 PP degree（每 stage 层数变多）**（batch 无用） |

> **V9 可直接用的预期**：TP 和 PP 的通信占比**与 batch size 无关**。所以在 nano 上扫 batch 时，如果观察到「加大 batch 让 TP 或 PP 的通信占比明显下降」，**第一怀疑不是理论错了，而是测量把别的东西（如 kernel launch 开销、未 warmup、未 `cuda.synchronize()`）算进了通信时间**。这是一条**判别力很强的自检**。
