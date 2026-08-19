# 01 · 单卡：显存账本、激活重算、梯度累积

> 对应原文：High-Level Overview / First Steps: Training on One GPU / Memory usage in transformers / Activation recomputation / Gradient accumulation / Profiling

---

## 1. 全书只围绕三个约束

1. **显存（memory）**——**硬约束**。一步训练放不下，训练就无法进行，没有折中余地。
2. **计算效率（compute efficiency）**——希望硬件绝大部分时间在算，而不是在搬数据或等别的 GPU。
3. **通信开销（communication overhead）**——通信时 GPU 空转。手段是：**尽量用节点内（快）带宽、少用节点间（慢）带宽**，并**尽量让通信与计算重叠**。

> 全书反复出现的一句话：**这三者可以互相交换**（重算拿算力换显存、TP 拿通信换显存……）。找到平衡点就是 scaling 的全部工作。

---

## 2. batch size：先定它，其它都是从它推出来的

- 记号：`bs` = 样本数，`seq` = 序列长度，`bst = bs × seq` = token 数。LLM 社区**按 token 报 batch size**（与序列长度解耦）。
- 近年常见甜点：**每批 4M–60M tokens**。Llama 1 ≈ 4M（训 1.4T tokens），DeepSeek ≈ 60M（训 14T tokens）。
- 小 batch：早期能快速穿过 loss 地形，但后期梯度噪声大、收敛不到最优。
- 大 batch：梯度估计准，但每个 token 的利用率下降、可能浪费算力。
- **在最优点附近，模型最终性能对 batch size 其实不敏感**——所以可以放心用它当调吞吐的旋钮。
- 实践：DeepSeek-V3 前 469B token 把 batch 从 3072 条**渐进**升到 15360 条，之后固定。
- 预训练序列长度普遍 **2–8k**（Llama/DeepSeek 主预训练阶段用 4k），理由很朴素：**网上比这更长的文档很罕见**。长上下文在训练末期用少量长样本追加。

---

## 3. 显存账本（一）：权重 / 梯度 / 优化器状态

参数量（简单 transformer，不含固定位置编码）：

$$N = h\,v + L\,(12h^2 + 13h) + 2h$$

`h`=hidden、`v`=vocab、`L`=层数。**大 h 时 `h²` 项主导**。

**FP32 全精度**：

$$m_{params}=4N,\quad m_{grad}=4N,\quad m_{opt}=(4+4)N \;\Rightarrow\; 16N$$

**BF16 混合精度**（今天的默认）：

$$m_{params}=2N,\quad m_{grad}=2N,\quad m_{params\_fp32}=4N,\quad m_{opt}=(4+4)N \;\Rightarrow\; 16N$$

那份 FP32 参数副本就是文献里说的 **master weights**。若再用 FP32 累积梯度（Nanotron 就这么做，因为 BF16 对小值有损、优先稳定性），再加 `4N` → **20N**。

> **反直觉但重要的一条**：**混合精度本身不省显存**——它只是把同样的量重新分配，甚至比 FP32 多 4 字节。它的价值在于 ①能用 GPU 上更快的低精度算子；②**前向的激活显存减半**（而激活往往才是大头）。

| 参数量 | FP32 或 BF16（无 FP32 梯度累积） | BF16 + FP32 梯度累积 |
|---|---|---|
| 1B | 16 GB | 20 GB |
| 7B | 112 GB | 140 GB |
| 70B | 1120 GB | 1400 GB |
| 405B | 6480 GB | 8100 GB |

**7B 就已经装不进单张 80GB H100 了。** 速算技巧：参数量 ×2 = 权重本身的最低显存（70B → 140GB）。

（另有两块不好精确算的：CUDA context 通常吃 **1–2 GB**；buffer/中间结果/碎片。全书忽略这两项。）

---

## 4. 显存账本（二）：激活——真正会爆的那部分

$$m_{act} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5\,n_{heads}\,seq}{h}\right)$$

**随 batch 线性增长、随序列长度二次增长。**

- 短序列 / 小 batch 时激活几乎可忽略；
- **约 2–4k token 起**开始显著；
- 大 batch / 长序列时，**激活是最大的一块**（而参数/梯度/优化器状态与 bs、seq 无关）。

---

## 5. 合起来：总显存等式（`m_act` 和 `m_opt` 到底是什么关系）

§3 和 §4 是**两本分开记的账**，最后要相加。但注意两者**不是同一层级的概念**——`m_opt` 是「静态三件」里的**一项**，`m_act` 是与那一整组并列的**另一大类**：

$$\underbrace{m_{params} + m_{grad} + m_{params\_fp32} + m_{opt}}_{\text{静态：只随 } N \text{ 变，与 } bs/seq \text{ 无关}} \;+\; \underbrace{m_{act}}_{\text{动态}} \;+\; \underbrace{\text{CUDA context}(1\text{–}2\text{GB}) + \text{碎片}}_{\text{书里明说忽略}}$$

前一个大括号就是 §3 那张 `16N / 20N` 表的内容（1B→16GB、7B→112GB…），后面的 `m_act` 是 §4 那个单独的公式。

### 两者的性质完全不同

| | `m_opt`（及整个静态组） | `m_act` |
|---|---|---|
| 存的是什么 | Adam 的 momentum + variance（各 4 字节/参数） | 反向求梯度时要用到的中间张量 |
| 大小取决于 | **只取决于参数量 `N`** | `L·seq·bs·h·(34 + 5·n_heads·seq/h)` |
| 与 batch / 序列的关系 | **完全无关** | **随 bs 线性、随 seq 二次** |
| 生命周期 | 第一次 optimizer step 之后常驻 | 前向涨、反向逐段释放，一步内起落 |
| 能不能分片 | **能**（ZeRO-1/2/3 正是干这个的） | **不能**（各 DP rank 吃不同数据，本来就不冗余）→ 只能靠重算 / 梯度累积 / TP / CP |

### 一个具体例子：Qwen3-0.6B（N ≈ 0.6B，L=28，h=1024，n_heads=16）

- **静态部分**：`16N` = **9.6 GB**（开 FP32 梯度累积则 `20N` = 12 GB）—— **batch 开多大这个数都不动**
- **激活**（bs=1）：
  - `seq=1024` → `28·1024·1·1024·(34+80)` = **3.35 GB**
  - `seq=4096` → `28·4096·1·1024·(34+320)` = **41.6 GB**

序列长 4 倍，激活涨 **12.4 倍** —— 因为 `5·n_heads·seq/h` 这个二次项从 80 涨到 320，早已反超常数项 34。

> 这正是 nano 在 32G 的 5090 上的实际处境：**参数才 0.6B，爆的从来不是权重，是激活。**

### 一个记号陷阱（两章里「优化器状态」指的不是一回事）

- **§3（单卡显存）**：`m_params_fp32 = 4N` 与 `m_opt = (4+4)N` 是**分开列**的；
- **[02 章 ZeRO](02-data-parallelism-zero.md)**：`k = 12` **把 master weights 也算进去了**（`4Ψ + 4Ψ + 4Ψ`）。

所以看到 `12Ψ` 时它**含** FP32 主权重，看到 `m_opt = 8N` 时**不含**。两处加总都是 `16N`，不矛盾。

### 最后一点：峰值 ≠ 简单相加

激活在**前向末尾**达峰、梯度在**反向末尾**达峰，两者错峰（见 §8 那条一步显存曲线）。把各项相加是个**偏保守的容量估算**，用来判断"装不装得下"够用，但别把它当成 profiler 会报出的峰值数字。

---

## 6. 激活重算（gradient checkpointing）

前向时丢掉部分激活，反向时从最近的 checkpoint **重算**——拿算力换显存。

| 策略 | 做法 | 代价 / 收益 |
|---|---|---|
| **Full** | 每个 transformer 层边界存一次 | 相当于反向时多做一次完整前向；**+30~40% 计算时间**，很明显 |
| **Selective** | 只丢「显存大但重算便宜」的部分——**attention 计算正是这类** | GPT-3 175B：**激活显存 -70%，计算成本仅 +2.7%** |

DeepSeek-V3 用 selective + MLA（Multi-Head Latent Attention）把 attention 激活压得更小。

> **今天大多数人已经在无意识地用 selective 重算了**：FlashAttention 内部就是在反向时重算 attention scores 而不是存下来。

### HFU vs MFU（V9 会直接用到的口径定义）

- **hardware FLOPs**：把重算也算进去的真实算子数 → `hardware FLOPs / step_time` = 实际达到的 FLOPS → 除以硬件峰值 = **HFU**（hardware FLOPs utilization）。
- **MFU**（model FLOPs utilization）：**只算模型前向+反向必需的 FLOPs，不含重算**。

为什么要有 MFU：如果一张卡显存大到不用重算，它做的总操作更少（hardware FLOPs 更低）但训得更快——**这应该被奖励而不是被惩罚**。HFU 会惩罚它，MFU 不会。**MFU 更贴近「训完一个数据集要多久」这个真正重要的问题。**

> 对本项目：V9 计划里 `tflops = 3 × fwd_flops / train_time` 用的是**模型 FLOPs 口径**（不含重算），即 MFU 侧。若之后开了重算而不改公式，MFU 数字不变（正确），但别把它当 HFU 解读。

---

## 7. 梯度累积

$$gbs = mbs \times grad\_acc$$

把一个 batch 切成若干 micro-batch，逐个 fwd/bwd 累加梯度，最后**按平均**（不是求和）做一次 optimizer step——这样结果与累积步数无关。

- 好处：**显存占用恒定**的前提下把 batch size 拉到任意大；与激活重算可叠加。
- 代价 1：**串行**多次 fwd/bwd，拖慢训练。天下没有免费午餐。
- 代价 2：需要**常驻的梯度累积 buffer**；而不累积时，梯度是边算边释放激活的，峰值反而更低。

> 关键一句：**这些 micro-batch 的 fwd/bwd 其实彼此独立、可以并行** —— 这正是数据并行的由来。**DP 就是梯度累积的并行版本。**

---

## 8. Profiler：分布式训练工具箱里最有用的那个

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    with_stack=True,
) as prof:
    for step in range(steps):
        train_step()
        prof.step()
```

trace 里能看到：CPU 线程**异步**发射 kernel、多个 CUDA stream 上计算与通信并行、各 kernel 耗时与显存分配。用来抓：**本可重叠却串行的计算/通信**、等数据传输的空转、CUDA sync 与 H2D/D2H 搬运、kernel launch 开销。

### 一步显存曲线的形状（很实用的排障知识）

一步内：前向时激活快速上涨 → 反向时梯度堆积、同时已用完的激活被逐步释放 → optimizer 阶段需要全部梯度、并更新优化器状态。

**但第一步长得不一样**：激活涨上去后会平一段——因为 PyTorch caching allocator 在做准备工作，后续步就不必再找空闲块了。**优化器状态也是第一步之后才出现的。**

> 由此解释一个常见现象：**第一步跑得好好的，第二步 OOM** —— 就是优化器状态在第一步之后才建起来。

---

## 9. 与本项目的关系

- V7.5/V8 用的 `MixedPrecisionPolicy(param_dtype=bf16)` 正对应 §3 那条「混合精度不省参数显存、省的是激活和算子速度」——所以 nano 做 fp32 精确门时必须显式改 `compute_dtype`，否则前向永远是 bf16。
- **§5 的总账**给了一个可直接套用的容量估算：nano 在 32G 卡上，0.6B 模型的静态部分只占 9.6GB，剩下的全归激活——所以调 `seq` 比调任何别的旋钮都更能决定"装不装得下"。
- V9 要报 MFU，§6 给了 HFU/MFU 的官方口径分界；nano 目前没开重算，两者数值相同，但文档里应写明用的是哪个。
- §7 的「累积 N 个微批 == 一次大批」正是 V7.2 那条 `max_diff=0.000e+00` 等价门在证的东西；playbook 特别点出**要按平均而非求和**，与 nano `_microbatch_backward` 里乘 `dp_size/global_batch_size` 的缩放同源。
