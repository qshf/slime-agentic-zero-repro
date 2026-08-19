# 05 · 流水线并行 PP：AFAB / 1F1B / interleaved / zero bubble

> 全书索引见 [00-index.md](00-index.md)。
> 对应原文：Pipeline Parallelism（Splitting layers on various nodes - All forward, all backward / One forward, one backward and Llama 3.1 schemes / Interleaving stages / Zero bubble and DualPipe）
> 读这章的目的：V8.2 已经把 PP=2 跑通了（正确性），V9 要问「PP 到底浪费了多少」，需要一个能**先写下预期再去测**的 bubble 模型。

---

## 1. 为什么要 PP：它解决的是「模型本身太大」

CP/SP 解决的是**序列太长**；但如果瓶颈是**权重本身**（70B+ 的参数量单节点 4–8 卡放不下），切序列没用。PP 的做法极简单：**按层切**——GPU1 拿 layer 1-4，GPU2 拿 layer 5-8……每卡只存/算一部分层。

两个必须记住的性质：

**① PP 省参数显存，但不省激活显存。**
文中那张 8B 模型的图里，参数被切开了，**每卡激活显存却几乎不变**。原因（原文注释）：每卡要先做 PP 次 forward 才轮到第一次 backward，每卡只管 1/PP 的层，但要囤 PP 个 micro-batch 的激活 → `PP × (activs/PP) ≈ activs`。

> 这一条是后面所有调度演进的**唯一驱动力**：AFAB → 1F1B 不是为了减 bubble（bubble 一样大），是为了把这个 `≈ activs` 压到 `p/m` 倍。

**② PP 的通信量极低。**
只在模型深度方向的少数几个切点传激活张量，而 TP 是**每层内部**要通信好几次。所以 PP 天然适合跨节点（低带宽互联），TP 应该关在节点内。

---

## 2. Naive PP：bubble = p-1

先不做任何切分，一个 batch 顺序流过所有卡。记：

- $t_f$ / $t_b$：**一个 micro-batch、一个 stage** 的前向/反向时间（常用近似 $t_b ≈ 2·t_f$）
- `p`：PP degree（stage 数）

理想时间 $t_{id} = t_f + t_b$；实际因为等待多出 $t_{pb} = (p-1)·(t_f + t_b)$。

$$r_{bubble} = \frac{(p-1)(t_f+t_b)}{t_f+t_b} = p-1$$

p=4 就是 300% 的额外开销。灾难。

---

## 3. AFAB（all forward, all backward）：bubble = (p-1)/m

把 batch 切成 `m` 个 micro-batch。GPU2 在处理 mb1 时，GPU1 已经可以开始 mb2 了。先跑完**所有** forward，再跑完**所有** backward——这就是 AFAB。

理想时间变成 `t_id = m·(t_f + t_b)`，bubble 绝对量不变：

$$r_{bubble} = \frac{(p-1)(t_f+t_b)}{m(t_f+t_b)} = \frac{p-1}{m}$$

**优点**：forward/backward 仍然是干净的两段，训练代码组织几乎不用改，是 PP 里最好实现的一种。
**代价**：所有 micro-batch 的激活都得囤到 backward 才能释放 → **激活显存爆炸**（正是第 1 节那条性质的最坏形态）。

---

## 4. 1F1B：bubble 一样大，但激活显存从 m 降到 p

稳态里交替做「一次 forward、一次 backward」，**尽早开始 backward** 好尽早丢激活。

- **bubble 大小不变**（数一数格子就知道）→ 训练效率本身没直接提升；
- 但只需要存 **p** 个 micro-batch 的激活（而不是 m 个）；
- 省下来的显存**允许你把 m 开得更大** → 通过 `(p-1)/m` 间接把 bubble 压下去。

> 逻辑链要理清楚：**1F1B 不是直接减 bubble 的，它是通过省显存来解锁「加大 m」这条减 bubble 的路。**

代价是复杂度：forward/backward 不再是全局同步的两段，而是**各设备各自决定何时从 forward 切到 backward**。原文一句话点破了 PP 工程上最烦人的地方——

> 这就是为什么实现 PP 通常要对**训练代码和建模代码**都做相当大的改动。

（对照 V8.2：nano 必须从 `_microbatch_backward` 改成 `forward_step` + `loss_func` + `get_forward_backward_func()`，正是这句话的实证。）

**实测规律（原文 benchmark）**：
- `m ≤ p-1` 时 bubble 灾难性，性能随 PP 增大反而**下降**；
- `m = 32 >> p-1` 时低 PP degree 段明显改善，但很大的 PP degree 依旧受限；
- 现实里 m 不能无限加——**受 global batch size 上限约束**。PP degree 一大，能给的 m 就不够，bubble 必然回涨。
- 有意思的一条：小 m 时从 1 节点（p=8）扩到 2 节点（p=16）**只掉 14%**，而 TP 在同样跨节点场景通常掉 **43%**。→ **跨节点用 PP，节点内用 TP**，这条经验数字值得记住。

---

## 5. Interleaved stages：bubble = (p-1)/(v·m)

**一句话：interleaved 不是换了个调度，是换了个切层方式**——把「每卡一段连续深度」换成「每卡 v 段离散深度」，用 v 倍通信把阶梯高度削到 `1/v`。

### 5.1 层是怎么分的（读懂那张四色图的前提）

原文那张 `pp_1f1b_interleaved.svg` 的 caption 说得很清楚：

> Numbers still correspond to the micro-batch IDs, but for clarity we've colored the **first and last layers of the model** differently to illustrate how layers are spread across GPUs.

所以四种颜色里，**关键信息不是"前向/反向"，而是深浅**——深浅代表**同一张卡上的两个不同 model chunk**。图是 **p=4, v=2**。设模型 16 层：

**非 interleaved**（前面几节的做法）——按连续深度切：

```
GPU1: 1-4    GPU2: 5-8    GPU3: 9-12   GPU4: 13-16
```

**interleaved**——先切成 `p·v = 8` 个 chunk（各 2 层），再**轮转发牌**：

```
chunk: c0   c1   c2   c3   c4    c5    c6    c7
层:    1-2  3-4  5-6  7-8  9-10 11-12 13-14 15-16
卡:    G1   G2   G3   G4   G1    G2    G3    G4
       └──── 第 1 圈 ────┘└──── 第 2 圈 ────┘
```

| | GPU1 | GPU2 | GPU3 | GPU4 |
|---|---|---|---|---|
| **深色**（first layers，v=0 那圈） | 层 1-2 | 3-4 | 5-6 | 7-8 |
| **浅色**（last layers，v=1 那圈） | 层 9-10 | 11-12 | 13-14 | 15-16 |

> 原文举的例子「奇数层 (1,3,5,7) 给卡0、偶数层 (2,4,6,8) 给卡1」，就是这条轮转规则在 p=2, v=4 时的样子，同一件事。

### 5.2 一个 micro-batch 走的路：绕圈

```
前向：G1(c0) → G2(c1) → G3(c2) → G4(c3) ─┐
      ┌─────────── 绕回来 ──────────────┘
      G1(c4) → G2(c5) → G3(c6) → G4(c7) → loss

反向：G4(c7) → G3(c6) → G2(c5) → G1(c4) ─┐
      ┌─────────── 绕回来 ──────────────┘
      G4(c3) → G3(c2) → G2(c1) → G1(c0)
```

**这就是 "looping pipeline" 的字面意思**：一个 micro-batch 要在这 4 张卡的环上跑 `v=2` 圈才走完前向。

对照颜色就通了：同一个 mb 在 GPU1 上会出现**两次前向**（先深青 c0、后浅青 c4，中间隔着它在 G2/G3/G4 上跑一圈的时间）、**两次反向**（先浅紫 c4、后粉红 c0——反向倒着走，先碰到的是外圈）。

### 5.3 怎么读那条时间线

1. **左边暖机段**：GPU1 先连做深青 `1 2 3 4`（mb1-4 过 c0）；做到第 4 个时 mb1 已绕完一圈回来 → 接着做浅青 `1 2 3 4`（mb1-4 过 c4）；然后再深青 `5 6`……**同卡上深浅交替出现，就是"绕圈"在时间轴上的投影。**
2. **阶梯状起始**：G2 晚一格、G3 晚两格、G4 晚三格，左下那片灰就是这个阶梯留下的 bubble。
3. **中段四色密集交错** = 1F1B 稳态：每卡一会儿推某个 mb 往前、一会儿把另一个 mb 往回 backward。这正是 §4 引的原文那句「各设备各自决定何时翻面」的画面。
4. **中间偏右那片大灰块** = 上一批 micro-batch 排空（drain）到下一批填充（fill）的交界，bubble 的另一半。

### 5.4 为什么 bubble 除以 v（全部动机在这）

记 `t_f` = **一张卡上全部层**的前向时间。

- **非 interleaved**：GPU4 要等 G1/G2/G3 各做完 4 层 → 空转 `3 · t_f`；
- **interleaved v=2**：GPU4 的 c3 只等 G1/G2/G3 各做完 **2 层** → 空转 `3 · t_f/2`。

**每个 chunk 的粒度变成 `1/v`，阶梯的每一级就矮 v 倍：**

$$t_{pb} = \frac{(p-1)(t_f+t_b)}{v}, \qquad r_{bubble} = \frac{p-1}{v\,m}$$

**代价直接可数**：非 interleaved 前向跨卡 `p-1 = 3` 跳；interleaved 是 `p·v - 1 = 7` 跳。同一份计算，激活多穿了几次卡 → **通信量 ×v**。这就是 bubble ↔ 通信的直接 trade-off。

### 5.5 由此才有 depth-first / breadth-first

稳态里 GPU1 手上常同时有两件事可做：mb5 的 c0 前向、和 mb1 绕回来的 c4 前向。选哪个？

- **depth-first**：先做 mb1 的 c4 —— 让老的 micro-batch 尽快走完、尽早释放激活（**省显存**）；
- **breadth-first**：先做 mb5 的 c0 —— 尽快把流水线灌满（**减 bubble**）。

**Llama 3.1 的 PP = 1F1B + interleaved stages + depth/breadth 优先级可调**，那个旋钮就是为这个选择准备的。

---

## 6. Zero Bubble / DualPipe：把 backward 拆成 B 和 W

关键观察（Sea AI Lab 的 Zero Bubble 工作，DualPipe 的前身）：一次矩阵乘的 backward 其实是**两个独立操作**——

- **B**：对**输入**求梯度 → 下游（更浅的）层等着它，**有依赖，必须及时做**；
- **W**：对**权重**求梯度 → 谁也不等它，**只需在 optimizer step 之前做完**。

所以 W 可以在同 stage 的 B 之后**任意位置**灵活插入 → 拿它去**填 bubble**。ZB-H2 就是靠这个做出（理论上的）零 bubble 调度。

**DualPipe**（DeepSeek-V3/R1）在此之上再加一层：**两条流从 PP 维度的两端同时推进**并互相交错，进一步挤掉空闲。

代价：这类调度要精确测量各细粒度算子耗时，并解一个 **ILP** 来最小化最终 bubble——复杂到原文都不给代码了。

---

## 7. 公式速查 & 与 Megatron 口径的对账（V9 直接要用）

| 调度 | bubble / ideal | 激活显存（按 micro-batch 计） | 通信 |
|------|----------------|------------------------------|------|
| naive | `p-1` | 1 | 低 |
| AFAB | `(p-1)/m` | **m** | 低 |
| 1F1B | `(p-1)/m` | **p** | 低 |
| interleaved 1F1B | `(p-1)/(v·m)` | ~p | **×v** |
| zero bubble / DualPipe | ≈ 0 | ~p | 高 + 调度复杂 |

**对账（重要，别把两个公式当成矛盾）**：CLAUDE.md 里 V9 计划写的 PP bubble 是 `(pp-1)/(m+pp-1)`，本文写的是 `(p-1)/m`。两者一致，只是**分母口径不同**：

- ideal（纯计算时间）：`t_id = m(t_f+t_b)`
- total（含 bubble 的墙钟）：`t_total = (m+p-1)(t_f+t_b)`
- bubble 绝对量：`t_pb = (p-1)(t_f+t_b)`

于是
- 本文口径（**相对理想时间的额外开销**）：`t_pb / t_id = (p-1)/m`
- Megatron/V9 口径（**bubble 占总墙钟的比例**，即空转率）：`t_pb / t_total = (p-1)/(m+p-1)`

> **V9 落地提醒**：实测量到的一定是 **`t_total`**，所以要验证「PP bubble 是否符合预期」，该对的是 **`(p-1)/(m+p-1)`**；把它错当成 `(p-1)/m` 会得到一个系统性偏高的「预期」，然后误判成实现有问题。

---

## 8. 对本项目（V8.2 已做 / V9 要做）的直接映射

1. **V8.2 的 PP=2 用的是 mcore `get_forward_backward_func()`**，在 `m > 1` 时就是 1F1B。本文解释了为什么源项目必须把训练代码拆成 `forward_step` + `loss_func` 两件——**调度器要能独立控制每个 stage 何时翻面**，一个合成好的大函数交不出这个控制权。
2. **V9 的 PP 实验预期**（先写死再测）：固定 p=2，扫 m ∈ {1, 2, 4, 8}，空转率应贴近 `(p-1)/(m+p-1) = 1/(m+1)`，即 50% / 33% / 20% / 11%。若实测显著高于此，怀疑点排序：① 每 micro-batch 的 p2p 通信没被隐藏；② stage 切分不均（层数不能整除 → `t_f` 各 stage 不等，公式假设是均等的）；③ 计时没 `cuda.synchronize()`。
3. **激活显存那条能省一次实验**：如果 V9 观察到「加大 m 后显存不涨」，那正好是 1F1B 生效的证据（AFAB 会随 m 线性涨）。这是一条**免费的、判别力很强的**旁证。
4. **不做的**：interleaved（`virtual_pipeline_model_parallel_size`）和 zero bubble。按 §5.1 的分法，nano 的 p=2、28 层、v=2 会切成 4 个 7 层 chunk（G1 拿 1-7 和 15-21、G2 拿 8-14 和 22-28），理论上把 bubble 减半；但跨卡跳数从 1 变 3、通信 ×2，在 4 卡小模型上大概率吃掉收益。且 V9 的纪律是**只测量不优化**。若 V9 实测 bubble 确实是大头，把「上 interleaved」记进待办即可。
5. **拓扑结论可直接采信**：跨节点 PP 掉 14% vs TP 掉 43% → 印证 V8.2 G5 那条「TP 组必须同 NUMA/同节点」的拓扑检查是对的方向。

---

## 9. 一句话总结

> PP 沿层切模型来解决**参数放不下**，代价是引入 bubble；`m` 个 micro-batch 把 bubble 摊薄到 `(p-1)/m`（AFAB），1F1B 不改 bubble 但把激活显存从 `m` 压到 `p`、从而**解锁更大的 m**，interleaved 用 v 倍通信换 `1/v` 的 bubble，zero bubble/DualPipe 则靠拆开 backward 的 B/W 把 W 塞进空隙。
