# Ultra-Scale Playbook 读书笔记（全书）

> 原文：[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)（HuggingFace / nanotron）
> 说明：HF Space 是动态渲染的，直接抓页面只能拿到 loading 占位符；正文实际来自该 Space 仓库的静态 `dist/index.html`。
> 读它的动机：nano 项目走到 V8.2 已经手搓了 DP×TP×PP×CP 四根轴，V9 要第一次问"多快"。这本书是**唯一一份把五种并行、显存账本、重叠数学和真实 benchmark 教训写在一起**的系统材料。

---

## 目录

| # | 文件 | 内容 | 对 nano 最有用的一条 |
|---|---|---|---|
| 01 | [单卡：显存账本 / 重算 / 梯度累积](01-single-gpu-memory.md) | 三大约束、参数与激活显存公式、**总显存等式（静态 + 激活）**、full vs selective 重算、**HFU vs MFU 口径**、梯度累积 | MFU 的官方定义（V9 要报的就是它） |
| 02 | [数据并行与 ZeRO-1/2/3](02-data-parallelism-zero.md) | DP 三优化（重叠/分桶/no_sync）、`gbs = mbs×grad_acc×dp`、ZeRO 三阶段显存与通信 | **激活不可能被 ZeRO 分片**；V7 FSDP = ZeRO-3 |
| 03 | [张量并行 TP 与序列并行 SP](03-tensor-sequence-parallelism.md) | column/row linear 与对应原语、MLP/MHA 的切法、f/g 共轭对、形状速查表 | LN 梯度何时天然同步、何时必须 all-reduce（V8 qk-norm 坑的正解） |
| 04 | [上下文并行 CP](04-context-parallelism.md) | CP vs SP 的作用域差别、Ring Attention 三步、因果 mask 的负载倾斜、Zig-Zag | **"CP 组内要 all-reduce 梯度"** —— V8.2 那个"计划外新发现"的出处 |
| 05 | [流水线并行 PP](05-pipeline-parallelism.md) | naive → AFAB → 1F1B → interleaved → zero bubble/DualPipe，bubble 公式全链 | `(p−1)/m` 与 `(p−1)/(m+p−1)` 的口径对账 |
| 06 | [专家并行 EP + 5D 总览](06-expert-parallelism-and-5d.md) | EP 与 DP 的关系、**PP vs ZeRO-3 对比表**、组合原则、总表 | 传权重 vs 传激活 —— 解释 FSDP 与 Megatron 后端的性格差异 |
| 07 | [怎么选配置](07-finding-best-config.md) | 三步决策法、上千配置的热力图洞察、benchmark 工程血泪 | **"TP 和 PP 谁快"主要量的是你代码的成熟度** |
| 08 | [GPU 内幕](08-gpu-internals.md) | SM/warp/内存层级、coalescing/tiling/coarsening、融合、FlashAttention、混合精度与 FP8 | bf16 尾数只有 7 位 —— nano fp32/bf16 分层验收门的物理依据 |
| 09 | [附录：集合通信 / 尺度 / 重叠数学](09-appendix-collectives-and-overlap-math.md) | Ring AllReduce 成本、`16h²` 速算、**四个 `t_comm/t_compute` 公式** | FLOPs 公式 `6·tokens·params` 及其被丢掉的二次项 |

---

## 全书的一句话骨架

> 训练受制于**显存、计算效率、通信开销**三者，而它们可以互相交换。
> 单卡上先用**重算**（算力换显存）和**梯度累积**（时间换显存）；
> 多卡上，**DP** 沿 batch 切、**ZeRO** 沿 DP 轴消冗余、**TP** 沿 hidden 切、**SP/CP** 沿序列切、**PP** 沿层切、**EP** 沿专家切；
> 每一种都是"用某种通信换某种显存"，因此**组合原则由带宽拓扑决定**：TP 关在节点内，PP/ZeRO-3 跨节点；
> 最后，所有这些的收益都受制于**你能不能把通信藏到计算后面**——而这件事在 GPU 内部还要和 SM 争抢资源。

---

## 五条最反直觉的结论（挑出来单独记）

1. **混合精度本身不省显存**（16N vs 16N，开 FP32 梯度累积还多 4N）。它省的是**激活**，赚的是**算子速度**。→ [01](01-single-gpu-memory.md)
2. **PP 不省激活显存**。每卡只管 1/PP 的层，但要囤 PP 个 micro-batch 的激活，`PP × activs/PP ≈ activs`。这才是 AFAB→1F1B 的唯一动力。→ [05](05-pipeline-parallelism.md)
3. **1F1B 并不直接减 bubble**（bubble 大小与 AFAB 相同）。它省显存，从而**解锁更大的 m**，再由 `(p−1)/m` 间接减 bubble。→ [05](05-pipeline-parallelism.md)
4. **TP 和 PP 的通信/计算比值与 batch size 和序列长度无关**，只取决于 `h`、并行度、带宽。加 batch 救不了它们。→ [09](09-appendix-collectives-and-overlap-math.md)
5. **"重叠通信与计算"不是免费的**：NCCL 通信 kernel 会占用与计算相同的 SM，重叠会拉低计算吞吐。全书前面章节都建立在"重叠免费"这个略微失真的假设上。→ [07](07-finding-best-config.md)

---

## 本书证实 / 补上 nano 已踩过的坑

| nano 的经历 | 书里对应的位置 |
|---|---|
| V7.0 tie 时 embedding 不能单独 `fully_shard` | ZeRO-3 的按 unit 分片语义（[02](02-data-parallelism-zero.md)） |
| V7.2 累积 N 微批 == 一次大批（`max_diff=0`） | 梯度累积要按**平均**而非求和（[01](01-single-gpu-memory.md)） |
| V7.5/V8 的 fp32 精确门 vs bf16 舍入门 | bf16 尾数 7 位、epsilon 比 fp32 大 4 个数量级（[08](08-gpu-internals.md)） |
| V7.6/V8.2 "FA2 拒收 fp32" | FA2/FA3 是为半精度 Tensor Core 路径深度定制的实现（[08](08-gpu-internals.md)） |
| V8 qk_layernorm 必须跨 TP SUM all-reduce | "TP 区 LN 梯度天然同步"的**例外**：qk-norm 作用在被切开的头上（[03](03-tensor-sequence-parallelism.md)） |
| V8 dropout 未关导致等价门假阳性 | TP 下必须同步 dropout 种子；mcore **故意**让各 rank RNG 不同（[03](03-tensor-sequence-parallelism.md)） |
| **V8.2 "计划外新发现"：全部梯度要跨 DP×CP 规约** | **CP 组内要 all-reduce 梯度，就像 DP 一样**（[04](04-context-parallelism.md)）—— 书里明写了，只是当时没读到 |
| V8.2 G3c 反例：换成朴素连续切分 cosine 掉到 0.588 | 书只讲 Zig-Zag 的**负载均衡**动机；nano 实测它还是**正确性契约**（[04](04-context-parallelism.md)）—— 这条**超出**了书 |
