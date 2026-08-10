# V9 · 吞吐实验 —— 迭代计划（roadmap 承诺终点）

> 状态：**计划待审**（2026-08-10）。分支 `v9`。**依赖 V8.2**（见 §0.2）。
> 一句话：把前八版建起来的每根旋钮（TP/PP/CP/DP × 微批 × packing × 同步/异步）**扫一遍**，
> 用源项目的 FLOPs/MFU 口径量化，**定位瓶颈在哪一段**——这是 roadmap 主线三的最后一版。

---

## 0. 定位

### 0.1 前八版都在问「对不对」，V9 第一次问「多快」

V0–V8 的每一个门都是**正确性门**：loss/grad 逐值等价、往返 max_abs_diff==0、negative control 判别力。
唯一碰过时间的是 V5（sync 209.3s vs async 193.7s），而那还是**用 `fake_train_seconds` 模拟出来的**。

V9 换一类问题：**同样一批样本，换旋钮，wall-clock 怎么变，为什么**。
这需要三样 nano 至今没有的东西，而**源项目三样都有**：

| 源组件 | 干什么 | nano 现状 |
|---|---|---|
| `slime/utils/timer.py:15` `Timer`（`SingletonMeta` 单例，`start/end/add/log_dict` + `context` 上下文管理器 + `:55 timer` 双用装饰器） | 分段累加计时，`log_dict()` 汇总 | ❌ 无（各版本散落 `time.time()`） |
| `slime/utils/flops_utils.py:66 calculate_fwd_flops(seqlens, args)` | 按 seqlen **逐条**算前向 FLOPs = Σ层(qkv 投影 + attention + output + mlp)（含 GQA `num_query_groups`、MoE、vocab 投影） | ❌ 无 |
| `slime/utils/train_metric_utils.py:13 log_perf_data_raw` | `actor_train_tflops = 3 × fwd_flops / train_time`、`tok_per_s`、`step_time`、`wait_time_ratio` | ❌ 无 |

两处细节要照抄：

- **`3 ×` 那个系数**（`train_metric_utils.py:35`）：前向 1 份 FLOPs，反向 2 份（对权重的梯度 + 对激活的梯度），
  所以训练一步 ≈ 3× 前向。这是 MFU 口径的行业惯例。注意源对 `log_probs` 那条**不乘 3**
  （`:29`，只有前向），nano 复写时别一律乘。
- **seqlen 从哪来**：源在 `slime/utils/data.py:292` 把整个 rollout batch 的 `total_lengths`
  挂到 `Timer().seq_lens` 上，metric 层再取用（`train_metric_utils.py:26,36`）。
  **FLOPs 必须按真实逐条长度算**，用 `max_len × B` 会把 padding 的浪费算成有效算力 —— Q3 就白测了。

### 0.2 为什么必须排在 V8.2 之后

V8 的偏离 #1 是 **gloo**（单卡多 rank 被 NCCL 硬拒）。gloo 走 TCP、经 CPU 中转，
**现在测吞吐等于在测 TCP 栈，不是在测并行策略**。V8 文档已明确写下「本版不做吞吐断言，留 V9」。
V8.2 换 NCCL 之后，扫参数才有意义。

同理，V8.2 补的 PP/CP 是 V9 扫参数的**对象**——没有它们，V9 只能扫 TP 一根轴。

---

## 1. 要回答的五个问题（每个都对应一组扫描）

> 每个问题都必须有**可证伪的预期**。「跑一遍看看」不是实验，「预期 X，实测 Y，差异因为 Z」才是。

### Q1 · TP 扩展效率：TP=1→2→4，加速比是多少？为什么达不到线性？

**预期**：远低于线性。TP 每层两次 all-reduce **激活**，通信量 ∝ `batch × seqlen × hidden`，
而 0.6B 模型每层的计算量很小 → **通信占比高**。且本机**无 NVLink**（§V8.2 §1 实测拓扑：
0-1 同 NUMA、2-3 同 NUMA、跨组 SYS），TP=4 必然跨 NUMA，比 TP=2 差得更多。

**要产出的数字**：`TP=2` 同 NUMA vs `TP=2` 跨 NUMA（`CUDA_VISIBLE_DEVICES=0,1` vs `0,2`）的对比——
**这是「拓扑感知的 rank 映射」从口头常识变成实测数字的地方**（V8.2 §3.4 只断言了组的落位，没测代价）。

### Q2 · PP 的 bubble 有多大？符合 `(pp-1)/(m+pp-1)` 吗？

1F1B 的理论 bubble 比例 = `(pp_size - 1) / (num_microbatches + pp_size - 1)`。
`pp=2, m=4` → 1/5 = 20%；`m=16` → 1/17 ≈ 6%。

**预期**：实测 bubble 略高于理论（p2p 传输本身有开销、stage 负载不完全均衡）。
**这是 V9 最漂亮的一个实验**：改一个旋钮（`num_microbatches`），bubble 按公式缩小，**理论与实测对得上**。

**延伸**：若实测与理论差得远，最可能的原因是 **stage 负载不均**——PP 按层均分，
但 first stage 多一个 embedding、last stage 多一个 vocab 投影（V=151936，**在 0.6B 上这是大头**）。
这会引出源的 `--decoder-first-pipeline-num-layers` 那类不均分旋钮。

### Q3 · packing 到底省了多少？

V7.5/V7.6 只证了 packing **正确**（loss/grad 等价），**从未测过它快多少**。
padding 的浪费 = `(max_len × B - Σlen) / (max_len × B)`，取决于长度分布的方差。

**预期**：GSM8K 真实长度分布下浪费约 30-50%；`train_packing="fa2"`（varlen 块级稀疏）
应比 `"mask"`（显式 O(T²) 块对角）更快——**这正是 V7.6 退休偏离 #7 时承诺但没测的那部分**
（v7.md 偏离 #7 原文：「装 flash-attn 后换 varlen 得全部吞吐」——**V9 就是兑现这句话的地方**）。

### Q4 · rollout 与 train 的比例是多少？异步真省了吗（这次用真 trainer）？

V5 的 async<sync 是用 `fake_train_seconds=0.2` 模拟出来的。V6 之后 train 是**真的**了
（V6 实测 `train=1.7-11s`、`sync≈18-20s`、`gen=30-70s`），但**从没在真 trainer 上重跑过 V5 的对比**。

**预期**：gen 仍占大头 → 异步收益显著，但**受 `update_weights_interval` 限制**
（换权重前必须 sync 在途 gen，那一刻 overlap 断掉）。
**这会第一次让 V5 登记的「off-policy staleness」痛点在真实数字上显形**
（v5.md：「nano fake reward 感知不到，留主线二真 agent 时显现」——至今仍未显现）。

### Q5 · 权重同步（disk reload）占多少？值得换 NCCL broadcast 吗？

V6/V7.3 实测 `sync≈18-29s`，而 train 只要 1-13s —— **同步比训练还贵**。
这是 V6/V7 偏离 #4（disk reload vs 源 NCCL broadcast）一直挂着的那条。

**预期**：sync 占比高到足以证明「该换 broadcast」。
**V9 不实现 broadcast**（那是独立一版的工作量），但要给出**该不该做的量化依据**——
这正是把「登记的偏离」转成「有优先级的待办」的正确方式。

---

## 2. 范围

### 做

- **V9.0 · 计时与 FLOPs 基础设施**（忠实重写源三件套）：
  `toy_rl/utils/timer.py`（≡ 源 `slime/utils/timer.py`，`Timer` 单例 + `@timer` 装饰器 + `log_dict()`）、
  `toy_rl/utils/flops_utils.py`（≡ 源 `calculate_fwd_flops`，含 GQA 与 vocab 投影项）、
  `mini_slime/perf_metrics.py`（≡ 源 `log_perf_data_raw`，出 `tflops` / `tok_per_s` / `wait_time_ratio`）。
  **接进 V7.4 已建的 `learner_metrics` 分相位 trace**（那里已有 phase 骨架，V9 把时间填进去）。
- **V9.1 · 扫描脚本**：`scripts/bench_v9_sweep.py`，按配置矩阵跑、落 CSV/JSON、出表。
- **V9.2 · 实验报告**：`docs/decisions/v9.md` 给出 Q1–Q5 的**预期 vs 实测 vs 解释**三段式。

### 不做

- **不追求绝对 MFU 数字好看**。0.6B 模型 + 4 张消费卡 + 无 NVLink，MFU 必然低。
  **V9 的产出是「哪根轴是瓶颈、为什么」的相对结论**，不是刷 benchmark。
- **不实现新的优化**（NCCL broadcast 权重同步、VPP、`DistributedOptimizer`、动态批）。
  V9 只**测量并给出优先级**；实现是后续版本的事。混进来会让「测量」和「优化」互相污染。
- **不做多机**。单机 4 卡。

---

## 3. 实验矩阵（初稿，实施时按 §1 的问题裁剪）

| 组 | 固定 | 扫 | 观测 |
|---|---|---|---|
| A（Q1） | PP=1 CP=1 | TP ∈ {1,2,4}；TP=2 分同 NUMA / 跨 NUMA | `train_time`、`tflops`、加速比 |
| B（Q2） | TP=1 CP=1 PP=2 | `num_microbatches` ∈ {1,2,4,8,16} | bubble 实测 vs `(pp-1)/(m+pp-1)` |
| C（Q3） | TP=2 PP=1 | `train_packing` ∈ {False, "mask", "fa2"} | `tok_per_s`、padding 浪费率 |
| D（Q4） | 最优静态配置 | sync vs async × `update_weights_interval` ∈ {1,2,4} | `wait_time_ratio`、总 wall-clock |
| E（Q5） | 同 D | — | `sync_time / step_time` 占比 |

**每格重复 ≥3 次取中位数**，并报告离散度。单次计时在共享 GPU 上噪声很大——
V5 就吃过这个亏（`saving=15.6s` 远超理论 10s，多出部分事后归因为 gen 方差）。

---

## 4. 方法学纪律（V9 特有的坑）

正确性实验和性能实验的失效模式完全不同。V7.5 的「度量教训」在这里换了一副面孔：

1. **必须 `torch.cuda.synchronize()` 再停表**。CUDA 是异步的，不同步的计时**测的是 kernel launch 时间不是执行时间**——
   会得到「TP=4 比 TP=1 快」这种荒谬结论。源在权重同步的关键处就显式 sync
   （`hf_weight_iterator_direct.py:64`）。**注意源的 `Timer` 自己不 sync**（`timer.py:20-32` 只调 `time()`）——
   它测的是 rollout/train 这类**跨 `ray.get` 的粗粒度阶段**，边界天然被同步点隔开。
   nano 的 V9 要测**层级更细的段**（单次 forward、单次 all-reduce），**必须自己加 sync**，
   否则精度不够。这是一条要写进代码注释的偏离。
2. **必须丢弃 warmup 轮**。首轮含 CUDA context 创建、cuDNN autotune、NCCL 建链、内存池预热，
   能比稳态慢数倍。至少丢前 2 轮。
3. **必须报告卡是否被别人占**。实测 4 张卡常被 `vllm-qwen36-27b` 占 28.5G（V8 就因此 OOM）。
   **共享 GPU 上的吞吐数字不可比** —— 每次记录 `nvidia-smi` 快照进结果文件。
4. **理论值先写下来再测**。§1 每个 Q 都先写预期。**先测后编解释**是性能工程最常见的自欺——
   任何数字都能事后编出理由，只有预先写下的预期才能被证伪。
5. **区分「测量」与「优化」**。V9 不改被测代码（除了加计时）。发现瓶颈 → 记进待办，不当场改。

---

## 5. 风险

| # | 风险 | 预案 |
|---|---|---|
| R1 | **GPU 被 vllm 占**（实测各卡剩 ~3.5G）→ 数字不可比甚至 OOM | 阻塞验收不阻塞写码；跑前必须协调独占 |
| R2 | 0.6B 太小，通信占比过高 → 所有并行轴都「不划算」，结论平淡 | **这本身就是结论**（小模型不该上多维并行）；用 `--layers` 放大或造合成大 config 做对照 |
| R3 | 扫描组合爆炸 | 按 §1 的五个问题裁剪，每组只扫**一根**轴（其余固定） |
| R4 | `calculate_fwd_flops` 的 nano 版与源口径不一致 → tflops 不可比 | 忠实复写源公式并用小 config 手算校验一次 |

---

## 6. 与 roadmap 的关系

V9 是 roadmap（`docs/system-roadmap.md` §2.4）主线三的最后一版，**跑完即三条主线全部收官**。
它的产出（Q5 的量化依据）会把两条长期挂着的偏离转成有优先级的后续：
**权重同步换 NCCL broadcast**（V6/V7 偏离 #4）与 **`DistributedOptimizer`**（V8 偏离 #7）。
