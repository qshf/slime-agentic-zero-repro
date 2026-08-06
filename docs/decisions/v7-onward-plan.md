# V7 收尾 + Infra 落地 — 后续方案建议

> 状态：建议（2026-08-06）
> 取代关系：**接续 `v7-plan.md`**（那份写于 V7.0 实施前、已过时），保留其 V7.3 内核，废弃其 packing 顺序与 V7.4 框架。
> 前提共识：infra（`agentic-rl-infra-lab`）作为 **规范/铺垫**；其代码是参考，**本 repo 里按 slime-agentic 源忠实重写**，用 infra 的 CPU 不变量当验收测试。

---

## 0. 为什么 `v7-plan.md` 要接续而非照做

| 维度 | `v7-plan.md` 假设 | 现实（`v7.md`） | 结论 |
|---|---|---|---|
| tie 处理 | 未提 | V7.0 已做忠实 tie（meta device 会 broadcast hang） | 计划漏了 0.6B 上最硬的正确性点 |
| packing 难度 | "低风险，逻辑清晰"，排在 V7.1 | **暂缓**：5090 无 flash-attn，varlen 无隔离会静默算错 loss | 计划把最大风险当最小 |
| V7.2 顺序 | 依赖 packing 先落 | 实际在无 packing 的微批上做，`max_diff=0` | 计划的子版本依赖链错 |
| infra 落地 | **完全没提** | 对齐说明已给出每个模块的落点+时机 | 计划盲区 = 本次真正工作量 |
| V7.4 "broadcast<disk" | 当作要通过的契约 | — | 单机 2 卡未必复现；是测量非不变量 |

**仍然有效的内核**：V7.3（Ray + 权重同步，disk reload 先行）。其余重新推导。

---

## 1. 铁律下的落地原则（infra as spec）

1. **不把 infra 的 V10–V14 当第二条平行版本线克隆进 repo。** infra 是 learner 基础设施的独立课程；repo 是 Agentic RL 训练系统。两者版本号不对应。
2. **每个 infra 契约在它的 seam 落地、按"何时需要"的时机接**（对齐说明 §seam map），源锚点仍是 slime-agentic 对应位置，infra 只提供"落地后语义不变"的可判定不变量。
3. **每处落地要么退休一条已登记偏离，要么重新登记偏离**（铁律三要素）。

seam map（来自 `infra-slime-alignment.md`，作为本方案的落点依据）：

| infra 契约 | repo 落点 | 触发时机 |
|---|---|---|
| learner_contract | `_convert_samples_to_train_data` 之后、`_pad_batch` 之前 | 最先接，收益最大 |
| learner_metrics | 替换 `Trainer` 的 metric dict | 零风险，尽早 |
| packing | padding→packing 迁移时顶上 | 上 packing 时 |
| weight_publishing | 泛化 `WeightUpdater`（单 engine = 1-replica） | 多 rollout replica 时 |
| batch_planning | 多卡 DP 的等样本数/等累积步 | learner 多卡 DP 时 |
| checkpointing | `save_pretrained` → DCP + manifest | learner 上 FSDP checkpoint 时 |

---

## 2. 建议的迭代顺序（子版本）

顺序原则：**先关掉 V7 的未闭合项，再接零风险 infra，最后按需要接重的**。每步都能独立在 5090 验证、独立回归。

### V7.3 · 收尾 V7：Ray + 权重同步（disk reload）+ 版本戳

保留 `v7-plan.md` 的 V7.3 内核，**多做一件小事**：补 infra 缺口 1（per-sample rollout 版本戳）。

- **实施**：
  1. 新建 `mini_slime/train_fsdp.py`（复制 `train_ray.py` 结构，TrainRayActor 内换 `FSDPTrainer`）。
  2. `update_weights()` 先用 **disk reload**（复用 V6 `WeightUpdater`，风险最低）。
  3. **缺口 1**：rollout 生成时把当时 `weight_version` 写进 `Sample`，由 `_convert_samples_to_train_data` 带出。改动极小，却是所有 staleness/版本语义的前提。
- **验收**：weight_version 递增、rollout engine reload 成功、reward_mean 与 V6 同量级；每条样本带 `rollout_policy_version`。
- **源锚点**：`update_weight_utils.py`（同步）、`train.py`（编排）。
- **偏离**：权重同步仍 disk reload（非 NCCL broadcast），登记，留后续泛化到 weight_publishing。
- **废弃**：`v7-plan.md` 的 V7.4「broadcast<disk 当契约」——改为多 replica 出现时通过 weight_publishing 契约落，不作为独立性能断言。

### V7.4 · 接零风险 infra：learner ABI + 阶段 trace

infra 明言这两个"零风险、无新 GPU 行为、尽早接"。

- **实施**：
  1. **learner_contract**：在 seam 处按源忠实重写 `LearnerSample` 语义（token→T-1 causal target、mask/old_lp 同步右移、batch 单一 rollout 版本校验）。用 infra `07_learner_batch_contract.py` 的不变量当测试。
     - **在此定缺口 2**：GRPO 标量 advantage vs 逐 token。建议选 infra 的选项 b（sample-level 标量字段），更贴 GRPO 逐序列信用分配、打包时不多搬数据。
  2. **learner_metrics**：把 `Trainer` 的 metric dict 换成阶段 trace 超集（batch/log-prob/forward-backward/optimizer/发布分相位计时；token/s 分母 = 总 trainable token）。infra 已证 `sum(target_mask)` 与 repo `sum(loss_mask)` 恒等，可直接对上。
- **验收**：learner_contract 的 mask 右移与 `_pad_batch` 逐值一致；metrics 分相位、分母对得上 V6；全回归绿。
- **风险**：低（纯输入适配层 + metric 超集，不改训练数学）。

### V7.5（或 V8 头）· 退休偏离 #1：packing 正确性路径

`v7.md` 偏离 #1（无 packing）是**硬件缺失导致的暂缓，不是永久墙**。infra 已建 FA docker sandbox + `04_megatron_te_thd_spike.py`（THD + `cu_seqlens` 注意力正确性 spike）+ 块对角 mask 路径。

- **实施**：先**核验** infra 那条 spike 的正确性断言（不轻信"已验证"），确认后在 repo 按源 `data_packing.pack_sequences` 重写，隔离用**块对角 mask 显式实现**（能装 FA 时再切 varlen 内核）。
- **验收**：packed 与 padding 两路 loss 逐值等价（正是偏离 #1 声称"不等价（吞吐差）"里唯一要守的语义等价）。
- **收益**：honestly 退休偏离 #1；补上 V7 原定的"packing 提高吞吐"成功标准。

### V8 · Megatron / 多维并行（repo 自有线，按需接重 infra）

- weight_publishing（V12）：出现多 rollout replica 时，把 `WeightUpdater` 泛化成"分桶 + 全或无激活"状态机。
- checkpointing（V13）：FSDP checkpoint 时 `save_pretrained` → DCP + manifest。
- batch_planning（V11）：learner 上多卡 DP 时管每 rank 等样本数/等累积步。

---

## 3. 与 `v7-plan.md` 的差异摘要（供审阅）

- **删**：V7.1 packing 排在早期（改为 V7.5，且先验证 infra 正确性路径）；V7.4「broadcast<disk 当契约」。
- **改**：V7.3 折入缺口 1 版本戳；权重同步的"演进方向"从裸 NCCL broadcast 改为 weight_publishing 契约。
- **加**：V7.4 infra 落地（learner_contract + learner_metrics）；缺口 2 advantage 表示决策；每步的 infra 不变量当验收。
- **纠**：把 tie 处理、packing 硬件风险写进风险表（原表遗漏）。

---

## 4. 待办 / 决策点

- [ ] 缺口 2：advantage 用 sample-level 标量（建议）还是逐 token 广播 —— 在 V7.4 落 learner_contract 时定。
- [ ] V7.5 前先跑一遍 infra `04_megatron_te_thd_spike.py`，核验块对角/varlen 隔离的 loss 等价断言。
- [ ] 每个子版本仍遵分支准则（落在 `v7` 或新 `vN` 分支）与偏离登记铁律。
