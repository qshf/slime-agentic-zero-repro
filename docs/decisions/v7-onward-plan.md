# V7 收尾 + Infra 落地 — 后续方案（FA2 双机统一版）

> 状态：建议（2026-08-06 初版 5090/A100 分叉 → 2026-08-09 统一 → 2026-08-09 V7.6 按 5090 实测重写）
> 取代关系：**接续 `v7-plan.md`**（写于 V7.0 实施前、已过时），保留其 V7.3 内核，废弃其 packing 顺序与 V7.4 框架。
> **2026-08-09 合并**：原 `v7-onward-plan-a100.md` 已并入本文并删除——见 §0.1「为什么不再分 A100/5090 两版」。
> **2026-08-09 二次更新**：V7.6（§2）在单卡 5090 上**逐条实测**后重写。两处实质修正：①决策点「auto-dispatch vs 手动 varlen」定案为 auto（实测它就是源路径）；②**旧计划的 fp32 验收门作废**——FA2 内核拒绝 fp32，改用 bf16 + negative control（§2.3）。V7.4 已完成，进度表随之更正。
> 前提共识：infra（`agentic-rl-infra-lab`）作为 **规范/铺垫**；其代码是参考，**本 repo 里按 slime-agentic 源忠实重写**，用 infra 的 CPU 不变量当验收测试。

---

## 0.1 为什么不再分 A100/5090 两版（关键更新）

初版把后续方案分成两支：`v7-onward-plan.md`（5090，用块对角 mask）与 `v7-onward-plan-a100.md`（A100，用 FA2 varlen）。**分叉的唯一前提是「5090 无 FA2」**。这个前提在 2026-08-09 被 infra 重检推翻：

| 维度 | 分叉时的假设 | 2026-08-09 实测 | 结论 |
|---|---|---|---|
| 5090 FA2 可用性 | ❌ sm_120 源码编译产物丢失、未知 | ✅ **官方 `cu13torch2.10` 预编译 wheel 已含 sm_120 kernel image**，`pip install --no-deps` 秒装，5090 varlen forward+backward PASSED | 前提推翻 |
| 覆盖架构 | A100/5090 各需不同产物 | **同一个 wheel 覆盖 sm_80 + sm_120 两架构** | 无需按卡切分支 |
| packing 隔离机制 | 5090 只能显式块对角 mask（偏离 #7）| 两机都可 FA2 varlen（O(N) IO-aware，真块级稀疏）| **#7 双机均可退休** |

> 依据：infra `docs/04-flash-attn-blackwell-recheck-plan.md` §3（A100 sm_80 与 5090 sm_120 均 FA2 PASSED，同一 wheel）与 §4 决策表。

**结果**：不再需要"A100 用 FA2 / 5090 用块对角 mask"的双分支代码。**同一份 FA2 varlen 代码在两机通用**，无需为不同显卡切分支。分支策略也随之简化（见 §5）。5090 仍有一条**专属**的显卡相关后续（FA4 b25 备选、TE cuDNN backward bug 规避），单列于 [`v8-5090-followup.md`](v8-5090-followup.md)，不再污染主线。

---

## 0. 为什么 `v7-plan.md` 要接续而非照做

| 维度 | `v7-plan.md` 假设 | 现实（`v7.md`） | 结论 |
|---|---|---|---|
| tie 处理 | 未提 | V7.0 已做忠实 tie（meta device 会 broadcast hang） | 计划漏了 0.6B 上最硬的正确性点 |
| packing 难度 | "低风险，逻辑清晰"，排在 V7.1 | **曾暂缓**（5090 当时无 flash-attn）→ V7.5 已用块对角 mask 退休偏离 #1 | 计划把最大风险当最小 |
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
| packing | padding→packing 迁移时顶上 | 上 packing 时（V7.5 已用块对角、V7.6 换 FA2 varlen） |
| weight_publishing | 泛化 `WeightUpdater`（单 engine = 1-replica） | 多 rollout replica 时 |
| batch_planning | 多卡 DP 的等样本数/等累积步 | learner 多卡 DP 时 |
| checkpointing | `save_pretrained` → DCP + manifest | learner 上 FSDP checkpoint 时 |

---

## 2. 建议的迭代顺序（子版本）

顺序原则：**先关掉 V7 的未闭合项，再接零风险 infra，最后按需要接重的**。每步都能独立在双机验证、独立回归。

进度速览：

| 子版本 | 内容 | 状态 |
|---|---|---|
| V7.3 | Ray + 权重同步（disk reload）+ 版本戳 | ✅ 5090 双卡 PASSED |
| V7.4 | 零风险 infra：learner_contract + learner_metrics + 版本戳 | ✅ 纯 CPU 6/6 PASSED（见 `v7.4.md`）|
| V7.5 | packing 正确性路径（块对角 mask，退休 #1、新增 #7） | ✅ 5090 world=1 PASSED |
| **V7.6** | **FA2 varlen packing（单机 5090，退休 #7）** | ✅ **5090 world=1 PASSED**（判别力 8721×，见 `v7.6.md`）|
| V8 | Megatron 多维并行 | 后续 |

### V7.3 · 收尾 V7：Ray + 权重同步（disk reload）+ 版本戳 ✅

保留 `v7-plan.md` 的 V7.3 内核，**多做一件小事**：补 infra 缺口 1（per-sample rollout 版本戳）。**已完成**（`v7.md` §V7.3：5090 --world 1 & 2 端到端 PASSED）。

- **缺口 1**：rollout 生成时把当时 `weight_version` 写进 `Sample`，由 `_convert_samples_to_train_data` 带出——所有 staleness/版本语义的前提。
- **偏离**：权重同步仍 disk reload（非 NCCL broadcast），登记 #4，留后续泛化到 weight_publishing。
- **废弃**：`v7-plan.md` 的 V7.4「broadcast<disk 当契约」——改为多 replica 出现时通过 weight_publishing 契约落，不作为独立性能断言。

### V7.4 · 接零风险 infra：learner ABI + 阶段 trace ✅

**已完成**（`v7.4.md`：纯 CPU 6/6 PASSED，无新 GPU 行为、无需服务器）。infra 明言这两个"零风险、无新 GPU 行为、尽早接"。

- **已实施**：
  1. **learner_contract**（`mini_slime/learner_contract.py`）：`LearnerSample` token→T-1 causal target 坐标系 + batch 单一 rollout 版本校验。落地时定为**校验视图而非数据变换**——训练器照旧内部 `[1:]` 右移消费，`validate_train_data` 只 opt-in 把同批数据过一遍校验，故零回归（改成变换要重写两条 hot path，风险高，已否决）。
  2. **learner_metrics**（`mini_slime/learner_metrics.py`）：`Trainer` 扁平 metric → 分相位 trace，`sum(target_mask)` 与 `sum(loss_mask)` 恒等已验。
  3. **缺口 1 版本戳**：`Sample.rollout_policy_version`（源 = `WeightUpdater.version` int），`custom_convert` 拆 turn 时每 turn-row 继承父版本。
- **缺口 2 的落定**：GRPO advantage 仍为**逐序列标量**（未逐 token 物化），登记为 `v7.4.md` 偏离 gap 2。V7.6 packing 沿用该语义（`pack["rewards"]` 逐段标量），无需改动。
- **验收**（已过）：learner_contract 的 mask 右移与 `_pad_batch` 逐值一致；trace `policy_version_gap=1` 是真 staleness；全回归绿。

### V7.5 · 退休偏离 #1：packing 正确性路径（块对角 mask）✅

**已完成**（`v7.md` §V7.5 + `v7.5.md`：5090 world=1 PASSED，fp32 精确 loss 2.4e-7 / grad rel_L2 1.7e-4 / cosine 0.99999999）。用**显式块对角 4D mask** 证明 packed 与 padding 两路 loss/grad 逐值等价，诚实退休偏离 #1，**新登记偏离 #7**（显式 O(T²) mask 非 varlen O(N) 内核）。

> 这是当时"5090 无 FA2"约束下的忠实替代。V7.6 现在用 FA2 varlen 替换它、退休 #7——见下。

### V7.6 · 退休偏离 #7：FA2 varlen packing ★本次新方案（环境已实测）

**背景变化**：V7.5 时 5090 无 FA2，只能显式块对角 mask（偏离 #7：消除 padding 浪费✔、不得 varlen 块级稀疏✘）。infra 2026-08-09 重检证明官方预编译 wheel 已含 sm_120 kernel——**5090 上 FA2 varlen 可用，#7 可退休**。

> **本节的每条断言都在 `root@223.109.239.36 -p 15016`（单卡 5090）实测过**，实测记录见 §2.6。这与 V7.5 计划阶段的"纸面推导"不同——V7.5 的 fp32 硬门到 FA2 上直接失效（见 §2.3），若不先实测就写计划，会照抄一个跑不起来的验收门。

#### 2.1 关键实测结论：走 transformers 自动 dispatch（决策点 2 定案）

计划旧版留了个决策点：「`_attn_implementation="flash_attention_2"` 自动 dispatch vs 手动调 `flash_attn_varlen_func`」。**实测定案：走自动 dispatch**，因为它恰好就是源的那条路径。

transformers 5.12.1 的链路（容器内源码逐层确认）：

| 层 | 文件:行 | 行为 |
|---|---|---|
| ① packed 检测 | `masking_utils.py:883-890` | `attention_mask is None` 且有 `position_ids` 且无 cache → `find_packed_sequence_indices(position_ids)` |
| ② 段边界推导 | `masking_utils.py:775-786` | `diff(position_ids) != 1` 处 cumsum → 段 id；**每段 reset 的 position_ids 就是边界信息本身** |
| ③ varlen 判定 | `modeling_flash_attention_utils.py:528-541` | `_is_packed_sequence`：batch==1 且 position_ids 非单调递增 → True |
| ④ cu_seqlens | 同上 `:452-487` | `prepare_fa_kwargs_from_position_ids`：`(position_ids==0).nonzero()` 反推 `cu_seq_lens` + `max_length` |
| ⑤ 内核调用 | 同上 `:788-809` | `flash_varlen_fn(..., cu_seqlens_q/k, max_seqlen_q/k)` |

**实测验证 dispatch 真的发生**（spy 掉 `mfu._flash_varlen_fn`，2 层 Qwen3，lens=[5,3,4]）：
```
varlen calls: 2                      # 每层一次，确实逐层走 varlen
kwargs: {'cu_seqlens_q': [0,5,8,12], 'cu_seqlens_k': [0,5,8,12],
         'max_seqlen_q': 5, 'max_seqlen_k': 5}     # 与手工 cu_seqlens 完全一致
```

**这正是源的写法**：源 `_get_model_inputs_args`（actor.py:801-812）传的就是 `{input_ids:[1,T], position_ids:[1,T], attention_mask:None}`，源 `arguments.py:29` 默认 `attn_implementation="flash_attention_2"`。**手动调 `flash_attn_varlen_func` 反而偏离源**（要 monkey-patch attention 层）。故 nano 走自动 dispatch = **既贴源又更简洁**，`pack_sequences` 产出的 `cu_seqlens` 退化为**校验/切 loss 用**，不再喂给内核。

> 铁律记账：这条**不是偏离，是回归对齐**——V7.5 的显式 mask 才是偏离（#7），V7.6 把它换回源写法。

#### 2.2 实施（改动很小）

`data_packing.py`：`pack_sequences` **完全不动**（flat 1D + reset position_ids + cu_seqlens 全部照用）；`build_block_diagonal_causal_mask` **保留**作 fallback。

`fsdp_trainer.py::_packed_backward` 只改「怎么喂前向」这一处，loss 段切分逻辑（`logits[start:end-1]` 预测 `tokens[start+1:end]`）**一行不动**：

```python
# V7.5（train_packing="mask"）：显式块对角 4D mask，后端须 eager/sdpa
#   attn_mask = build_block_diagonal_causal_mask(cu, total, self.device, self.compute_dtype)
#   logits = self.model(input_ids, position_ids=position_ids, attention_mask=attn_mask)
# V7.6（train_packing="fa2"，默认）：attention_mask=None，靠 position_ids 反推 cu_seqlens 隔离
logits = self.model(
    input_ids,                    # [1,T]
    position_ids=position_ids,    # [1,T] 每段 reset —— **既定位 RoPE 又定义段边界**
    attention_mask=None,          # 对齐源 actor.py:807
).logits[0].float()
```

`args.train_packing` 从 `bool` 扩成三态字符串（`False`/`"mask"`/`"fa2"`），保持 `False` 默认 → 零回归。

**模型构造需带 attn_implementation**：`FSDPTrainer.__init__` 的 `from_pretrained` 加 `attn_implementation=self.attn_implementation`（对齐源 actor.py:96 的 `attn_implementation=self.args.attn_implementation`）——这是目前 nano 相对源缺的一个参数，顺手补齐。

#### 2.3 ⚠️ 验收门必须重设计：**FA2 不支持 fp32**（推翻旧计划的门槛）

旧计划写的 V7.6 验收门是：
```
fp32: loss diff < 1e-5，grad cosine > 0.9999     ← 【实测不可达，作废】
bf16: loss diff < 5e-4，grad cosine > 0.99
```

**实测结果**：
```
C. fp32 FA2 REJECTED: RuntimeError FlashAttention only support fp16 and bf16 data type
```

这不是小事——**V7.5 的整个正确性论证建立在 fp32 硬门上**（`v7.5.md`：「fp32 才是精确证明、bf16 只到舍入」；bf16 下 magnitude rel_L2 可达 10.9%，是 28 层舍入累积，无判别力）。FA2 路径拿不到 fp32，等于**失去了 V7.5 那个硬门**。若照抄旧计划的 fp32 门，V7.6 会在第一次跑就 crash；若只留 bf16 rel_L2 门，则**门槛松到无法区分"隔离正确"与"隔离失效"**。

**替代方案：negative control（阴性对照）——用判别力代替精度**

不再追求"误差足够小"，改为证明**度量本身有判别力**：同一个 bf16 度量，正例（正确隔离）与反例（故意泄漏）必须相差若干数量级。反例构造极简——**把 reset 的 position_ids 换成单调递增**，`_is_packed_sequence` 立刻返回 False，整个 pack 退化成一条长序列（真串扰）。

**实测判别力**（4 层 Qwen3-0.6B，lens=[37,23,41]，bf16，对照组 = 逐条独立前向）：

| 组 | 构造 | rel_L2 | cosine |
|---|---|---:|---:|
| **正例**：varlen packed（reset pos） | `position_ids=[0..36,0..22,0..40]` | **7.68e-03** | **0.99997056** |
| **反例**：故意泄漏（单调 pos） | `position_ids=[0..100]` | **1.07e+00** | **0.42443368** |

**差 140×（rel_L2）/ 0.58 绝对差（cosine）**——bf16 舍入噪声与真泄漏之间隔着两个数量级，门槛可以站得住。故 V7.6 验收门定为：

```
【主门 · bf16，全部为硬门】
  正例 logits/loss：  rel_L2 < 5e-2      cosine > 0.999
  反例（negative control）必须 FAIL：rel_L2 > 0.5 且 cosine < 0.9
      —— 反例若"通过"，说明度量无判别力，本轮验收作废（防 V7.5 那类假阴性）
  grad：             cosine > 0.99       rel_L2 仅报 info（同 V7.5，bf16 magnitude 无判别力）
  loss：             |loss_fa2 - loss_mask路径| < 5e-2

【交叉验证 · 三路一致（替代 fp32 的第二重保险）】
  padding(V7.0) ≈ mask packing(V7.5) ≈ fa2 packing(V7.6)
  其中 padding vs mask 这一对**已有 V7.5 的 fp32 精确证明**（loss 2.4e-7 / cosine 0.99999999），
  故 fa2 只需在 bf16 下与"已被 fp32 认证过"的 mask 路径对齐，即间接继承那份精确性。
```

> 这是本次计划相对旧版最实质的修正：**旧版的 fp32 门在 FA2 上根本跑不起来**，而 negative control 是在"没有 fp32"的约束下仍能守住正确性的唯一诚实做法。同时它顺带修了 infra doc 04 §3b.4 记录的那类假阴性（"绝对阈值 + 无对照 → 错误梯度蒙混过关"）。

#### 2.4 环境准备（已实测通过，非纸面推导）

`bench5090` 常驻容器**当前不能直接用**，两个坑都已实测确认并有解：

| 坑 | 实测现象 | 解 |
|---|---|---|
| `flash_attn` namespace 被 FA4 b15 占 | `import flash_attn` 成功但**无 `__version__`**（是 FA4 的包） | 先 `pip uninstall -y flash-attn-4` 再装 FA2 wheel |
| **`accelerate` 缺失** | `ModuleNotFoundError: No module named 'accelerate'` —— `get_init_weight_context` 的 tie 分支依赖它，**FSDPTrainer 起不来** | `pip install accelerate`（实测装到 1.14.0） |
| wheel 挂载方式 | 单文件 bind mount 报 `ERROR: Invalid wheel filename (wrong number of parts): 'fa2'`（挂载改名破坏 wheel 命名规则） | 挂**目录** `-v /root:/host:ro`，用原始文件名 |

**实测通过的完整配方**：
```bash
docker run --rm --gpus all \
  -v /root:/host:ro \
  -v /root/slime-agentic-zero-repro:/workspace/slime \
  -w /workspace/slime agentic-rl-infra-lab:te-cudnn-system-spike bash -lc "
    pip uninstall -y flash-attn-4 -q
    pip install --no-deps -q /host/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    pip install -q accelerate
    NANO_MODEL_PATH=/host/models/Qwen/Qwen3-0.6B \
      torchrun --nproc_per_node=1 scripts/test_v7.6_fa2_packing.py"
```
实测输出：`FA2 version: 2.8.3` / `is_flash_attn_2_available: True` / `accelerate: 1.14.0` / `varlen func OK`。

> 装完 FA2 后 **TE import 会崩**（FA2 覆盖 `flash_attn` namespace，TE `backends.py` 无条件 import FA4 的 `cute.interface`）。V7.6 不用 TE，无影响；但**别在 `bench5090` 常驻容器里装**，会把 infra 的 TE 对照环境搞坏——用上面的 `--rm` 临时容器。

#### 2.5 硬断言：严禁静默回退

FA2 路径的失败模式比 V7.5 更隐蔽——`attention_mask=None` 若落到非 FA2 后端，就是**满因果注意力、跨样本串扰、loss 静默算错**（正是 V7.5 注释里警告的那个）。故 `_packed_backward` 的 fa2 分支需三重断言：

```python
assert attn_impl == "flash_attention_2", f"fa2 packing 要求 FA2 后端，当前={attn_impl}"
assert is_flash_attn_2_available(), "flash_attn 不可用——attention_mask=None 会退化成满因果注意力"
# position_ids 必须非单调（否则 _is_packed_sequence 返回 False → 静默退化成单条长序列）
assert num_segments == 1 or not _is_monotonic(position_ids), "position_ids 未按段 reset，varlen 隔离不会生效"
```

第三条正是 §2.3 反例的构造方式——**把反例变成运行时断言**，比只在测试里查更可靠。

#### 2.6 实测记录汇总（2026-08-09，`root@223.109.239.36 -p 15016`，单卡 RTX 5090 sm_120）

| 项 | 结果 |
|---|---|
| 环境 | torch 2.11.0+cu130 / transformers **5.12.1** / python 3.12.3 / sm_120 / 32607 MiB 空闲 |
| FA2 安装 | ✅ 2.8.3，`--no-deps` 秒装，`is_flash_attn_2_available: True` |
| varlen auto-dispatch | ✅ 每层触发，`cu_seqlens=[0,5,8,12]` `max_seqlen=5` 与手工一致 |
| 隔离正确性（2 层） | ✅ packed vs 逐条 rel_L2=5.66e-03 cosine=0.99998403 |
| 隔离正确性（4 层） | ✅ rel_L2=7.68e-03 cosine=0.99997056 |
| **negative control** | ✅ 有判别力：泄漏态 rel_L2=1.07 cosine=0.424（差 140×）|
| backward | ✅ 完成，全参数 grad finite |
| **fp32** | ❌ **FA2 拒绝**（`only support fp16 and bf16`）→ 验收门重设计，见 §2.3 |
| accelerate | ❌ 容器缺 → 必须补装（否则 FSDPTrainer 起不来）|

#### 2.7 偏离处理

- **退休偏离 #7**：✅ FA2 varlen 上线，得块级 compute 稀疏（O(N) HBM），且**回到源的 `attention_mask=None` 写法**。
- **保留块对角 mask 作 fallback**：`train_packing="mask"` 仍可用（无 FA2 环境 / 数值对照 / 需要 fp32 精确证明时）。诚实登记为"两实现并存、默认 fa2"。
- **新登记偏离 #8（本版新增）**：**验收精度从 fp32 精确降级到 bf16 + negative control**。①源无此问题（源不做 padding-vs-packing 等价验收，直接信 flash-attn）；②nano 偏离是因 FA2 内核不接受 fp32，V7.5 的 fp32 硬门物理不可达；③语义**不完全等价**——bf16 证不到"逐值精确"，只证到"与已被 fp32 认证的 mask 路径一致 + 与泄漏态差两个数量级"。补齐路径：`train_packing="mask"` 路径永久保留 fp32 精确证明，fa2 通过与它交叉验证间接继承。
- **不再登记**："双机统一"相关偏离作废——见 §5，本次只有 5090 单机。

### V7.7（可选）· 全模型切 FA2

V7.6 只在 packing 路径用 FA2 varlen；可再把非 packing 的单样本前向也切 FA2（`config._attn_implementation="flash_attention_2"` 全局）。动机是端到端吞吐——但 nano 教学目标已由 V7.6 达成，此步按需。

> **注意（TE 互斥）**：FA2 wheel 覆盖 `flash_attn` namespace，装后 **TE import 崩溃**（TE `backends.py` 无条件 `from flash_attn.cute.interface import`）。同容器若还要跑 TE，先 `pip uninstall flash-attn -y`。若频繁切换，考虑两个镜像 tag（见 infra doc 04 §3b.0 环境冲突矩阵）。

### V8 · Megatron / 多维并行（repo 自有线，按需接重 infra）

- weight_publishing（infra V12）：多 rollout replica 时把 `WeightUpdater` 泛化成"分桶 + 全或无激活"状态机。
- checkpointing（infra V13）：FSDP checkpoint 时 `save_pretrained` → DCP + manifest。
- batch_planning（infra V11）：learner 多卡 DP 时管每 rank 等样本数/等累积步。
- packing 在 THD layout 下与 CP（context parallel）组合。

---

## 3. TE cuDNN 的定位（双机统一）

TE cuDNN 在**两机上都不再是首选**，降级为对照基线：

| 机器 | TE cuDNN THD 状态 | 定位 |
|---|---|---|
| **A100 sm_80** | 不支持 THD（`get_fused_attn_backend` 对 THD 返回 `No_Backend`）→ 回退 Unfused O(N²) | 仅 forward 对照；FA2 快 ~4.7–40× |
| **5090 sm_120** | 支持 THD（cuDNN≥9.18.1）**但 FusedAttention backward 内核损坏**——`padding_causal` 下梯度错误（单条/packed 均崩），forward 正确、backward 不可用于训练 | 仅 forward 对照；**训练严禁用**；FA2 快 ~1.14–1.85× |

> 5090 TE cuDNN backward bug 详见 infra `docs/investigations/te-sm120-cudnn-bwd/` 与上游 [TE#3333](https://github.com/NVIDIA/TransformerEngine/issues/3333)。Workaround（若非用 TE 不可）：`NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0` 强制 Unfused，梯度正确但 O(N²) 慢。**结论：两机训练都走 FA2 varlen，TE cuDNN 仅留作 forward 数值对照 / 镜像层。**

---

## 4. 与初版（含 a100 分支）的差异摘要（供审阅）

- **删**：`v7-onward-plan-a100.md` 整份（分叉前提消失，FA2 双机通用）——内容并入本文 V7.6 + §3。
- **删**：原 V7.1 packing 排在早期的假设（已由 V7.5 落地、V7.6 升级）。
- **改**：V7.5 是「块对角 mask」（已完成、退休 #1）；**V7.6 新增「FA2 varlen」，退休 #7**，取代原 a100 版的 V7.5-a100/V7.6-a100 两步。
- **改**：TE cuDNN 定位从"5090 唯一可用"改为"两机均降为 forward 对照"，并记入 5090 backward bug。
- **加**：§0.1 说明为何不再分双版；§3 双机 TE 定位表；5090 专属后续单列 `v8-5090-followup.md`。

### 4.1 本次（2026-08-09 二次更新）相对上一版的修正

上一版 V7.6 是**未实测的纸面方案**，本次在单卡 5090 上逐条验证后改了四处：

| # | 上一版写法 | 实测结果 | 本版修正 |
|---|---|---|---|
| 1 | 验收门含 `fp32: loss diff < 1e-5` | ❌ **FA2 拒绝 fp32**（`only support fp16 and bf16`）| 门槛重设计：bf16 + **negative control**（实测判别力 140×），§2.3 |
| 2 | 决策点悬空：auto-dispatch vs 手动 varlen | ✅ auto-dispatch 实测吐出 `cu_seqlens=[0,5,8,12]`，与手工一致，且**就是源写法** | 定案 auto-dispatch，决策点关闭，§2.1 |
| 3 | 安装只写 `pip install --no-deps <wheel>` | ❌ 漏两步：`flash_attn` 被 FA4 b15 占；**`accelerate` 缺失**导致 FSDPTrainer 起不来 | 补完整配方 + 三个坑，§2.4 |
| 4 | "双机各跑一遍（A100 + 5090）" | A100 实例已释放，当前只有单卡 5090 | 改为 5090 单机验收，A100 列入待办不作完成条件，§5 |

> 教训与 V7.5 同源：**验收门必须先在目标硬件上试过再写进计划**。V7.5 那次是把 infra 单层 spike 的绝对阈值搬到 28 层整模型（虚高）；这次若不实测，会写一个内核根本不接受的 dtype 门。

---

## 5. 分支策略（简化）

初版计划 A100/5090 各开独立分支（`v8-a100` / `v8-5090`），因两者 attention 后端不同。**FA2 双机通用后，这个理由消失**：

- **主线单分支**：FA2 varlen packing，代码与显卡无关，只是 `NANO_MODEL_PATH` / `CUDA_VISIBLE_DEVICES` 等环境差异。
- **5090 专属后续**（FA4 b25 备选、TE bug 规避、镜像区分）不需要独立代码分支，作为文档记在 [`v8-5090-followup.md`](v8-5090-followup.md)，代码改动（若有）以 opt-in flag 落主线。

**当前实际状况（2026-08-09）**：手上只有 **`root@223.109.239.36 -p 15016` 单卡 5090**，A100 实例已释放（infra doc 04 §3b.4 已记"A100 实例已释放，无法交叉确认"）。因此：

- 当前分支 `v8` 已从 v7 末端切出，**V7.6 就在 `v8` 上做**，不再新建 `v8-a100`/`v8-5090`（memory 里"双分支"的记录已过时，随本次更新）。
- V7.6 验收**只在 5090 跑**。旧计划"双机各跑一遍"改为**待 A100 可用时补跑**，记入 §6 待办——不假装跑过。
- 单卡 ⇒ V7.6 只验 `--world 1`（与 V7.5 一致）。这本来就是对的：world=1 去掉 sharding-reduce 噪声，才能隔离 packing 本身的正确性；多卡 DP 与 packing 正交，已由 V7.3 验过。

---

## 6. 待办 / 决策点

- [x] ~~缺口 2：advantage 用 sample-level 标量还是逐 token 广播~~ —— **已定**：V7.4 落地时选 sample-level 标量（逐序列信用，登记为 `v7.4.md` gap 2）。
- [x] ~~V7.6：transformers 5.x 自动 dispatch vs 手动调 `flash_attn_varlen_func`~~ —— **已定：自动 dispatch**。实测确认它就是源 `attention_mask=None` + reset position_ids 那条路径，`cu_seqlens` 由 transformers 从 position_ids 反推且与手工值一致（§2.1）；手动调反而要 monkey-patch attention 层、更偏离源。
- [x] ~~**V7.6 实现**~~ —— ✅ 已完成：`train_packing` 三态 + `attn_implementation` 参数 + fa2 分支 + 三重硬断言。
- [x] ~~**V7.6 验收脚本**~~ —— ✅ `scripts/test_v7.6_fa2_packing.py`：dispatch + negative control + 等价，**5090 PASSED**（判别力 8721×，实测详见 [`v7.6.md`](v7.6.md)）。
- [x] ~~V7.6 在 5090 跑通~~ —— ✅ world=1 PASSED；V7.5 两路 + 离线全回归绿。
- [ ] **A100 补跑**：待再租到 sm_80 实例时，同一份代码跑一遍 FA2 varlen 等价，确认双机结论。当前无 A100，**不作为 V7.6 的完成条件**。
- [ ] 是否固化 `fa2-system-spike` 镜像 tag，避免每次临时 `pip uninstall/install`（当前用 `--rm` 临时容器，未污染 `bench5090`）。
- [x] ~~V7.6 收官文档~~ —— ✅ [`v7.6.md`](v7.6.md) 已写；`v7.md` 偏离表 **#7 已标退休、#8 已新增**。
- [ ] 每个子版本仍遵分支准则与偏离登记铁律。**下一步 → V8 Megatron 多维并行**（§2 末节）。
