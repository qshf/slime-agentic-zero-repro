# V8 · Megatron 多维并行 —— 迭代计划（单卡 5090，可行性已实测）

> 状态：**计划待审**（2026-08-09）。分支 `v8`（V7.6 已在其上收官）。
> 一句话：把 nano 的训练后端从 **FSDP（1D DP 分片）** 扩到 **Megatron TP（模型内张量切分）**，
> 并补上 Megatron 特有、FSDP 没有的那条 RL 接缝——**Megatron 分片权重 → HF 格式 → 推理引擎**。
> **本计划每条可行性断言都在单卡 5090 实测过**（见 §1），不是纸面推导。

---

## 0. 为什么 V8 值得做（痛点承接）

V7 把训练后端做到了 FSDP：**1D DP，每张卡都持有完整模型的一个分片，前向时 all-gather 回全量层**。
它的边界是 **单层必须放得进一张卡**。Megatron 的 TP 解决的是另一件事——**把单层内部的矩阵切开**
（qkv/mlp 按列/行切），让单层也能跨卡。这是两种正交的并行轴，也是 V8 的教学核心。

| | V7 FSDP | V8 Megatron TP |
|---|---|---|
| 切什么 | 参数**存储**按 rank 分片 | 参数**计算**按行/列切分 |
| 前向时 | all-gather 回全量层再算 | **不 gather**，各 rank 算部分结果 + all-reduce |
| 通信位置 | 层边界（gather 参数） | **层内部**（col→row 之间 all-reduce 激活） |
| 边界 | 单层须放得下 | 单层也可跨卡 |

对 RL 系统而言还有第二个痛点，**这才是 slime-agentic 相对纯 Megatron 教程的真正增量**：
FSDP 的 `save_pretrained` 天然吐 HF 格式，disk reload 就能喂给 SGLang（V6/V7.3 就是这么做的）；
**Megatron 的参数名和形状都不是 HF 的**（`decoder.layers.0.self_attention.linear_qkv.weight`
是 qkv 融合且按 TP 切开的），要同步回推理引擎必须先 **TP-gather + 拆 qkv + 改名**。
源项目为此写了整个 `megatron_to_hf/` 目录（10 个模型各一份）+ `update_weight/`（2232 行）。
**这条接缝是 V8 的第二个教学核心，也是 nano 之前完全没有的东西。**

---

## 1. 单卡可行性 —— 已实测（2026-08-09，`root@223.109.239.36 -p 15016`）

用户确认「本次仅单卡即可」。单卡跑多维并行的关键约束已逐条实测，**结论：TP 完全可做，PP/CP/EP 不可做**。

环境：容器 `agentic-rl-infra-lab:te-cudnn-system-spike`，已自带 **megatron-core 0.18.2** + torch 2.11+cu130。

| # | 实测项 | 结果 | 对 V8 的含义 |
|---|---|---|---|
| 1 | **NCCL 双 rank 共卡** | ❌ `Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 80` | 单卡多 rank **不能用 NCCL** |
| 2 | **gloo 双 rank 共卡 + CUDA allreduce** | ✅ `[3,3,3,3]` 正确 | **gloo 是单卡多 rank 的唯一通路**（TP 的 all-reduce 可走） |
| 3 | **`mpu.initialize_model_parallel(TP=2)`** | ✅ `TP=2 PP=1 DP=1` | Megatron 并行组能在单卡建起来 |
| 4 | **`ColumnParallelLinear` 真分片** | ✅ 权重 `(128,64)` → `(64,64)` | TP **真切开了**，不是假分片 |
| 5 | **TP=2 的 col→row MLP 数值等价** | ✅ `max_abs_err=1.9e-8` `rel_L2=1.3e-7` | **TP 数学正确**（与非分片全量 matmul 逐值一致） |
| 6 | **真 `GPTModel` TP=2 前向+反向** | ✅ `local_params=93.5M`、qkv 分片 `(2048,1024)`、vocab 分片 `75968=151936/2`、logits finite、backward OK | **V8 的地基成立** |
| 7 | **gloo CUDA p2p（PP 靠它传激活）** | ❌ 进程 abort：`gloo::IoException writev Bad address` | **PP 不能用 CUDA 张量走 gloo** |
| 8 | PP：CUDA 激活经 CPU 中转 send/recv | ✅ 可传 | PP 只能靠 CPU 中转（**极慢且偏离源，不做，见 §4**） |
| 9 | **gloo all_to_all（EP 靠它路由 token）** | ❌ `Backend gloo does not support alltoall` | **EP 在单卡上物理不可做** |
| 10 | numpy 版本 | ⚠️ 容器是 **2.3.5**，而源 `initialize.py:66` 断言 `numpy 1.x` | 见 §5 风险 R1 |

**结论**：V8 = **TP 做实**（1/4/5/6 全绿），**PP/CP/EP 记录设计不实现**（7/9 是硬阻断）。
这与 roadmap 对主线三「记录设计、按精力复现」的定位一致，且**偏离有实测依据、不是偷懒**。

---

## 2. 范围（做什么 / 不做什么）

### 做（单卡可验、有硬门）

- **V8.0 Megatron TP 训练一步**：建 `toy_rl/trainer/megatron_trainer.py`，TP=2 跑通
  「forward → GRPO loss → backward → step」，**loss 数学复用 V7 那一份**（`_grpo_sample_loss` 的语义）。
  硬门：**TP=2 与 TP=1 的 loss/grad 等价**（同 V7.5/V7.6 的等价验收范式）。
- **V8.1 Megatron → HF 权重转换 + 同步回 SGLang**：建 `toy_rl/trainer/megatron_to_hf.py`
  （对齐源 `megatron_to_hf/qwen2.py`，只做 Qwen3 dense 一种）+ TP-gather（对齐源
  `update_weight/common.py:15 all_gather_param` 的 `partition_dim` 逻辑）。
  硬门：**转换出的 HF 权重与 HF 原始 checkpoint 逐值一致**（这是可判定的强不变量，见 §3.2）。
- **V8.2 接进 Ray 闭环**：`train_backend="megatron"`，复用 `train_ray.train` 主循环（同 V7.3 的做法，
  不新建 `train_megatron.py`）。硬门：3 轮 rollout→train→update_weights，`weight_v` 递增、未 hang。

### 不做（记录设计，附实测理由）

| 轴 | 为何不做 | 记录在哪 |
|---|---|---|
| **PP**（流水并行） | 实测 7：gloo 不支持 CUDA p2p，单卡传激活只能经 CPU 中转——那既慢又**偏离源**（源走 NCCL p2p）。假装做等于教错。 | v8.md 偏离表 + §4 概念笔记 |
| **CP**（上下文并行） | 需 ≥2 卡切 seq 维；单卡 CP=1 等于没开，无可验内容 | 同上 |
| **EP**（专家并行） | 实测 9：gloo 无 all_to_all，**物理不可做**；且 Qwen3-0.6B 是 dense 模型，没有 expert | 同上 |

> 若日后租到 4 卡/8 卡，这三轴的验证脚本可直接照 §4 的设计补——**代码接缝按源留好，只是不启用**。

---

## 3. 实施细节与硬门

### 3.1 V8.0 · TP 训练一步

对齐源锚点：`megatron_utils/model.py:initialize_model_and_optimizer`（建模型）、
`actor.py` 的 train 路径、`initialize.py:_initialize_distributed`（建并行组）。

关键落点：
- **并行组初始化**走 `mpu.initialize_model_parallel(tensor_model_parallel_size=N)`，与源同一入口。
- **模型构造**用 `megatron.core.models.gpt.GPTModel` + `get_gpt_layer_local_spec`（实测 6 已通）。
  `share_embeddings_and_output_weights=True` —— 与 V7.0 同一个 tie 事实（Qwen3-0.6B `tie_word_embeddings=True`）。
- **backend=gloo**（实测 2）——这是**登记偏离**：源用 NCCL，nano 因单卡多 rank 被 NCCL 硬拒而改 gloo。
  语义等价（集合通信结果相同），性能不等价（gloo 走 TCP，慢）。**V8 的目的是教并行语义不是测吞吐**，可接受。

**硬门（沿用 V7.5/V7.6 已验证有效的等价范式）**：
```
TP=2 的 loss/grad  ==  TP=1 的 loss/grad
  loss:  |Δ| < 1e-4
  grad:  cosine > 0.999（全局 rel_L2 报 info——V7.5 的度量教训）
```
> 为什么这个门站得住：TP 是**数学恒等切分**（col-parallel 切输出维、row-parallel 切输入维 + all-reduce），
> 理论上应精确等价。实测 5 已在 MLP 层面拿到 `1.3e-7`，整模型该门是合理的。
> 若 TP 切错（比如 qkv 按错误维度切、忘了 all-reduce），误差是 O(1) 级——门有判别力。

### 3.2 V8.1 · Megatron→HF 转换（本版最有价值的部分）

对齐源 `megatron_to_hf/qwen2.py`（Qwen3 走的就是这个分支，见 `__init__.py:46`
`elif "qwen2" in model_name or "qwen3" in model_name`）。要处理的三类映射：

| Megatron 参数 | HF 参数 | 变换 |
|---|---|---|
| `embedding.word_embeddings.weight` | `model.embed_tokens.weight` | 改名 + **去 vocab padding**（源 `remove_padding`） |
| `...self_attention.linear_qkv.weight` | `q_proj/k_proj/v_proj.weight` | **按 num_query_groups 拆 GQA 三份**（源 qwen2.py:26-36） |
| `...mlp.linear_fc1.weight` | `gate_proj/up_proj.weight` | **chunk(2) 拆门控**（源 qwen2.py:53-57） |

外加 **TP-gather**：每个参数按 `param.partition_dim` 在 TP 组内 `all_gather` + `torch.cat` 拼回全量
（对齐源 `update_weight/common.py:15-47`）。

**硬门（可判定的强不变量）**：
```
convert(gather(megatron_params))  ==  HF 原始 checkpoint 的同名张量
  逐参数 max_abs_diff == 0（应精确相等——这是纯 reshape/split/rename，不涉及浮点运算）
  且参数名集合完全一致（无遗漏、无多余）
```
> 这个门比 V7 系列的等价门更强：**要求逐值精确为 0**，因为转换全是无损重排。
> 做法：从 HF checkpoint 载入 → 转成 Megatron 格式建模型 → 再转回 HF → 与原始比。
> 名字集合的双向比对能抓住"漏了某层"这类静默错误（源那 10 个转换文件最常见的 bug 类型）。

### 3.3 V8.2 · 接 Ray 闭环

复用 V7.3 已验证的骨架：`args.train_backend="megatron"` 分支、`RayTrainGroup` 运行时包装、
DP-split、集体 save + rank0-only POST。**权重同步路径改为**：
TP-gather → convert_to_hf → 落盘 HF 格式 → SGLang disk reload（沿用 V6/V7.3 的 disk reload，
仍是登记偏离 #4，源用 NCCL broadcast/IPC）。

硬门：3 轮闭环、`weight_v` 递增、有真 backward（组内有方差的那轮）、未 hang。

---

## 4. 不实现的三轴 —— 概念笔记（写进 v8.md，不写代码）

按铁律「记录设计」，每轴记：源怎么做、单卡为何做不了、需要什么条件补齐。

- **PP**：源按层切 stage，stage 间 NCCL p2p 传激活，用 1F1B 调度掩盖 bubble
  （`mpu.is_pipeline_first_stage/last_stage` 门控 loss 只在最后 stage 算，见 `model.py:279,463`）。
  单卡阻断 = 实测 7。补齐条件：≥2 卡 + NCCL。
- **CP**：源按 seq 维切，attention 处做 all-gather/ring 交换 KV（`cp_utils.py:slice_with_cp`）。
  与 V7.5/V7.6 的 packing 正交但会组合（THD layout）。补齐条件：≥2 卡。
- **EP**：源按 expert 切，all_to_all 路由 token（`expert_model_parallel_size`）。
  单卡阻断 = 实测 9（gloo 无 all_to_all）+ 0.6B 是 dense 无 expert。补齐条件：≥2 卡 NCCL + MoE 模型。

---

## 5. 风险与预案（实测已暴露的）

| # | 风险 | 现状 | 预案 |
|---|---|---|---|
| **R1** | **numpy 2.x**：源 `initialize.py:66` 断言 `np.__version__.startswith("1.")`，容器是 2.3.5 | megatron-**core** 实测可用（探针 3/4/5/6 全过，未触发该断言）——那行断言在源的 `megatron.training.init` 路径里 | nano **只用 `megatron.core`，不走 `megatron.training.init`**（源那套还要 `_build_tokenizer`/微批计算器等训练框架件，nano 用不上）。这本身是一条**登记偏离**。若必须走 training 路径 → 降 numpy 1.x |
| **R2** | gloo 慢 | TP all-reduce 每层两次，gloo 走 TCP | 接受——V8 教语义不测吞吐。**明确不做吞吐断言**（吞吐留 V9） |
| **R3** | 显存：TP=2 两 rank 共 32GB | 实测 6 的 2 层模型没问题；28 层全模型两份 optimizer state 未测 | 先用 **2-4 层缩减配置** 打通等价门（等价性与层数无关），再试全 28 层；顶不住就 `num_layers` 减半并登记 |
| **R4** | `megatron.bridge` 缺失（实测 `ModuleNotFoundError`） | 源 `update_weight/hf_weight_iterator_bridge.py` 依赖它 | nano 走 **`hf_weight_iterator_direct` 那条路**（源本就有两条），不需要 bridge |
| **R5** | HF→Megatron 载权重 | 源用 checkpoint 转换工具链，nano 需自己写反向映射才能做 §3.2 的硬门 | §3.2 的门是**双向的**，反向映射（HF→Megatron）本就要写；若成本过高，退一步只验 **Megatron→HF→再载回 HF 模型能正常前向** |

---

## 6. 交付物

| 文件 | 内容 |
|---|---|
| `toy_rl/trainer/megatron_trainer.py` | MegatronTrainer：并行组 + GPTModel + GRPO 训练一步 |
| `toy_rl/trainer/megatron_to_hf.py` | TP-gather + qkv/gate 拆分 + 改名（对齐源 qwen2.py） |
| `mini_slime/args.py` | `+train_backend="megatron"`、`tensor_model_parallel_size` |
| `mini_slime/trainer.py` | megatron 分支（同 fsdp 分支的写法） |
| `scripts/test_v8.0_megatron_tp.py` | TP=2 vs TP=1 等价硬门 |
| `scripts/test_v8.1_megatron_to_hf.py` | 转换逐值精确门 + 名字集合双向比对 |
| `scripts/test_v8.2_megatron_ray.py` | Ray 闭环（镜像 test_v7.3） |
| `docs/decisions/v8.md` | 决策日志 + 偏离表（gloo/仅 TP/只用 megatron.core/disk reload…）|

**零回归保证**：全部新增文件 + `train_backend` 分支，FSDP/torch/fake 三条既有路径一行不改。

---

## 7. 已拍板（2026-08-09）

1. **收敛点 = V8.1**：做到「TP 训练一步 + Megatron→HF 权重转换」，`torchrun` 验证即收官。
   两个教学核心（层内切分 / 权重转换接缝）都覆盖。**V8.2 Ray 闭环不做**——与 V7.3 高度同构、
   新知识少，需要时再单开一版。相应地 §6 交付物表里的 `test_v8.2_megatron_ray.py` 与
   `trainer.py` megatron 分支**移出本版范围**。
2. **模型规模 = 先缩减层数（2-4 层）拿硬门，再冲全 28 层**。等价性与层数无关，缩减配置快且避开
   R3 显存风险；缩减版绿了再用真实 Qwen3-0.6B 配置复跑一次，跑不动则记入偏离。
