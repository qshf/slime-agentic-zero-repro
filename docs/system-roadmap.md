# slime-agentic 复现 — 子系统实现先后顺序考量

> 受众：未来的自己 / 协作 LLM。每开新版本前 review；新增子系统插入对应位置。
> 目的：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
> 本文件是项目级活文档（决定"做哪个子系统、什么顺序"）。单系统内部拆分见 `docs/<system>-system/iteration-plan.md`。

## 核心原则（来自源项目 README，被反复验证）

```
先复现数据如何流动，再学习系统如何加速。
```

一切排序都服从这条：先把 `Sample → generate → reward → trainer → update_weights → rollout engine` 这条数据链跑通（纯 CPU / fake trainer 即可），再逐层换上真实的分布式部件（Ray → SGLang → FSDP → Megatron）。

## 硬件与复现深度（已与用户对齐）

- **硬件**：多卡服务器。工作流 = 本地开发 → 推 git → 服务器拉取运行/调试。
- **深度**：全部尽量复现。因此没有"只能读源码"的死档 —— SGLang / FSDP / Megatron 都在服务器上真跑。
- **推论**：每个版本本地能跑的部分（数据契约、Ray 单机、fake trainer）在本地验证；需要 GPU 的部分（真 SGLang engine、真 FSDP step、Megatron 并行）在服务器验证。CLAUDE.md 的验活 cheatsheet 分"本地档 / 服务器档"两栏。

---

## 1. 候选子系统全景（源项目对照）

已核对，路径与行数均真实存在于 `/Users/qshf/my-project/slime-agentic`。

| 候选子系统 | 源项目位置 | 规模 | 定位 | 教学价值 | 复现档位 |
|-----------|-----------|------|------|---------|---------|
| **RL 数据契约** | `slime/utils/types.py:8` (Sample) | 175 行 | tokens/loss_mask/reward 的载体，连接 agent 与 trainer | 最高（一切的地基）| 本地可跑 |
| **Agent rollout** | `agentic/agentflow/rollout.py:104` (generate) | 22 文件 | 多轮 agent 交互 → 一条训练样本 | 最高 | 本地可跑 |
| **custom hook 接口** | `slime/utils/arguments.py` (三个 `--custom-*-path`) | — | generate/reward/eval 的可插拔契约 | 高（对齐源项目的扩展方式）| 本地可跑 |
| **最小训练闭环** | `train.py` (100 行主循环) | 100 行 | rollout→train→update_weights 编排 | 最高 | 本地可跑(fake trainer) |
| **Ray 调度层** | `slime/ray/{rollout,actor_group,placement_group}.py` | rollout.py 1260 行 | 进程/GPU 资源分配，rollout 与 train 分进程 | 高 | 本地单机可跑 |
| **同步 vs 异步循环** | `train.py` vs `train_async.py:31,39,62` | 100/77 行 | 提前发起下一轮 rollout，overlap train | 高（吞吐叙事的核心）| 本地可跑(fake) |
| **SGLang rollout 引擎** | `slime/backends/sglang_utils/sglang_engine.py` | 21.9K | 独立推理服务 + update_weights 权重同步 | 高 | 服务器 GPU 真跑 |
| **FSDP 训练后端** | `slime/backends/fsdp_utils/actor.py` | 42K | 包模型/offload/packing/一步真训练 | 中高 | 服务器 GPU 真跑 |
| **Megatron 并行** | `slime/backends/megatron_utils/{actor,model}.py` | 26K+31K | TP/PP/CP/EP 多维并行 | 中（概念为主，配置成本大）| 服务器多卡真跑 |
| **吞吐率实验** | `docs/zh/developer_guide/profiling.md` | — | 系统扫参数，定位瓶颈 | 高（飞轮：一次搭好后续实验免费）| 服务器真跑 |

**已主动放弃的方向**（写清"在什么条件下值得做"，未来不重复评估）：

- **Critic / PPO value model**（`train.py` 里 `use_critic` 分支）—— 值得做的条件：主线跑通后想对比 GRPO vs PPO 的算法差异时。当前用无 critic 的 GRPO 式流程即可，砍掉 critic 减少一半编排复杂度。
- **MemAgent / ToolOrchestra 两个 agentic 实现**（`agentic/memagent`、`agentic/ToolOrchestra`）—— 值得做的条件：AgentFlow 那条 calculator 主线跑通后，想理解"长上下文压缩"或"多 agent 路由"这两种不同的 rollout 形态时，各自独立开档。当前只复现最容易讲清 loss_mask 的 calculator agent。
- **on-policy distillation / PD 分离**（`docs/zh/advanced/pd-disaggregation.md`）—— 值得做的条件：SGLang 主线跑通、想优化多轮 agent 的推理吞吐时。当前只需理解"为什么 rollout 要独立推理服务"，不需要 prefill/decode 分离。
- **多机训练脚本 / MoE 配置** —— 源项目 README 明确列为避坑项，永不作为起点。值得做的条件：单机 Megatron 概念完全吃透后，且有真实多机需求时。

---

## 2. 决策：版本排序

### 2.1 候选 & 痛点起源

源项目是**一条紧耦合的训练流水线**，不是像 hermes 那样几个独立子系统。所以不按"独立系统"并列，而是用**一条全局版本链 V0→V9** 讲一个故事：每版解决上一版暴露的具体痛点，逐层从"写死的单样本 + fake trainer"演进到"真 SGLang + 真 FSDP + Megatron 多并行"。

### 2.2 为什么 V0 = 硬编码单样本闭环（不是先搭 Ray、不是先读 Megatron）

- **数据契约是一切的地基**：不先把 `Sample{tokens, loss_mask, reward}` 这个结构和"哪些 token 被训练"想清楚，后面 Ray/SGLang/FSDP 都是在搬运一个你还没理解的数据。源项目 README 的第一原则就是这个。
- **V0 必须"能跑但痛苦"**：一个写死的数学题，手工构造 tokens/loss_mask，跑一次 fake trainer step 打印统计。痛点立刻暴露——样本只有一条、reward 写死、loss_mask 靠手搓。这些痛点正好牵引出 V1/V2。
- **反例**：如果先搭 Ray，你会在还不理解 rollout_data_ref 里装什么的情况下就去调度它，属于"搬运不理解的数据"。

### 2.3 为什么不先做 Ray / 不先读 Megatron（反向决策成本）

- **先做 Ray 的代价**：Ray 是"把已有的串行闭环拆进程"的手段，没有闭环就没有可拆的东西。先做 Ray = 为空管道搭脚手架，V3 之前它无事可调度。
- **先读 Megatron 的代价**：Megatron 是最后一档不是偶然——它解决的是"大模型单卡放不下"的痛点，而这个痛点只有在你已经有一个能真训练的小模型闭环（V7 FSDP）之后才可感知。跳过前面直接啃 Megatron 源码 = README 明确列出的头号大坑。

### 2.4 路线图总览

```text
V0 硬编码单样本闭环 ──暴露: 样本写死/loss_mask手搓/reward写死
  → V1 Sample契约 + calculator agent rollout ──暴露: reward规则化/generate同步阻塞
  → V2 custom generate/reward hook 化 ──暴露: 单进程串行/rollout挤占train
  → V3 mini_slime 最小闭环(rollout_manager+fake trainer+weight_sync) ──暴露: 全串行/trainer空转
  → V4 Ray 化(actor 拆进程 + ray.get 同步点) ──暴露: ray.get强制等待
  → V5 同步 vs 异步(提前发起 rollout N+1 overlap train N) ──暴露: rollout用假引擎/无真吞吐
  ────────── 以上纯 CPU/fake，本地可跑；以下上服务器真跑 ──────────
  → V6 SGLang rollout 引擎(真推理服务 + update_weights 同步) ──暴露: 训练侧仍fake
  → V7 FSDP 训练后端(小模型真训一步 + offload/packing) ──暴露: 单并行维度/大模型放不下
  → V8 Megatron 并行概念(TP/PP/CP, slime 如何调 Megatron)
  → V9 吞吐率实验(扫参数定位瓶颈)  ← 飞轮档
```

**承诺范围**：确定承诺到 **V5**（本地纯 CPU 可完整跑通的最小 Agentic RL 闭环，这是 nano 项目的核心价值）。V6-V9 标"上服务器后按精力追加"，因为它们依赖 GPU 环境且投入产出比逐档递减。

### 2.5 排序依据原则

- **痛点驱动**（主逻辑）：每版标题后面那句"暴露:"就是下一版的存在理由。写不出暴露的新痛点 = 该档该合并或是终点档。
- **依赖优先**：数据契约(V0-V2) 是闭环(V3) 的前置；闭环是 Ray(V4) 的前置；Ray 是异步(V5) 的前置。后做就要给前面打补丁。
- **飞轮**：V3 的 fake trainer + 验证脚本一旦建好，V4-V9 每版回归测试免费；V9 的实验脚手架建好后所有调参实验自动产出对比。
- **难度避连击**：V5(异步逻辑，烧脑) 之后是 V6(接 SGLang，偏工程配置)，密度中等，缓冲一下再进 V7/V8 的分布式深水区。

### 2.6 如果改主意先做 calculator agent（V1 提前到 V0）会怎样

诱惑：agent 多轮交互最直观、最有成就感。代价：你会在还没有"fake trainer 消费样本"的出口时就造了一个复杂的 agent，无法验证生成的 loss_mask 到底喂给训练对不对——等于先造上游不接下游。所以 V0 坚持用**写死的单样本**把"样本如何被 trainer 消费"跑通，V1 再换上真 agent 产生样本。先通链路，再增真实度。

---

## 3. 各档最小切片

> V0-V5 是本地承诺范围，写详细；V6-V9 上服务器后再各自开 iteration-plan，这里只列骨架。

### V0: 硬编码单样本闭环
- **核心问题**：trainer 到底消费一个什么样的数据结构？哪些 token 被训练？
- **切片**：写死一道 `2+3=?`，手工填 `tokens/loss_mask/reward`，`fake_train_step` 只打印 batch 统计（reward_mean、trainable_token 数）。
- **验证**：`scripts/test_v0_contract.py` 断言 loss_mask 长度 == tokens 长度、trainable 数 == 预期。
- **简化掉的**：真 tokenizer（用 char/word 级假 token）、真 loss、多样本。
- **对应源项目**：`slime/utils/types.py:8` (Sample)、`train.py:80` (async_train 消费 rollout_data_ref)。

### V1: Sample 契约 + calculator agent rollout
- **上一版痛点**：样本写死、loss_mask 手搓，不知道真实多轮交互怎么生成它。
- **切片**：`toy_rl/agent/calculator_agent.py` 跑一个真 agent loop（模型生成工具调用 → python calculator 执行 → 结果拼回 → 继续生成 → 最终答案），程序化构造 loss_mask（agent token=1，tool 返回=0，final answer 按目标设）。模型可先用规则/桩代替。
- **验证**：`test_v1_rollout.py` 打印完整 trajectory + 断言 tool 返回段 loss_mask 全 0。
- **对应源项目**：`agentic/agentflow/rollout.py:104` (generate)、其 loss_mask 构造。

### V2: custom generate / reward hook 化
- **上一版痛点**：generate/reward 是写死的函数，没对齐 slime 的可插拔契约。
- **切片**：定义 `async def generate(args, sample, ...) -> sample` 和 `async def reward_func(args, sample) -> dict`，通过配置路径加载（模拟 `--custom-generate-function-path` / `--custom-rm-path`）。
- **验证**：`test_v2_hooks.py` 从路径动态加载并跑通。
- **对应源项目**：`slime/utils/arguments.py` 三个 hook flag、`agentic/agentflow/rollout.py:209` (reward_func)。

### V3: mini_slime 最小训练闭环
- **上一版痛点**：generate/reward 有了，但没有编排它们的循环，rollout 和 train 挤在一起。
- **切片**：`mini_slime/{rollout_manager,trainer,weight_sync}.py`，跑多轮 `rollout→fake train→fake update_weights`，输出 rollout_time/train_time/update_weights_time/reward_mean/tokens_per_rollout。
- **验证**：`test_v3_loop.py` 跑 2 轮，断言日志含全部指标。
- **对应源项目**：`train.py:65-93` 主循环。

### V4: Ray 化
- **上一版痛点**：全串行，rollout 时 trainer 空转。
- **切片**：RolloutManager / Trainer 改 `@ray.remote` actor，`ray.get` 做同步点，单机多进程。
- **验证**：`test_v4_ray.py` 确认两 actor 在不同进程 + 闭环仍通过。
- **对应源项目**：`slime/ray/{rollout.py:441 (RolloutManager), placement_group.py:181}`。

### V5: 同步 vs 异步循环
- **上一版痛点**：`ray.get` 强制等待，rollout 和 train 串行。
- **切片**：`mini_slime/train_sync.py`（严格串行）vs `train_async.py`（提前 `generate.remote(N+1)` overlap train N，update_weights 按 interval）。日志对比总耗时、互相等待时间。
- **验证**：`test_v5_async.py` 断言异步版总耗时 < 同步版。
- **对应源项目**：`train.py` vs `train_async.py:31,39,62`（提前发起 + interval 更新权重）。

### V6-V9（服务器档，骨架）
- **V6 SGLang rollout 引擎**：接真 SGLang，理解独立推理服务 + `update_weights` 把训练权重同步到引擎。源：`slime/backends/sglang_utils/sglang_engine.py`。
- **V7 FSDP**：小模型真训一步，offload/gradient checkpointing/sequence packing/dynamic batch。源：`slime/backends/fsdp_utils/actor.py`。
- **V8 Megatron 并行**：TP/PP/CP/EP 概念 + slime 如何调 Megatron。源：`slime/backends/megatron_utils/{actor,model}.py`。
- **V9 吞吐实验**：扫 rollout_batch_size / n_samples_per_prompt / max_tokens_per_gpu / colocate / sync-vs-async，定位瓶颈。源：`docs/zh/developer_guide/profiling.md`。

---

## 4. 维护规则

- 顺序变更必须留档（标日期 + 理由），旧决策只增不删。
- 新候选按 教学价值×自包含度 插入对应位置。
- 状态字段三处同步：本文档、`CLAUDE.md` 进度表、`docs/decisions/` 索引。
- 每完成一版，进度打 ✅ 并在 CLAUDE.md 决策日志写"选了 X 不选 Y 的 why + 真实踩坑"。
- V5 收尾后回到本文件，按场景 B 重审 V6-V9（那时已知服务器环境实况，排序可能调整）。
