# slime-agentic 复现 — 子系统实现先后顺序考量

> 受众：未来的自己 / 协作 LLM。每开新版本前 review；新增子系统插入对应位置。
> 目的：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
> 本文件是项目级活文档（决定"做哪条主线、什么顺序"）。单主线内部拆分见 `docs/<line>/iteration-plan.md`。

## 核心原则（来自源项目 README，被反复验证）

```
先复现数据如何流动，再学习系统如何加速。
```

一切排序都服从这条：先把 `Sample → generate → reward → trainer → update_weights → rollout engine` 这条数据链跑通，再逐层换上真实的分布式部件（Ray → SGLang → FSDP → Megatron），再逐个复现越来越复杂的真实 agent。

## 硬件、模型与工作流（已与用户对齐 2026-07-20）

- **工作流**：本地只负责**开发**（写代码、推 git）；所有**跑通/验证/调试**在 **SSH 9934 多卡服务器**上进行。
- **模型桩**：从 V1 起接 **Qwen3-0.6B** 真推理（不再用规则桩）。因此 V1 及以后都需要 GPU → 都在服务器验证。V0（纯 fake）可本地验证。
- **深度**：主线一 V0-V5 + 主线二三个 agent **做扎实**；主线三 V6-V9（分布式后端）**先记录设计、后续复现**。
- **推论**：CLAUDE.md 验活 cheatsheet 分"本地开发"和"服务器 9934 运行"两栏；每版验证脚本在服务器上跑。

---

## 1. 候选子系统全景（源项目对照）

已核对，路径与行数均真实存在于 `/Users/qshf/my-project/slime-agentic`。

### 主线一：最小训练流水线（系统骨架）

| 子系统 | 源项目位置 | 规模 | 定位 |
|--------|-----------|------|------|
| RL 数据契约 | `slime/utils/types.py:8` (Sample) | 175 行 | tokens/loss_mask/reward 载体 |
| custom hook 接口 | `slime/utils/arguments.py` (三个 `--custom-*-path`) | — | generate/reward/eval 可插拔契约 |
| 最小训练闭环 | `train.py` (100 行主循环) | 100 行 | rollout→train→update_weights 编排 |
| Ray 调度层 | `slime/ray/{rollout,actor_group,placement_group}.py` | rollout.py 1260 行 | rollout 与 train 分进程 |
| 同步 vs 异步 | `train.py` vs `train_async.py:31,39,62` | 100/77 行 | 提前发起下一轮 rollout overlap train |

### 主线二：三个真实 Agent（rollout 方法，按难度递增）

难度已实测（见 `docs/decisions/` 附的 explore 报告），排序 **MemAgent < AgentFlow < ToolOrchestra**：

| Agent | 源项目位置 | 规模 | 引擎 | 工具 | reward | loss_mask 特点 |
|-------|-----------|------|------|------|--------|---------------|
| **MemAgent** | `agentic/memagent/rollout.py` | 2.0K LOC | 1 | 无 | 纯规则(math 归一化) | 全 1（无工具边界，最简）|
| **AgentFlow** | `agentic/agentflow/{rollout.py,core/}` | 2.5K LOC | 3-5 | 2-N | LLM-as-judge | **executor token=0**（工具边界精华课）|
| **ToolOrchestra** | `agentic/ToolOrchestra/{orchestra_solver.py,reward.py,tau2/}` | 31K LOC | 1+N专家 | 4 | 规则+LLM+成本+延迟 | orchestrator=1，专家/工具=0 |

### 主线三：分布式后端（V6-V9，记录设计后续复现）

| 子系统 | 源项目位置 | 规模 | 定位 |
|--------|-----------|------|------|
| SGLang rollout 引擎 | `slime/backends/sglang_utils/sglang_engine.py` | 21.9K | 独立推理服务 + update_weights 权重同步 |
| FSDP 训练后端 | `slime/backends/fsdp_utils/actor.py` | 42K | 包模型/offload/packing/真训练一步 |
| Megatron 并行 | `slime/backends/megatron_utils/{actor,model}.py` | 26K+31K | TP/PP/CP/EP 多维并行 |
| 吞吐率实验 | `docs/zh/developer_guide/profiling.md` | — | 扫参数定位瓶颈（飞轮档）|

**已主动放弃 / 降级的方向**（写清"在什么条件下值得做"）：

- **Critic / PPO value model**（`train.py` `use_critic` 分支）—— 主线一跑通后想对比 GRPO vs PPO 时再加。当前用无 critic 的 GRPO 式流程，减一半编排复杂度。
- **ToolOrchestra 的 `func_call` 路径 + tau2 环境模拟器**（`agentic/ToolOrchestra/tau2/`，含 10+ 领域仿真、子进程文件协议）—— **主线二先只复现 QA 路径**（多轮检索+推理）。tau2 环境模拟器**延后处理**（非永久放弃）：A3 收尾后单独评估如何复现或学习其细节（子进程文件协议、领域状态机、reward_info 解析），可能的路径是"精读 + 画图理解"或"直接 import 原 tau2 包驱动一个最小 func_call rollout"，届时在本文件新开一节记录。
- **AgentFlow / MemAgent 的 math 答案归一化细节**（`_strip_string` LaTeX/分数处理）—— 复现主流程即可，边界 case 用最简版，不追平。

---

## 2. 决策：主线顺序与版本排序

### 2.1 为什么拆成三条主线（而不是一条 V0-V9）

上一版路线图把所有东西塞进一条线，是错的。实际有**两个正交的演进轴**：
- **系统轴**：闭环怎么从串行 fake 长成真分布式（主线一 + 主线三）。
- **方法轴**：rollout 里的 agent 从最简长到最复杂（主线二）。

强行拧成一条线会让"这一版到底在教系统还是教 agent"变模糊。拆开后每条主线内部才是干净的痛点驱动链。

### 2.2 主线执行顺序：一 → 二 → 三

- **先主线一**：没有能消费样本的闭环，agent 生成的样本无处验证（先造上游不接下游是反模式）。主线一 V1 用一个**自造的极简 calculator agent** 打通"Qwen3-0.6B 真推理 → 样本 → 闭环"，controllable、最快见效。
- **再主线二**：闭环跑通后，把自造 toy agent 逐步换成三个真实 agent，按难度 MemAgent → AgentFlow → ToolOrchestra，一次做完一个再下一个（用户明确要求）。
- **最后主线三**：把 fake trainer / 假引擎换成真 SGLang + FSDP + Megatron。这些解决的痛点（大模型放不下、吞吐低）只有在前两条主线跑通后才可感知。

### 2.3 为什么主线二内部是 MemAgent → AgentFlow → ToolOrchestra

- **MemAgent 最先**：单引擎、无工具、loss_mask 全 1，是"真实 agent"里最接近主线一 toy 的，平滑过渡。痛点：loss_mask 太简单，教不了工具边界。
- **AgentFlow 居中**：Planner→Executor→Verifier，**executor 的 token 不参与训练**——这正是 README 问题 #2（loss_mask 如何区分模型输出 vs 工具返回）的精华课。痛点：需要协调 3-5 个引擎。
- **ToolOrchestra 最后**：最复杂（31K LOC、多专家、tau2）。**只复现 QA 路径**，func_call/tau2 记录不实现。放最后因为它暴露"多 agent 路由 + 多组件 reward（正确性+成本+延迟）"，是集大成。

**难度避连击**：AgentFlow（协调多引擎，烧脑）之后如果直接上 ToolOrchestra 全量会崩，所以 ToolOrchestra 砍到只做 QA 路径，密度可控。

### 2.4 路线图总览

```text
【主线一 · 系统骨架】本地开发 + 服务器验证
  V0 硬编码单样本+fake trainer(本地可跑) ──暴露: 样本写死/loss_mask手搓
   → V1 极简 calculator agent + Qwen3-0.6B 真推理(服务器) ──暴露: generate/reward 写死没对齐 slime
   → V2 custom generate/reward hook 化 ──暴露: 单进程串行, rollout 挤占 train
   → V3 mini_slime 闭环(rollout_manager+fake trainer+weight_sync) ──暴露: 全串行 trainer 空转
   → V4 Ray 化(actor 拆进程+ray.get 同步点) ──暴露: ray.get 强制等待
   → V5 同步 vs 异步(提前发起 rollout N+1 overlap train N)   ★主线一承诺终点

【主线二 · 三个真实 Agent】插进主线一闭环, 按难度各自做完整
   A1 MemAgent(单引擎/无工具/loss_mask全1) ──暴露: 教不了工具边界
   → A2 AgentFlow(Planner→Executor→Verifier, executor token 不训练) ──暴露: 多引擎协调
   → A3 ToolOrchestra 仅 QA 路径(多专家路由+多组件reward)   ★主线二承诺终点

【主线三 · 分布式后端】先记录设计, 后续复现
   V6 SGLang 真引擎+update_weights → V7 FSDP 真训一步 → V8 Megatron 并行概念 → V9 吞吐实验
```

**承诺范围**：主线一 V0-V5 + 主线二 A1-A3 **做扎实**。主线三 V6-V9 先在文档里记录设计与源项目对照，具备服务器大模型环境后按精力复现。

### 2.5 排序依据原则

- **痛点驱动**：每版"暴露:"那句就是下一版的存在理由。
- **依赖优先**：数据契约→闭环→Ray→异步；闭环是三个 agent 的前置。
- **飞轮**：V3 的 fake trainer + 验证脚本一旦建好，后续每版回归免费；A1 的 agent-loop 骨架被 A2/A3 复用。
- **难度避连击**：AgentFlow 之后 ToolOrchestra 砍到只 QA。

### 2.6 反向决策成本：如果先做主线二（跳过主线一直接复现 agent）会怎样

诱惑：真实 agent 最有成就感。代价：agent 生成的 tokens/loss_mask 没有 trainer 消费它、没有闭环验证它对不对，等于造了上游不接下游。所以坚持先用自造 toy agent 把闭环通了，再上真 agent。

---

## 3. 各档最小切片

> 主线一 V0-V5、主线二 A1-A3 写详细；主线三 V6-V9 只列骨架，服务器环境就绪后各自开 iteration-plan。

### 主线一

**V0 硬编码单样本闭环**（本地）
- 核心问题：trainer 消费什么数据结构？哪些 token 被训练？
- 切片：写死 `2+3=?`，手工填 tokens/loss_mask/reward，`fake_train_step` 打印 batch 统计。
- 验证：`scripts/test_v0_contract.py` 断言 loss_mask 长度==tokens 长度、trainable 数==预期。
- 对应源项目：`slime/utils/types.py:8`、`train.py:80`。

**V1 极简 calculator agent + Qwen3-0.6B 真推理**（服务器）
- 上一版痛点：样本写死，不知道真实交互怎么生成。
- 切片：Qwen3-0.6B 生成工具调用 → python calculator 执行 → 结果拼回 → 继续 → 最终答案；程序化构造 loss_mask（agent=1, tool 返回=0）。
- 验证：`scripts/test_v1_rollout.py`（服务器）打印完整 trajectory + 断言 tool 段 loss_mask 全 0。
- 对应源项目：`agentic/agentflow/rollout.py:104`（generate 形态参考）。

**V2 custom generate/reward hook 化**
- 上一版痛点：generate/reward 写死，没对齐 slime 可插拔契约。
- 切片：`async def generate(args, sample)->sample` / `async def reward_func(args, sample)->dict`，配置路径动态加载。
- 验证：`test_v2_hooks.py` 从路径加载并跑通。
- 对应源项目：`slime/utils/arguments.py` 三 hook、`agentic/agentflow/rollout.py:209`。

**V3 mini_slime 最小闭环** ✅ 完成（5090 端到端 4/4，reward=1.0；本地离线 3/3）
- 上一版痛点：有 generate/reward 但无编排循环。
- 切片：`mini_slime/{args,rollout_manager,trainer,weight_sync,train}.py`，多轮 rollout→fake train→fake update_weights，输出各阶段耗时+reward_mean+tokens_per_rollout。
- 验证：`test_v3_loop.py` 跑 2 轮断言指标齐全。
- 对应源项目：`train.py:65-93`。

**V4 Ray 化**
- 上一版痛点：全串行，rollout 时 trainer 空转。
- 切片：RolloutManager/Trainer 改 `@ray.remote`，`ray.get` 做同步点，单机多进程。
- 验证：`test_v4_ray.py` 确认两 actor 不同进程 + 闭环通过。
- 对应源项目：`slime/ray/rollout.py:441`、`placement_group.py:181`。

**V5 同步 vs 异步**（主线一承诺终点）✅ 完成（5090 端到端 async 193.7s < sync 209.3s；本地离线 async 1.492s < sync 2.018s）
- 上一版痛点：ray.get 强制等待。
- 切片：同步基线复用 `train_ray.py`（≡ 源 train.py）vs 新建 `train_async.py`（提前 `generate.remote(N+1)` overlap train N，update_weights 按 interval）。为让 fake trainer 的 overlap 可观测，加 `fake_train_seconds`/`fake_gen_seconds` 模拟耗时旋钮（默认 0，见 v5.md 偏离表）。
- 验证：`test_v5_async.py --offline` 断言异步版总耗时 < 同步版（异步 rollout 1/2 `wait_gen=0.000s` 是 overlap 铁证）。
- 对应源项目：`train.py` vs `train_async.py:29/31/39/62`。

### 主线二

**A1 MemAgent**  ✅ 完成（5090 端到端 single reward=1.0、闭环 reward_mean=0.5；离线 loss_mask 全 1 + boxed reward + 闭环）
- 引入：真实 agent 的 chunk 记忆更新循环（单引擎，无工具）。loss_mask 全 1（无工具边界，最简）。
- 简化：math 归一化只做主流程；char 级分 chunk、硬编码 mini QA（见 docs/decisions/a1.md 偏离表）。
- 对应源项目：`agentic/memagent/rollout.py`（298 行，reward 内嵌 272-297）。

**A2 AgentFlow**（工具边界精华课）✅ 完成（5090 端到端 reward=1.0、闭环 reward_mean=1.0；真 4B planner + 真 DeepSeek coder）
- 上一版痛点：MemAgent loss_mask 全 1，教不了工具边界。
- 引入：Planner→Executor→Verifier，**只有 planner 进 training turns**（executor/verifier/final_output 不训练）；LLM-as-judge reward 回退。
- 简化：引擎数从源 3-5 降到**双引擎**（训练 planner / 固定其余）；单玩具工具 calculator；命令解析只留正则（见 docs/decisions/a2.md 偏离表）。
- 对应源项目：`agentic/agentflow/core/{solver,planner,executor,verifier,rewarder}.py`。

**A3 ToolOrchestra（仅 QA 路径）** ✅ 离线完成；服务器待验（主线二承诺终点）
- 上一版痛点：AgentFlow 单任务；缺多 agent 路由和多组件 reward。
- 引入：orchestrator 路由到多专家（QA 路径），reward = 正确性+成本+延迟。
- 简化：**func_call 路径 + tau2 环境模拟器不实现**（记录设计）。
- 对应源项目：`agentic/ToolOrchestra/{orchestra_solver.py,reward.py}`（QA 分支）。

### 主线三（骨架，服务器就绪后展开）
- V6 SGLang 真引擎 + update_weights 权重同步；V7 FSDP 真训一步(offload/packing)；V8 Megatron TP/PP/CP/EP 概念；V9 扫参数吞吐实验。

---

## 4. 维护规则

- 顺序变更必须留档（标日期 + 理由），旧决策只增不删。
- 状态字段三处同步：本文档、`CLAUDE.md` 进度表、`docs/decisions/` 索引。
- 每完成一版，进度打 ✅ 并在 CLAUDE.md 决策日志写"选了 X 不选 Y 的 why + 真实踩坑"。
- 主线一 V5 收尾后回到本文件，按场景 B 重审主线二细节；主线二收尾后重审主线三是否具备服务器条件。
