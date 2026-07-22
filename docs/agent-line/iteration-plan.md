# 主线二：三个真实 Agent — 迭代拆分计划

> 上一级是 [../system-roadmap.md](../system-roadmap.md)（项目级三主线排序）。本文件拆分主线二内部 A1→A3。
> 目标：把主线一闭环里跑的**自造 toy calculator agent**，逐步换成三个**真实 agent**，按难度递增，
> 一次做完一个再下一个（用户明确要求）。难度序：MemAgent < AgentFlow < ToolOrchestra（explore 实测）。

## 前置：主线一已交付的闭环（A1-A3 的插座）

主线一 V0-V5 已把系统骨架打通：`Sample 契约 → data_source/generate/reward hook → RolloutManager 编排
→ Ray 分进程 → 同步/异步 overlap`。三个 hook 全部按路径动态加载（`data_source_path` /
`custom_generate_function_path` / `custom_rm_path`）。**主线二的每个 agent = 换这三条路径 + agent 专属
Args 字段**，闭环骨架一行不改——这正是 V2 hook 化的兑现。

## 版本规划

| 版本 | 标题 | 核心概念（新引入） | loss_mask 特点 | 对应源项目 | 跑在哪 |
|------|------|------------------|---------------|-----------|--------|
| A1 | MemAgent | chunk 记忆更新循环（单引擎、无工具）| **全 1**（无工具边界，最简） | `agentic/memagent/rollout.py` | 服务器 |
| A2 | AgentFlow | Planner→Executor→Verifier 多引擎协调 | **executor token=0**（工具边界精华课）| `agentic/agentflow/core/*` | 服务器 |
| A3 | ToolOrchestra（仅 QA）| 多专家路由 + 多组件 reward | orchestrator=1，专家/工具=0 | `agentic/ToolOrchestra/{orchestra_solver,reward}.py` | 服务器 |

难度避连击：A2（协调多引擎，烧脑）之后 A3 砍到只做 QA 路径（func_call/tau2 记录设计不实现）。

## 各版本详细设计

### A1: MemAgent  ✅ 本地离线通过（loss_mask 全 1 + boxed reward + 闭环），待服务器
- **解决的问题：** 主线一跑的是 toy agent，没见过"真实 agent"怎么产样本。MemAgent 是真实 agent 里最简的
  （单引擎、无工具），最接近 toy，平滑过渡。
- **引入的概念：** chunk-by-chunk 记忆更新循环——`for chunk: memory = LLM(problem, memory, chunk)`，
  最后 `answer = LLM(problem, memory)`；**每轮是一条独立训练序列**（memory 以文本跨轮传递，非增长上下文）；
  reward = 抽 `\boxed{}` 答案 + is_equiv 归一化。
- **loss_mask 特点（教学核心）：** response **全 1**——没有工具返回段要置 0。这是最简 loss_mask，
  和 A2 的"executor token=0"（工具边界）形成对照，为 A2 铺垫。
- **对应源项目：** `agentic/memagent/rollout.py`（chunk 循环 in generate + boxed reward，无 solver）。
- **暴露的新问题：** loss_mask 太简单（全 1），教不了"模型输出 vs 工具返回"的边界；且单任务单引擎。→ 引出 A2。

### A2: AgentFlow（工具边界精华课）
- **上一版痛点：** MemAgent loss_mask 全 1，教不了工具边界。
- **引入的概念：** Planner→Executor→Verifier 多引擎；**executor 生成的 token loss_mask=0**（工具/执行段
  不训练）；LLM-as-judge reward。这是 README 问题 #2（loss_mask 如何区分模型输出 vs 工具返回）的精华课。
- **对应源项目：** `agentic/agentflow/core/{solver,planner,executor,verifier,rewarder}.py`。
- **暴露的新问题：** 单任务；缺多 agent 路由和多组件 reward。→ 引出 A3。

### A3: ToolOrchestra（仅 QA 路径，主线二终点）
- **上一版痛点：** AgentFlow 单任务，缺多专家路由 + 多组件 reward。
- **引入的概念：** orchestrator 路由到多专家（QA 路径），reward = 正确性 + 成本 + 延迟。
- **简化：** func_call 路径 + tau2 环境模拟器**不实现**（记录设计，见 system-roadmap 已弃/降级方向）。
- **对应源项目：** `agentic/ToolOrchestra/{orchestra_solver.py,reward.py}`（QA 分支）。

## 关于 fake trainer 的延续

主线二仍复用主线一的 **fake trainer**（不算真梯度）：主线二的教学目标是**不同 agent 的 rollout 形态 +
loss_mask 语义 + reward 形态**，不是训练数学。真训练一步在主线三 V7（FSDP）引入。

## 反模式自查
- A1 不一次引入多概念：只引入"记忆循环 + loss_mask 全 1 + boxed reward"，工具边界留 A2。✅
- 复用主线一闭环，不重造编排：A1 只加 data/generate/reward，train/RolloutManager/Trainer 零改。✅
