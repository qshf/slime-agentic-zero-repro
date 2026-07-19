# slime-agentic-zero-repro — 项目上下文 primer

> 新会话起手必读。每完成一版更新一次。详细路线见 [docs/system-roadmap.md](docs/system-roadmap.md)。

## 1. TL;DR

- **项目**：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
- **源项目**：slime-agentic —— 基于 Ray + SGLang + Megatron/FSDP 的 Agentic RL 训练框架（58K LOC）。
- **当前阶段**：阶段一（路线图已重构为三条主线，等用户对齐）。代码尚未开始，下一步 V0。

## 2. 路径与仓库

- **源项目**：`/Users/qshf/my-project/slime-agentic`（github: LMIS-ORG/slime-agentic，分支 main。只读，用于对照）
- **nano 项目**：`/Users/qshf/my-project/slime-agentic-zero-repro`（git 已 init，主分支 main）
- **当前活跃分支**：`main`（V0 开始时切 `v0` 分支）
- **工作流**：**本地只开发**（写码+推 git）→ **SSH 9934 多卡服务器**跑通/验证/调试。V0（纯 fake）本地可验；V1 起接 Qwen3-0.6B，都在服务器验证。

## 3. 进度状态

三条主线，详见 [docs/system-roadmap.md](docs/system-roadmap.md)。

**主线一 · 最小训练流水线**（做扎实）

| 版本 | 标题 | 跑在哪 | 状态 |
|------|------|--------|------|
| V0 | 硬编码单样本 + fake trainer | 本地 | 计划中（下一步）|
| V1 | 极简 calculator agent + Qwen3-0.6B | 服务器 | 计划中 |
| V2 | custom generate/reward hook 化 | 服务器 | 计划中 |
| V3 | mini_slime 最小闭环 | 服务器 | 计划中 |
| V4 | Ray 化 | 服务器 | 计划中 |
| V5 | 同步 vs 异步 | 服务器 | 计划中（**主线一终点**）|

**主线二 · 三个真实 Agent**（做扎实，按难度）

| 版本 | 标题 | 关键学点 | 状态 |
|------|------|---------|------|
| A1 | MemAgent | 单引擎/无工具/loss_mask 全 1 | 计划中 |
| A2 | AgentFlow | executor token 不训练（工具边界精华）| 计划中 |
| A3 | ToolOrchestra（仅 QA 路径）| 多专家路由 + 多组件 reward | 计划中（**主线二终点**）|

**主线三 · 分布式后端**（记录设计，后续复现）：V6 SGLang / V7 FSDP / V8 Megatron / V9 吞吐实验。

## 4. 环境前置

- **本地**：纯开发。仅 V0 fake trainer 可本地跑（纯 Python，uv 管理）。
- **服务器 SSH 9934（多卡）**：V1 起全部在此验证。
  - V1 起需 **Qwen3-0.6B**（真推理生成工具调用）。
  - V4 起需 ray。V6 起需 SGLang engine、FSDP/Megatron（env 待 V6 补充）。

## 5. 验活 cheatsheet

```bash
# 本地：当前分支
git -C /Users/qshf/my-project/slime-agentic-zero-repro branch --show-current
# 本地：源项目对照(只读)
ls /Users/qshf/my-project/slime-agentic/train.py /Users/qshf/my-project/slime-agentic/train_async.py
# 服务器 9934：跑当前版本回归(V1 起每版一个)
# ssh -p 9934 <server> 'cd <repo> && python scripts/test_v1_rollout.py'
```

## 6. 决策日志（按版本累加）

### 路线图阶段（2026-07-20）
- **选了痛点驱动版本链，不选按周学主题**：原 README 是"按周读源码记笔记"，改成"每版解决上一版暴露的具体痛点 + 可运行验证"，契合"复现中学"目标。
- **拆成三条主线（系统骨架 / 三个 agent / 分布式后端）**：系统轴和方法轴正交，拧成一条 V0-V9 会让"教系统还是教 agent"变模糊。
- **V0 用写死单样本，不先做 calculator agent**：先通链路（样本如何被 trainer 消费）再增真实度，避免先造上游不接下游。
- **三个 agent 全复现，难度序 MemAgent<AgentFlow<ToolOrchestra**：实测 LOC/引擎数/工具/reward 复杂度定序（explore 报告）。ToolOrchestra **只做 QA 路径**，func_call+tau2 环境模拟器忠实复现基本不可能，记录设计不实现。
- **V1 起接 Qwen3-0.6B 真模型**：用户要求；因此 V1 起都需 GPU，都在服务器验证。
- **砍 critic/PPO**：见 system-roadmap 重启条件。

## 7. 待办 / 已知问题

- [ ] 与用户对齐 system-roadmap 三主线结构（阶段一检查点）。
- [ ] 对齐后写主线一 V0-V3 的 iteration-plan（数据契约+闭环可合并成一份）。
- [ ] 确认 Qwen3-0.6B 在服务器 9934 的部署方式（SGLang 起服务？transformers 直接加载？）。
- [ ] 确认本地→服务器的代码同步方式（git push/pull 还是 rsync）。
