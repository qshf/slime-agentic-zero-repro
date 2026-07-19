# slime-agentic-zero-repro — 项目上下文 primer

> 新会话起手必读。每完成一版更新一次。详细路线见 [docs/system-roadmap.md](docs/system-roadmap.md)。

## 1. TL;DR

- **项目**：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
- **源项目**：slime-agentic —— 基于 Ray + SGLang + Megatron/FSDP 的 Agentic RL 训练框架（58K LOC）。
- **当前阶段**：阶段一已完成（system-roadmap 路线图对齐中）。代码尚未开始，下一步是 V0。

## 2. 路径与仓库

- **源项目**：`/Users/qshf/my-project/slime-agentic`（github: LMIS-ORG/slime-agentic，分支 main。只读，用于对照）
- **nano 项目**：`/Users/qshf/my-project/slime-agentic-zero-repro`（git 已 init，主分支 main）
- **当前活跃分支**：`main`（V0 开始时切 `v0` 分支）
- **工作流**：本地开发 → 推 git → 服务器拉取运行/调试（多卡服务器）。

## 3. 进度状态

| 版本 | 标题 | 复现档位 | 状态 |
|------|------|---------|------|
| V0 | 硬编码单样本闭环 | 本地 | 计划中（下一步）|
| V1 | Sample 契约 + calculator agent rollout | 本地 | 计划中 |
| V2 | custom generate/reward hook 化 | 本地 | 计划中 |
| V3 | mini_slime 最小训练闭环 | 本地 | 计划中 |
| V4 | Ray 化 | 本地单机 | 计划中 |
| V5 | 同步 vs 异步循环 | 本地 | 计划中（**承诺范围终点**）|
| V6 | SGLang rollout 引擎 | 服务器 GPU | 按精力追加 |
| V7 | FSDP 训练后端 | 服务器 GPU | 按精力追加 |
| V8 | Megatron 并行概念 | 服务器多卡 | 按精力追加 |
| V9 | 吞吐率实验 | 服务器 | 按精力追加 |

## 4. 环境前置

- **本地档 (V0-V5)**：纯 Python，无 GPU。用 uv 管理。fake trainer 不需要真模型。
  - 依赖：ray（V4 起）；calculator agent 的"模型"先用规则/桩，无需真 LLM。
- **服务器档 (V6-V9)**：需要 GPU。SGLang engine、FSDP/Megatron 真训练。具体 env 待 V6 开始时补充。

## 5. 验活 cheatsheet

```bash
# 当前分支
git -C /Users/qshf/my-project/slime-agentic-zero-repro branch --show-current
# 源项目对照(只读)
ls /Users/qshf/my-project/slime-agentic/train.py /Users/qshf/my-project/slime-agentic/train_async.py
# 跑当前版本回归(V0 起每版一个)
# python scripts/test_v0_contract.py
```

## 6. 决策日志（按版本累加）

### 路线图阶段（2026-07-20）
- **选了痛点驱动版本链，不选按周学主题**：原 README 是"按周读源码记笔记"，改成"每版解决上一版暴露的具体痛点 + 可运行验证"，契合"复现中学"目标。
- **V0 用写死单样本，不先做 calculator agent**：先通链路（样本如何被 trainer 消费）再增真实度，避免先造上游不接下游。
- **砍掉 critic/PPO、MemAgent、ToolOrchestra、PD 分离**：见 system-roadmap "已主动放弃" 各自的"重启条件"。
- **承诺到 V5（本地纯 CPU 闭环）**：V6-V9 依赖 GPU 且投入产出比递减，标"按精力追加"。

## 7. 待办 / 已知问题

- [ ] 与用户对齐 system-roadmap（阶段一检查点）。
- [ ] 对齐后写 V0-V3 的 iteration-plan（数据契约+闭环这条主线可合并成一份）。
- [ ] 确定本地"模型桩"的形态：V1 calculator agent 的模型用规则模拟还是接一个小 LLM。
