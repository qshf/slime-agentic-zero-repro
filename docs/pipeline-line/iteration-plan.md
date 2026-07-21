# 主线一：最小训练流水线 — 迭代拆分计划

> 上一级是 [../system-roadmap.md](../system-roadmap.md)（项目级三主线排序）。本文件拆分主线一内部 V0→V5。
> 目标：从"写死单样本 + fake trainer"演进到"Ray 化 + 同步/异步对比"的最小 Agentic RL 闭环。

## 源项目概述

slime-agentic 的训练是一条流水线：`train.py` 主循环调 `RolloutManager.generate()`（`slime/ray/rollout.py:441`）产生一批 `Sample`（`slime/utils/types.py:8`），交给 actor model 的 `async_train()`，再 `update_weights()` 把新权重同步回 SGLang 推理引擎。custom generate/reward 通过 `--custom-generate-function-path` / `--custom-rm-path`（`slime/utils/arguments.py`）注入。`train.py`（同步）与 `train_async.py`（提前发起下一轮 rollout）是同一循环的两种调度。

主线一 nano 化这条流水线的**编排骨架**，暂不接真分布式后端（那是主线三）。

## 版本规划

| 版本 | 标题 | 核心概念 | 对应源项目 | 跑在哪 |
|------|------|---------|-----------|--------|
| V0 | 硬编码单样本 + fake trainer | Sample 数据契约 / loss_mask 语义 | `types.py:8`, `train.py:80` | 本地 |
| V1 | 极简 calculator agent + Qwen3-0.6B | 多轮 rollout → 样本 / 程序化 loss_mask | `agentic/agentflow/rollout.py:104` | 服务器 9934 |
| V2 | custom generate/reward hook 化 | 可插拔契约 (动态加载) | `arguments.py` 三 hook, `rollout.py:209` | 服务器 |
| V3 | mini_slime 最小闭环 | rollout_manager 编排 / 指标 | `train.py:65-93` | 服务器 |
| V4 | Ray 化 | @ray.remote actor / ray.get 同步点 | `ray/rollout.py:441`, `placement_group.py:181` | 服务器 |
| V5 | 同步 vs 异步 | 提前发起 rollout N+1 overlap train N | `train.py` vs `train_async.py:31,39,62` | 服务器 |

## 各版本详细设计

### V0: 硬编码单样本 + fake trainer
- **解决的问题：** trainer 到底消费一个什么数据结构？哪些 token 被训练、哪些不被训练？不先想清楚这个，后面所有搬运都是搬运一个没理解的数据。
- **引入的概念：** `Sample` dataclass（对齐源项目字段子集：`tokens/response/loss_mask/reward/label`）；`fake_train_step` 只做统计（不真训练），打印 batch 的 reward_mean、trainable_token 数、total_token 数。
- **暴露的问题：** 样本写死一条、reward 写死、loss_mask 手工填——无法规模化，也没见过真实交互怎么生成这些字段。→ 引出 V1。
- **对应源项目：** `slime/utils/types.py:8`（Sample 全字段，nano 只取子集）、`train.py:80`（`async_train(rollout_id, rollout_data_ref)` 消费样本）。

### V1: 极简 calculator agent + Qwen3-0.6B
- **上一版痛点：** 样本写死，不知道真实多轮交互如何生成 tokens/loss_mask。
- **解决方法：** 一个真 agent loop —— Qwen3-0.6B 生成工具调用 → python calculator 执行 → 结果拼回上下文 → 模型继续 → 最终答案。程序化构造 loss_mask（agent 生成段=1，tool 返回段=0）。
- **新引入的概念：** 真推理引擎（服务器）、多轮 trajectory、工具边界与 loss_mask 的对应关系。
- **对应源项目：** `agentic/agentflow/rollout.py:104`（`generate()` 的多轮形态）。
- **暴露的新问题：** generate 和 reward 是写死的函数，没对齐 slime 的可插拔契约，换任务要改代码。→ 引出 V2。

### V2: custom generate/reward hook 化
- **上一版痛点：** generate/reward 硬编码，换 agent/任务要改主循环代码。
- **解决方法：** 定义标准签名 `async def generate(args, sample) -> Sample` / `async def reward_func(args, sample) -> dict`，通过配置路径动态 import 加载（模拟三个 `--custom-*-path`）。
- **新引入的概念：** 可插拔 hook 契约 —— 主循环不认识具体 agent，只认签名。这是主线二能"插进闭环"的前置。
- **对应源项目：** `slime/utils/arguments.py`（三 hook flag）、`agentic/agentflow/rollout.py:209`（reward_func 签名）。
- **暴露的新问题：** 有了 generate/reward，但还是单进程串行手动调用，没有编排循环，rollout 和 train 挤在一个函数里。→ 引出 V3。

### V3: mini_slime 最小闭环  ✅ 完成（5090 端到端 4/4）
- **上一版痛点：** 缺一个编排层把 rollout→train→update_weights 串成多轮循环。
- **解决方法：** `mini_slime/{rollout_manager,trainer,weight_sync}.py`，跑多轮 `rollout → fake train → fake update_weights`，输出 rollout_time / train_time / update_weights_time / reward_mean / tokens_per_rollout。
- **新引入的概念：** RolloutManager 编排职责、weight_sync 抽象（fake）、每轮指标采集（V9 吞吐实验的雏形）。
- **对应源项目：** `train.py:65-93`（主循环编排）。
- **暴露的新问题：** 全串行 —— rollout 时 trainer 空转，train 时 rollout 空转，单进程无法并行。→ 引出 V4。

### V4: Ray 化  ✅ 完成（5090 端到端 3/3，actor 分进程）
- **上一版痛点：** 单进程串行，rollout 和 train 无法并行，资源利用率低。
- **解决方法：** RolloutManager / Trainer 改 `@ray.remote` actor，用 `ray.get` 显式标注同步点，单机多进程运行。
- **新引入的概念：** Ray actor、ObjectRef、`.remote()` / `ray.get()`、rollout 与 train 分进程。
- **对应源项目：** `slime/ray/rollout.py:441`（RolloutManager actor）、`placement_group.py:181`（create_rollout_manager）。
- **暴露的新问题：** `ray.get` 强制等待——同步循环里 rollout N+1 必须等 train N 完全结束才能开始，串行瓶颈仍在。→ 引出 V5。

### V5: 同步 vs 异步（主线一承诺终点）
- **上一版痛点：** `ray.get` 的位置导致 rollout 和 train 严格串行。
- **解决方法：** 两个版本对比 —— `train_sync.py`（严格串行）vs `train_async.py`（在 train N 之前就 `generate.remote(N+1)` 提前发起下一轮 rollout，overlap；update_weights 按 interval 而非每轮）。日志对比总耗时、rollout 等 train 时间、train 等 rollout 时间。
- **新引入的概念：** 异步 pipeline、future 提前发起、update_weights_interval 权衡（权重新鲜度 vs 吞吐）。
- **对应源项目：** `train.py`（同步）vs `train_async.py:31`（`rollout_data_next_future = generate.remote(start)`）、`:39`（提前发起 N+1）、`:62`（interval 更新权重）。
- **这是主线一终点：** 闭环的系统骨架已完整；再往下是把 fake 部件换成真部件（主线三），或把 toy agent 换成真 agent（主线二）。

## 关于 fake trainer 的说明

V0-V5 的 trainer 全程是 **fake**（只算统计、sleep 模拟耗时），因为主线一的教学目标是**数据如何流动 + 如何编排**，不是"如何真训练"。真训练一步在主线三 V7（FSDP）引入。这符合源项目 README 原则"先复现数据如何流动，再学习系统如何加速"。

## 反模式自查

- V0 不写成 hello world：它必须"能完成核心场景（跑通一次 fake 训练统计）但痛苦（样本写死）"。✅
- 不一次引入多概念：V1 只引入"真 agent 生成样本"，hook 化留给 V2。✅
- 不抄源项目演进历史：nano 按教学逻辑（先通链路再增真实度），非源项目真实提交顺序。✅
