# slime-agentic-zero-repro — 项目上下文 primer

> 新会话起手必读。每完成一版更新一次。详细路线见 [docs/system-roadmap.md](docs/system-roadmap.md)。

## 1. TL;DR

- **项目**：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
- **源项目**：slime-agentic —— 基于 Ray + SGLang + Megatron/FSDP 的 Agentic RL 训练框架（58K LOC）。
- **当前阶段**：主线一进行中。V0、V1、V2、V3、V4 均已验证通过；下一步 V5（同步 vs 异步，主线一终点）。

> **铁律（每版必须遵守）**：**代码范式遵从原项目**。nano 的代码结构 / 接口签名 / 命名 / 数据流必须对齐 slime-agentic 源项目在对应位置的写法。**只允许在其基础上做得更清晰（更好），不允许比源项目更乱、更 hack、更偏离（更差）**。判断法：写任一段前先问"源项目对应位置怎么做的"，对齐它；要偏离只能朝"更清晰且语义等价"的方向，并在注释里写明为何偏离。反例（已修）：把 agent loop 抄成两份塞进 generate() 里——源项目 rollout.py 是薄适配器，loop 在 solver.py。
>
> **锚点优先级**：**slime-agentic 是唯一首要锚点**；其它 nano 项目（如 nano_hermes_agent 的工具系统）只是**次级借鉴**，仅在"不与 slime-agentic 冲突、且能让代码更清晰"时借其工程手法（如 registry/dispatch 接缝、统一返回格式），**绝不能借来它与 slime-agentic 相悖的部分**（如 native tool-calling `tools=` schema——那会破坏 RL 的 token/log_prob 保真度）。
>
> **偏离必须写明**：迭代任一段代码时，只要**没有对齐 slime-agentic 对应位置的写法**（无论是简化、次级借鉴、还是暂缓实现），都必须在**代码注释 + 当版 `docs/decisions/*.md`** 里写清楚三件事：①源项目对应位置怎么做的；②nano 这里为何偏离；③语义是否等价 / 何时补齐。没有说明的偏离视为违反铁律。

## 2. 路径与仓库

- **源项目**：`/Users/qshf/my-project/slime-agentic`（github: LMIS-ORG/slime-agentic，分支 main。只读，用于对照）
- **nano 项目**：`/Users/qshf/my-project/slime-agentic-zero-repro`（git 已 init，主分支 main）
- **当前活跃分支**：`v4`
- **分支准则（每版必须遵守）**：**每个版本切一个 `vN` 分支，从上一版分支的末端切出；当版的全部提交——实施计划 doc + 代码实现 + 验证结果——都落在 `vN` 上，绝不提交到别的版本分支**。判断法：提交前先 `git branch --show-current`，确认在当版分支。反例（已修）：V4 的计划/实现/调试提交错落在 `v3` 分支上——已把 `v3` 回退到其最后一个 V3 提交、V4 全部收进 `v4` 分支。
- **工作流**：**本地只开发**（写码+推 git）→ **SSH 5090 服务器**（RTX 5090，路径 `/home/ubuntu/zj`）拉取/跑通/验证。V0（纯 fake）本地可验；V1 起接 Qwen3-0.6B SGLang，都在服务器验证。
- **git 远端**：`git@github.com:qshf/-slime-agentic-zero-repro.git`

## 3. 进度状态

三条主线，详见 [docs/system-roadmap.md](docs/system-roadmap.md)。

**主线一 · 最小训练流水线**（做扎实）

| 版本 | 标题 | 跑在哪 | 状态 |
|------|------|--------|------|
| V0 | 硬编码单样本 + fake trainer | 本地 | ✅ 完成（5/5 验证通过）|
| V1 | 极简 calculator agent + Qwen3-0.6B | 服务器 | ✅ 完成（5090 上 4/4 过，reward=1.0）|
| V2 | custom generate/reward hook 化 | 服务器 | ✅ 代码完成+离线测过，待服务器端到端 |
| V3 | mini_slime 最小闭环 | 服务器 | ✅ 完成（5090 端到端 4/4，reward=1.0）|
| V4 | Ray 化 | 服务器 | ✅ 完成（5090 端到端 3/3，actor 分进程 + reward=1.0）|
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
- **服务器 SSH 5090（4×RTX，各 32G，V1 时四卡各仅剩 ~2.8G free）**：V1 起全部在此验证。
  - **模型**：Qwen3-0.6B 已用 modelscope 下到 `/home/ubuntu/models/Qwen/Qwen3-0.6B`（HF/github 均不可达）。
  - **SGLang**：`lmsysorg/sglang:latest` 镜像已在服务器；起容器挂本地模型路径，走 OpenAI 接口端口 30000。
    - 显存紧张：必须 `--mem-fraction-static 0.62 --context-length 4096 --disable-cuda-graph`，否则 OOM。
  - **venv**：`uv venv` 建在项目内 `.venv`（server 无 openai，装 openai 即可）。python 用 `python3`。
  - V4 起需 ray。V6 起需 SGLang engine、FSDP/Megatron（env 待 V6 补充）。

## 5. 验活 cheatsheet

```bash
# 本地：当前分支
git -C /Users/qshf/my-project/slime-agentic-zero-repro branch --show-current

# 本地→服务器同步代码（服务器连不上 github，用 rsync 不用 git pull）
rsync -az --exclude '.venv' --exclude '.git' \
  /Users/qshf/my-project/slime-agentic-zero-repro/ \
  5090:/home/ubuntu/zj/

# 服务器 5090：起 SGLang 容器（4 卡都被占，必须调 mem-fraction-static 避免 OOM）
# docker run -d --name sglang-qwen3 --gpus device=0 --network host \
#   -v /home/ubuntu/models/Qwen/Qwen3-0.6B:/models/Qwen3-0.6B:ro \
#   lmsysorg/sglang:latest python -m sglang.launch_server \
#   --model-path /models/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
#   --port 30000 --host 0.0.0.0 --trust-remote-code \
#   --mem-fraction-static 0.62 --context-length 4096 --disable-cuda-graph
# 就绪判断：/health 会持续 503（warmup），改用真实 chat completion 探活

# 服务器 5090：跑版本验证（uv 建的 .venv）
# ssh 5090 'cd /home/ubuntu/zj && uv run python scripts/test_v1_rollout.py'
```

## 6. 决策日志（按版本累加）

### 路线图阶段（2026-07-20）
- **选了痛点驱动版本链，不选按周学主题**：原 README 是"按周读源码记笔记"，改成"每版解决上一版暴露的具体痛点 + 可运行验证"，契合"复现中学"目标。
- **拆成三条主线（系统骨架 / 三个 agent / 分布式后端）**：系统轴和方法轴正交，拧成一条 V0-V9 会让"教系统还是教 agent"变模糊。
- **V0 用写死单样本，不先做 calculator agent**：先通链路（样本如何被 trainer 消费）再增真实度，避免先造上游不接下游。
- **三个 agent 全复现，难度序 MemAgent<AgentFlow<ToolOrchestra**：实测 LOC/引擎数/工具/reward 复杂度定序（explore 报告）。ToolOrchestra **只做 QA 路径**，func_call+tau2 环境模拟器忠实复现基本不可能，记录设计不实现。
- **V1 起接 Qwen3-0.6B 真模型**：用户要求；因此 V1 起都需 GPU，都在服务器验证。
- **砍 critic/PPO**：见 system-roadmap 重启条件。

### V0（2026-07-20）详见 docs/decisions/v0.md
- Sample 砍到 6 字段（不照搬源项目 30+）；用词级假 token 让 loss_mask 肉眼可读；fake_train_step 只统计不算 loss。
- loss_mask 规则：agent=1 / prompt=0 / tool 返回=0。
- 踩坑：直接跑脚本 ModuleNotFoundError → train_loop.py 顶部插 sys.path。

### V1（2026-07-20）详见 docs/decisions/v1.md
- 推理引擎 SGLang（核对源项目 requirements 无 vllm）；Docker 部署走 OpenAI 接口端口 30000。
- loss_mask 仍程序化按段打；reward 先规则匹配（LLM-judge 留 A2）。
- 本地只过离线测；真实 rollout 待 5090：server_setup.sh → start_sglang.sh → test_v1_rollout.py。

### V3（2026-07-21）详见 docs/decisions/v3.md
- 建 `mini_slime/{args,rollout_manager,trainer,weight_sync,train}.py`：把「产数据/训/同步权重」三角色拆开，`train.py` 串成 rollout→train→update_weights 同步闭环。fake trainer/weight_sync（不算真梯度、不广播真张量），真 backend 留主线三。
- **对齐源做的三处修正（都朝源更靠拢）**：①命名改回与源一致——hook 存 `self.generate_rollout`、编排方法叫 `generate`（核对 rollout.py:455/:539，v3.md 草稿曾命名反了）；②`_convert_samples_to_train_data` 做成 RolloutManager 方法（源:737 就是方法）；③Args 收归框架层 `mini_slime/args.py`，calculator_hooks re-export（V2 测试零改）。
- **登记偏离**：generate hook 是 per-sample，源 rollout_function 是 batch 级——nano 把「遍历 batch」放 RolloutManager，语义等价。无 Ray/无 DP/fake train 均沿用 v3.md 偏离表。
- 验证：`scripts/test_v3_loop.py --offline` 本地 3/3 过（monkeypatch `_chat` 不连 SGLang）；V0/V2 回归全绿。5090 真 SGLang 端到端 **4/4 过**，两轮 reward=1.0、weight_version 递增到 2。实测 gen=43/27s 而 train/sync≈0s——全串行痛点的实证，引出 V4。

### V4（2026-07-21）详见 docs/decisions/v4.md
- 建 `mini_slime/ray/{placement_group,actor_group}.py` + `train_ray.py`：把 RolloutManager 包成单 `@ray.remote` actor、Trainer 收进 `RayTrainGroup`+`TrainRayActor`（DP=1），主循环 `.remote()`/`ray.get()` 拆进程。镜像源 `train.py:9-95` 装配三步 + 主循环。
- **对齐源做的落定**：①含训练前初始 `update_weights`（源 train.py:26），故 `weight_version==num_rollout+1`（V3 是 num_rollout）；②RolloutManager 加 `pid()` introspection（测试确认分进程，不改数据流）；③RolloutManager 用 `ray.remote(cls).remote()` 构造时包装——同一个类 V3 单进程 / V4 actor 两用，不重写。
- **登记偏离**：无 GPU placement group（fake trainer 不占 GPU、SGLang 在独立 docker）、`ray.init` 进程内起（源用 `ray start` 集群）、world_size=1（DP=1）。均见 v4.md 偏离表。
- **离线测试改用"换 hook 路径"**：rollout 在 actor 独立进程、主进程 monkeypatch 不到，故 `--offline` 把 `custom_generate_function_path` 指到新增的 `stub_hooks`（no-SGLang）——恰好实证 V2 hook 可插拔。
- 验证：`scripts/test_v4_ray.py --offline` 本地 3/3 过（三者不同 PID）；V0/V2/V3 回归全绿。5090 端到端 **3/3 过**，main/rollout/trainer 三独立进程，reward=1.0、weight_v→3。实测 gen=37/28s 而 train/sync≈0.005s——**分了进程仍串行**（ray.get 强制等待），且 train_time 从 V3 的 0s 变 0.005s（ray 跨进程 IPC 新增成本），引出 V5 overlap。

## 7. 待办 / 已知问题

- [x] 与用户对齐 system-roadmap 三主线结构。
- [x] 写主线一 V0-V5 的 iteration-plan。
- [x] Qwen3-0.6B 部署：modelscope 下到 `/home/ubuntu/models/Qwen/Qwen3-0.6B`，SGLang Docker 挂载本地路径起服务（HF/github 服务器都不可达）。
- [x] 本地→服务器同步：rsync（github 在服务器上 443 超时，改 rsync 推代码）。
- [x] 远端仓库已配：`git@github.com:qshf/-slime-agentic-zero-repro.git`（SSH）。
- [ ] **已知坑（V1）**：4 张卡都被别的进程占满（各剩 ~2.8G free），SGLang `--mem-fraction-static=0.62 --context-length 4096 --disable-cuda-graph` 才起得来。换空闲卡时可调回默认。
- [ ] **已知坑（V1）**：Qwen3-0.6B 默认开思考，每轮先吐 `<think>`（已正确打成 loss_mask=1）；小模型格式不稳会吐 `<answer>5</</answer>` 脏尾，解析截到第一个 `<` 之前。
