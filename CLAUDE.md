# slime-agentic-zero-repro — 项目上下文 primer

> 新会话起手必读。每完成一版更新一次。详细路线见 [docs/system-roadmap.md](docs/system-roadmap.md)。

## 1. TL;DR

- **项目**：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
- **源项目**：slime-agentic —— 基于 Ray + SGLang + Megatron/FSDP 的 Agentic RL 训练框架（58K LOC）。
- **当前阶段**：主线一已打通（V0-V5 全验证）。**主线二进行中**：A1 MemAgent 本地离线通过（loss_mask 全 1 + boxed reward + 闭环），待服务器端到端。

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
- **工作流**：**本地只开发**（写码+推 git）→ **SSH 5090 服务器**（RTX 5090，路径 `/home/ubuntu/slime-agentic-zero-repro`，git 管理）拉取/跑通/验证。V0（纯 fake）本地可验；V1 起接 Qwen3-0.6B SGLang，都在服务器验证。
- **服务器 git 恢复准则（每版必须遵守）**：服务器 checkout 是 **git 管理**的（当前目录 `/home/ubuntu/slime-agentic-zero-repro`，旧的 `zj` 已弃用）。**当服务器所在版本与要跑的版本不匹配时，用 git 从远程仓库恢复到目标版本**（`git fetch origin && git reset --hard origin/<vN>`），**绝不用 rsync 往 git 工作树上盖**——rsync 会把本地其它版本的文件混进 checkout（tracked 被改、v4 文件混进 v3），污染分支状态。反例（已修）：本次调试把本地 v4 工作树 rsync 盖到服务器 v3 checkout，事后 `git restore` + `git clean` 才复原成干净 v3。**已落地（2026-07-22）**：服务器已配 github SSH key（`~/.ssh/id_ed25519`，公钥已加到 github）、remote 已换 SSH，`git fetch origin && git reset --hard origin/<vN>` 实测可用（HTTPS 443 仍超时，故必须走 SSH）。
- **git 远端**：`git@github.com:qshf/slime-agentic-zero-repro.git`（SSH；注意仓库名无前导横杠。旧 CLAUDE.md 曾误写 `-slime...`）

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
| V5 | 同步 vs 异步 | 服务器 | ✅ 完成（5090 端到端 async 193.7s < sync 209.3s；本地离线 async<sync；**主线一终点**）|

**主线二 · 三个真实 Agent**（做扎实，按难度）

| 版本 | 标题 | 关键学点 | 状态 |
|------|------|---------|------|
| A1 | MemAgent | 单引擎/无工具/loss_mask 全 1 | ✅ 本地离线通过（loss_mask 全 1 + boxed reward + 闭环），待服务器 |
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

### V5（2026-07-22）详见 docs/decisions/v5.md
- 建 `mini_slime/train_async.py`（≡ 源 train_async.py）：预取 gen(0) → 循环里先 `generate.remote(N+1)` 提前发起下一轮、再 `ray.get(async_train(N))`，让 train N 与 gen N+1 **跨进程并行**；`update_weights` 按 interval、换权重前先 sync 在途 gen。这就是源注释的"改 ray.get 位置"。
- **核心洞察 + 偏离**：fake trainer `train≈0` → overlap 无可省、断言必挂。故加两个"模拟耗时"旋钮（`args.py`，默认 0.0，V3/V4 零影响）：`fake_train_seconds`（Trainer.train sleep，代表训练一步 wall-clock）、`fake_gen_seconds`（离线 stub gen sleep，代表 SGLang 推理；服务器真 SGLang 时=0）。
- **对齐源命名的落定**：同步基线**复用 V4 的 train_ray.py**（扮演源 train.py 角色），不按 roadmap 草稿名另建 `train_sync.py`（会重复）——朝"源命名 train.py↔train_async.py + 不重复"更清晰。
- 验证：`scripts/test_v5_async.py --offline` 本地 **PASSED**——`sync_total=2.018s`、`async_total=1.492s`、`saving=0.526s`（理论 (3-1)·0.2=0.4s）。异步 rollout 1/2 的 `wait_gen=0.000s` 是 overlap 的直接证据（gen(N+1) 早在上轮 train 期间跑完）。V0/V2/V3/V4 回归全绿。5090 端到端 **PASSED**：真 SGLang gen 50-70s，`sync_total=209.3s`、`async_total=193.7s`、`saving=15.6s`（>理论 10s，多出为 gen 方差），异步 rollout 1/2 同样 `wait_gen=0.000s`、weight_v→4。
- **登记待显现的痛点**：异步引入 off-policy staleness（train 用 gen(N) 旧权重、gen(N+1) 已在跑），`update_weights_interval` 即调新旧度的旋钮；nano fake reward 感知不到，留主线二真 agent 时显现。**主线一系统骨架到此打通** → 主线二 A1 MemAgent。

### A1 MemAgent（2026-07-22）详见 docs/decisions/a1.md
- 主线二第一个真实 agent。建 `toy_rl/agent/memagent/{rollout,data,stub_rollout}.py`：镜像源 `agentic/memagent/rollout.py` 的 chunk 记忆循环（`for chunk: memory=LLM(problem,memory,chunk)` → `answer=LLM(problem,memory)` with `\boxed{}`），**每轮独立训练序列**、memory 文本跨轮传递。
- **A1 教学核心 = loss_mask 全 1**：每轮 response 全训练、无工具返回段置 0（对齐源 `[1]*len(token_ids)`）。这是最简 loss_mask，和 A2 的"executor token=0"（工具边界）形成对照。reward = 抽 `\boxed{}` + is_equiv 归一化。
- **主线一改造（都朝更对齐源 + 零回归）**：data_source 做成 `data_source_path` hook（对齐源 data_source_cls），RolloutManager 改为动态加载数据源、产出带 `metadata["context"]` 的 Sample；calculator `_PROMPTS` 抽到 `calculator_data.py`（产出与旧题库一致，主线一回归零改动）。
- **范式对齐落定**：源 memagent 的循环**就在 generate 里**（无 solver），故 A1 把循环放 `_run_memory_loop`、real/stub 只注入不同 chat_fn——按对应源结构对齐，不套 calculator 的"loop 独立模块"（那是因 agentflow 有 solver）。
- 验证：`scripts/test_a1_memagent.py --offline` **PASSED**——3 chunk→4 turn、loss_mask 全 1（trainable=167=各轮 response 之和）、`\boxed{Lyonar}`→reward=1.0、闭环 reward_mean=1.0。V0/V2/V3/V4/V5 回归全绿（data_source 改造对 calculator 透明）。服务器端到端待跑。
- **偏离**：char 级分 chunk（无真 tokenizer）、假 token 无真 log_probs、硬编码 mini QA、reward 归一化取最简、reward 不做 turn 均摊、data_source 返回 Sample 列表——均见 a1.md 偏离表。**暴露痛点**：loss_mask 全 1 教不了工具边界 → A2 AgentFlow。

## 7. 待办 / 已知问题

- [x] 与用户对齐 system-roadmap 三主线结构。
- [x] 写主线一 V0-V5 的 iteration-plan。
- [x] Qwen3-0.6B 部署：modelscope 下到 `/home/ubuntu/models/Qwen/Qwen3-0.6B`，SGLang Docker 挂载本地路径起服务（HF/github 服务器都不可达）。
- [x] 本地→服务器同步：rsync（github 在服务器上 443 超时，改 rsync 推代码）。
- [x] 远端仓库已配：`git@github.com:qshf/-slime-agentic-zero-repro.git`（SSH）。
- [ ] **已知坑（V1）**：4 张卡都被别的进程占满（各剩 ~2.8G free），SGLang `--mem-fraction-static=0.62 --context-length 4096 --disable-cuda-graph` 才起得来。换空闲卡时可调回默认。
- [ ] **已知坑（V1）**：Qwen3-0.6B 默认开思考，每轮先吐 `<think>`（已正确打成 loss_mask=1）；小模型格式不稳会吐 `<answer>5</</answer>` 脏尾，解析截到第一个 `<` 之前。
