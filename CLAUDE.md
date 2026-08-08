# slime-agentic-zero-repro — 项目上下文 primer

> 新会话起手必读。每完成一版更新一次。详细路线见 [docs/system-roadmap.md](docs/system-roadmap.md)。

## 1. TL;DR

- **项目**：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
- **源项目**：slime-agentic —— 基于 Ray + SGLang + Megatron/FSDP 的 Agentic RL 训练框架（58K LOC）。
- **当前阶段**：主线一（V0-V5）✅、主线二（A1/A2/A3）✅ 收官。**主线三启动**：V6 真训练闭环（GSM8K）✅ 全链路跑通——真 log_probs + 真 GRPO 组归一 + 真 torch 训练一步 + 真权重同步，5090 端到端验证机制全真实发生（acc 0.4→0.2 变化证明权重真被改，稳定提升属训练规模/超参问题留后续）。

> **铁律（每版必须遵守）**：**代码范式遵从原项目**。nano 的代码结构 / 接口签名 / 命名 / 数据流必须对齐 slime-agentic 源项目在对应位置的写法。**只允许在其基础上做得更清晰（更好），不允许比源项目更乱、更 hack、更偏离（更差）**。判断法：写任一段前先问"源项目对应位置怎么做的"，对齐它；要偏离只能朝"更清晰且语义等价"的方向，并在注释里写明为何偏离。反例（已修）：把 agent loop 抄成两份塞进 generate() 里——源项目 rollout.py 是薄适配器，loop 在 solver.py。
>
> **锚点优先级**：**slime-agentic 是唯一首要锚点**；其它 nano 项目（如 nano_hermes_agent 的工具系统）只是**次级借鉴**，仅在"不与 slime-agentic 冲突、且能让代码更清晰"时借其工程手法（如 registry/dispatch 接缝、统一返回格式），**绝不能借来它与 slime-agentic 相悖的部分**（如 native tool-calling `tools=` schema——那会破坏 RL 的 token/log_prob 保真度）。
>
> **偏离必须写明**：迭代任一段代码时，只要**没有对齐 slime-agentic 对应位置的写法**（无论是简化、次级借鉴、还是暂缓实现），都必须在**代码注释 + 当版 `docs/decisions/*.md`** 里写清楚三件事：①源项目对应位置怎么做的；②nano 这里为何偏离；③语义是否等价 / 何时补齐。没有说明的偏离视为违反铁律。

## 2. 路径与仓库

- **源项目**：`/Users/qshf/my-project/slime-agentic`（github: LMIS-ORG/slime-agentic，分支 main。只读，用于对照）
- **nano 项目**：`/Users/qshf/my-project/slime-agentic-zero-repro`（git 已 init，主分支 main）
- **当前活跃分支**：`v7`
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
| A1 | MemAgent | 单引擎/无工具/loss_mask 全 1 | ✅ 完成（5090 端到端 single reward=1.0、闭环 reward_mean=0.5；离线 4/4）|
| A2 | AgentFlow | executor token 不训练（工具边界精华）| ✅ 完成（5090 端到端 reward=1.0、闭环 reward_mean=1.0；真 4B planner + 真 DeepSeek coder）|
| A3 | ToolOrchestra（仅 QA 路径）| 多专家路由 + 多组件 reward | ✅ 完成（5090 端到端 turns=2、reward=0.880、真 4B search→answer(expert_fast)；离线 reward_mean=0.991）（**主线二终点**）|

**主线三 · 分布式后端**

| 版本 | 标题 | 关键学点 | 状态 |
|------|------|---------|------|
| V6 | 真训练闭环（GSM8K）| 真 log_probs + 真 GRPO 组归一 + 真 torch 训练一步 + 真权重同步 | ✅ 全链路跑通（5090 端到端：真 log_probs nonzero、真 backward、weight_v 递增、真同步 disk reload；acc 0.4→0.2 变化证明权重真被改，稳定提升属训练规模/超参问题留后续）|
| V7 | FSDP 真训一步 | 分片 / tie / 梯度累积 | ✅ V7.0+V7.2+V7.3 收官（5090：2卡真分片+tie忠实+累积 max_diff=0；V7.3 Ray+FSDP+权重同步 --world 1&2 端到端 PASSED——真 dist 组+DP-split+集体 save/rank0 POST，weight_v 递增、真 backward、未 hang）；V7.5 补 packing 退休偏离 #1 |
| V7.4 | 接零风险 infra + 版本戳 | learner ABI / 阶段 trace / staleness | ✅ 代码完成+离线全回归绿（learner_contract 校验视图 + learner_metrics 阶段 trace + 缺口1版本戳；全 opt-in/additive，纯 CPU 可验无需服务器；6/6 CPU 不变量测试 PASSED，gap=1 真 staleness）|
| V7.5 | sequence packing（退休偏离 #1）| flat + 块对角 mask 隔离 / loss-grad 等价 | 🚧 代码完成+离线全回归绿（opt-in `train_packing`，默认 off 零回归；显式块对角 4D mask 替 flash-attn varlen；HF 4D-mask early-exit 直通 + eager/sdpa 硬断言）；服务器等价验证待跑（world=1，bf16<0.08 / fp32<1e-3）|
| V8 | Megatron 并行 | TP/PP/CP/EP 概念 | 记录设计 |
| V9 | 吞吐实验 | 扫参数定位瓶颈 | 记录设计 |

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
- 验证：`scripts/test_a1_memagent.py --offline` **PASSED**——3 chunk→4 turn、loss_mask 全 1（trainable=167=各轮 response 之和）、`\boxed{Lyonar}`→reward=1.0、闭环 reward_mean=1.0。V0/V2/V3/V4/V5 回归全绿（data_source 改造对 calculator 透明）。**5090 端到端 PASSED**：single_sample reward=1.0（真答出 `\boxed{Lyonar}`）、闭环 reward_mean=0.5（4B 两条 QA 答对一条，真实模型行为）。
- **服务器踩坑（已修）**：Qwen3.5-4B 混合推理默认吐 `<think>`，把 token 预算耗尽、对着模板 meta-rambling → reward=0（tokens 11376、gen 274s）。修：`enable_thinking=False` + `_strip_think`（源用微调模型无此问题，nano 接现成模型显式关思考）。修后 tokens→2629、gen→32s、reward=1.0。
- **偏离**：char 级分 chunk（无真 tokenizer）、假 token 无真 log_probs、硬编码 mini QA、reward 归一化取最简、reward 不做 turn 均摊、data_source 返回 Sample 列表——均见 a1.md 偏离表。**暴露痛点**：loss_mask 全 1 教不了工具边界 → A2 AgentFlow。

### A2 AgentFlow（2026-07-23）详见 docs/decisions/a2.md
- 主线二第二个真实 agent。建 `toy_rl/agent/agentflow/{memory,planner,executor,verifier,rewarder,solver,rollout,data,stub_rollout,tools/{base,base_generator,python_coder}}.py`：镜像源 `agentic/agentflow/` 的 Planner→Executor→Verifier ReAct 循环（plan → for step(next_step→executor→verifier→STOP?) → final_output）。
- **A2 教学核心 = "executor token 不训练"**：`solver.py` 里**只有 planner 的 `plan()`/`generate_next_step()` 被 `_emit_turn`（loss_mask=1）**，executor/verifier/final_output 的生成完全不进 turns（进 `sample.response` 供日志/reward 但不进 `sample.tokens`）。
- **系统层落地 = 忠实源"三模型分工"**（engine_map，rollout.py:136-144）：①planner=训练引擎（policy，唯一训练目标）；②executor/verifier/final_output=固定 base 引擎（纯环境）；③**python_coder 内部 coder 模型=独立模型**（纯环境）。executor 出**自然语言**命令→python_coder 工具内部再调 coder 模型翻成 Python→subprocess 执行。写代码那次 LLM 调用不在 executor、不进 trajectory，故"规划"（训练）与"写代码"（不训练）解耦——不把两关注点耦合、不篡改训练目标（曾考虑砍掉 python_coder 内层 LLM 让 executor 直接产代码，**已否决**：违源可分离性）。nano coder 模型 = **外部 DeepSeek API**（密钥/base_url/model 走环境变量、不进 git；DeepSeek v4 是推理模型，只取 content 忽略 reasoning_content）。
- **reward 新增 LLM-as-judge 回退**：boxed 精确匹配（命中 1.0）→ miss 回退 `Rewarder`（固定引擎判 `VERDICT: True/False`）。
- **偏离**：base 角色共享 planner 端点（不单起 base 实例，三角色都不训练、语义无关）、coder=外部 DeepSeek API（不起第二个 SGLang，coder 是独立纯环境模型，外部 API 最贴源三分）、dict 注册表替代 importlib 目录扫描（executor 真分发、只简化工具发现方式）、命令解析只留正则、Solver 收 chat_fn 而非 engine_map、字符级假 token、硬编码多步计算 QA——均见 a2.md 偏离表。
- **分发修正（2026-07-23 收官后回补）**：初版 executor 只持单 coder_tool、`execute_command` 收了 `tool_name` 却从不使用（假分发）。已恢复**真分发**——`Executor` 收 `toolbox: dict[str,BaseTool]`、按 tool_name 查表选实例（未命中回退+warning，对齐源 `_resolve_tool_mapping`）；新增 `tools/{base,base_generator}.py`（base_generator=固定引擎直接答），与 python_coder 一起注册（对齐源 engine_map 两工具）。`generate_tool_command` **保留**（源保真 + "executor token 不训练"活教材；功能因两工具同 `execute(query=)` 签名而冗余，但注释改诚实，不暗示它承重）。新增 `test_executor_dispatch` 断言分发真按 tool_name 生效。详见 a2.md「分发修正」。
- 验证：`scripts/test_a2_agentflow.py --offline` **PASSED**——turns=2、每轮 loss_mask 全 1、`sum(loss_mask)==sum(response_length)`（executor/verifier/final/coder 对可训练 token 零贡献）、turns 全为 kind=="planner"、python_coder **真跑 subprocess**（stub coder `print(165)`→stdout 165 进 response）、`\boxed{165}`→reward=1.0、闭环 reward_mean=1.0、weight_v→2。V0/V2/V3/V4/V5/A1 回归全绿。**5090 端到端 PASSED**（服务器直连，非隧道）：真 4B planner 吐 Context/Sub-Goal/Tool Name → 真 DeepSeek coder 翻 NL→Python→subprocess→`\boxed{165}`、reward=1.0、trainable=3018=各 planner turn 之和、闭环 reward_mean=1.0、gen=292s。**踩坑修正**：早先误记"5090 DNS 解析不了 DeepSeek 需 SSH 隧道"——**实际服务器可直连**（curl 返 401=能连只缺 key、DNS 正常），直接服务器侧跑通。**暴露痛点**：单 reward + 固定单工具路由 → A3 ToolOrchestra（多专家路由 + 多组件 reward）。

### A3 ToolOrchestra（QA 路径，2026-07-23）详见 docs/decisions/a3.md
- 建 `toy_rl/agent/toolorchestra/{data,prompt,solver,rollout,stub_rollout}.py`：采用源 QA 分支的 `messages -> assistant tool call -> role=tool observation -> messages` 循环；只有 orchestrator 输出进入 turns，search/expert 结果只进后续 prompt。
- 两个逻辑专家按 sample metadata 的模型映射、价格与偏好向量路由；reward 同时计算正确性、专家成本、延迟、角色偏好。因 mini hook 是 per-sample，未复刻源 custom_convert 的同题多 rollout min-max/GRPO，改为单样本确定性 utility，偏离记在 `docs/decisions/a3.md`。
- 验证：`scripts/test_a3_toolorchestra.py --offline` **PASSED**——search→answer 的 tool message 进入第二轮 prompt、loss_mask 仅覆盖 orchestrator、expert 失败后 error message 触发改选专家、多组件 reward 与 `train_ray` 闭环通过（reward_mean=0.991）。
- **5090 端到端 PASSED**（真 4B orchestrator @30001）：`turns=2 events=2 reward=0.880`。真实轨迹——orchestrator 第一轮自主选 `search`（拿证据）→ 第二轮 `answer` 路由 `expert_fast` → 专家算出 `\boxed{165}`。契约全绿（turns 全 orchestrator、`len(tokens)==len(loss_mask)`、`sum(loss_mask)==sum(response_length)`、reward∈[0,1]）。多组件 reward 生效：correctness=1.0、total_cost=0.000656、total_latency≈10.4s、tool_counts={expert_fast:1} → 加权 reward=0.880（**非满分正是成本/延迟组件起作用**，这是 A3 相对 A2 单 reward 的核心增量）。**主线二收官。**
- **收官前小修**（本次评估回补）：补齐 a3.md「偏离登记（按铁律三要素）」表（8 条，最大偏离=单样本预算归一 vs 源组内 GRPO min-max，留主线三）；删 solver.py 未使用的 `import asyncio`。

### V6 真训练闭环（GSM8K，2026-07-24）详见 docs/decisions/v6.md
- **主线三第一版**：把 A3 后 nano 的四根假支柱（假 token / 无真 log_probs / fake trainer / fake weight_sync / 单样本 reward）换真。内部四刀：V6.0 GSM8K 数据源 + calculator 确定性工具（复用 V1 CalculatorTool）；V6.1 orchestrator 走 SGLang 原生 `/generate` 拿真 token+log_probs（+ 真 tokenizer apply_chat_template，补 A3「手工渲染」偏离）；V6.2 移植源 `custom_convert` 的 GRPO 组归一（同题多 rollout min-max + `(r-mean)/(std+eps)` clip[-3,3] + std<0.1 过滤，opt-in via `custom_convert_path`，补 A3 最大偏离）；V6.3 torch 真训练一步（loss 对齐源 `ppo_utils.compute_policy_loss`）+ 真权重同步（disk reload）。
- 数据集 GSM8K（答案 `#### 数字` 可严格验证、小模型裸答会错、天然需 calculator）；服务器连不上 HF，parquet scp 到 `/home/ubuntu/data/gsm8k`，`gsm8k_data` 优先本地 parquet。
- 关键一致性：rollout 采样模型 == 训练重算 log_prob 的模型（否则 importance ratio 无意义）→ orchestrator/expert/train_model_path 统一 0.6B@30000。
- **5090 端到端全链路 PASSED**：`real_logprobs OK (tokens=241, nonzero_lp=16)`（真 log_probs 工作）；真 backward（某轮 `reward_mean=-0.346` 组内有对有错→真算梯度，`train=1.7-11s`）；weight_version 递增 2→3→4（真权重同步，`sync≈18-20s`=save+SGLang disk reload 真耗时）。`acc_before=0.400 acc_after=0.200`——acc **变化**证明权重真被改（机制全链路真实发生）；未稳定提升是训练量微不足道（4 题×3 轮×lr 1e-6）+ 多数轮组内无方差被 mask，属**训练规模/超参问题非机制问题**，留后续（调温度制造方差/换适中难度/加大 n 和轮数）。
- ray worker 需带 venv site-packages 才能 import torch（V4/V5 只用标准库没暴露；见 placement_group 修正）。torch/transformers 收进 pyproject optional `[train]`（仅服务器装）。
- **偏离**（v6.md 8 条三要素表）：单卡纯 torch 无 FSDP 多维并行、关 KL/entropy 取最纯 GRPO、权重同步 disk reload vs 源 tensor 广播、逐样本无 packing、GSM8K+calculator 替代 STEM/HLE+FAISS、只两工具、本地 parquet、acc 未稳定提升。

### V7 FSDP 真训一步（2026-07-29 / V7.3 收官 2026-08-06）详见 docs/decisions/v7.md
- **主线三第二版**：把 V6 的「单卡纯 torch 整模型」换成 **FSDP2 分片后端**（对齐源 `fsdp_utils/actor.py`）。建 `toy_rl/utils/fsdp_utils.py`（mesh/wrap/init-context/broadcast）+ `toy_rl/trainer/fsdp_trainer.py`（FSDPTrainer）。内部切刀：V7.0 基础设施 → V7.2 梯度累积 →（V7.1 packing 暂缓）→ V7.3 Ray+FSDP+权重同步收官。
- **V7.0**：教学核心=**同一份 loss 数学换到分片后端**（loss 公式与 V6 torch_actor 完全相同，只换执行后端）。**忠实 tie 处理**（本版最关键对齐）：Qwen3-0.6B `tie_word_embeddings=True`（实测确认）→ ①embedding 不单独 `fully_shard`（tie 时单独 wrap 破坏权重共享，对齐 actor.py:956）；②建模型全 rank load 全量到 CPU（tie 走 meta device 会 broadcast hang，对齐 actor.py:217-226）。**5090 2卡 PASSED**：q_proj local shard=1048576=全量的一半（真分片）、loss=-2.0（数学精确 -1×dp2/gbs1）、grad_norm=308、权重 L2 变化 0.14、tie 未 hang。
- **V7.2**：教学核心=**累积多微批梯度 == 一次大批梯度**。`train_batch`：zero_grad 一次→逐微批 `_micro_backward` 累积→末尾单次 clip+step（对齐源 `_train_core` + step 门控）。每微批 loss 乘 `dp_size/global_batch_size`：与 FSDP 反向的 dp 平均相抵→累积 N 微批==全局批「逐样本 mean 之和/gbs」单次梯度。`train_step` 保留为 gbs=1 薄封装。**5090 2卡 PASSED**：**累积等价性 max_diff=0.000e+00**（两独立累积路径梯度逐参数精确相等）；`loss=0.0` 是预期正确（两样本 adv=+1/-1 缩放后 loss 相消，但梯度不相消 grad_norm=166→演示「loss 抵消≠梯度抵消」）。
- **V7.3 Ray+FSDP+权重同步（收官刀）**：把 `FSDPTrainer` 从 torchrun 单脚本**搬进 Ray actor**，形成真 torch.distributed 进程组，闭合 rollout→train→update_weights，权重 disk reload 同步回 SGLang。对齐源 `train_actor.py:29-71`（actor `__init__` 设 MASTER_*/RANK/LOCAL_RANK env、init_process_group 留 init() 集体 rendezvous）+ `actor_group.py:94-106`（rank0 暴露自由端口→广播给 rank>0）。**两个老 v7-plan.md 漏掉的正确性点**：①**LOCAL_RANK 恒 0 不是 rank**（Ray 独占 CUDA_VISIBLE_DEVICES，用 rank 会 invalid device）；②**权重同步=集体 save + 仅 rank0 落盘/POST**（save_pretrained 是集体 gather，每 rank 都进但落盘/POST/version 只 rank0，避免重复 reload/version 分叉）。DP-split=`samples[rank::world_size]`（对齐源 `_split_train_data_by_dp`）。**复用 train_ray.train 不建 train_fsdp.py**（同步主循环 backend 无关，纠老计划 trap）。**5090 端到端 PASSED**：`--world 1`（FSDP@GPU2）+ `--world 2`（FSDP@GPU2,3 双卡收官）都 3 轮闭环、weight_v 2→3→4、rollout 2 真 backward（loss≈-0.33、train≈13s）、rollout 0/1 组内无方差被 mask（loss=0，同 V6 GRPO 正确语义）、sync≈29s（集体 save+disk reload）、未 hang。离线全回归绿（V0-A3+V6）。
- **V7.1 packing 暂缓（登记偏离）**：源 `pack_sequences` flat 1D + position_ids reset + `attention_mask=None`，**靠 flash-attn varlen 隔离样本**；5090 无 flash-attn 且 Blackwell sm_120 难编译，缺它会让样本间注意力**静默串扰、loss 算错**（比不做更危险）。忠实替代=用 cu_seqlens 物化**块对角 mask** 显式隔离，留能装 flash-attn 时补。不假装做了。
- **踩坑（已修）**：①`accelerate` 缺失（`init_empty_weights` 依赖）→补进 `[train]` extra；②`_no_split_modules` 在 transformers 5.x 是 `set` 不可 `[0]` 索引→改成员测试。
- **偏离**（v7.md 6 条三要素表）：无 packing（#1，最大，缺 flash-attn）、无 KL/entropy、无 TP/PP/CP 多维并行（留 V8）、权重同步 disk reload vs 源 NCCL broadcast（留 V7.4/泛化 infra weight_publishing）、单节点 master=127.0.0.1、单样本/小微批+无 placement group。
- **后续路线**（详见 `docs/decisions/v7-onward-plan.md`）：V7.4 接 infra learner_contract+trace（as spec，在 repo 忠实重写）+ 补版本戳；V7.5 退休 packing 偏离；V8 Megatron 多维并行。

### V7.4 接零风险 infra + 版本戳（2026-08-08）详见 docs/decisions/v7.4.md
- **主线三·infra 落地第一刀**：把 infra（`agentic-rl-infra-lab`）明言「零风险、无新 GPU 行为」的两契约按「infra as spec」在 repo 内**忠实重写**，用其 CPU 不变量验收。三件事：①**learner_contract**（`mini_slime/learner_contract.py`，LearnerSample token→causal-target(T-1) 坐标系 + 单一 rollout 版本校验）；②**learner_metrics**（`mini_slime/learner_metrics.py`，Trainer 扁平 metric → 分相位 trace）；③**缺口 1 版本戳**（每条 Sample 戳 rollout 时权重版本 → off-policy staleness 有真实来源）。
- **关键设计（Plan agent 校验后定）**：①**learner_contract 是校验视图不是数据变换**——训练器照旧内部 `[1:]` 右移消费 loss_mask，`validate_train_data` 只 opt-in 把同批数据过一遍 LearnerSample 校验（右移一致+单一版本），失败即抛，**不改训练器消费字段** → 零回归（若改成变换要重写两条 hot path，高风险，否决）；②**版本源=`WeightUpdater.version` int**（repo 权威源，单调），主循环调 generate 前取 `weight_version()` 传入，**不依赖** SGLang meta 串（未验证、是 "default"），不动 solver/GenOutput；③**custom_convert 拆 turn → 每 turn-row 继承父版本**（版本列长度=turn-row 数，逐行对齐）。
- **全 opt-in/additive（零回归保证）**：`Sample.rollout_policy_version` 默认 None；新列只被 opt-in 校验消费；两新模块默认不引用；Args `learner_contract_validate`/`learner_trace` 默认 False；`generate` 新参带缺省 → V3/V5/测试零改。
- 验证：`test_v7.4_learner_abi.py` **纯 CPU PASSED（6/6）**——causal 右移逐值对齐 `_pad_batch`、单一版本拒混版本、`sum(target_mask)==sum(loss_mask)` 分母恒等、LearnerTrace 不变量+`policy_version_gap`、版本戳端到端（内置每行戳/custom_convert 拆 turn 每行继承）、未戳版本被拒、opt-in trace 闭环 `gap=1`（trainer_v2−rollout_v1 真 staleness）。V0/V2/V3/V4/V5/V6/A1/A2/A3 全回归绿。**本版无新 GPU 行为、无需服务器**（infra「零风险」正是此意）。
- **偏离**（v7.4.md 5 条三要素表）：advantages 逐 token 未物化（gap 2，repo GRPO 逐序列标量广播）、log_prob_seconds=0.0（repo 融合 log-prob 进 forward，无独立 pass）、optimizer_seconds=0.0（Trainer 层粗计未拆 fwd/bwd）、版本单 int vs 源 list[str]、learner_contract 为校验视图非变换。

### V7.5 sequence packing（退休偏离 #1，2026-08-08）详见 docs/decisions/v7.5.md
- **主线三·退休 V7 唯一未闭合的正确性偏离**：给 `FSDPTrainer` 加 **opt-in** packing 前向路径（`args.train_packing`，默认 off 零回归），用 infra spike 的验收范式（packed vs padded 的 **loss/grad 逐值等价**）证明隔离精确 → 诚实退休偏离 #1。建 `toy_rl/trainer/data_packing.py`（`pack_sequences` + `build_block_diagonal_causal_mask`）+ `FSDPTrainer._packed_backward`。
- **为什么必须偏离源（三方硬件都缺）**：源靠 **flash-attn varlen** 隔离（`attention_mask=None` + reset position_ids）；infra spike 靠 **TransformerEngine** cu_seqlens；5090 是 Blackwell sm_120 **两者皆无**。照搬 `attention_mask=None` → 整个 pack 走满因果注意力、**样本间静默串扰、loss 算错**。nano 改传**显式物化 `[1,1,T,T]` 块对角因果 mask**（infra 文档承认这是源 varlen 路径的 "conceptual equivalent"）。隔离语义等价、吞吐部分等价（消 padding 浪费，但付 O(T²) mask、不得 varlen 块级稀疏）。
- **两个必须对的正确性点（源级确认）**：①**transformers 5.x 直通 4D mask**——`create_causal_mask` → `_preprocess_mask_arguments` 对 `len(shape)==4` early-exit 原样返回（不重推因果），`eager_attention_forward` 当加性 bias（用 `finfo.min` 不用 `-inf`，避免全屏蔽行 NaN；dtype 匹配激活；position_ids 必传）；②**仅 eager/sdpa honor 4D mask**（flash/flex 无视 → 静默串扰）→ `_packed_backward` **硬断言**后端 ∈{eager,sdpa}。
- **loss 数学抽共用**：`_grpo_sample_loss` 从 `_microbatch_backward` 抽出，padding/packing 两路共用（bit-identical，是等价断言的根本）；两路唯一差别只在"怎么算 cur_lp"。**跨边界 target 不泄漏**：段内 `logits[start:end-1]` 预测 `tokens[start+1:end]`，与 padding `[1:]` 右移逐值一致（双重隔离：mask + shift）。
- **fp32 副证靠 `compute_dtype` 非 `model.float()`**：FSDP2 `MixedPrecisionPolicy(param_dtype=bf16)` 强制 bf16 前向无论存储 dtype；给 `apply_fsdp2` 加 `param_dtype`、`FSDPTrainer` 加 `compute_dtype`，fp32 副证传 `float32` 真关混合精度、mask dtype 随之 fp32 → 证明算法数学精确。
- 验证：离线全回归绿（opt-in 默认 off，V0/V2/V3/V4/V5/V6/A1/A2/A3/V7.4 各 `--offline` PASSED）。**服务器 5090（world=1）待跑**：`torchrun --nproc_per_node=1 scripts/test_v7.5_packing.py`（bf16 主门 `<0.08`）+ `--fp32`（副证 `<1e-3`）——断言 padding vs packing 的 loss 与逐参数 grad `max_abs_error < TOL` + 块对角 mask 结构。
- **偏离**（v7.md：#1 退休、#6 更新、**#7 新增**）：#7 = 隔离机制用显式 O(T²) 块对角 mask 非 varlen O(T) 内核（消 padding 浪费✔、不得块级稀疏✘；T≤~3k 无 OOM；装 flash-attn 后换 varlen 得全部吞吐）。

## 7. 待办 / 已知问题

- [x] 与用户对齐 system-roadmap 三主线结构。
- [x] 写主线一 V0-V5 的 iteration-plan。
- [x] Qwen3-0.6B 部署：modelscope 下到 `/home/ubuntu/models/Qwen/Qwen3-0.6B`，SGLang Docker 挂载本地路径起服务（HF/github 服务器都不可达）。
- [x] 本地→服务器同步：rsync（github 在服务器上 443 超时，改 rsync 推代码）。
- [x] 远端仓库已配：`git@github.com:qshf/-slime-agentic-zero-repro.git`（SSH）。
- [ ] **已知坑（V1）**：4 张卡都被别的进程占满（各剩 ~2.8G free），SGLang `--mem-fraction-static=0.62 --context-length 4096 --disable-cuda-graph` 才起得来。换空闲卡时可调回默认。
- [ ] **已知坑（V1）**：Qwen3-0.6B 默认开思考，每轮先吐 `<think>`（已正确打成 loss_mask=1）；小模型格式不稳会吐 `<answer>5</</answer>` 脏尾，解析截到第一个 `<` 之前。
