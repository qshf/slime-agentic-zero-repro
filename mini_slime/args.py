"""V3: mini_slime 框架的统一配置对象（single source of truth）。

对齐源项目 —— slime 的 args 是一个巨大的 argparse Namespace，从命令行构造后**一路透传**
给 RolloutManager / actor / 各 hook（generate/reward）。框架和 agent 都只**读** args 字段，
不各自 new 一份配置。

V2 时 Args 临时定义在 toy_rl/agent/calculator_hooks.py（agent 侧），这在角色上是反的：
配置的归属应是**框架**，agent 只是被透传进来读它。V3 把 Args 收归 mini_slime（框架层），
calculator_hooks 改为 re-export，`from toy_rl.agent.calculator_hooks import Args` 仍可用
（V2 测试零改）。这样只有一个 Args，避免"扩 loop 字段要改两处"。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Args:
    """极简配置对象，对齐源项目 args 的角色（庞大 Namespace 里只保留 nano 用到的字段）。"""

    # --- rollout / SGLang 推理配置（generate hook 读）---
    #     默认指向 5090 上的 Qwen3.5-4B（GPU1、端口 30001）；0.6B 仍在 30000 可切回对比。
    sglang_base_url: str = "http://localhost:30001/v1"
    model_name: str = "Qwen/Qwen3.5-4B"
    max_turns: int = 5
    # Qwen3-0.6B 是混合推理模型，每轮先吐一段 <think>。128 装不下"think + 闭合 + 动作"，
    # 会在 think 中途撞 max_tokens 被截断 → 该轮无 tool/answer，白耗一轮。提到 512 让整段
    # 思考+动作在一轮内跑完（512*max_turns=2560 < context-length 4096，不溢出）。
    max_tokens: int = 512
    temperature: float = 0.0

    # --- hook 路径（对齐 --custom-generate-function-path / --custom-rm-path / data_source）---
    #     RolloutManager 用 load_function 按这三个路径动态加载 data_source / generate / reward。
    #     默认是 calculator 路径（主线一）；主线二换 agent 只改这三条路径 + 下面的 agent 专属字段。
    custom_generate_function_path: str = "toy_rl.agent.calculator_hooks.generate"
    custom_rm_path: str = "toy_rl.agent.calculator_hooks.reward_func"
    data_source_path: str = "toy_rl.agent.calculator_data.load_data_source"  # 对齐源 args.data_source_path

    # --- 主循环编排配置（V3 新增，对齐 train.py 的 args.num_rollout / rollout_batch_size）---
    num_rollout: int = 2              # 主循环轮数，对齐 args.num_rollout（train.py:65）
    batch_size: int = 2              # 每轮取几条 prompt，对齐 args.rollout_batch_size
    update_weights_interval: int = 1  # 每几轮同步一次权重（对齐 train_async.py:62）；V3/V4 恒为 1

    # --- V5 异步 overlap 的"模拟耗时"旋钮（默认 0.0 → V3/V4 完全不受影响）---
    #     源项目 train 是真 FSDP/Megatron 一步（秒级）、gen 是真 SGLang 推理，故 overlap 能省真
    #     wall-clock。nano 是 fake：train≈0、离线 gen≈0，overlap 无可省、断言无法成立。这两个旋钮
    #     用 sleep **代表**我们没真跑的那段计算的 wall-clock，让 V5 的 overlap 可观测可断言。
    #     偏离登记见 docs/decisions/v5.md。
    fake_train_seconds: float = 0.0   # Trainer.train 模拟训练一步耗时；V5 打开以让 overlap 可观测

    # --- V9+ 权重同步优化 ---
    use_tensor_weight_sync: bool = False  # True=HTTP POST tensor（免落盘，2-5秒）；False=disk reload（默认，25秒）
    fake_gen_seconds: float = 0.0     # 离线 stub gen 模拟推理耗时；服务器真 SGLang 时恒为 0

    # --- A1 MemAgent 专属字段（对齐源 agentic/memagent/rollout.py 的 MEM_* 环境变量）---
    #     只在 data/generate/reward 路径指向 memagent 时生效；calculator 路径不读这些。
    mem_chunk_chars: int = 400   # 按**字符**切 chunk（nano 无真 tokenizer；源是 MEM_CHUNK_TOKENS 按 token）
    mem_max_chunks: int = 8      # 最多切几个 chunk（对齐源 MAX_CHUNKS）
    mem_max_memory: int = 512    # 记忆更新轮 max_tokens（对齐源 MAX_MEMORY_TOKENS）
    mem_max_final: int = 256     # 最终回答轮 max_tokens（对齐源 MAX_FINAL_TOKENS）

    # --- A2 AgentFlow 专属字段（对齐源 agentic/agentflow/rollout.py 的 engine_map 双引擎）---
    #     只在 data/generate/reward 路径指向 agentflow 时生效；calculator/memagent 路径不读这些。
    #     A2 精华 = "executor token 不训练"：planner 走**训练引擎**（权重每轮更新、要优化的策略），
    #     executor/verifier/final_output/rewarder 走**固定引擎**（不变权重的"环境"，对 loss 零贡献）。
    #     偏离登记见 docs/decisions/a2.md：默认两字段同指 30001+4B（0.6B 跑 judge/final_output 太弱），
    #     角色仍分两引擎（两 chat_fn）；服务器把 af_fixed_base_url 指 30000+0.6B 即成两真实引擎。
    af_planner_base_url: str = "http://localhost:30001/v1"   # 训练引擎（planner 唯一策略）
    af_planner_model: str = "Qwen/Qwen3.5-4B"
    af_fixed_base_url: str = "http://localhost:30001/v1"     # 固定引擎（executor/verifier/final_output/rewarder）
    af_fixed_model: str = "Qwen/Qwen3.5-4B"
    af_max_steps: int = 3        # ReAct 最多步数（对齐源 Solver.max_steps，nano 砍到 3 够说清概念）
    af_max_tokens: int = 512     # 每次 LLM 调用 max_tokens

    # --- A3 ToolOrchestra QA 专属字段（对齐源 ToolOrchestra QA solver）---
    # orchestrator 是唯一产训练轨迹的角色；expert/search 是工具环境，结果作为 role=tool observation
    # 写入下一轮 prompt。默认复用 4B 端点；逻辑专家角色/价格/偏好在每条数据 metadata 中定义。
    orchestra_orchestrator_base_url: str = "http://localhost:30001/v1"
    orchestra_orchestrator_model: str = "Qwen/Qwen3.5-4B"
    orchestra_expert_base_url: str = "http://localhost:30001/v1"
    orchestra_expert_model: str = "Qwen/Qwen3.5-4B"
    orchestra_max_steps: int = 4
    orchestra_max_tokens: int = 512

    # --- V6 GSM8K 真训练专属字段（主线三：真 log_probs + 真 GRPO + 真 torch 训练一步）---
    #     GSM8K 小学数学应用题（答案以 "#### 数字" 结尾，可严格验证）；4B 裸答多步算术会错，
    #     需 calculator 工具才能稳定答对——满足"调工具才答对"+"base 有提升空间"。
    gsm8k_num_train: int = 8      # 离线/训练取前 N 条 train（避免每次拉全量 7473）
    gsm8k_num_eval: int = 20      # eval 取前 N 条 test，量 before/after 答对率
    # 服务器连不上 HF（CLAUDE.md 已记 HF/github 不可达）：优先从本地 parquet 目录加载 GSM8K；
    # 该目录不存在时回退 datasets.load_dataset("openai/gsm8k")（本地开发有网时走这条）。
    gsm8k_local_dir: str = "/home/ubuntu/data/gsm8k"
    # GRPO 组归一：同题 n_samples_per_prompt 个 rollout → 组内 min-max + 标准化（对齐源 custom_convert）。
    n_samples_per_prompt: int = 4
    global_batch_size: int = 1    # custom_convert 裁剪到该倍数（对齐源；nano 单卡取 1）
    # custom_convert hook 路径（对齐源 --custom-convert-samples-to-train-data-path）：
    #   设了才走 GRPO 组归一；留空（默认）时 RolloutManager 用内置 per-sample 转换（V0-A3 不受影响）。
    custom_convert_path: str = ""
    # 真训练超参（trainer.py 真 torch 一步读；对齐源 fsdp_utils/actor.py + ppo_utils.py）。
    # "fake"=只统计（V0-A3/离线）; "torch"=单卡真 forward/backward/optimizer（V6.3）;
    # "fsdp"=FSDP2 分片后端跑在 Ray actor 里、真 torch.distributed 进程组（V7.3）。
    train_backend: str = "fake"
    fsdp_world_size: int = 1      # V7.3 FSDP 后端的 DP world_size（Ray 起几个训练 actor / 用几张卡）
    # V9 Megatron 并行度（对齐 MegatronTrainer.__init__ 同名参数；未启用时保持默认 1 零回归）。
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    megatron_data_parallel_size: int = 1
    # V9 Megatron：统一走 THD sequence packing，避免 BSHD 按 batch 最长样本 padding。
    megatron_qkv_format: str = "thd"
    # V7.5/V7.6 opt-in sequence packing（默认 False → V7.0/V7.2 padding 路径不变、零回归）。
    # 三态（V7.6 从 bool 扩成字符串；False/"" 仍是默认关闭，V7.5 的 True 等价于 "mask"）：
    #   False/""  padding：逐样本一行 [B,L]（V7.0/V7.2 路径）。
    #   "fa2"     FA2 varlen：flat [1,T] + attention_mask=None，靠 reset 的 position_ids 反推
    #             cu_seqlens 在内核里隔离段（**对齐源** actor.py:801-812 + arguments.py:29）。
    #   "mask"    显式块对角 4D mask（V7.5）：无 FA2 环境的 fallback，且是唯一能做 fp32
    #             精确等价证明的路径（FA2 内核只收 fp16/bf16）——见 v7.6.md。
    train_packing: "bool | str" = False
    train_model_path: str = "/home/ubuntu/models/Qwen/Qwen3-0.6B"  # 训练侧可训模型（单卡先用 0.6B）
    # HF attention 后端（对齐源 fsdp_utils/arguments.py:29 `attn_implementation`）。
    # 源默认 "flash_attention_2"；nano 默认 "sdpa" 因 V7.0-V7.5 环境无 FA2，且 "mask" packing
    # 路径要求 eager/sdpa（4D mask 只在这两个后端被 honor）。train_packing="fa2" 时须显式设成
    # "flash_attention_2"（_packed_backward 有硬断言拦截，不会静默退化）。
    attn_implementation: str = "sdpa"
    train_lr: float = 1e-6
    eps_clip: float = 0.2         # PPO clip 下界 1-eps_clip
    eps_clip_high: float = 0.2    # PPO clip 上界 1+eps_clip_high（对齐源 compute_policy_loss）
    clip_grad: float = 1.0        # grad norm 裁剪（对齐源 clip_grad_norm_）
    rollout_temperature: float = 1.0  # 训练侧重算 log_prob 的温度（须与 rollout 采样温度一致）
    # SGLang 原生 /generate 端点（真 token_ids + log_probs，替换 A3 chat 端点）。
    sglang_generate_url: str = "http://localhost:30001/generate"
    weight_save_path: str = "/home/ubuntu/models/nano_v6_ckpt"  # 权重同步走 disk reload 的落盘路径

    # --- V7.4 infra 落地开关（默认 off → V0-V7.3 零回归；opt-in 才启用）---
    #     learner_contract：把 train_data 过一遍 LearnerSample 校验（右移一致 + 单一 rollout 版本）。
    #     learner_trace：把 Trainer 扁平 metric 升级成分相位 trace（batch/fwd-bwd/opt/publish 计时）。
    learner_contract_validate: bool = False
    learner_trace: bool = False
