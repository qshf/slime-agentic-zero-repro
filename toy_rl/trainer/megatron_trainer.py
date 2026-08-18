"""V8.0/V8.2: MegatronTrainer —— Megatron 多维并行（TP × PP × CP）后端的真训练一步。

对齐源 slime/backends/megatron_utils/：
  - 并行组初始化 ← initialize.py:_initialize_distributed（mpu.initialize_model_parallel）
  - 建模型      ← model_provider.py:177-201（GPTModel + parallel_output=True + tie
                  + pre_process/post_process 按 PP stage 判定）
  - 调度        ← model.py:419-429（`get_forward_backward_func()` 跑 1F1B）
  - forward     ← model.py:380-416 `forward_step`（只前向，返回 logits + loss 偏函数）
  - loss        ← loss.py:943 `loss_function`（算 loss + 缩放以对接 Megatron 的梯度累积）
  - log_probs   ← loss.py:225 get_log_probs_and_entropy → ppo_utils.py:compute_log_probs
                  （**vocab-parallel**：logits 保持分片，跨 TP 组算 cross entropy）
  - loss 数学   ← ppo_utils.py:132-135 compute_policy_loss（与 V6/V7 完全同一份 GRPO 公式）
  - 梯度收尾    ← distributed/finalize_model_grads.py（qk-layernorm 跨 TP、word embedding 跨 PP）
  - CP 切分     ← cp_utils.py（见 toy_rl/trainer/cp_utils.py）

**三根轴切什么（V8/V8.2 的教学核心）——同一份 loss 数学，换到不同的并行轴上**：

  | 轴  | 切什么维度        | 通信在哪                          | 边界              |
  |-----|-------------------|-----------------------------------|-------------------|
  | DP  | 样本（V7 FSDP）   | 层边界：all-gather 参数           | 单层须放得下一张卡 |
  | TP  | 参数的行/列（层内）| 层内部：col→row 之间 all-reduce 激活 | 单层可跨卡       |
  | PP  | 层（模型纵向）    | stage 边界：p2p send/recv 激活    | 通信最小，但有 bubble |
  | CP  | 序列长度维        | attention 内部：ring 交换 KV      | 长序列不再受单卡显存限制 |

  `_grpo_sample_loss` 在四条路径上**逐字相同**——换的是并行轴，不是 loss 语义。

**vocab-parallel logits（V8 最关键的正确性点）**：
  源 model_provider.py:177 设 `parallel_output=True` —— 输出层也按 vocab 维切开，
  各 rank 只拿到 [S, V/tp] 的**分片 logits**，从不 gather 成全量 [S, V]（V=151936，
  gather 会白白吃显存+通信）。因此 log_prob **不能**用普通 log_softmax，必须用
  `fused_vocab_parallel_cross_entropy` 跨 TP 组做（内部 all-reduce max 与 sum-exp）。
  实测（5090 TP=2）：它与全量 log_softmax 的 max_abs_err = **0.000e+00**，梯度正常回流。

**跨 rank 的梯度规约有三处，缺一处就"前向对、梯度错"（V8.2 最硬的正确性点）**：
  1. **qk-layernorm 跨 TP**（V8 已做，源 finalize_model_grads.py:403-405）：
     q/k-layernorm 作用在**每个头**上而头被 TP 切开，各 rank 的梯度是部分和。
  2. **tie 的 word embedding 跨 PP first/last stage**（V8.2 新增，源 :164
     `_allreduce_word_embedding_grads`）：tie 让输入嵌入与输出投影是同一份权重，
     PP>1 时它们在**不同进程**里，各自梯度也只是部分和。
  3. **跨 DP/CP 的梯度规约**（V8.2 新增）：源靠 mcore 的 DDP wrapper 自动做；
     nano 用 torch AdamW 无 DDP（V8 偏离 #7），必须手写这一步，否则 CP>1 时
     每个 CP rank 只持有自己那几块 token 的贡献 —— 又是"前向对、梯度错"。
  三处的**指纹完全一样**：loss Δ 恰为 0，而梯度系统性偏、最差参数指向漏掉的那一类
  （分别是 `q_norm.weight` / `embed_tokens.weight` / 全体）。**只有 fp32 档抓得到**，
  bf16 会把它当舍入放过（V8 实测：漏 #1 时 bf16 的 5e-2 看不出来，fp32 从 1.8e-2 → 1.9e-5）。

偏离源项目（登记见 docs/decisions/v8.md / v8.2.md）：
  - **通信后端**：多卡走 NCCL（V8 的 gloo 偏离已退休）；**单卡多 rank 仍只能 gloo**
    （NCCL 硬拒 `Duplicate GPU detected: rank 0 and rank 1 both on CUDA device 80`），
    故 `backend` 默认保留 "gloo" 以免单卡回归被破坏，多卡验收显式传 "nccl"。
  - **只用 megatron.core，不走 megatron.training.init**：源 initialize.py:init 还要
    _build_tokenizer / 微批计算器 等训练框架件（且断言 numpy 1.x，容器是 2.3.5）；nano 只需
    并行组 + GPTModel，直接用 core 更清晰。
  - **不做 EP**：Qwen3-0.6B 是 **dense 模型，没有 expert**（V8 写的理由「gloo 无 all_to_all」
    已作废——不是做不到，是无对象）。
  - **不做 VPP / interleaved 1F1B**：它是减 bubble 的**调度优化**，不改 PP 语义，留 V9 当吞吐旋钮。
  - **`finalize_model_grads` 手写而非直接调 mcore 原版**：源 model.py:539 把
    mcore 的原版挂上去，但那份实现读的是 `param.main_grad`（DDP 的梯度缓冲）且要
    `model_chunk.ddp_config` —— nano 无 DDP wrapper（V8 偏离 #7 的连带后果），
    调不通。故按 mcore 同一份逻辑手写、改读 `param.grad`，语义等价。
  - 无 KL/entropy、无 ref model（沿用 V6/V7 的最纯 GRPO）；优化器用 torch AdamW 而非
    Megatron DistributedOptimizer（后者是 ZeRO-1 式优化器分片，与 TP/PP/CP 正交，nano 不铺）。
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# megatron 的 `--attention-backend X` 落到 `TransformerConfig.attention_backend`，
# 由 mcore 在建模型时翻译成 TE 的三个 NVTE_* env
# （mcore language_module.py:105-124 的 `_set_attention_backend` / `check_and_set_env_variable`）。
# 源**所有** megatron 训练脚本的 MISC_ARGS 都带 `--attention-backend flash`
# （run-qwen3-4B.sh:139 等），理由见源 docs/zh/developer_guide/debug.md:15：
# 「避免 CP 下 fused attention 的数值不稳定」。sm_120 上实测这不是"不稳定"而是**确定错**
# （fused 的 THD backward `dq` cosine 只有 0.356，上游 TE#3333），故 nano 把源的
# 配置约定**编码成代码里的断言**（见 `observed_attention_backend` + CP>1 的硬断言）。
#
# **不要自己去 set 那三个 env**（初版这么写，实测直接炸）：mcore 会**校验** env 与
# `config.attention_backend` 一致，不一致就 assert 退出；且本项目的容器镜像已经把
# `NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0` 烤进了镜像默认值，
# 与 mcore 的默认 `auto`（期望 1/1/1）冲突 —— 所以**必须显式设 config 字段**，
# 让 mcore 自己去写 env。这也正是源的做法（传 arg，不碰 env）。
_ATTN_BACKENDS = ("flash", "fused", "unfused", "local", "auto")


def _attn_backend_enum(name: str):
    """`"flash"` → `AttnBackend.flash`（对齐 megatron `--attention-backend` 的取值集合）。"""
    from megatron.core.transformer.enums import AttnBackend

    if name not in _ATTN_BACKENDS:
        raise ValueError(f"attention_backend={name!r} 须是 {_ATTN_BACKENDS} 之一")
    return getattr(AttnBackend, name)


def _clear_nvte_backend_env() -> None:
    """建模型前**清掉**继承来的 NVTE_* —— 让 mcore 按 `config.attention_backend` 自己写。

    这不是"自己设 env"（那正是上面注释里说不能干的事），而是相反：把**别处留下的陈旧值**
    清掉，好让 mcore 的 `check_and_set_env_variable`（它接受 `current_value is None`）
    唯一地由 config 决定结果。

    非清不可的实证：本项目的 `agentic-rl-infra-lab:fa2-mcore` 镜像把
    `NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0` 烤进了默认 env
    （给 CP 用的），于是任何**非 flash** 的请求都会撞上
    `AssertionError: NVTE_FLASH_ATTN set to 1, but expected 0 for attention backend type unfused`
    —— 而 fp32 档恰恰只能走 unfused（FA2 内核拒收 fp32）。清掉之后两条路径都能选。
    """
    for k in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(k, None)


def observed_attention_backend() -> str:
    """读回 **TE 实际选中的**后端（不是我们请求的那个）。跑过一次真前向后才有值。

    env 表达的是**请求**：TE 会按 dtype/head_dim/mask/CP/硬件能力否掉它
    （unfused 遇 CP 直接 `No dot product attention backend is available`；
     FA4 b15 在 sm_120 上 `Operation creation failed`）。只打印 `NVTE_FLASH_ATTN=1`
    而不验证结果，就可能"以为测的是 FA2、实际落到 fused"——而 fused 的错是**静默**的。
    与 scripts/probe_v8.2_cp_packing.py 的 C0 同一招（TE 2.17 把选择缓存在模块级 dict）。
    """
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as _dpa,
    )

    b = _dpa._attention_backends
    if b.get("use_flash_attention"):
        return f"FlashAttention({b.get('flash_attention_backend')})"
    if b.get("use_fused_attention"):
        return f"FusedAttention({b.get('fused_attention_backend')})"
    if b.get("use_unfused_attention"):
        return "UnfusedDotProductAttention"
    return "unknown(未选中任何后端)"


def _build_transformer_config(
    hf_config,
    tp_size: int,
    num_layers: Optional[int] = None,
    params_dtype=torch.bfloat16,
    pp_size: int = 1,
    cp_size: int = 1,
    variable_seq_lengths: bool = False,
    attention_backend: str = "flash",
):
    """HF config → Megatron TransformerConfig（对齐源 model_provider.py 的字段映射）。

    num_layers 可覆盖：等价性与层数无关，缩减层数能更快拿到硬门且避开单卡显存约束
    （v8-plan.md §7 已拍板"先缩减层数再冲全层"）。
    params_dtype 可覆盖：fp32 用于"并行切分是数学恒等"的**精确证明**（bf16 只能到舍入，
    见 v7.5.md 的度量教训）。源恒用 bf16（训练用），nano 多一条 fp32 副证路径。
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    return TransformerConfig(
        num_layers=num_layers if num_layers is not None else hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,  # GQA
        ffn_hidden_size=hf_config.intermediate_size,
        kv_channels=head_dim,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        # ≡ megatron 的 `--attention-backend`（源所有 megatron 脚本都传 flash）。
        # mcore 建模型时据此设 NVTE_*，并校验现有 env 与之一致 —— 故必须走这里而不是自己设 env。
        attention_backend=_attn_backend_enum(attention_backend),
        # PP 的 p2p 收发缓冲按这个 dtype 分配（不设则 1F1B 调度直接报错）。
        pipeline_dtype=params_dtype,
        # thd(packing) 下每个微批的 T 不同，PP 的 recv 形状必须动态协商。
        variable_seq_lengths=variable_seq_lengths,
        use_cpu_initialization=True,     # 建在 CPU 再 .cuda()，避开单卡多 rank 的显存峰值
        bf16=(params_dtype == torch.bfloat16),
        fp16=False,
        params_dtype=params_dtype,
        # Qwen3 结构：SwiGLU + RMSNorm + 无 bias + qk-layernorm
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        qk_layernorm=True,
        layernorm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-6),
        # **dropout 必须显式设 0**（TransformerConfig 默认 0.1！）。两个理由：
        #  ①**忠实**：HF Qwen3 config 的 `attention_dropout=0.0` 且无 hidden dropout，
        #    继承 Megatron 的 0.1 默认值等于凭空改了模型；RL 微调也从不开 dropout。
        #  ②**否则等价硬门失去意义**：`model_parallel_cuda_manual_seed` 会**故意**让各 TP rank
        #    持不同的 dropout RNG（TP 区域内 dropout 本就该各切片独立），于是 TP=1 与 TP=2
        #    丢弃的是不同单元 —— 实测 grad cosine 只有 0.25、grad_norm 差 11×，
        #    而这**不是 TP 切错**，纯粹是随机性。踩过这个坑，故写死 0 并留此注释。
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


class MegatronTrainer:
    """Megatron TP × PP × CP 后端训练器。对齐源 megatron_utils 的 init + train 路径。"""

    def __init__(
        self,
        model_path: str,
        lr: float = 1e-6,
        eps_clip: float = 0.2,
        eps_clip_high: float = 0.2,
        clip_grad: float = 1.0,
        global_batch_size: int = 1,
        tensor_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        num_layers: Optional[int] = None,
        backend: str = "gloo",
        load_hf_weights: bool = True,
        params_dtype=torch.bfloat16,
        use_te_spec: Optional[bool] = None,
        attention_backend: str = "flash",
        qkv_format: str = "bshd",
    ):
        self.model_path = model_path
        self.lr = lr
        self.eps_clip = eps_clip
        self.eps_clip_high = eps_clip_high
        self.clip_grad = clip_grad
        self.global_batch_size = global_batch_size
        self.qkv_format = qkv_format
        assert qkv_format in ("bshd", "thd"), f"未支持的 qkv_format={qkv_format!r}"

        # TE spec 默认跟着 CP 走：**CP 只有 TE 后端支持**（mcore
        # dot_product_attention.py:57-59 对 local spec 直接 `assert cp == 1`）。
        # TP-only 路径仍可用 local spec（V8 的既有路径，零回归）。
        self.use_te_spec = (context_parallel_size > 1) if use_te_spec is None else use_te_spec
        assert context_parallel_size == 1 or self.use_te_spec, (
            "CP>1 必须用 TE spec（local spec 的 DotProductAttention assert cp==1）"
        )
        # **CP>1 时后端必须是 flash**：对齐源所有 megatron 训练脚本的 `--attention-backend flash`。
        # 历史原因：TE#3333（fused THD backward 在 sm_120 上静默算错，dq cosine~0.36）。
        # TE 2.18 已修复该 bug（2026-08-14，commit 70957ad5），实测 G3b fused cosine=0.99894
        # vs flash=0.99921，两者均远离 negative control（0.604）——bug 已不存在。
        # 仍保留 flash 约束：（1）跟源；（2）flash 过门更干净；（3）换后端无吞吐收益证据。
        self.attention_backend = attention_backend
        if context_parallel_size > 1:
            assert attention_backend == "flash", (
                f"CP>1 必须走 flash 后端（当前 {attention_backend!r}）——对齐源项目约定"
            )

        # 通信后端：单卡多 rank 只能 gloo（NCCL 拒绝 duplicate GPU）；多卡传 "nccl"。
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        self.backend = backend

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # **device 绑定在单卡/多卡下语义相反，这一行两种情形都要对**：
        #  - 多卡 torchrun：`--nproc_per_node=4` 给每个进程不同的 LOCAL_RANK，必须真绑不同卡，
        #    否则 4 个进程全挤 GPU0 → 立刻 OOM，且 NCCL 会以 `Duplicate GPU detected` 拒绝。
        #  - 单卡多 rank（V8 的 gloo 场景）/ Ray actor（V7.3，Ray 独占 CUDA_VISIBLE_DEVICES）：
        #    LOCAL_RANK 可能超出可见设备数，fallback 到 0 才对。
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_count = torch.cuda.device_count()
        self.local_rank = local_rank if local_rank < device_count else 0
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda")

        self.tp_size = tensor_model_parallel_size or (
            self.world_size // (pipeline_model_parallel_size * context_parallel_size)
        )
        self.pp_size = pipeline_model_parallel_size
        self.cp_size = context_parallel_size
        model_parallel = self.tp_size * self.pp_size * self.cp_size
        assert self.world_size % model_parallel == 0, (
            f"world_size={self.world_size} 须被 tp×pp×cp={model_parallel} 整除"
        )

        # 1) 建并行组（对齐源 initialize.py:_initialize_distributed → mpu.initialize_model_parallel）。
        #    rank 布局默认 order='tp-cp-ep-dp-pp'：TP 相邻、PP 最外层 —— 这正好贴合本机拓扑
        #    （0-1 / 2-3 各自同 NUMA，跨组走 SYS 且无 NVLink）：**TP 通信量最大放同 NODE 内，
        #    PP 通信量最小（每 stage 边界一次 p2p）可跨组**。见 v8.2-plan.md §3.4。
        from megatron.core import mpu

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
                context_parallel_size=self.cp_size,
            )
        self.mpu = mpu
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.cp_rank = mpu.get_context_parallel_rank()
        self.cp_group = mpu.get_context_parallel_group()
        self.dp_size = mpu.get_data_parallel_world_size()
        # **缩放/规约用的是 DP×CP 组**（源 loss.py:1004 `get_data_parallel_world_size(
        # with_context_parallel=True)`）：CP rank 之间持有同一批样本的不同 token 段，
        # 它们的梯度也是部分和，与 DP 一样要规约。
        self.dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self.dp_cp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        # PP 角色标志：first stage 承载 input embedding，last stage 承载 final norm / output
        # layer 并产出 logits；它们随后传给 GPTModel 的 pre_process / post_process。
        # PP=1 时同一 rank 既是 first 也是 last，因而同时拥有模型的输入端和输出端。
        self.is_first_stage = mpu.is_pipeline_first_stage()
        self.is_last_stage = mpu.is_pipeline_last_stage()

        # Megatron 的 TP 随机数状态（各 TP rank 要有不同的 dropout 种子、相同的权重种子）。
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(1234)

        # 2) 顺序 load config/tokenizer（同 V7：避开 HF cache 竞争）。
        from transformers import AutoConfig, AutoTokenizer

        for i in range(self.world_size):
            if i == self.rank:
                self.hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            dist.barrier()

        self.config = _build_transformer_config(
            self.hf_config,
            self.tp_size,
            num_layers,
            params_dtype=params_dtype,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            variable_seq_lengths=(qkv_format == "thd" and self.pp_size > 1),
            attention_backend=attention_backend,
        )
        self.params_dtype = params_dtype
        self.num_layers = self.config.num_layers

        # 3) 建 GPTModel（对齐源 model_provider.py:177-201）。
        #    parallel_output=True → 输出 **vocab 分片** logits，log_prob 走 vocab-parallel 路径。
        #    share_embeddings_and_output_weights = tie（Qwen3-0.6B tie=True，同 V7.0 的事实）。
        #    pre/post_process 按 PP stage 判定：**只有 first stage 有 embedding、
        #    只有 last stage 有 output_layer + final_layernorm**（PP 把模型纵向切开）。
        from megatron.core.models.gpt import GPTModel

        _clear_nvte_backend_env()  # 见该函数 docstring：清陈旧 env，让 config 唯一决定后端
        if self.use_te_spec:
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_with_transformer_engine_spec as _layer_spec,
            )
        else:
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_local_spec as _layer_spec,
            )

        self.tie = bool(self.hf_config.tie_word_embeddings)
        model = GPTModel(
            config=self.config,
            transformer_layer_spec=_layer_spec(qk_layernorm=True),
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=getattr(self.hf_config, "max_position_embeddings", 4096),
            pre_process=self.is_first_stage,
            post_process=self.is_last_stage,
            position_embedding_type="rope",
            rotary_base=getattr(self.hf_config, "rope_theta", 10000),
            parallel_output=True,
            share_embeddings_and_output_weights=self.tie,
        )
        self.model = model.cuda()
        self.model.train()

        # 源 model.py:539 把 mcore 的 finalize_model_grads 挂到 config 上，由
        # `get_forward_backward_func()` 在反向结束后自动调用。nano 挂自己那份
        # （原因见模块 docstring 的偏离登记：mcore 原版读 param.main_grad，需要 DDP wrapper）。
        self.config.finalize_model_grads_func = lambda *a, **kw: self._finalize_model_grads()

        # 4) 载 HF 预训练权重（对齐源：训练从预训练 ckpt 起，不从随机初始化起）。
        #    源用离线 checkpoint 转换工具链（HF → mcore 格式）在训练进程外做；nano 在进程内
        #    走 megatron_to_hf.load_hf_into_megatron（登记偏离，见该模块 docstring）。
        #    **这对等价硬门是必需的**：TP=1 与 TP=2 是两次独立进程，随机初始化不可能一致，
        #    只有都从同一份 HF 权重载入，「loss/grad 等价」才是在比切分本身。
        if load_hf_weights:
            from toy_rl.trainer.megatron_to_hf import load_hf_into_megatron
            from toy_rl.trainer.megatron_to_hf import load_hf_weights as _read_hf

            for i in range(self.world_size):
                if i == self.rank:
                    hf_state = _read_hf(model_path)
                    load_hf_into_megatron(self.model, hf_state, self.hf_config, self.tie)
                    del hf_state
                dist.barrier()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )
        logger.info(
            f"Rank {self.rank}: MegatronTrainer ready (tp={self.tp_size}/{self.tp_rank}, "
            f"pp={self.pp_size}/{self.pp_rank}, cp={self.cp_size}/{self.cp_rank}, "
            f"dp={self.dp_size}, layers={self.num_layers}, tie={self.tie}, "
            f"spec={'te' if self.use_te_spec else 'local'}, qkv={self.qkv_format})"
        )

    # ---- 拓扑 introspection（G5：把"拓扑感知的 rank 映射"变成可验证的知识点）----------

    def parallel_group_ranks(self) -> dict[str, list[int]]:
        """打印各并行组的全局 rank 列表。

        `TP=2 × PP=2` 的默认布局（order='tp-cp-ep-dp-pp'）恰好是 `[0,1]` 一个 TP 组、
        `[2,3]` 另一个 —— 正好落在同 NUMA node 内，PP 组 `[0,2]`/`[1,3]` 跨组。
        这不是巧合而是**该被断言的设计**：TP 每层两次 all-reduce 激活（通信量最大）
        必须放同 NODE，PP 每 stage 边界一次 p2p（通信量最小）跨组可接受。
        """
        return {
            "tp": dist.get_process_group_ranks(self.tp_group),
            "pp": dist.get_process_group_ranks(self.pp_group),
            "cp": dist.get_process_group_ranks(self.cp_group),
            "dp_cp": dist.get_process_group_ranks(self.dp_cp_group),
        }

    # ---- 批构造（对齐源 data.py:get_batch）--------------------------------

    def _build_batch(
        self,
        samples: list[dict],
        num_microbatches: int,
        global_batch_size: int,
        seq_length: Optional[int] = None,
        packed_length: Optional[int] = None,
    ) -> Optional[dict]:
        """样本 list → 本 rank 的一个微批（含 CP 切分 / thd packing）。对齐源 `get_batch`。

        **坐标系（本方法唯一需要想清楚的事）**：把每条样本物化成三个与 token 流**等长**的
        「full」数组，位置 g 的语义统一为「logits[g] 预测 tokens[g+1]」：
            full_targets[g]  = tokens[g+1]          （g == L-1 时填 pad，被 mask 掉）
            full_tgt_mask[g] = loss_mask[g+1]       （g == L-1 时填 0）
            full_old_lp[g]   = old_log_probs[g+1]   （同上）
        这与 V6/V7/V8 的 `[1:]` 右移**逐值一致**，只是把右移写进了数组本身。
        好处是三者与 tokens 对齐后可以用**同一个** `slice_with_cp` 切 —— CP 下不必再做
        偏移量算术（源 `get_logits_and_tokens_offset_with_cp` 做的就是那套算术，
        nano 用等长坐标系把它消掉，见 cp_utils.py 的偏离登记）。
        源 data.py:137 对 loss_mask 做 `F.pad(loss_mask, (prompt_length-1, 1))` 也是同一件事。

        **分母取全量**：`denom = sum(loss_mask[1:])` 是这条样本的**全部**可训练 token 数，
        与本 rank 分到几块无关 —— 这正是源 `get_sum_of_sample_mean` 的 CP 分支
        （分子部分和 / 分母全量）。各 CP rank 的部分和最后由梯度规约加起来。
        """
        import torch.nn.functional as F

        from toy_rl.trainer.cp_utils import cp_padded_length, slice_with_cp

        trainable = [s for s in samples if len(s["tokens"]) >= 2 and sum(s["loss_mask"][1:]) > 0]
        if not trainable:
            return None
        if self.pp_size > 1:
            assert len(trainable) == len(samples), (
                "PP 下各微批的 micro_batch_size 必须一致（1F1B 的 recv 形状是按它算的），"
                "不能有样本被过滤掉"
            )

        dev = self.device
        toks, tgts, masks, olps, denoms, advs = [], [], [], [], [], []
        for s in trainable:
            t = torch.tensor(s["tokens"], dtype=torch.long, device=dev)
            L = t.size(0)
            m = torch.tensor(s["loss_mask"], dtype=torch.float32, device=dev)
            full_t = torch.cat([t[1:], t.new_full((1,), self.pad_id)])
            full_m = torch.cat([m[1:], m.new_zeros(1)])
            old = s.get("old_log_probs")
            if old is not None:
                o = torch.tensor(old, dtype=torch.float32, device=dev)
                full_o = torch.cat([o[1:], o.new_zeros(1)])
            else:
                # on-policy：真正的 old_lp = cur_lp.detach()，在 loss_func 里补；
                # 这里放等长的 0 只是为了让三个 full 数组能走**同一条**切分/打包路径。
                full_o = torch.zeros_like(full_m)
            toks.append(t)
            tgts.append(full_t)
            masks.append(full_m)
            olps.append(full_o)
            denoms.append(torch.clamp_min(full_m.sum(), 1.0))
            advs.append(float(s["reward"]))

        batch = {
            "num_microbatches": num_microbatches,
            "global_batch_size": global_batch_size,
            "denoms": denoms,
            "advantages": advs,
            "on_policy": [s.get("old_log_probs") is None for s in trainable],
        }

        if self.qkv_format == "bshd":
            # 一行一条样本，各行须等长；CP 再把每行切成 2 chunk（故须被 2*cp 整除）。
            unit = 2 * self.cp_size
            max_len = seq_length or (
                (max(t.size(0) for t in toks) + unit - 1) // unit * unit
            )
            assert max_len % unit == 0, f"seq_length({max_len}) 须被 2*cp({unit}) 整除"
            batch["tokens"] = torch.stack(
                [slice_with_cp(t, self.pad_id, "bshd", max_len) for t in toks]
            )
            batch["targets"] = torch.stack(
                [slice_with_cp(t, self.pad_id, "bshd", max_len) for t in tgts]
            )
            batch["tgt_masks"] = torch.stack([slice_with_cp(m, 0.0, "bshd", max_len) for m in masks])
            batch["old_log_probs"] = torch.stack(
                [slice_with_cp(o, 0.0, "bshd", max_len) for o in olps]
            )
            batch["packed_seq_params"] = None
            # **全局**长度（mcore 的 get_tensor_shapes 内部再 // cp_size 得到本 rank 的 S）
            batch["seq_length"] = max_len
            return batch

        # ---- thd（sequence packing）：所有样本拼成一条 [1, T] flat 流 ----
        from megatron.core.packed_seq_params import PackedSeqParams

        local = [slice_with_cp(t, self.pad_id, "thd") for t in toks]
        # cu_seqlens 要的是**原始（全局）长度**，故 CP 下按 cp_padded_length 累加
        # （源 data.py:112 用 `cu_seqlens * cp_size` 表达同一件事；nano 直接算全局长度，
        #  因为各条样本的 pad 量不同，逐条算比整体乘更不易错）。
        cu = [0]
        for t in toks:
            cu.append(cu[-1] + cp_padded_length(t.size(0), self.cp_size))
        flat = torch.cat(local)
        target = packed_length or flat.size(0)
        pad = target - flat.size(0)
        assert pad >= 0, f"packed_length={target} 小于实际打包长度 {flat.size(0)}"
        if pad > 0:
            flat = F.pad(flat, (0, pad), value=self.pad_id)
            cu.append(cu[-1] + pad * self.cp_size)  # padding 也算一"段"，对齐源 data.py:109
        cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
        max_seqlen = int((cu_t[1:] - cu_t[:-1]).max().item())

        def _pack(seqs, pad_value):
            x = torch.cat([slice_with_cp(v, pad_value, "thd") for v in seqs])
            return F.pad(x, (0, pad), value=pad_value) if pad > 0 else x

        batch["tokens"] = flat.unsqueeze(0)
        batch["targets"] = _pack(tgts, self.pad_id).unsqueeze(0)
        batch["tgt_masks"] = _pack(masks, 0.0).unsqueeze(0)
        batch["old_log_probs"] = _pack(olps, 0.0).unsqueeze(0)
        # 本 rank 的段边界（用于从 flat logits 里切每条样本）—— 与 cu_seqlens 不同：
        # cu_seqlens 是**全局**长度（TE 需要），这里是**本 rank 分片**的长度。
        local_cu = [0]
        for v in local:
            local_cu.append(local_cu[-1] + v.size(0))
        batch["local_cu_seqlens"] = local_cu
        batch["packed_seq_params"] = PackedSeqParams(
            cu_seqlens_q=cu_t,
            cu_seqlens_kv=cu_t,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
        batch["seq_length"] = flat.size(0)
        return batch

    # ---- 前向 / loss（对齐源 model.py:380-416 + loss.py:943）---------------

    def _forward_logits(self, batch: dict) -> torch.Tensor:
        """跑一次前向，返回 **vocab 分片** logits（不 gather，对齐 parallel_output=True）。

        三条 attention mask 路径，**必须分清**：
          - thd(packing)：`attention_mask=None` + `position_ids=None` + `packed_seq_params`
            —— 与源 model.py:395-402 逐字一致。段隔离与 RoPE 全由 cu_seqlens 定义，
            TE 内部 dispatch 到 varlen 内核（这与 V7.6 在 FSDP 侧走 FA2 varlen 是同一件事）。
          - bshd + TE spec：`attention_mask=None`，TE 按 spec 里的 `AttnMaskType.causal`
            内部生成因果 mask。**CP 必须走这条**——显式 [B,1,S,S] mask 与序列被切开不兼容。
          - bshd + local spec（V8 既有路径）：mcore 的 DotProductAttention 要显式 mask，
            形状 [B,1,S,S] 且语义是 **True=屏蔽**（与 HF 的 True=保留相反，最易踩反的一处）。
        """
        input_ids = batch["tokens"]
        if self.qkv_format == "thd":
            return self.model(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=batch["packed_seq_params"],
            )

        b, s = input_ids.shape
        position_ids = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        if self.use_te_spec:
            return self.model(
                input_ids=input_ids, position_ids=position_ids, attention_mask=None
            )
        causal = torch.tril(torch.ones(s, s, device=input_ids.device, dtype=torch.bool))
        attention_mask = (~causal).view(1, 1, s, s).expand(b, 1, s, s)
        return self.model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

    def _vocab_parallel_log_probs(
        self, logits_shard: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """分片 logits [S, V/tp] + targets [S] → 每 token log_prob [S]（跨 TP 组）。

        对齐源 ppo_utils.py:compute_log_probs —— 用 `fused_vocab_parallel_cross_entropy`，
        它内部跨 tp_group all-reduce（max 与 sum-exp），故**无需把 logits gather 成全量**。
        实测 TP=2 与全量 log_softmax 的 max_abs_err = 0.000e+00。

        源把 logits 转成 [S,B,V] 喂 fused kernel，这里逐样本处理故 B=1（unsqueeze(1)），
        与源 `logits.unsqueeze(1)` 逐字对应。

        **必须传独立副本（fp32 副证路径踩出来的坑）**：该 fused kernel 会 **in-place** 减去
        max（`vocab_parallel_logits -= logits_max`）。nano 是**逐样本**切 logits 的，
        各样本的切片共享同一块 base storage —— 原地改动会撞上 autograd 版本计数
        （实测报 `modified by an inplace operation ... is at version 9; expected version 6`），
        且会污染产生 logits 那一步自身的反向。bf16 时 `.float()` 恰好**隐式**拷了一份把问题盖住，
        只有 fp32（`.float()` 是 no-op）才暴露 —— 故这里显式 clone，与 dtype 无关。
        源不需要这一步：它把整批 pack 成一条 [S,B,V] 一次性喂进 kernel，不做逐样本切片。
        """
        from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

        logits_shard = logits_shard.float().clone()
        lp = -fused_vocab_parallel_cross_entropy(
            logits_shard.unsqueeze(1).contiguous(), targets.unsqueeze(1), self.tp_group
        )
        return lp.squeeze(-1) if lp.dim() > 1 else lp

    def _grpo_sample_loss(
        self,
        cur_lp: torch.Tensor,
        old_lp: torch.Tensor,
        adv: torch.Tensor,
        tgt_mask: torch.Tensor,
        denom: torch.Tensor,
    ) -> torch.Tensor:
        """单样本 GRPO policy-gradient loss —— **与 V7 FSDPTrainer._grpo_sample_loss 同一份数学**。

        对齐 ppo_utils.py:132-135 + sum_of_sample_mean：
          ratio = exp(cur_lp - old_lp); pg = min(ratio·adv, clip(ratio,1±eps)·adv)
          loss  = -(pg·tgt_mask).sum() / denom
        刻意与 FSDP 路径逐字相同：V8/V8.2 换的是**并行轴**，不是 loss 语义。

        `denom` 单独传而不是就地 `tgt_mask.sum()`：CP>1 时本 rank 的 `tgt_mask` 只是
        全量的一部分，分母必须取**全量** loss_mask 的和（源 cp_utils.py 的
        `get_sum_of_sample_mean` CP 分支）。cp_size==1 时两者恒等，故无回归。
        """
        ratio = (cur_lp - old_lp).exp()
        pg = torch.min(
            ratio * adv,
            ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip_high) * adv,
        )
        return -(pg * tgt_mask).sum() / denom

    def loss_func(self, batch: dict, logits: torch.Tensor):
        """logits → (缩放后的 loss, num_tokens, 日志 dict)。对齐源 loss.py:943 `loss_function`。

        **本函数显式执行的缩放**（对齐源 loss.py:1002-1005）：
            loss × num_microbatches / global_batch_size × dp_size(with_cp)
        注意：这里的 `× num_microbatches` 只有乘法；对应的
        `output_tensor /= num_microbatches` 不在本函数，而是在 Megatron Core 的
        pipeline schedule（`schedules.py:274`）中执行。两者合起来才是完整缩放。
        三个因子各自的理由：
          - `× num_microbatches`：抵消 mcore schedule 随后执行的
            `output_tensor /= num_microbatches`（schedules.py:274）。
          - `/ global_batch_size`：GRPO 的 loss 是「全局批的逐样本 mean 之和 / gbs」。
          - `× dp_size(with_cp)`：抵消梯度规约那一步的 `1/dp_cp`（源靠 mcore DDP 的
            `gradient_scaling_factor`，nano 靠 `_allreduce_dp_cp_grads` 的 AVG）。
            净效果 = 各 DP/CP rank 的贡献**求和**。
        """
        losses = []
        for i, denom in enumerate(batch["denoms"]):
            if self.qkv_format == "thd":
                lo, hi = batch["local_cu_seqlens"][i], batch["local_cu_seqlens"][i + 1]
                lg, tg = logits[0, lo:hi], batch["targets"][0, lo:hi]
                mask, old = batch["tgt_masks"][0, lo:hi], batch["old_log_probs"][0, lo:hi]
            else:
                lg, tg = logits[i], batch["targets"][i]
                mask, old = batch["tgt_masks"][i], batch["old_log_probs"][i]
            cur_lp = self._vocab_parallel_log_probs(lg, tg)
            if batch["on_policy"][i]:
                old = cur_lp.detach()
            adv = torch.tensor(batch["advantages"][i], device=logits.device)
            losses.append(self._grpo_sample_loss(cur_lp, old, adv, mask, denom))

        loss = torch.stack(losses).sum()
        loss = loss * batch["num_microbatches"] / batch["global_batch_size"] * self.dp_cp_size
        # 3-tuple 的第二项是 per-token loss 的分母，nano 走 per-sample 路径故恒为 1
        # （源 loss.py:1011 在 `calculate_per_token_loss=False` 时同样返回 1）。
        return (
            loss,
            torch.tensor(1, device=logits.device),
            {"loss": loss.detach(), "trained_samples": len(losses)},
        )

    def forward_step(self, data_iterator, model):
        """一个微批的**纯前向**，返回 (logits, loss 偏函数)。对齐源 model.py:380-416。

        这是 `get_forward_backward_func()` 要求的签名 —— **前向与 loss 必须拆开**，
        因为 PP 的 1F1B 调度里「前向」发生在每个 stage、而「算 loss」只发生在 last stage
        （中间 stage 的 `output_tensor` 是 hidden states，直接 p2p 送给下一个 stage）。
        V8 因为 PP=1 才能把两者合成一个 `_microbatch_backward`，V8.2 把这笔技术债还了。
        """
        batch = next(data_iterator)
        output_tensor = self._forward_logits(batch)
        return output_tensor, partial(self.loss_func, batch)

    def _microbatch_backward(
        self, samples: list[dict], global_batch_size: int
    ) -> tuple[float, int]:
        """一个微批的 forward + loss + backward（累积梯度，不 step、不 zero_grad）。

        = `forward_backward_no_pipelining` 在 num_microbatches=1 时的手写等价物
        （mcore schedules.py:600 那个循环体）。保留它有两个用处：
          ①V8 的门脚本要「只跑一个微批、拿到未被 clip 修改过的梯度」；
          ②把「forward_step + loss_func + backward」这三步的关系摊平给人看。
        **PP>1 时不能用它**（会绕过 1F1B 调度而死锁），故硬断言。
        """
        assert self.pp_size == 1, "_microbatch_backward 不支持 PP>1，请走 train_batch"
        batch = self._build_batch(samples, 1, global_batch_size)
        if batch is None:
            return 0.0, 0
        output_tensor, loss_fn = self.forward_step(iter([batch]), self.model)
        loss, _, log = loss_fn(output_tensor)
        loss.backward()
        return float(loss.detach()), int(log["trained_samples"])

    # ---- 梯度收尾（对齐源 distributed/finalize_model_grads.py）------------

    def _allreduce_qk_layernorm_grads(self) -> None:
        """跨 TP 组 SUM-all-reduce **qk-layernorm** 的梯度。

        对齐源 `finalize_model_grads.py:402-405`（`_allreduce_layernorm_grads` 里那条
         `elif (config.sequence_parallel and param.sequence_parallel) or
              (config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name))`
        分支，reduce op = **SUM**）。

        **为什么必须有这一步（V8 踩出来的正确性坑）**：q/k-layernorm 作用在**每个头**上，
        而头是被 TP 切开的 —— 各 rank 只见到自己那份头，算出的 `q_norm.weight` 梯度是
        **部分和**，加起来才是全量。其余 layernorm（`input_layernorm`/`final_layernorm`）
        不需要：它们在 TP 区域**外**，输入本就是各 rank 相同的全量激活
        （row-parallel 的 all-reduce 保证反向传回的梯度也相同），故各 rank 已持全量梯度。
        源那条 `sequence_parallel` 并列条件 nano 未开 SP，恒 False，故不实现（登记偏离）。

        实测判别力：漏掉这一步时 TP=2 vs TP=1 的 grad rel_L2=1.8e-2、cosine=0.99984，
        逐参数最差的正是 `q_norm.weight`（8.5e-4）—— 而同一次 loss 差**恰为 0**。
        「前向逐值相同、梯度却有系统性偏差」正是这类漏规约的指纹。
        """
        if self.tp_size <= 1 or not self.config.qk_layernorm:
            return
        for name, p in self.model.named_parameters():
            if p.grad is not None and ("q_layernorm" in name or "k_layernorm" in name):
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.tp_group)

    def _allreduce_word_embedding_grads(self) -> None:
        """跨 PP first/last stage SUM-all-reduce **tie 的 word embedding** 梯度。

        对齐源 `finalize_model_grads.py:164 _allreduce_word_embedding_grads`
        （→ `:204 _allreduce_embedding_grad`，组用 `mpu.get_embedding_group()`，
        取权重用 `model_module.shared_embedding_or_output_weight()`）。

        **V8.2 最硬的正确性点**：Qwen3-0.6B `tie_word_embeddings=True`（V7.0 已实测）。PP>1 时
          - first stage 持 `embedding.word_embeddings.weight`（输入嵌入）
          - last  stage 持 `output_layer.weight`（输出投影）
        tie 意味着这**是同一份权重**，但它们物理上在**不同 stage、不同进程**里，
        各自算出的梯度都只是部分和 —— 必须跨 first/last stage all-reduce 才是全量。

        **与 V8 坑③（漏 qk-layernorm 规约）是同一类错误、同一个指纹**：
        loss Δ 恰为 0 而梯度系统性偏，只不过最差参数从 `q_norm.weight` 变成
        `embed_tokens.weight`。**fp32 精确门能抓到、bf16 会当舍入放过**（V8 已证 5e-2 被放过）。

        nano 手写而非调 mcore 原版的理由见模块 docstring（原版读 `param.main_grad`，
        且要 `model_chunk.ddp_config` —— 那是 DDP wrapper 的东西，nano 无）。
        """
        if self.pp_size <= 1 or not self.tie:
            return
        embd_group = self.mpu.get_embedding_group(check_initialized=False)
        if embd_group is None or dist.get_world_size(embd_group) <= 1:
            return
        if self.rank not in dist.get_process_group_ranks(embd_group):
            return
        weight = self.model.shared_embedding_or_output_weight()
        if weight is None or weight.grad is None:
            return
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM, group=embd_group)

    def _allreduce_dp_cp_grads(self) -> None:
        """跨 DP×CP 组 AVG-all-reduce **全部**梯度 —— 即 mcore DDP 做的那一步。

        源靠 `DistributedDataParallel` 的梯度缓冲自动完成：它以
        `gradient_scaling_factor = 1/dp_cp_size` 累积再 all-reduce，净效果 = 各 rank 贡献求和
        （所以源在 loss.py:1004 先 `× dp_size(with_cp)` 把这个 1/n 抵消掉）。
        nano 用 torch AdamW、无 DDP wrapper（V8 偏离 #7 的连带后果），**必须手写这一步**。

        注意：本函数只负责 DP×CP 梯度的 AVG 规约；`num_microbatches` 的除法由
        Megatron Core 的 pipeline schedule 完成，不在这里执行。

        **CP 下这不是可选项**：各 CP rank 只持有序列的一部分 token，它们对同一份权重的
        梯度是**部分和**。漏掉 → 又一个"前向对、梯度错"（且 loss 值也只是部分和）。
        `dp_cp_size == 1`（V8 的全部既有配置）时整个方法是 no-op → 零回归。
        """
        if self.dp_cp_size <= 1:
            return
        for p in self.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.dp_cp_group)

    def _finalize_model_grads(self) -> None:
        """反向之后、clip/step 之前的三处跨 rank 梯度规约。对齐源 `finalize_model_grads`。

        顺序对齐源：先 DP/CP 的梯度同步（源里是 DDP 的 `finish_grad_sync`），
        再 layernorm（TP），再 embedding（PP）。三者作用在不相交的通信组上，
        故顺序其实不影响结果；照源排是为了对照时一眼能找到对应位置。
        """
        self._allreduce_dp_cp_grads()
        self._allreduce_qk_layernorm_grads()
        self._allreduce_word_embedding_grads()

    # ---- 训练一步 --------------------------------------------------------

    def train_batch(
        self,
        samples: list[dict],
        global_batch_size: Optional[int] = None,
        microbatch_size: int = 1,
    ) -> dict:
        """梯度累积一步：zero_grad → `get_forward_backward_func()` → clip + step。

        **对齐源 model.py:419-429**：调度交给 megatron-core，它按 `pp_size` 自动选
        `forward_backward_no_pipelining`（PP=1）或 `forward_backward_pipelining_without_
        interleaving`（1F1B）。**手写 send/recv 调度是重复造轮子且必错** —— 1F1B 的
        warmup/steady/cooldown 三段与跨 stage 的梯度累积边界很容易搞反。
        `_finalize_model_grads` 由 `config.finalize_model_grads_func` 在反向结束后自动调用
        （同源 model.py:539），不在这里显式调。
        """
        if microbatch_size < 1:
            raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
        gbs = global_batch_size if global_batch_size is not None else self.global_batch_size

        chunks = [samples[i : i + microbatch_size] for i in range(0, len(samples), microbatch_size)]
        num_microbatches = len(chunks)
        if self.pp_size > 1:
            assert len(samples) % microbatch_size == 0, (
                f"PP 下 len(samples)={len(samples)} 须被 microbatch_size={microbatch_size} 整除"
                "（1F1B 的 recv 缓冲按固定 micro_batch_size 预分配）"
            )

        # **各微批的形状必须一致**（PP 的 recv 缓冲按 seq_length/micro_batch_size 预分配；
        # 即使 PP=1 也统一按全局最长 pad，省得两条路径的数值有差别）。
        seq_length = packed_length = None
        if self.qkv_format == "bshd":
            longest = max(len(s["tokens"]) for s in samples)
            unit = 2 * self.cp_size
            seq_length = ((longest + unit - 1) // unit) * unit
        else:
            from toy_rl.trainer.cp_utils import cp_padded_length

            packed_length = max(
                sum(cp_padded_length(len(s["tokens"]), self.cp_size) // self.cp_size for s in c)
                for c in chunks
            )

        batches = []
        for c in chunks:
            b = self._build_batch(c, num_microbatches, gbs, seq_length, packed_length)
            if b is not None:
                batches.append(b)
        if not batches:
            return {"loss": 0.0, "grad_norm": 0.0, "trained_samples": 0, "num_microbatches": 0}

        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        self.optimizer.zero_grad(set_to_none=True)
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=self.forward_step,
            data_iterator=iter(batches),
            model=[self.model],
            num_microbatches=len(batches),
            # bshd：`seq_length` 要传**全局**长度（mcore get_tensor_shapes 内部 // cp_size）。
            # thd：`variable_seq_lengths=True` 时它被忽略（各微批的 T 动态协商）。
            seq_length=batches[0]["seq_length"],
            micro_batch_size=microbatch_size,
            forward_only=False,
        )

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        # loss 只在 last stage 算出；CP>1 时各 rank 只有部分和 —— 用与梯度同样的
        # AVG 规约还原成全量（× dp_cp 的缩放已在 loss_func 里做过）。
        loss_sum = sum(float(x["loss"]) for x in losses_reduced)
        trained = sum(int(x["trained_samples"]) for x in losses_reduced)
        if self.dp_cp_size > 1:
            t = torch.tensor([loss_sum, float(trained)], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG, group=self.dp_cp_group)
            loss_sum, trained = float(t[0]), int(round(float(t[1])))
        if self.pp_size > 1:
            # **PP 下非 last stage 的 `losses_reduced` 是空的**（mcore schedules 只在
            # last stage 调 loss_func），照直返回会让 rank 0（first stage）报 loss=0 ——
            # 那不是算错，是"这个数在别的进程里"。源在 last stage 上记日志
            # （loss.py:431 / model.py:279 都有 `is_pipeline_last_stage` 守卫）；
            # nano 多广播一步，让任一 rank 都能拿到同一个数，验收脚本才不必关心自己是哪个 stage。
            # **纯报告用**，与梯度无关。
            t = torch.tensor([loss_sum, float(trained)], device=self.device)
            dist.broadcast(t, src=dist.get_process_group_ranks(self.pp_group)[-1],
                           group=self.pp_group)
            loss_sum, trained = float(t[0]), int(round(float(t[1])))

        return {
            "loss": loss_sum,
            "grad_norm": float(grad_norm),
            "trained_samples": trained,
            "num_microbatches": len(batches),
        }

    def train_step(
        self,
        tokens: list[int],
        loss_mask: list[int],
        reward: float,
        old_log_probs: Optional[list[float]] = None,
    ) -> dict:
        """单样本一步（与 V7 同名入口对齐）= global_batch_size=1 的单微批 train_batch。"""
        return self.train_batch(
            [{"tokens": tokens, "loss_mask": loss_mask, "reward": reward,
              "old_log_probs": old_log_probs}],
            global_batch_size=1,
        )

    # ---- 权重导出（V8.1 + V8.2 的 PP broadcast）---------------------------

    def to_hf_state_dict(self) -> dict:
        """Megatron 分片权重 → HF state_dict（TP-gather + PP-broadcast + 拆 qkv/gate + 改名）。

        这是 Megatron 后端**特有**的一步：FSDP 的 `save_pretrained` 天然吐 HF 格式，
        Megatron 必须先过这层转换才能喂给 SGLang。**集体操作**（含 all_gather + broadcast），
        所有 rank 都要进；各 rank 拿到的结果相同。对齐源 `update_weight/` 的路径。
        """
        from toy_rl.trainer.megatron_to_hf import megatron_to_hf_state_dict

        return megatron_to_hf_state_dict(self.model, self.hf_config, self.tie)

    def save_pretrained(self, save_path: str) -> None:
        """HF 格式落盘，供 SGLang disk reload 重载权重。对齐 WeightUpdater.update_weights 的调用约定。

        偏离源（登记见 docs/decisions/v9.md）：
          源 megatron_utils 走 tensor 广播到 SGLang 内存（update_weights_from_distributed），
          不落盘。nano 沿用 V6/V7 的 disk reload 最小路径：先 to_hf_state_dict 转换 +
          gather（集体操作，所有 rank 必须进），再 rank0 写盘。语义等价；disk I/O 是额外开销。

        集体操作约定（对齐 FSDPTrainer.save_pretrained / WeightUpdater rank 门控）：
          to_hf_state_dict 内含 all_gather + broadcast，所有 rank 都要调；
          只有 rank0 真正写盘（与 fsdp 路径一致，WeightUpdater 会 fan-out 到每个 rank）。
        """
        import os

        from transformers import AutoTokenizer

        # 集体操作：所有 rank 参与（TP-gather + PP-broadcast 在内部）
        state_dict = self.to_hf_state_dict()

        if self.rank == 0:
            os.makedirs(save_path, exist_ok=True)
            # HF 格式写盘：config + tokenizer 从原始模型路径复制，权重用转换后的 state_dict
            self.hf_config.save_pretrained(save_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            tokenizer.save_pretrained(save_path)
            torch.save(state_dict, os.path.join(save_path, "pytorch_model.bin"))
