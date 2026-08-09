"""V8.0: MegatronTrainer —— Megatron TP（张量并行）后端的真训练一步。

对齐源 slime/backends/megatron_utils/：
  - 并行组初始化 ← initialize.py:_initialize_distributed（mpu.initialize_model_parallel）
  - 建模型      ← model_provider.py:177-201（GPTModel + parallel_output=True + tie）
  - log_probs   ← loss.py:225 get_log_probs_and_entropy → ppo_utils.py:649
                  calculate_log_probs_and_entropy → ppo_utils.py:compute_log_probs
                  （**vocab-parallel**：logits 保持分片，跨 TP 组算 cross entropy）
  - loss 数学   ← ppo_utils.py:132-135 compute_policy_loss（与 V6/V7 完全同一份 GRPO 公式）

与 V7 FSDPTrainer 的关系（V8 教学核心）：**同一份 GRPO loss 数学，换到另一根并行轴上**。
  - V7 FSDP = 1D DP：参数按 rank 分片**存储**，前向时 all-gather 回**全量层**再算。
    边界是「单层必须放得进一张卡」。通信发生在**层边界**（gather 参数）。
  - V8 Megatron TP = 层内切分：qkv/mlp 按列/行切开，各 rank 只算**部分结果**，
    col→row 之间 all-reduce **激活**。单层也能跨卡。通信发生在**层内部**。
  两者正交（源里可以同时开）。nano 分两版讲清楚每一根轴。

**vocab-parallel logits（本版最关键的正确性点）**：
  源 model_provider.py:177 设 `parallel_output=True` —— 输出层也按 vocab 维切开，
  各 rank 只拿到 [S, V/tp] 的**分片 logits**，从不 gather 成全量 [S, V]（V=151936，
  gather 会白白吃显存+通信）。因此 log_prob **不能**用普通 log_softmax，必须用
  `fused_vocab_parallel_cross_entropy` 跨 TP 组做（内部 all-reduce max 与 sum-exp）。
  实测（5090 TP=2）：它与全量 log_softmax 的 max_abs_err = **0.000e+00**，梯度正常回流。

偏离源项目（登记见 docs/decisions/v8.md）：
  - **通信后端 gloo 而非 NCCL**：单卡多 rank 被 NCCL 硬拒（实测
    `Duplicate GPU detected: rank 0 and rank 1 both on CUDA device 80`），gloo 是单卡跑
    多维并行的唯一通路。集合通信语义等价、性能不等价（TCP）。V8 教并行语义不测吞吐（吞吐留 V9）。
  - **只用 megatron.core，不走 megatron.training.init**：源 initialize.py:init 还要
    _build_tokenizer / 微批计算器 等训练框架件（且断言 numpy 1.x，容器是 2.3.5）；nano 只需
    并行组 + GPTModel，直接用 core 更清晰。
  - **只做 TP，不做 PP/CP/EP**：单卡实测阻断——gloo 不支持 CUDA p2p（PP 传激活）与
    all_to_all（EP 路由 token）。记录设计不实现，见 v8.md。
  - 无 KL/entropy、无 ref model（沿用 V6/V7 的最纯 GRPO）；优化器用 torch AdamW 而非
    Megatron DistributedOptimizer（后者是 ZeRO-1 式优化器分片，与 TP 正交，nano 不铺）。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _build_transformer_config(
    hf_config, tp_size: int, num_layers: Optional[int] = None, params_dtype=torch.bfloat16
):
    """HF config → Megatron TransformerConfig（对齐源 model_provider.py 的字段映射）。

    num_layers 可覆盖：等价性与层数无关，缩减层数能更快拿到硬门且避开单卡显存约束
    （v8-plan.md §7 已拍板"先缩减层数再冲全层"）。
    params_dtype 可覆盖：fp32 用于"TP 是数学恒等切分"的**精确证明**（bf16 只能到舍入，
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
        pipeline_model_parallel_size=1,  # 单卡只做 TP，见模块 docstring 偏离说明
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
    """Megatron TP 后端训练器。对齐源 megatron_utils 的 init + train 路径。"""

    def __init__(
        self,
        model_path: str,
        lr: float = 1e-6,
        eps_clip: float = 0.2,
        eps_clip_high: float = 0.2,
        clip_grad: float = 1.0,
        global_batch_size: int = 1,
        tensor_model_parallel_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        backend: str = "gloo",
        load_hf_weights: bool = True,
        params_dtype=torch.bfloat16,
    ):
        self.model_path = model_path
        self.lr = lr
        self.eps_clip = eps_clip
        self.eps_clip_high = eps_clip_high
        self.clip_grad = clip_grad
        self.global_batch_size = global_batch_size

        # 通信后端：单卡多 rank 只能 gloo（NCCL 拒绝 duplicate GPU）；多卡时传 "nccl"。
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # 单卡：所有 rank 共用 device 0（LOCAL_RANK 恒 0 的同一个道理，见 v7.md V7.3 正确性点 1）。
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(local_rank if local_rank < device_count else 0)
        self.device = torch.device("cuda")

        self.tp_size = tensor_model_parallel_size or self.world_size
        assert self.world_size % self.tp_size == 0, (
            f"world_size={self.world_size} 须被 tp_size={self.tp_size} 整除"
        )

        # 1) 建并行组（对齐源 initialize.py:_initialize_distributed → mpu.initialize_model_parallel）。
        from megatron.core import mpu

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=1,
            )
        self.mpu = mpu
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.dp_size = mpu.get_data_parallel_world_size()

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
            self.hf_config, self.tp_size, num_layers, params_dtype=params_dtype
        )
        self.params_dtype = params_dtype
        self.num_layers = self.config.num_layers

        # 3) 建 GPTModel（对齐源 model_provider.py:177-201）。
        #    parallel_output=True → 输出 **vocab 分片** logits，log_prob 走 vocab-parallel 路径。
        #    share_embeddings_and_output_weights = tie（Qwen3-0.6B tie=True，同 V7.0 的事实）。
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        self.tie = bool(self.hf_config.tie_word_embeddings)
        model = GPTModel(
            config=self.config,
            transformer_layer_spec=get_gpt_layer_local_spec(qk_layernorm=True),
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=getattr(self.hf_config, "max_position_embeddings", 4096),
            position_embedding_type="rope",
            rotary_base=getattr(self.hf_config, "rope_theta", 10000),
            parallel_output=True,
            share_embeddings_and_output_weights=self.tie,
        )
        self.model = model.cuda()
        self.model.train()

        # 4) 载 HF 预训练权重（对齐源：训练从预训练 ckpt 起，不从随机初始化起）。
        #    源用离线 checkpoint 转换工具链（HF → mcore 格式）在训练进程外做；nano 在进程内
        #    走 megatron_to_hf.load_hf_into_megatron（登记偏离，见该模块 docstring）。
        #    **这对等价硬门是必需的**：TP=1 与 TP=2 是两次独立进程，随机初始化不可能一致，
        #    只有都从同一份 HF 权重载入，「loss/grad 等价」才是在比 TP 切分本身。
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
            f"Rank {self.rank}: MegatronTrainer ready (tp={self.tp_size}, tp_rank={self.tp_rank}, "
            f"dp={self.dp_size}, layers={self.num_layers}, tie={self.tie})"
        )

    # ---- 前向 / loss ----------------------------------------------------

    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """跑一次前向，返回 **vocab 分片** logits [B,S,V/tp]（不 gather，对齐 parallel_output=True）。

        Megatron 的 GPTModel 要 attention_mask 形状 [B,1,S,S] 且语义是 **True=屏蔽**
        （与 HF 的 True=保留相反——这是最易踩反的一处）。
        """
        b, s = input_ids.shape
        position_ids = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
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
        max（`vocab_parallel_logits -= logits_max`）。nano 是**逐样本**切 `logits[row, :L-1]`，
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
    ) -> torch.Tensor:
        """单样本 GRPO policy-gradient loss —— **与 V7 FSDPTrainer._grpo_sample_loss 同一份数学**。

        对齐 ppo_utils.py:132-135 + sum_of_sample_mean：
          ratio = exp(cur_lp - old_lp); pg = min(ratio·adv, clip(ratio,1±eps)·adv)
          loss  = -(pg·tgt_mask).sum() / clamp_min(tgt_mask.sum(), 1)
        刻意与 FSDP 路径逐字相同：V8 换的是**并行轴**，不是 loss 语义。
        """
        ratio = (cur_lp - old_lp).exp()
        pg = torch.min(
            ratio * adv,
            ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip_high) * adv,
        )
        return -(pg * tgt_mask).sum() / torch.clamp_min(tgt_mask.sum(), 1.0)

    def _microbatch_backward(
        self, samples: list[dict], global_batch_size: int
    ) -> tuple[float, int]:
        """一个微批的 forward + loss + backward（累积梯度，不 step、不 zero_grad）。

        与 V7 FSDPTrainer._microbatch_backward 结构一一对应，**唯一差别**是
        cur_lp 走 vocab-parallel 路径（logits 是分片的，不能直接 log_softmax）。

        缩放：loss * dp_size / global_batch_size（对齐源 actor.py:677）。TP 组内各 rank
        算的是**同一批样本的不同分片**（不是不同样本），故 TP 不参与这个缩放——
        这正是 TP 与 DP 的区别所在。
        """
        trainable = [s for s in samples if len(s["tokens"]) >= 2 and sum(s["loss_mask"]) > 0]
        if not trainable:
            return 0.0, 0

        max_length = max(len(s["tokens"]) for s in trainable)
        input_ids = torch.full(
            (len(trainable), max_length), self.pad_id, dtype=torch.long, device=self.device
        )
        for row, s in enumerate(trainable):
            input_ids[row, : len(s["tokens"])] = torch.tensor(
                s["tokens"], dtype=torch.long, device=self.device
            )

        logits = self._forward_logits(input_ids)  # [B,S,V/tp]，vocab 分片

        sample_losses = []
        for row, s in enumerate(trainable):
            length = len(s["tokens"])
            tgt_mask = torch.tensor(s["loss_mask"][1:], dtype=torch.float32, device=self.device)
            targets = input_ids[row, 1:length]
            # logits[:length-1] 预测 tokens[1:length]（与 V6/V7 的 [1:] 右移逐值一致）。
            # 转 fp32 + clone 在 _vocab_parallel_log_probs 内做（该 kernel 原地改输入，见其 docstring）。
            cur_lp = self._vocab_parallel_log_probs(logits[row, : length - 1], targets)

            old_log_probs = s.get("old_log_probs")
            old_lp = (
                torch.tensor(old_log_probs[1:], dtype=torch.float32, device=self.device)
                if old_log_probs is not None
                else cur_lp.detach()
            )
            adv = torch.tensor(float(s["reward"]), device=self.device)
            sample_losses.append(self._grpo_sample_loss(cur_lp, old_lp, adv, tgt_mask))

        loss = torch.stack(sample_losses).sum() * self.dp_size / global_batch_size
        loss.backward()
        return float(loss.detach()), len(trainable)

    def _finalize_model_grads(self) -> None:
        """反向之后、clip/step 之前，跨 TP 组 SUM-all-reduce **qk-layernorm** 的梯度。

        对齐源 `megatron/core/distributed/finalize_model_grads.py:403-405`
        （`_allreduce_non_tensor_model_parallel_grads` 里那条
         `elif ... or (config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name))`
         分支，reduce op = **SUM**）。源在 `finalize_model_grads` 里做，每个 step 一次
        （不是每微批），nano 同样放在 `train_batch` 的累积循环之后。

        **为什么必须有这一步（本版踩出来的正确性坑）**：q/k-layernorm 作用在**每个头**上，
        而头是被 TP 切开的 —— 各 rank 只见到自己那份头，算出的 `q_norm.weight` 梯度是
        **部分和**，加起来才是全量。其余 layernorm（`input_layernorm`/`final_layernorm`）
        不需要：它们在 TP 区域**外**，输入本就是各 rank 相同的全量激活
        （row-parallel 的 all-reduce 保证反向传回的梯度也相同），故各 rank 已持全量梯度。
        源还有一条 `config.sequence_parallel and param.sequence_parallel` 的并列条件——
        nano 未开 sequence parallel，那条恒 False，故不实现（登记偏离）。

        实测判别力：漏掉这一步时 TP=2 vs TP=1 的 grad rel_L2=1.8e-2、cosine=0.99984，
        逐参数最差的正是 `q_norm.weight`（8.5e-4）—— 而同一次 loss 差**恰为 0**。
        「前向逐值相同、梯度却有系统性偏差」正是这类漏规约的指纹。
        """
        if self.tp_size <= 1:
            return
        grads = [
            p.grad
            for n, p in self.model.named_parameters()
            if p.grad is not None
            and self.config.qk_layernorm
            and ("q_layernorm" in n or "k_layernorm" in n)
        ]
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=self.tp_group)

    def train_batch(
        self,
        samples: list[dict],
        global_batch_size: Optional[int] = None,
        microbatch_size: int = 1,
    ) -> dict:
        """梯度累积一步：zero_grad → 逐微批 backward 累积 → **finalize** → 单次 clip+step。

        与 V7 FSDPTrainer.train_batch 同构（对齐源 _train_core 的累积循环），**多一步
        `_finalize_model_grads`** —— 那是 TP 特有的跨 rank 梯度规约（FSDP 路径不需要）。
        """
        if microbatch_size < 1:
            raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
        gbs = global_batch_size if global_batch_size is not None else self.global_batch_size

        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, trained, num_microbatches = 0.0, 0, 0
        for start in range(0, len(samples), microbatch_size):
            l, n = self._microbatch_backward(samples[start : start + microbatch_size], gbs)
            loss_sum += l
            trained += n
            num_microbatches += 1

        self._finalize_model_grads()  # 对齐源 finalize_model_grads（每 step 一次，不是每微批）
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        return {
            "loss": loss_sum,
            "grad_norm": float(grad_norm),
            "trained_samples": trained,
            "num_microbatches": num_microbatches,
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

    # ---- 权重导出（V8.1）------------------------------------------------

    def to_hf_state_dict(self) -> dict:
        """Megatron 分片权重 → HF state_dict（TP-gather + 拆 qkv/gate + 改名）。

        这是 Megatron 后端**特有**的一步：FSDP 的 `save_pretrained` 天然吐 HF 格式，
        Megatron 必须先过这层转换才能喂给 SGLang。**集体操作**（含 all_gather），
        所有 TP rank 都要进；各 rank 拿到的结果相同。对齐源 `update_weight/` 的路径。
        """
        from toy_rl.trainer.megatron_to_hf import megatron_to_hf_state_dict

        return megatron_to_hf_state_dict(self.model, self.hf_config, self.tie)
