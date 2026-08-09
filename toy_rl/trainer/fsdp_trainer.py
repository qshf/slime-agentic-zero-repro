"""V7.0/V7.2/V7.5: FSDPTrainer —— FSDP2 后端的真训练一步（含梯度累积 + 可选 packing）。

对齐源 slime/backends/fsdp_utils/actor.py：
  - init      ← actor.py:48-154：setup mesh → 建模型(tie 分支) → apply_fsdp2 → rank0 广播 → AdamW
  - train_step← actor.py:551-722：forward → log_probs → GRPO policy loss → 缩放 → backward → step
  - loss 数学 ← actor.py:606 compute_policy_loss + actor.py:897 sum_of_sample_mean（与 V6 TorchActor 同一公式）

与 V6 mini_slime/torch_actor.py 的关系：**loss 公式完全相同**（同一 GRPO policy gradient），
差别只在 **执行后端**——V6 是单卡整模型 + padding batch，V7 是 FSDP 分片 + 单样本前向
（V7.5 起补 packing）。这样对照能看清"同一份 loss 数学，换到分布式后端上跑"。

V7.0 最小切片（单样本，2 卡真分片）：
  - 忠实 tie 处理（Qwen3-0.6B tie=True → 全 rank CPU 全量建模，见 fsdp_utils.get_init_weight_context）
  - 单样本前向 + 后向（暂不 packing）
  - loss 缩放对齐源：loss * dp_size / global_batch_size（actor.py:677）

V7.2 增量（梯度累积）：
  - train_batch：zero_grad 一次 → 逐微批 backward 累积梯度 → 累积窗口末尾单次 clip+step
    （对齐源 _train_core actor.py:521-532 + _train_step 的 step 门控 actor.py:684-690）
  - 每微批 loss 乘 dp_size/global_batch_size：与 FSDP 反向的 dp 平均相抵，
    累积 N 微批 == 全局批「逐样本 mean 之和 / gbs」的单次梯度（V7.2 教学核心）
  - train_step 保留为 gbs=1 单微批的 train_batch 薄封装（V7.0 入口兼容）

V7.5 增量（sequence packing，opt-in train_packing）：
  - _grpo_sample_loss：把 GRPO 单样本 loss 抽成 padding/packing 共用方法（各路 loss 数学 bit-identical）。
  - _packed_backward：多样本 flat 成 [1,T] 单微批 + 显式块对角 4D mask 隔离段间注意力
    （源靠 flash-attn varlen，当时 5090 无 → 物化 mask，见 toy_rl/trainer/data_packing）。
  - 与 padding 路径 loss/grad 逐值等价（scripts/test_v7.5_packing 验：fp32 loss 2.4e-7 / cosine 0.99999999）；
    退休 v7.md 偏离 #1、新增偏离 #7（显式 O(T²) mask vs varlen O(T)）。

V7.6 增量（FA2 varlen packing，退休偏离 #7）：
  - train_packing 三态化：False(padding) / "mask"(V7.5 块对角) / "fa2"(FA2 varlen，**对齐源**)。
  - "fa2" 路径传 attention_mask=None + 每段 reset 的 position_ids，transformers 5.x 由此反推
    cu_seqlens 并 dispatch 到 flash_attn_varlen_func —— 与源 actor.py:801-812 逐字对应。
  - 新增 attn_implementation 参数（对齐源 actor.py:96 / arguments.py:29）。
  - 三重硬断言（后端==FA2、flash-attn 可用、position_ids 真被判成 packed）：fa2 路径的
    attention_mask=None 一旦落到非 FA2 后端就是**静默**满因果注意力、段间串扰、loss 错。
  - 验收改用 negative control（FA2 内核拒收 fp32 → V7.5 那套 fp32 精确门在此路径不可用），
    见 scripts/test_v7.6_fa2_packing.py 与 docs/decisions/v7.6.md 偏离 #8。

偏离源项目（登记见 docs/decisions/v7.md）：
  - V7.5 前无 packing；V7.5 用显式块对角 mask 替代 varlen（无 flash-attn/TE），隔离等价、吞吐部分等价。
    **V7.6 起 "fa2" 模式回到源写法，#7 退休**；"mask" 降为无 FA2 环境的 fallback + fp32 数值对照。
  - cpu_offload=False（显存够）；无 ref model / KL / entropy（继承 V6 --no-ref，取最纯 GRPO）；无 Ray（V7.3 接）。
  - 一个 train_batch 即一个累积窗口（源用 grad_accum 边界列表支持窗口内多次 step，nano 简化为窗口=批）
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from toy_rl.utils.fsdp_utils import (
    apply_fsdp2,
    get_init_weight_context,
    load_full_state_dict_fsdp,
    setup_device_mesh,
)

logger = logging.getLogger(__name__)


def _normalize_packing_mode(train_packing: "bool | str") -> Optional[str]:
    """三态 train_packing → None(padding) / "mask" / "fa2"（V7.6）。

    V7.5 用 bool（True=块对角 mask）；V7.6 加 FA2 varlen 后扩成字符串，故 True 归一到 "mask"
    保持旧调用点语义不变（零回归）。
    """
    if train_packing is False or train_packing is None or train_packing == "":
        return None
    if train_packing is True:
        return "mask"  # V7.5 兼容：布尔 True == 显式块对角 mask
    mode = str(train_packing).lower()
    if mode not in ("mask", "fa2"):
        raise ValueError(f"train_packing 只支持 False/'mask'/'fa2'，收到 {train_packing!r}")
    return mode


def _assert_varlen_will_engage(position_ids: torch.Tensor, num_segments: int) -> None:
    """断言 transformers 真会把这条 pack 判成 packed（→ 走 flash_attn_varlen_func）。

    **为什么不自己写单调性判据**：varlen 是否启用完全由 transformers 的
    `_is_packed_sequence`（modeling_flash_attention_utils.py:528-541）说了算——它检查
    `batch==1 且 position_ids 非「从 min 开始逐 1 递增」`。自己另写一套"看着等价"的判据，
    一旦与上游不一致就会出现**断言过了但 varlen 没启用**的最坏情况（attention_mask=None
    落到满因果注意力 → 段间静默串扰 → loss 悄悄算错）。故直接调上游那个函数，判据永远同源；
    上游不可导入时才退回**镜像其语义**的等价实现（同样是 diff-based，不是"严格递增"）。

    **长度=1 的段不会误判**（本函数与上游共同保证）：pack 的 position_ids 每段从 0 重数，
    故段边界处 diff = 0 - (L_prev - 1) = 1 - L_prev。它等于 1 仅当 L_prev == 0（空段，
    pack_sequences 已过滤 len<2，更不会产生空段）。L_prev == 1 时 diff = 0 ≠ 1 → 照样被检出。
    真正**不该**触发断言的只有 num_segments == 1（整条 pack 就是一条序列，position_ids
    本就是 0..T-1 单调，此时无段可隔离、退化成普通因果前向是正确行为）——故单段直接放行。
    """
    if num_segments <= 1:
        return  # 单段：无需隔离，position_ids 本就单调，非 packed 是正确语义。

    try:
        from transformers.modeling_flash_attention_utils import _is_packed_sequence

        detected = bool(_is_packed_sequence(position_ids, batch_size=position_ids.shape[0]))
    except ImportError:  # 上游改名/换版本：镜像其语义（diff!=1 处即段边界）。
        first = position_ids[:, :1] - 1
        detected = bool((torch.diff(position_ids, prepend=first, dim=-1) != 1).any())

    assert detected, (
        f"position_ids 未被 transformers 判成 packed（num_segments={num_segments}），"
        "varlen 隔离不会生效、attention_mask=None 会退化成满因果注意力 → 段间串扰、loss 算错。"
        "检查 pack_sequences 是否按段 reset 了 position_ids。"
    )


class FSDPTrainer:
    """FSDP2 后端训练器。对齐源 actor.py 的 init + _train_step。"""

    def __init__(
        self,
        model_path: str,
        lr: float = 1e-6,
        eps_clip: float = 0.2,
        eps_clip_high: float = 0.2,
        clip_grad: float = 1.0,
        global_batch_size: int = 1,
        fp16: bool = False,
        cpu_offload: bool = False,
        train_packing: "bool | str" = False,
        compute_dtype: Optional[torch.dtype] = None,
        attn_implementation: str = "sdpa",
    ):
        self.model_path = model_path
        self.lr = lr
        self.eps_clip = eps_clip
        self.eps_clip_high = eps_clip_high
        self.clip_grad = clip_grad
        self.global_batch_size = global_batch_size
        self.fp16 = fp16
        self.cpu_offload = cpu_offload
        # V7.5/V7.6：opt-in sequence packing（默认 off → V7.0/V7.2 padding 路径 byte-for-byte 不动）。
        # 三态归一：False/""→None(padding)、True→"mask"（V7.5 兼容）、"mask"/"fa2" 原样。
        self.packing_mode = _normalize_packing_mode(train_packing)
        self.train_packing = self.packing_mode is not None  # V7.5 布尔语义保留给外部只读
        self.attn_implementation = attn_implementation
        # V7.5：forward 计算 dtype = FSDP MixedPrecision 的 param_dtype（默认 fp16/bf16 由 fp16 定；
        # compute_dtype=torch.float32 关混合精度做 fp32 等价副证）。packing 的显式 mask dtype 须匹配它。
        self.compute_dtype = compute_dtype or (torch.float16 if fp16 else torch.bfloat16)

        # 启动器已在每个训练进程内设好 RANK/WORLD_SIZE/LOCAL_RANK：
        #   - V7.0/V7.2 torchrun 路径由 torchrun 设置；
        #   - V7.3 Ray 路径由 TrainRayActor 在各 actor 进程内设置。
        # init_process_group 读取这些 env 完成 rendezvous。
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()  # = dp_size（1D mesh）
        self.dp_size = self.world_size
        # 每个进程绑一张卡（对齐源 LOCAL_RANK 语义）。
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        logger.info(f"Rank {self.rank}/{self.world_size}: init FSDPTrainer")

        # 1) 1D dp mesh（对齐 actor.py:57）。
        self.mesh = setup_device_mesh(self.world_size)

        # 2) 顺序 load config/tokenizer（rank-by-rank barrier，避免 HF cache 竞争，对齐 actor.py:81-88）。
        for i in range(self.world_size):
            if i == self.rank:
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            dist.barrier()

        # 3) 建模型：按 tie 选设备上下文（tie=True 全 rank CPU 全量；否则 rank0 CPU + 其余 meta）。
        #    对齐 actor.py:90-97。
        tie = self.config.tie_word_embeddings
        init_context = get_init_weight_context(tie, self.rank)
        with init_context():
            # attn_implementation 对齐源 actor.py:96（源默认 flash_attention_2）。
            # 后端决定 packing 能用哪条隔离路径："fa2" 要 flash_attention_2（varlen 内核隔离），
            # "mask" 要 eager/sdpa（4D mask 只在这两个后端被当加性 bias）——两者都在
            # _packed_backward 里硬断言，避免静默走成满因果注意力。
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
            )
        model.train()

        # 4) FSDP wrap 前先抓全量 state_dict（对齐 actor.py:101；meta rank 上此 dict 是空/meta，
        #    真权重只在 rank0）。
        full_state = model.state_dict()

        # 5) apply_fsdp2（对齐 actor.py:103）。
        logger.info(f"Rank {self.rank}: apply_fsdp2")
        model = apply_fsdp2(model, mesh=self.mesh, cpu_offload=cpu_offload, fp16=fp16, param_dtype=self.compute_dtype)

        # 6) rank0 广播全量权重进各 rank 分片（对齐 actor.py:105-107）。
        model = load_full_state_dict_fsdp(model, full_state, cpu_offload=cpu_offload)
        self.model = model

        # 7) AdamW（对齐 actor.py:114-121）。
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )
        logger.info(f"Rank {self.rank}: FSDPTrainer ready (tie={tie}, dp_size={self.dp_size}, "
                    f"packing={self.packing_mode or 'off'}, attn={self.attn_implementation})")

    def _grpo_sample_loss(
        self,
        cur_lp: torch.Tensor,
        old_lp: torch.Tensor,
        adv: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """单样本 GRPO policy-gradient loss（padding 与 packing 两路共用，V7.5）。

        对齐 ppo_utils.py:132-135 + sum_of_sample_mean（actor.py:911）：
          ratio = exp(cur_lp - old_lp)
          pg    = min(ratio·adv, clip(ratio,1±eps)·adv)      # GRPO 标量 adv 广播到每 token
          loss  = -(pg·tgt_mask).sum() / clamp_min(tgt_mask.sum(), 1)
        抽成共用方法，让 padding/packing 两条前向路径的**loss 数学 bit-identical**——V7.5 等价
        断言（loss/grad max_abs<TOL）成立的根本保证，也让两路更清晰（唯一差别只在"怎么算出 cur_lp"）。
        """
        ratio = (cur_lp - old_lp).exp()  # 对齐 ppo_utils.py:132。
        pg = torch.min(
            ratio * adv,
            ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip_high) * adv,
        )
        return -(pg * tgt_mask).sum() / torch.clamp_min(tgt_mask.sum(), 1.0)

    def _microbatch_backward(
        self,
        samples: list[dict],
        global_batch_size: int,
    ) -> tuple[float, int]:
        """一个多样本微批的 forward + loss + backward（累积梯度，**不 step、不 zero_grad**）。

        对齐源 actor.py:551-678 的 _train_step 主体（去掉 packing / step 门控）。
        loss 数学与 V6 TorchActor 完全一致：
          cur_lp = logprob(model(tokens)[:-1], tokens[1:])   # actor.py:554-562
          ratio  = exp(cur_lp - old_lp)                       # ppo_utils.py:132
          pg     = min(ratio·adv, clip(ratio,1±e)·adv)        # ppo_utils.py:133-135
          loss   = -masked_mean(pg)                           # sum_of_sample_mean, actor.py:911
          loss   = loss * dp_size / global_batch_size         # actor.py:677（累积缩放）

        缩放的意义（V7.2 教学核心）：每个微批的逐样本 loss 先求和，再乘 dp_size/gbs；
        FSDP 反向自动按 dp_size 平均梯度，两者相抵 → 累积所有微批的梯度等于全局批
        的「逐样本 mean 之和 / gbs」。

        源项目在 packing 后以一个 packed microbatch 做前向；nano 尚未实现 varlen packing，
        所以此处按本微批的最长序列做 padding。两者的逐样本 loss / 累积语义等价，吞吐优化留 V7.1。
        """
        trainable_samples = [
            s for s in samples if len(s["tokens"]) >= 2 and sum(s["loss_mask"]) > 0
        ]
        if not trainable_samples:
            return 0.0, 0

        max_length = max(len(s["tokens"]) for s in trainable_samples)
        input_ids = torch.full(
            (len(trainable_samples), max_length), self.pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, s in enumerate(trainable_samples):
            length = len(s["tokens"])
            input_ids[row, :length] = torch.tensor(s["tokens"], dtype=torch.long, device=self.device)
            attention_mask[row, :length] = 1

        # Forward（对齐 actor.py:554）：每行 logits[:-1] 预测对应行 tokens[1:]。
        logits = self.model(input_ids, attention_mask=attention_mask).logits.float()
        sample_losses = []
        for row, s in enumerate(trainable_samples):
            length = len(s["tokens"])
            tgt_mask = torch.tensor(s["loss_mask"][1:], dtype=torch.float32, device=self.device)
            targets = input_ids[row, 1:length]
            cur_lp = torch.log_softmax(logits[row, : length - 1], dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)

            old_log_probs = s.get("old_log_probs")
            old_lp = (
                torch.tensor(old_log_probs[1:], dtype=torch.float32, device=self.device)
                if old_log_probs is not None
                else cur_lp.detach()
            )
            adv = torch.tensor(float(s["reward"]), device=self.device)
            # GRPO 单样本 loss（padding/packing 共用同一数学，V7.5）。
            sample_losses.append(self._grpo_sample_loss(cur_lp, old_lp, adv, tgt_mask))

        # 梯度累积缩放（对齐 actor.py:677）：本地先乘 dp_size / global_batch_size。
        # 注意 FSDP2 的 `fully_shard(...)` 会在 loss.backward() 的 autograd hook 里做
        # DP 维梯度 reduce-scatter/平均；因此最终全局缩放是：
        #   local_sum * dp_size / global_batch_size / dp_size = local_sum / global_batch_size
        # 若这里不乘 dp_size，FSDP 再平均一次后会变成 1 / (global_batch_size * dp_size)。
        loss = torch.stack(sample_losses).sum() * self.dp_size / global_batch_size
        loss.backward()  # 累积到 .grad，不清零
        return float(loss.detach()), len(trainable_samples)

    def _micro_backward(
        self,
        tokens: list[int],
        loss_mask: list[int],
        reward: float,
        old_log_probs: Optional[list[float]],
        global_batch_size: int,
    ) -> float:
        """单样本兼容入口，委托给多样本 `_microbatch_backward()`。"""
        loss, _ = self._microbatch_backward(
            [{"tokens": tokens, "loss_mask": loss_mask, "reward": reward, "old_log_probs": old_log_probs}],
            global_batch_size,
        )
        return loss

    def _packed_backward(
        self,
        samples: list[dict],
        global_batch_size: int,
    ) -> tuple[float, int]:
        """一个 flat pack 的 forward + loss + backward（V7.5/V7.6，packing 路径；累积梯度、不 step）。

        与 `_microbatch_backward`（padding 路径）**唯一差别** = "怎么算 cur_lp"：
          - padding：每样本一行 [B,L]、按行 2D mask、逐行 logits[:len-1]。
          - packing：所有样本 flat 成一条 [1,T]、position_ids 每段 reset，段间隔离二选一：
              * "fa2"（V7.6，**对齐源**）：attention_mask=None，transformers 从 reset 的
                position_ids 反推 cu_seqlens → flash_attn_varlen_func 在内核里跳过跨段计算（O(N)）。
              * "mask"（V7.5，fallback）：显式物化 [1,1,T,T] 块对角因果 mask（O(T²)），
                无 FA2 环境可用，且是唯一能做 **fp32** 精确等价证明的路径（FA2 只收 fp16/bf16）。
        loss 数学（GRPO ratio/clip/masked-mean/dp_size 缩放）走**同一** `_grpo_sample_loss`，
        故三路（padding / mask / fa2）loss/grad 等价（V7.5/V7.6 等价断言的根本）。

        源对照（slime actor.py:801-812 _get_model_inputs_args）：input_ids=[1,T]、position_ids=[1,T]、
        attention_mask=None（靠 varlen 内核隔离）——**"fa2" 模式与之逐字对应**；"mask" 是 V7.5 在
        无 FA2 硬件下的忠实替代（登记偏离 #7，V7.6 起降为 fallback）。
        """
        from toy_rl.trainer.data_packing import (
            build_block_diagonal_causal_mask,
            pack_sequences,
        )

        pack = pack_sequences(samples)
        if pack is None:  # 无可训练样本，与 _microbatch_backward 空返回对称。
            return 0.0, 0

        cu = pack["cu_seqlens"]
        total = cu[-1]
        num_segments = len(cu) - 1
        input_ids = torch.tensor(pack["tokens"], dtype=torch.long, device=self.device)[None]  # [1,T]
        position_ids = torch.tensor(pack["position_ids"], dtype=torch.long, device=self.device)[None]

        attn_impl = getattr(self.model.config, "_attn_implementation", "eager")
        if self.packing_mode == "fa2":
            # 三重硬断言：fa2 路径传 attention_mask=None，任一环节没到位都会**静默**退化成
            # 满因果注意力（段间串扰、loss 错）——本版最危险的失败模式，宁可炸也不静默算错。
            assert attn_impl == "flash_attention_2", (
                f"train_packing='fa2' 要求 FA2 后端，当前 _attn_implementation={attn_impl}"
                "（attention_mask=None 落到 eager/sdpa 会变成满因果注意力 → 段间串扰）"
            )
            from transformers.utils import is_flash_attn_2_available

            assert is_flash_attn_2_available(), (
                "flash-attn 不可用（未安装/版本不符）——attention_mask=None 会退化成满因果注意力。"
                "容器内需先 `pip uninstall -y flash-attn-4` 再装 FA2 wheel，见 docs/ops/server-env-bench5090.md"
            )
            _assert_varlen_will_engage(position_ids, num_segments)
            # 对齐源 actor.py:807：attention_mask=None，段边界信息全在 reset 的 position_ids 里。
            attn_mask = None
        else:
            # V7.5 "mask"：4D 显式 mask 仅在 eager/sdpa 被当加性 bias（flash/flex 会无视 → 串扰）。
            assert attn_impl in ("eager", "sdpa"), (
                f"packing 的显式块对角 mask 只在 eager/sdpa 被 honor，当前后端={attn_impl}"
                "（flash/flex 会无视 4D mask、样本间串扰 → loss 错）"
            )
            # mask dtype 须匹配激活（= FSDP MixedPrecision param_dtype = self.compute_dtype）。
            attn_mask = build_block_diagonal_causal_mask(cu, total, self.device, self.compute_dtype)

        logits = self.model(
            input_ids, position_ids=position_ids, attention_mask=attn_mask
        ).logits[0].float()  # [T,V]

        old_lp_flat = pack["old_log_probs"]  # flat 长 T 或 None
        sample_losses = []
        for seg in range(num_segments):
            start, end = cu[seg], cu[seg + 1]
            # 段内因果：logits[start:end-1] 预测 tokens[start+1:end]（边界 token 对下一段首 token
            # 的预测**从不形成** → 跨段 target 不泄漏，与 padding 路径 [1:] 右移逐值一致）。
            seg_mask = pack["loss_masks"][start:end]
            tgt_mask = torch.tensor(seg_mask[1:], dtype=torch.float32, device=self.device)
            targets = input_ids[0, start + 1 : end]
            cur_lp = torch.log_softmax(logits[start : end - 1], dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)

            if old_lp_flat is not None:
                old_lp = torch.tensor(
                    old_lp_flat[start + 1 : end], dtype=torch.float32, device=self.device
                )
            else:
                old_lp = cur_lp.detach()
            adv = torch.tensor(float(pack["rewards"][seg]), device=self.device)
            sample_losses.append(self._grpo_sample_loss(cur_lp, old_lp, adv, tgt_mask))

        # 缩放与 padding 路径完全相同（对齐 actor.py:677；FSDP dp 平均相抵，见 _microbatch_backward）。
        loss = torch.stack(sample_losses).sum() * self.dp_size / global_batch_size
        loss.backward()
        return float(loss.detach()), num_segments

    def train_batch(
        self,
        samples: list[dict],
        global_batch_size: Optional[int] = None,
        microbatch_size: int = 1,
    ) -> dict:
        """梯度累积一步：zero_grad → 逐微批 backward 累积 → 单次 clip+step。

        对齐源 actor.py:521-532 + 684-693 的 _train_core 累积循环：本地按微批遍历、
        梯度累积到 grad_accum 边界才 optimizer.step()。nano 的一个 train_batch 即一个累积窗口，
        `microbatch_size` 个样本组成一个 padding 微批，窗口末尾只 step 一次。

        samples: [{tokens, loss_mask, reward, old_log_probs?}, ...]（本 rank 的本地样本）。
        global_batch_size: 全局批大小（用于缩放）；默认取 self.global_batch_size。
        microbatch_size: 每个微批的样本数；默认 1，保持 V7.0 单样本入口兼容。

        V7.5/V7.6：`self.packing_mode` → 每个微批走 packing 前向（"fa2"=varlen / "mask"=块对角），
        None 则走 padding。三路 loss 数学一致（_grpo_sample_loss），只换"怎么算 cur_lp"；默认 padding 不变。
        """
        if microbatch_size < 1:
            raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
        gbs = global_batch_size if global_batch_size is not None else self.global_batch_size
        backward = self._packed_backward if self.packing_mode else self._microbatch_backward

        self.optimizer.zero_grad(set_to_none=True)  # 累积窗口开始，清零一次（actor.py:523）
        loss_sum, trained = 0.0, 0
        num_microbatches = 0
        for start in range(0, len(samples), microbatch_size):
            l, trained_in_microbatch = backward(
                samples[start:start + microbatch_size], gbs
            )
            loss_sum += l
            trained += trained_in_microbatch
            num_microbatches += 1

        # 累积窗口结束：clip + step 一次（actor.py:686-690）。
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
        """单样本一步（V7.0 兼容入口）= global_batch_size=1 的单微批 train_batch。"""
        return self.train_batch(
            [{"tokens": tokens, "loss_mask": loss_mask, "reward": reward, "old_log_probs": old_log_probs}],
            global_batch_size=1,
        )

    def save_pretrained(self, path: str) -> None:
        """聚合分片权重到 rank0 落盘（供 SGLang disk reload）。对齐源 checkpoint save 的最小版。

        FSDP2 下各 rank 只持分片 → get_model_state_dict(full_state_dict=True) 把它聚合成
        rank0 的完整 state_dict，再用 HF save_pretrained 写盘（V7.3 权重同步用）。
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        if self.rank == 0:
            logger.info(f"Rank 0: saving model to {path}")
            unwrapped = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.fp16 else torch.bfloat16,
            )
            unwrapped.load_state_dict(state_dict)
            unwrapped.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        dist.barrier()
