"""V7.0/V7.2: FSDPTrainer —— FSDP2 后端的真训练一步（含梯度累积）。

对齐源 slime/backends/fsdp_utils/actor.py：
  - init      ← actor.py:48-154：setup mesh → 建模型(tie 分支) → apply_fsdp2 → rank0 广播 → AdamW
  - train_step← actor.py:551-722：forward → log_probs → GRPO policy loss → 缩放 → backward → step
  - loss 数学 ← actor.py:606 compute_policy_loss + actor.py:897 sum_of_sample_mean（与 V6 TorchActor 同一公式）

与 V6 mini_slime/torch_actor.py 的关系：**loss 公式完全相同**（同一 GRPO policy gradient），
差别只在 **执行后端**——V6 是单卡整模型 + padding batch，V7 是 FSDP 分片 + 单样本前向
（packing 留 V7.1）。这样对照能看清"同一份 loss 数学，换到分布式后端上跑"。

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

偏离源项目（V7.0/V7.2，登记见 docs/decisions/v7.md）：
  - 无 sequence packing（每样本即一个微批，留 V7.1——需 flash-attn varlen，5090 未装）；无 Ray（留 V7.3）
  - cpu_offload=False（显存够）；无 ref model / KL / entropy（继承 V6 --no-ref，取最纯 GRPO）
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
    ):
        self.model_path = model_path
        self.lr = lr
        self.eps_clip = eps_clip
        self.eps_clip_high = eps_clip_high
        self.clip_grad = clip_grad
        self.global_batch_size = global_batch_size
        self.fp16 = fp16
        self.cpu_offload = cpu_offload

        # torchrun 已设好 RANK/WORLD_SIZE/LOCAL_RANK 环境变量；init_process_group 读它们。
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
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        model.train()

        # 4) FSDP wrap 前先抓全量 state_dict（对齐 actor.py:101；meta rank 上此 dict 是空/meta，
        #    真权重只在 rank0）。
        full_state = model.state_dict()

        # 5) apply_fsdp2（对齐 actor.py:103）。
        logger.info(f"Rank {self.rank}: apply_fsdp2")
        model = apply_fsdp2(model, mesh=self.mesh, cpu_offload=cpu_offload, fp16=fp16)

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
        logger.info(f"Rank {self.rank}: FSDPTrainer ready (tie={tie}, dp_size={self.dp_size})")

    def _micro_backward(
        self,
        tokens: list[int],
        loss_mask: list[int],
        reward: float,
        old_log_probs: Optional[list[float]],
        global_batch_size: int,
    ) -> float:
        """单个微批的 forward + loss + backward（累积梯度，**不 step、不 zero_grad**）。

        对齐源 actor.py:551-678 的 _train_step 主体（去掉 packing / step 门控）。
        loss 数学与 V6 TorchActor 完全一致：
          cur_lp = logprob(model(tokens)[:-1], tokens[1:])   # actor.py:554-562
          ratio  = exp(cur_lp - old_lp)                       # ppo_utils.py:132
          pg     = min(ratio·adv, clip(ratio,1±e)·adv)        # ppo_utils.py:133-135
          loss   = -masked_mean(pg)                           # sum_of_sample_mean, actor.py:911
          loss   = loss * dp_size / global_batch_size         # actor.py:677（累积缩放）

        缩放的意义（V7.2 教学核心）：每微批乘 dp_size/gbs，FSDP 反向自动按 dp_size 平均梯度，
        两者相抵 → 累积 gbs 个微批的梯度 == 对全局批取「逐样本 mean 之和 / gbs」的单次梯度。
        """
        if len(tokens) < 2 or sum(loss_mask) == 0:
            return 0.0

        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,L]
        tgt_mask = torch.tensor(loss_mask[1:], dtype=torch.float32, device=self.device)       # [L-1]
        adv = torch.tensor(float(reward), device=self.device)                                 # 标量广播

        old_lp = (
            torch.tensor(old_log_probs[1:], dtype=torch.float32, device=self.device)
            if old_log_probs is not None
            else None
        )

        # Forward（对齐 actor.py:554）：logits[:-1] 预测 tokens[1:]。
        logits = self.model(input_ids).logits.squeeze(0)[:-1, :].float()  # [L-1, V]
        targets = input_ids.squeeze(0)[1:]                                 # [L-1]
        cur_lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [L-1]

        # ratio = exp(cur - old)；无真 old 时退化 on-policy（ratio=1）。对齐 ppo_utils.py:132。
        old_lp = cur_lp.detach() if old_lp is None else old_lp
        ratio = (cur_lp - old_lp).exp()
        # GRPO：标量 adv 广播到每个 token。PPO clip（此时还是要最大化的 J）。ppo_utils.py:133-135。
        pg = torch.min(ratio * adv, ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip_high) * adv)  # [L-1]

        # masked mean over 可训 token（sum_of_sample_mean 单样本版，actor.py:911）。
        masked = (pg * tgt_mask).sum() / torch.clamp_min(tgt_mask.sum(), 1.0)
        loss = -masked  # optimizer 是 minimize，加负号把"最大化 J"转"最小化 L"。

        # 梯度累积缩放（对齐 actor.py:677）：除以 global_batch_size、乘 dp_size 抵消 FSDP dp 平均。
        loss = loss * self.dp_size / global_batch_size
        loss.backward()  # 累积到 .grad，不清零
        return float(loss.detach())

    def train_batch(
        self,
        samples: list[dict],
        global_batch_size: Optional[int] = None,
    ) -> dict:
        """梯度累积一步：zero_grad → 逐微批 backward 累积 → 单次 clip+step。

        对齐源 actor.py:521-532 + 684-693 的 _train_core 累积循环：本地按微批遍历、
        梯度累积到 grad_accum 边界才 optimizer.step()。nano V7.2 每样本即一个微批
        （无 packing，留 V7.1），一个 train_batch 即一个累积窗口 → 末尾 step 一次。

        samples: [{tokens, loss_mask, reward, old_log_probs?}, ...]（本 rank 的本地微批）。
        global_batch_size: 全局批大小（用于缩放）；默认取 self.global_batch_size。
        """
        gbs = global_batch_size if global_batch_size is not None else self.global_batch_size

        self.optimizer.zero_grad(set_to_none=True)  # 累积窗口开始，清零一次（actor.py:523）
        loss_sum, trained = 0.0, 0
        for s in samples:
            l = self._micro_backward(
                s["tokens"], s["loss_mask"], s["reward"], s.get("old_log_probs"), gbs
            )
            loss_sum += l
            trained += 1 if l != 0.0 else 0

        # 累积窗口结束：clip + step 一次（actor.py:686-690）。
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        return {
            "loss": loss_sum,
            "grad_norm": float(grad_norm),
            "trained_samples": trained,
            "num_microbatches": len(samples),
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
