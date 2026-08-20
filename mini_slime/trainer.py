"""V3/V6/V7.3/V9: Trainer —— "谁负责训" 的角色。

对齐源项目 actor（slime/backends/fsdp_utils/actor.py）的**角色与签名**：
  - train(rollout_id, rollout_data) : actor.py:437，消费 RolloutManager 产的 train_data dict
  - update_weights()                : actor.py:725，训练后把权重同步回推理引擎

四个后端（args.train_backend）：
  - "fake"（V0-A3/离线，默认）：不 forward/backward，只消费 train_data 结构 + 产可断言指标。
  - "torch"（V6.3）：单卡真 forward/backward/optimizer 一步，loss 对齐源 ppo_utils.compute_policy_loss
    + fsdp_utils/actor.py 的 sum_of_sample_mean。单卡纯 torch（不 FSDP/offload，见 v6.md 偏离表）。
  - "fsdp"（V7.3）：FSDP2 分片后端跑在 Ray actor 里、真 torch.distributed 进程组。loss 数学与
    torch 后端**完全相同**（同一 GRPO policy gradient），只换执行后端为分片 + 跨 rank DP。
  - "megatron"（V9）：MegatronTrainer TP×PP×CP 并行后端。dist.init_process_group 在
    MegatronTrainer.__init__ 内部完成（env 由 RayTrainGroup actor init 提前设好）。
    DP-split 方式与 fsdp 一致；WeightUpdater.save_pretrained 走 to_hf_state_dict 落盘。

偏离说明（docs/decisions/{v3,v6,v7,v9}.md 偏离表）：
  - fake 后端不算真梯度（主线一聚焦系统骨架）；torch 补单卡真训练一步；fsdp 补分布式分片。
  - 关 KL/entropy（nano 取最简，语义是纯 GRPO policy gradient）。
"""

from __future__ import annotations

import time

from mini_slime.args import Args
from mini_slime.weight_sync import WeightUpdater


def _rollout_data_to_samples(rollout_data: dict) -> list[dict]:
    """列式 rollout_data → 逐样本 list（FSDPTrainer.train_batch 的入参格式）。

    RolloutManager 产列式 `{tokens:[[..]], loss_masks, rewards, rollout_log_probs}`
    （rollout_manager.py:102-108）；FSDPTrainer.train_batch 要 `list[dict{tokens,loss_mask,
    reward,old_log_probs}]`（fsdp_trainer.py:227）。这层转置放在 Trainer（backend 适配层），
    让 FSDPTrainer 不认识 nano 列格式——与 V6 Trainer 把列 dict 交给 TorchActor 一致的分层。

    rollout_log_probs 空 [] → old_log_probs=None（退化 on-policy）：_microbatch_backward 对
    None 走 ratio=1，对 [] 会 tensor([]) 维度不匹配，故空列表须归一为 None。
    """
    tokens = rollout_data["tokens"]
    loss_masks = rollout_data["loss_masks"]
    rewards = rollout_data["rewards"]
    log_probs = rollout_data.get("rollout_log_probs") or [None] * len(tokens)
    samples = []
    for i in range(len(tokens)):
        lp = log_probs[i] if log_probs[i] else None  # 空 [] 或 None → None
        samples.append(
            {"tokens": tokens[i], "loss_mask": loss_masks[i], "reward": rewards[i], "old_log_probs": lp}
        )
    return samples


class Trainer:
    """训练器角色。对齐源 actor（train / update_weights 签名）。"""

    def __init__(self, args: Args, rank: int = 0, world_size: int = 1) -> None:
        self.args = args
        self.rank = rank            # V7.3 fsdp：本 rank 号（fake/torch 恒 0）
        self.world_size = world_size  # V7.3 fsdp：DP world_size（fake/torch 恒 1）
        self._torch_actor = None       # V6 单卡 torch 后端
        self._fsdp_trainer = None      # V7.3 FSDP 分片后端
        self._megatron_trainer = None  # V9 Megatron TP×PP×CP 后端

        if args.train_backend == "torch":
            # 延迟导入：torch/transformers 只在服务器装（optional [train]）。
            from mini_slime.torch_actor import TorchActor

            self._torch_actor = TorchActor(args)
        elif args.train_backend == "fsdp":
            # FSDPTrainer.__init__ 读 actor 设的 RANK/WORLD_SIZE/LOCAL_RANK env 并 dist.init_process_group
            #（各 rank 集体 rendezvous）。故本 Trainer 必须在 actor.init() 阶段构造（见 actor_group.py）。
            from toy_rl.trainer.fsdp_trainer import FSDPTrainer

            self._fsdp_trainer = FSDPTrainer(
                model_path=args.train_model_path,
                lr=args.train_lr,
                eps_clip=args.eps_clip,
                eps_clip_high=args.eps_clip_high,
                clip_grad=args.clip_grad,
                global_batch_size=args.global_batch_size,
                train_packing=args.train_packing,  # V7.5/V7.6：opt-in packing（默认 False）
                attn_implementation=args.attn_implementation,  # V7.6：对齐源 actor.py:96
            )
            # FSDPTrainer 自己知道真实 rank（从 dist 读），以它为准。
            self.rank = self._fsdp_trainer.rank
            self.world_size = self._fsdp_trainer.world_size
        elif args.train_backend == "megatron":
            # MegatronTrainer.__init__ 读 actor 设的 RANK/WORLD_SIZE/LOCAL_RANK env 并
            # dist.init_process_group（多卡 nccl / 单卡 gloo）。
            # 与 fsdp 路径对称：dist init 在 MegatronTrainer 内部，Trainer 只构造它。
            from toy_rl.trainer.megatron_trainer import MegatronTrainer

            self._megatron_trainer = MegatronTrainer(
                model_path=args.train_model_path,
                lr=args.train_lr,
                eps_clip=args.eps_clip,
                eps_clip_high=args.eps_clip_high,
                clip_grad=args.clip_grad,
                global_batch_size=args.global_batch_size,
                tensor_model_parallel_size=args.tensor_model_parallel_size,
                pipeline_model_parallel_size=args.pipeline_model_parallel_size,
                context_parallel_size=args.context_parallel_size,
                qkv_format=args.megatron_qkv_format,
            )
            self.rank = self._megatron_trainer.rank
            self.world_size = self._megatron_trainer.world_size

        # 权重同步委托给独立角色（对齐源 actor 持有权重同步机制）。
        # torch/fsdp/megatron 后端把可训模型交给 WeightUpdater 落盘供 SGLang reload。
        train_actor = (
            self._megatron_trainer
            if self._megatron_trainer is not None
            else self._fsdp_trainer
            if self._fsdp_trainer is not None
            else self._torch_actor
        )
        use_tensor_sync = getattr(args, "use_tensor_weight_sync", False)
        self.weight_updater = WeightUpdater(
            save_path=args.weight_save_path if args.train_backend in ("torch", "fsdp", "megatron") else None,
            torch_actor=train_actor,
            rank=self.rank,  # fsdp/megatron 的 save 是集体操作，但落盘 + SGLang POST 只 rank0
            use_tensor_http_sync=use_tensor_sync,
            sglang_url=args.sglang_generate_url if use_tensor_sync else None,
        )
        if args.train_backend in ("torch", "fsdp", "megatron") and not use_tensor_sync:
            # disk reload 路径：训练后落盘的权重由 SGLang 从 disk 热重载
            self.weight_updater.generate_url = args.sglang_generate_url

    @property
    def weight_version(self) -> int:
        """当前推理引擎上的权重版本号（由 WeightUpdater 维护）。"""
        return self.weight_updater.version

    def train(self, rollout_id: int, rollout_data: dict) -> dict:
        """对齐源 actor.train(rollout_id, rollout_data)：消费 train_data dict。

        fake：不 forward/backward，只消费数据结构、产出可断言指标（+ V5 sleep 旋钮模拟耗时）。
        torch：单卡真 PPO policy-gradient 一步。
        fsdp：本 rank 训自己那片（DP split）——真分片 + 跨 rank 梯度 reduce。
        """
        rewards = rollout_data["rewards"]
        raw_rewards = rollout_data.get("raw_rewards", rewards)
        n_trainable = sum(sum(m) for m in rollout_data["loss_masks"])
        n_total = sum(rollout_data["response_lengths"])
        metrics = {
            "rollout_id": rollout_id,
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "raw_reward_mean": sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0,
            "trainable_tokens": n_trainable,
            "total_tokens": n_total,
            "grpo_group_count": int(rollout_data.get("grpo_group_count", 0)),
            "grpo_active_group_count": int(rollout_data.get("grpo_active_group_count", 0)),
        }
        # V7.4 learner_trace（opt-in）：可测相位计时 + token 分母（源锚点 infra LearnerTrace）。
        # model_tokens = 全序列长度之和；compute 相位把 forward+backward 合并计（repo 未拆 log-prob 独立
        # 前向，见 learner_metrics 偏离）。default off → 不加这些键，V0-V7.3 metric dict 不变。
        trace = self.args.learner_trace
        t_prep = 0.0

        if self._torch_actor is not None:
            # CUDA launch is asynchronous. Synchronize around the update so
            # throughput reports device work, not Python launch latency.
            if trace and self._torch_actor.device == "cuda":
                self._torch_actor.torch.cuda.synchronize()
            t0 = time.perf_counter() if trace else 0.0
            metrics.update(self._torch_actor.train_step(rollout_data))
            if trace:
                if self._torch_actor.device == "cuda":
                    self._torch_actor.torch.cuda.synchronize()
                compute = time.perf_counter() - t0
                self._fill_trace(metrics, rollout_data, batch_prep=0.0, compute=compute)
                from toy_rl.utils.flops_utils import calculate_fwd_flops

                fwd_flops = calculate_fwd_flops(
                    [len(tokens) for tokens in rollout_data["tokens"]],
                    self._torch_actor.model.config,
                )
                metrics["fwd_flops"] = fwd_flops
                metrics["tflops"] = 3 * fwd_flops / compute / 1e12 if compute > 0 else 0.0
            return metrics

        if self._megatron_trainer is not None:
            # 列 dict → 逐样本 list，再按 rank DP-split（与 fsdp 路径完全对称）。
            # MegatronTrainer 内部有 TP/PP/CP；外层只需按 DP 维 rank-split。
            t0 = time.perf_counter() if trace else 0.0
            all_samples = _rollout_data_to_samples(rollout_data)
            gbs = max(len(all_samples), 1)
            # Megatron 的 global rank 包含 TP/PP/CP，不能用它做 DP 切分。
            # TP-only（V9 Cell C）dp_size=1，因此所有 TP rank 必须拿到同一批样本，
            # 保证 collective 的序列 shape 和 microbatch 顺序一致。
            dp_rank = self._megatron_trainer.dp_rank
            dp_size = self._megatron_trainer.dp_size
            local_samples = all_samples[dp_rank::dp_size]
            t_prep = (time.perf_counter() - t0) if trace else 0.0
            t1 = time.perf_counter() if trace else 0.0
            if local_samples:
                metrics.update(self._megatron_trainer.train_batch(local_samples, global_batch_size=gbs))
            else:
                metrics.update({"loss": 0.0, "grad_norm": 0.0, "trained_samples": 0})
            if trace:
                self._fill_trace(metrics, rollout_data, batch_prep=t_prep, compute=time.perf_counter() - t1)
            return metrics

        if self._fsdp_trainer is not None:
            # 列 dict → 逐样本 list，再按 rank DP-split（对齐源 _split_train_data_by_dp：
            # 跨步分片，rank r 拿 samples[r::world_size]）。gbs = 全局样本数，缩放让累积梯度=全局批 mean。
            t0 = time.perf_counter() if trace else 0.0
            all_samples = _rollout_data_to_samples(rollout_data)
            gbs = max(len(all_samples), 1)
            local_samples = all_samples[self.rank :: self.world_size]
            t_prep = (time.perf_counter() - t0) if trace else 0.0
            t1 = time.perf_counter() if trace else 0.0
            if local_samples:
                metrics.update(self._fsdp_trainer.train_batch(local_samples, global_batch_size=gbs))
            else:
                metrics.update({"loss": 0.0, "grad_norm": 0.0, "trained_samples": 0})
            if trace:
                self._fill_trace(metrics, rollout_data, batch_prep=t_prep, compute=time.perf_counter() - t1)
            return metrics

        if self.args.fake_train_seconds > 0:
            time.sleep(self.args.fake_train_seconds)
        if trace:
            self._fill_trace(metrics, rollout_data, batch_prep=0.0, compute=0.0)
        return metrics

    def _fill_trace(self, metrics: dict, rollout_data: dict, batch_prep: float, compute: float) -> None:
        """把可测相位 + 分母塞进 metrics，供主循环组装 LearnerTrace（V7.4，opt-in）。

        只填本层可测的：batch_preparation（列→list+DP-split）、forward_backward（compute 合并）。
        optimizer/log_prob 未拆 → 0.0（登记偏离）；weight_publish 由主循环包 update_weights 补。
        rollout_policy_version 取本批共享值（None=未戳时留 None，主循环判空跳过 trace）。
        """
        versions = rollout_data.get("rollout_policy_versions") or []
        shared = {v for v in versions if v is not None}
        metrics["_trace"] = {
            "model_tokens": sum(len(t) for t in rollout_data["tokens"]),
            "batch_preparation_seconds": batch_prep,
            "forward_backward_seconds": compute,
            "rollout_policy_version": (shared.pop() if len(shared) == 1 else None),
        }

    def update_weights(self) -> None:
        """对齐源 actor.update_weights()：训练后把权重同步回推理引擎（委托 WeightUpdater）。

        fsdp：save_pretrained 是**集体操作**（各 rank gather 分片到 rank0），故每个 rank 都要进
        （RayTrainGroup.update_weights fan-out 已保证）；WeightUpdater 内部只让 rank0 落盘 + POST。
        """
        self.weight_updater.update_weights()
