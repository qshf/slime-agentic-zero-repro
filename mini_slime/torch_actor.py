"""V6.3: TorchActor —— 真 torch 训练一步（GRPO policy gradient）。

对齐源 slime/backends/fsdp_utils/actor.py + slime/utils/ppo_utils.py 的 loss 数学：
  cur_log_probs = logprob(model(tokens), tokens)          # 训练侧重算（actor.py:869-894）
  ppo_kl        = old_log_probs - cur_log_probs           # actor.py:606
  ratio         = exp(-ppo_kl) = exp(cur - old)           # ppo_utils.py:132
  adv           = reward（GRPO：广播到每个 response token，ppo_utils.py:201-208）
  pg_loss       = max(-ratio·adv, -clip(ratio,1-e,1+e)·adv)   # ppo_utils.py:133-135
  loss          = sum_of_sample_mean(pg_loss, resp_len, loss_masks)   # actor.py:622-642

写法：**一次批量前向 + advantage 广播**（minimind train_grpo.py 风格），不用逐样本 for。
  变长序列右 padding 成 [B, max_len]，attention_mask 屏蔽 pad；advantage [B,1] 广播到 [B, L-1]；
  masked-mean over 序列 → mean over batch。与逐样本循环**数值等价**（pad 位 attention/loss 均 0）。

偏离（docs/decisions/v6.md 偏离表）：
  - 单卡纯 torch（无 FSDP mesh / TP·PP·CP·EP / CPU offload）——留 V7/V8。
  - 关 KL loss（use_kl_loss=False）+ entropy_coef=0：nano 取最纯 GRPO policy gradient，语义是
    源在这两项系数为 0 时的特例。ref 模型/KL 留后续。
  - batch 组装用 **padding + attention_mask**（对齐 minimind），源用 packing 拼接（变长首尾相连、
    无 pad）。二者对可训 token 的 loss 数值等价（pad 位被 attention_mask + loss_mask 双重屏蔽）；
    packing 省显存/免 pad 浪费是吞吐优化，留 V7。
"""

from __future__ import annotations

from mini_slime.args import Args


class TorchActor:
    """持有可训模型 + 优化器，执行真 PPO policy-gradient 一步。"""

    def __init__(self, args: Args) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.train_model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.train_lr)
        # padding 需要一个 pad id：pad 位被 attention_mask + loss_mask 双重屏蔽，取值不影响 loss，
        # 但要是合法词表 id（避免越界）。优先 tokenizer.pad_token_id，回退 eos，再回退 0。
        tok = AutoTokenizer.from_pretrained(args.train_model_path, trust_remote_code=True)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def _pad_batch(self, tokens_list, loss_masks, rewards, old_log_probs_list):
        """把变长样本右 padding 成规整 batch 张量（minimind 用 tokenizer padding，这里手工 pad）。

        返回（均已右移一位对齐 targets=tokens[1:]，长度 L-1）：
          input_ids   [B, L]        pad_id 填充；attn_mask 屏蔽
          attn_mask   [B, L]        真实 token=1、pad=0
          tgt_mask    [B, L-1]      loss_mask[1:]（哪些位置计 loss；pad 位=0）
          old_lp      [B, L-1]|None 采样时 old_log_probs[1:]；无真 old 时为 None（train_step 退化 on-policy）
          adv         [B, 1]        每条 rollout 的标量 advantage，供广播到 [B, L-1]
        """
        torch = self.torch
        # 只保留 len>=2 且有可训 token 的样本（对齐旧逐样本 continue 条件）。
        keep = [i for i, t in enumerate(tokens_list) if len(t) >= 2 and sum(loss_masks[i]) > 0]
        if not keep:
            return None
        seqs = [tokens_list[i] for i in keep]
        max_len = max(len(t) for t in seqs)

        def _row(vals, length, fill):
            return vals + [fill] * (length - len(vals))

        input_ids = torch.tensor([_row(s, max_len, self.pad_id) for s in seqs], device=self.device)
        attn_mask = torch.tensor(
            [_row([1] * len(s), max_len, 0) for s in seqs], dtype=torch.long, device=self.device
        )
        # loss_mask/old_log_probs 与 tokens 等长；重算 log_prob 是 tokens[1:] 的 → 整体丢首位再 pad 到 L-1。
        tgt_mask = torch.tensor(
            [_row(loss_masks[i][1:], max_len - 1, 0) for i in keep], dtype=torch.float32, device=self.device
        )
        has_old = old_log_probs_list and all(len(old_log_probs_list[i]) == len(tokens_list[i]) for i in keep)
        if has_old:
            old_lp = torch.tensor(
                [_row(old_log_probs_list[i][1:], max_len - 1, 0.0) for i in keep], device=self.device
            )
        else:
            old_lp = None  # 无真 old_log_probs（不该发生在 V6.3 真路径）：train_step 退化 on-policy
        adv = torch.tensor([[float(rewards[i])] for i in keep], device=self.device)  # [B,1] 广播源
        return input_ids, attn_mask, tgt_mask, old_lp, adv

    def train_step(self, rollout_data: dict) -> dict:
        """真 PPO policy-gradient 一步（批量前向 + 广播），返回 {loss, grad_norm, trained_samples}。

        rewards 已经过 GRPO 组归一（custom_convert）；被 mask 掉的组 reward=0，对 loss 无贡献。
        """
        torch = self.torch
        tokens = rollout_data["tokens"]
        loss_masks = rollout_data["loss_masks"]
        rewards = rollout_data["rewards"]
        old_log_probs = rollout_data.get("rollout_log_probs")
        valid = [i for i, sequence in enumerate(tokens) if len(sequence) >= 2 and sum(loss_masks[i]) > 0]
        if not valid:
            return {"loss": 0.0, "grad_norm": 0.0, "trained_samples": 0}

        self.optimizer.zero_grad(set_to_none=True)
        total_samples = len(valid)
        microbatch_size = self.args.torch_microbatch_size or total_samples
        weighted_loss = 0.0
        for start in range(0, total_samples, microbatch_size):
            indices = valid[start : start + microbatch_size]
            batch = self._pad_batch(
                [tokens[i] for i in indices],
                [loss_masks[i] for i in indices],
                [rewards[i] for i in indices],
                [old_log_probs[i] for i in indices] if old_log_probs else None,
            )
            assert batch is not None
            input_ids, attn_mask, tgt_mask, old_lp, adv = batch
            # logits[:, :-1] predicts tokens[:, 1:].
            logits = self.model(input_ids, attention_mask=attn_mask).logits[:, :-1, :].float()
            targets = input_ids[:, 1:].unsqueeze(-1)
            cur_lp = torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)
            old_lp = cur_lp.detach() if old_lp is None else old_lp
            ratio = (cur_lp - old_lp).exp()
            pg_losses1 = ratio * adv
            pg_losses2 = ratio.clamp(1 - self.args.eps_clip, 1 + self.args.eps_clip_high) * adv
            pg_loss = torch.min(pg_losses1, pg_losses2)
            per_sample = (pg_loss * tgt_mask).sum(dim=1) / torch.clamp_min(tgt_mask.sum(dim=1), 1.0)
            loss = -per_sample.mean()
            # Accumulated gradients equal the historical mean over every valid
            # sample, while each padded activation tensor stays microbatch-sized.
            scale = len(indices) / total_samples
            (loss * scale).backward()
            weighted_loss += float(loss.detach()) * scale

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        self.optimizer.step()
        return {
            "loss": weighted_loss,
            "grad_norm": float(grad_norm),
            "trained_samples": total_samples,
        }

    def save_pretrained(self, path: str) -> None:
        """落盘当前权重供 SGLang reload（对齐源 update_weights 的 disk 路径最小版）。"""
        self.model.save_pretrained(path)
