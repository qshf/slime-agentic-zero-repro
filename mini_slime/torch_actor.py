"""V6.3: TorchActor —— 真 torch 训练一步（GRPO policy gradient）。

对齐源 slime/backends/fsdp_utils/actor.py + slime/utils/ppo_utils.py 的 loss 数学：
  cur_log_probs = logprob(model(tokens), tokens)          # 训练侧重算（actor.py:869-894）
  ppo_kl        = old_log_probs - cur_log_probs           # actor.py:606
  ratio         = exp(-ppo_kl) = exp(cur - old)           # ppo_utils.py:132
  adv           = reward（GRPO：广播到每个 response token，ppo_utils.py:201-208）
  pg_loss       = max(-ratio·adv, -clip(ratio,1-e,1+e)·adv)   # ppo_utils.py:133-135
  loss          = sum_of_sample_mean(pg_loss, resp_len, loss_masks)   # actor.py:622-642

偏离（docs/decisions/v6.md 偏离表）：
  - 单卡纯 torch（无 FSDP mesh / TP·PP·CP·EP / CPU offload / dynamic packing）——留 V7/V8。
  - 关 KL loss（use_kl_loss=False）+ entropy_coef=0：nano 取最纯 GRPO policy gradient，语义是
    源在这两项系数为 0 时的特例。ref 模型/KL 留后续。
  - 逐样本前向（不做 packing 拼 batch）：nano batch 小，清晰优先；packing 是吞吐优化留 V7。
"""

from __future__ import annotations

from mini_slime.args import Args


class TorchActor:
    """持有可训模型 + 优化器，执行真 PPO policy-gradient 一步。"""

    def __init__(self, args: Args) -> None:
        import torch
        from transformers import AutoModelForCausalLM

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

    def _token_log_probs(self, tokens: list[int]):
        """重算模型对给定 token 序列每个位置的 log_prob（对齐源 get_logprob_and_entropy）。

        logits[t] 预测 token[t+1]，故 shifted_logits=logits[:-1] 对齐 targets=tokens[1:]。
        返回长度 len(tokens)-1 的 log_prob 向量（第 i 个是 tokens[i+1] 的 log_prob）。
        """
        torch = self.torch
        ids = torch.tensor([tokens], device=self.device)
        logits = self.model(ids).logits.squeeze(0).float()   # [seq, vocab]
        shifted = logits[:-1, :]                              # [seq-1, vocab]
        targets = ids.squeeze(0)[1:]                          # [seq-1]
        log_probs = torch.log_softmax(shifted, dim=-1)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [seq-1]

    def train_step(self, rollout_data: dict) -> dict:
        """真 PPO policy-gradient 一步，返回 {loss, grad_norm}。

        逐样本算 pg_loss 的 sample-mean（只在 loss_mask=1 处），累加求和后 backward。
        rewards 已经过 GRPO 组归一（custom_convert）；被 mask 掉的组 reward=0，对 loss 无贡献。
        """
        torch = self.torch
        tokens_list = rollout_data["tokens"]
        loss_masks = rollout_data["loss_masks"]
        rewards = rollout_data["rewards"]
        old_log_probs_list = rollout_data.get("rollout_log_probs")

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device)
        n_samples = 0

        for i, tokens in enumerate(tokens_list):
            if len(tokens) < 2 or sum(loss_masks[i]) == 0:
                continue
            # loss_mask/old_log_probs 与 tokens 等长；重算 log_prob 是 tokens[1:] 的，故整体右移一位对齐。
            mask = torch.tensor(loss_masks[i][1:], dtype=torch.float32, device=self.device)
            cur_log_probs = self._token_log_probs(tokens)     # [seq-1]

            if old_log_probs_list and len(old_log_probs_list[i]) == len(tokens):
                old_log_probs = torch.tensor(old_log_probs_list[i][1:], device=self.device)
            else:
                # 无真 old_log_probs（不该发生在 V6.3 真路径）：退化为 on-policy（ratio=1）。
                old_log_probs = cur_log_probs.detach()

            ppo_kl = old_log_probs - cur_log_probs            # 源 actor.py:606
            ratio = (-ppo_kl).exp()                           # exp(cur-old)，源 ppo_utils.py:132
            adv = float(rewards[i])                           # GRPO：标量 reward 广播到每 token
            pg_losses1 = -ratio * adv
            pg_losses2 = -ratio.clamp(1 - self.args.eps_clip, 1 + self.args.eps_clip_high) * adv
            pg_loss = torch.maximum(pg_losses1, pg_losses2)   # 源 ppo_utils.py:135

            # sample-mean：只在 loss_mask=1 处（源 sum_of_sample_mean，actor.py:897-914）。
            sample_loss = (pg_loss * mask).sum() / torch.clamp_min(mask.sum(), 1.0)
            total_loss = total_loss + sample_loss
            n_samples += 1

        if n_samples == 0:
            return {"loss": 0.0, "grad_norm": 0.0, "trained_samples": 0}

        loss = total_loss / n_samples
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "trained_samples": n_samples,
        }

    def save_pretrained(self, path: str) -> None:
        """落盘当前权重供 SGLang reload（对齐源 update_weights 的 disk 路径最小版）。"""
        self.model.save_pretrained(path)
