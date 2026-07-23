"""V6.2: GRPO 组归一 —— 源 ToolOrchestra custom_convert.py 的 nano 移植。

对齐源 `agentic/ToolOrchestra/custom_convert.py`：
  1. 每 n_samples_per_prompt 个连续样本 = 同一题的多个 rollout（一组）。
  2. 组内对每个 feature 做 min-max 相对归一（cost/latency 取负——越低越好）。
  3. preference-weighted reward = Σ pref_vec[f] · normalized(f)（错误答案 reward=0）。
  4. 组内 GRPO 标准化：(r - mean) / (std + eps)，clip[-3,3]；std<0.1 的组无学习信号 → mask 掉。
  5. 按 turn 拆成独立训练样本（orchestrator 每轮一条，工具轮 loss_mask 已为 0）。

这补上 A3 登记的**最大偏离**（单样本预算归一 → 真同题多 rollout 组内归一）。

偏离（写入 docs/decisions/v6.md）：
  - nano features 存在 sample.metadata["reward_features"]（reward_func 已回填），源存 train_metadata。
  - nano 每个 sample 的 turns 存在 metadata["turns"]，与源 train_metadata["turns"] 同构。
  - 不做源的 rollout_log_probs 落 train_data 的 has_rollout_log_probs 分支复杂度——nano 直接
    按 turn 带 log_probs（若 turn 里有），保持契约简单。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_STD_THRESHOLD = 0.1
REWARD_CLIP = 3.0


def _compute_preference_rewards(group_features: list[dict], group_pref_vecs: list[dict]) -> list[float]:
    """组内 min-max 归一 + 偏好加权（对齐源 _compute_preference_rewards）。

    - 错误 rollout（correctness<0.5）reward=0。
    - 正确 rollout：reward = Σ pref_vec[f]·normalized(f)，f 含 tool_counts 各角色、accuracy、
      cost（取负）、latency（取负）；normalized 是组内 min-max。
    """
    n = len(group_features)
    if n == 0:
        return []

    all_keys: set[str] = set()
    feature_vectors: list[dict[str, float]] = []
    for feat in group_features:
        fv: dict[str, float] = {}
        for role, count in feat.get("tool_counts", {}).items():
            fv[role] = float(count)
            all_keys.add(role)
        fv["accuracy"] = feat["correctness"]
        fv["cost"] = -feat["total_cost"]
        fv["latency"] = -feat["total_latency"]
        all_keys.update(["accuracy", "cost", "latency"])
        feature_vectors.append(fv)

    feat_min = {k: min(fv.get(k, 0.0) for fv in feature_vectors) for k in all_keys}
    feat_max = {k: max(fv.get(k, 0.0) for fv in feature_vectors) for k in all_keys}

    rewards: list[float] = []
    for i, feat in enumerate(group_features):
        if feat["correctness"] < 0.5:
            rewards.append(0.0)
            continue
        pref = group_pref_vecs[i] if i < len(group_pref_vecs) else {}
        reward = 0.0
        fv = feature_vectors[i]
        for key in all_keys:
            if feat_max[key] > feat_min[key]:
                normalized = (fv.get(key, 0.0) - feat_min[key]) / (feat_max[key] - feat_min[key])
                reward += float(pref.get(key, 0.0)) * normalized
        rewards.append(reward)
    return rewards


def _grpo_normalize_and_filter(rewards: list[float], n: int) -> tuple[list[float], list[bool]]:
    """组内 GRPO 标准化 (r-mean)/(std+eps) clip[-3,3]，std<0.1 的组无信号 → mask（对齐源）。"""
    num_examples = len(rewards) // n
    normalized: list[float] = []
    keep_mask: list[bool] = []
    for g in range(num_examples):
        group = rewards[g * n : (g + 1) * n]
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        std = var ** 0.5
        has_signal = std > MIN_STD_THRESHOLD
        for r in group:
            nr = max(-REWARD_CLIP, min(REWARD_CLIP, (r - mean) / (std + 1e-6))) if has_signal else 0.0
            normalized.append(nr)
            keep_mask.append(has_signal)
    for _ in range(len(rewards) - num_examples * n):
        normalized.append(0.0)
        keep_mask.append(False)
    return normalized, keep_mask


def custom_convert(args, samples: list) -> dict:
    """Sample 列表 → train_data dict，含 GRPO 组归一 reward + 按 turn 拆样本。

    对齐源 custom_convert(args, samples)：n_samples_per_prompt 连续样本为一组。
    """
    n = getattr(args, "n_samples_per_prompt", 1)

    # 1. 提 features + pref_vec，组内算 preference reward。
    all_features = [s.metadata.get("reward_features", {"correctness": 0.0}) for s in samples]
    all_pref = [s.metadata.get("pref_vec", {}) for s in samples]

    pref_rewards: list[float] = []
    num_groups = len(samples) // n
    for g in range(num_groups):
        pref_rewards.extend(
            _compute_preference_rewards(
                all_features[g * n : (g + 1) * n], all_pref[g * n : (g + 1) * n]
            )
        )
    for j in range(len(samples) - num_groups * n):
        pref_rewards.append(all_features[num_groups * n + j].get("correctness", 0.0))

    # 2. 组内 GRPO 标准化 + 过滤。
    normalized, keep_mask = _grpo_normalize_and_filter(pref_rewards, n)

    # 3. 按 turn 拆独立训练样本（orchestrator 每轮一条）。
    tokens_list, response_lengths, loss_masks = [], [], []
    rewards, raw_rewards, log_probs_list = [], [], []
    for i, sample in enumerate(samples):
        turns = sample.metadata.get("turns")
        norm_r = normalized[i]
        should_mask = not keep_mask[i]
        if not turns:
            # 无 turns（不该发生在 A3/V6 路径）：整条按单序列处理。
            tokens_list.append(sample.tokens)
            response_lengths.append(sum(sample.loss_mask) if sample.loss_mask else len(sample.tokens))
            loss_masks.append([0] * len(sample.loss_mask) if should_mask else sample.loss_mask)
            rewards.append(norm_r)
            raw_rewards.append(pref_rewards[i])
            log_probs_list.append(sample.rollout_log_probs)
            continue
        for turn in turns:
            prompt_len = len(turn["tokens"]) - turn["response_length"]
            if prompt_len < 1 or turn["response_length"] < 1:
                continue
            lm = [0] * turn["response_length"] if should_mask else turn["loss_mask"]
            tokens_list.append(turn["tokens"])
            response_lengths.append(turn["response_length"])
            loss_masks.append([0] * prompt_len + lm)
            rewards.append(norm_r)
            raw_rewards.append(pref_rewards[i])
            log_probs_list.append(
                [0.0] * prompt_len + turn.get("rollout_log_probs", [0.0] * turn["response_length"])
            )

    return {
        "tokens": tokens_list,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "response_lengths": response_lengths,
        "rollout_log_probs": log_probs_list,
    }
