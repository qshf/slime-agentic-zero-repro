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

import numpy as np

logger = logging.getLogger(__name__)

MIN_STD_THRESHOLD = 0.1
REWARD_CLIP = 3.0


def _compute_preference_rewards(group_features: list[dict]) -> list[float]:
    """简化版：直接返回 correctness（0.0 或 1.0）。

    原版做组内 min-max 归一 + 多组件偏好加权（cost/latency/tool_counts），但 nano 已简化
    reward_func 只返回 correctness。GRPO 的教学核心是"同题多 rollout 的组归一化"，不是
    "多目标优化"，所以这里直接取 correctness 作为 preference reward。

    组内 min-max 在只有 0/1 二值时退化为恒等（max=min 或 已归一），真正的区分度来自下游
    _grpo_normalize_and_filter 的 (r-mean)/std 标准化（同组内有对有错才有梯度信号）。
    """
    return [feat.get("correctness", 0.0) for feat in group_features]


def _grpo_normalize_and_filter(rewards: list[float], n: int) -> tuple[list[float], list[bool]]:
    """组内 GRPO 标准化 A=(r-mean)/(std+eps) clip[-3,3]，std<0.1 的组无信号 → mask。

    偏离登记（对齐首要锚点 slime-agentic，登记见 docs/decisions/v6.md）：
      ① 源怎么做：源 ToolOrchestra/custom_convert.py:135-149 用 Python 双层 for（外层遍历组、
         内层遍历组内 rollout）逐个算 (r-mean)/std。
      ② nano 为何偏离：那段循环思路正确但写法不够清晰。改成 numpy 广播——把展平 reward
         reshape 成 [组数, n]，mean/std 沿 axis=1 一把算，克隆 minimind train_grpo.py:121-124 的
         向量化可读性。numpy 本地已装，不破 --offline 路径（torch 才会破，故不用 torch）。
      ③ 语义是否等价：**完全等价**。std 用 ddof=0（有偏，==源 var=Σ(r-mean)²/len）；clip[-3,3]、
         eps=1e-6、std<0.1 整组 mask、余数样本补 0 —— 全部逐位对齐源。
    """
    num_examples = len(rewards) // n
    remainder = len(rewards) - num_examples * n

    normalized: list[float] = []
    keep_mask: list[bool] = []
    if num_examples > 0:
        groups = np.asarray(rewards[: num_examples * n], dtype=float).reshape(num_examples, n)  # [组数, n]
        mean = groups.mean(axis=1, keepdims=True)                        # 组均值 [组数, 1]
        std = groups.std(axis=1, keepdims=True)                          # 组标准差(ddof=0，==源有偏方差)
        has_signal = std[:, 0] > MIN_STD_THRESHOLD                       # [组数] 组内是否有学习信号

        adv = np.clip((groups - mean) / (std + 1e-6), -REWARD_CLIP, REWARD_CLIP)  # A=(r-mean)/(std+eps)
        adv[~has_signal] = 0.0                                           # std<0.1 的组整组置 0（无梯度）
        normalized = adv.reshape(-1).tolist()
        keep_mask = np.repeat(has_signal, n).tolist()                    # 组标记广播回每个样本

    # 尾部余数样本（不足一组）：无组可归一 → reward 置 0、mask 掉（对齐源）。
    normalized.extend([0.0] * remainder)
    keep_mask.extend([False] * remainder)
    return normalized, keep_mask


def custom_convert(args, samples: list) -> dict:
    """Sample 列表 → train_data dict，含 GRPO 组归一 reward + 按 turn 拆样本。

    对齐源 custom_convert(args, samples)：n_samples_per_prompt 连续样本为一组。
    """
    n = getattr(args, "n_samples_per_prompt", 1)

    # 1. 提 features，组内算 preference reward（简化版直接取 correctness）。
    all_features = [s.metadata.get("reward_features", {"correctness": 0.0}) for s in samples]

    pref_rewards: list[float] = []
    num_groups = len(samples) // n
    for g in range(num_groups):
        pref_rewards.extend(
            _compute_preference_rewards(all_features[g * n : (g + 1) * n])
        )
    for j in range(len(samples) - num_groups * n):
        pref_rewards.append(all_features[num_groups * n + j].get("correctness", 0.0))

    # 2. 组内 GRPO 标准化 + 过滤。
    normalized, keep_mask = _grpo_normalize_and_filter(pref_rewards, n)

    # 3. 按 turn 拆独立训练样本（orchestrator 每轮一条）。
    tokens_list, response_lengths, loss_masks = [], [], []
    rewards, log_probs_list = [], []
    policy_versions = []  # V7.4：每个 turn-row 继承父 Sample 的 rollout 版本（与其它列 1:1 对齐）
    for i, sample in enumerate(samples):
        turns = sample.metadata.get("turns")
        norm_r = normalized[i]
        should_mask = not keep_mask[i]
        pv = sample.rollout_policy_version  # 父样本版本，拆出的每个 turn-row 都继承它

        if not turns:
            # 无 turns（不该发生在 A3/V6 路径）：整条按单序列处理。
            tokens_list.append(sample.tokens)
            response_lengths.append(sum(sample.loss_mask) if sample.loss_mask else len(sample.tokens))
            loss_masks.append([0] * len(sample.loss_mask) if should_mask else sample.loss_mask)
            rewards.append(norm_r)
            log_probs_list.append(sample.rollout_log_probs)
            policy_versions.append(pv)
            continue

        # 有 turns：每个 turn 拆成独立训练样本。
        # 注意：turn["generated_loss_mask"] 只覆盖 generated 部分，需要前面补 prompt 的 0。
        for turn in turns:
            gen_len = turn["generated_length"]
            prompt_len = len(turn["full_token_ids"]) - gen_len
            if prompt_len < 1 or gen_len < 1:
                continue

            # turn["generated_loss_mask"] 只有 generated 部分，长度 = generated_length
            generated_loss_mask = turn["generated_loss_mask"]
            generated_log_probs = turn.get("generated_log_probs", [0.0] * gen_len)

            # 构建完整序列的 loss_mask 和 log_probs（prompt 段补 0）
            if should_mask:
                full_loss_mask = [0] * (prompt_len + gen_len)
            else:
                full_loss_mask = [0] * prompt_len + generated_loss_mask

            full_log_probs = [0.0] * prompt_len + generated_log_probs

            tokens_list.append(turn["full_token_ids"])
            response_lengths.append(gen_len)
            loss_masks.append(full_loss_mask)
            rewards.append(norm_r)
            log_probs_list.append(full_log_probs)
            policy_versions.append(pv)  # V7.4：本 turn-row 继承父样本版本

    return {
        "tokens": tokens_list,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "response_lengths": response_lengths,
        "rollout_log_probs": log_probs_list,
        "rollout_policy_versions": policy_versions,  # V7.4：与其它列逐行对齐（长度 = turn-row 数）
    }
