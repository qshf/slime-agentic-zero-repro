"""V0: 硬编码单样本 + fake trainer。

主线一最小起点。目标不是"训练"，而是把一条训练样本的数据契约立起来，
并让一个 fake trainer 消费它、打印统计——从而回答:
  "trainer 到底吃什么? 哪些 token 参与训练?"

对齐源项目:
  - Sample 结构      -> slime/utils/types.py:8
  - trainer 消费样本  -> train.py:80  actor_model.async_train(rollout_id, rollout_data_ref)

V0 的痛点(故意留着, 引出 V1):
  - 样本是写死的一条, reward 写死, loss_mask 手工填
  - 没有真实 agent 交互, 不知道这些字段在多轮对话里怎么生成
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python3 toy_rl/train_loop.py` 直接运行(把项目根加入 path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toy_rl.sample import Sample


def build_hardcoded_sample() -> Sample:
    """手工构造一条 "2 + 3 = ?" 的样本。

    用「词」当 token(真 tokenizer 留到 V1 接 Qwen3 时再上), 这样 loss_mask 的语义肉眼可读。
    轨迹:  [prompt] 2 + 3 = ?   [agent] 我调用 calculator   [tool] 5   [agent] 答案是 5
    """
    # (词, 段类型) —— 段类型决定 loss_mask
    segments: list[tuple[str, str]] = [
        # prompt 段: 模型不该学复述题目 -> loss_mask 0
        ("2", "prompt"), ("+", "prompt"), ("3", "prompt"), ("=", "prompt"), ("?", "prompt"),
        # agent 生成段: 要强化的行为 -> loss_mask 1
        ("我", "agent"), ("调用", "agent"), ("calculator", "agent"),
        # tool 返回段: 环境产生的, 不是模型生成 -> loss_mask 0
        ("[tool:5]", "tool"),
        # agent 最终答案段 -> loss_mask 1
        ("答案", "agent"), ("是", "agent"), ("5", "agent"),
    ]

    # 假 tokenizer: 每个词映射成一个稳定 id(仅 V0 用, 够验证契约即可)
    vocab: dict[str, int] = {}
    tokens: list[int] = []
    loss_mask: list[int] = []
    for word, seg in segments:
        tokens.append(vocab.setdefault(word, len(vocab)))
        loss_mask.append(1 if seg == "agent" else 0)

    response_words = [w for w, seg in segments if seg != "prompt"]

    return Sample(
        prompt="2 + 3 = ?",
        label="5",
        response=" ".join(response_words),
        tokens=tokens,
        loss_mask=loss_mask,
        reward=1.0,  # 写死: 最终答案 "5" 正确
    )


def fake_train_step(batch: list[Sample], rollout_id: int) -> dict:
    """假的训练步: 不更新任何权重, 只计算并打印 batch 统计。

    对齐 train.py:80 的 async_train —— 真实版会拿 rollout_data 算 loss 反传;
    V0 只关心"数据流对不对", 所以退化成统计。真训练一步留到主线三 V7(FSDP)。
    """
    total_tokens = sum(len(s.tokens) for s in batch)
    trainable_tokens = sum(s.trainable_token_count() for s in batch)
    rewards = [s.reward for s in batch if s.reward is not None]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0

    stats = {
        "rollout_id": rollout_id,
        "batch_size": len(batch),
        "total_tokens": total_tokens,
        "trainable_tokens": trainable_tokens,
        "trainable_ratio": round(trainable_tokens / total_tokens, 3) if total_tokens else 0.0,
        "reward_mean": reward_mean,
    }
    return stats


def main() -> None:
    print("=== V0: 硬编码单样本 + fake trainer ===\n")

    sample = build_hardcoded_sample()

    # 打印完整样本, 让"哪些 token 训练"肉眼可见
    print("样本轨迹(token 级):")
    print(f"  prompt : {sample.prompt}")
    print(f"  label  : {sample.label}")
    print(f"  response: {sample.response}")
    print(f"  reward : {sample.reward}")
    print(f"  tokens   ({len(sample.tokens)}): {sample.tokens}")
    print(f"  loss_mask({len(sample.loss_mask)}): {sample.loss_mask}")
    print(f"  -> 参与训练的 token 数: {sample.trainable_token_count()} / {len(sample.tokens)}\n")

    # fake trainer 消费一个 batch(V0 里 batch 就一条)
    stats = fake_train_step([sample], rollout_id=0)
    print("fake_train_step 统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
