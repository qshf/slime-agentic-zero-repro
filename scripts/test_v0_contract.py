"""V0 数据契约行为验证脚本。

验证 4 个关键不变量:
1. loss_mask 与 tokens 等长(契约硬约束, 否则训练对不齐)。
2. tool 返回段的 token loss_mask 必须为 0(不训练环境输出)。
3. agent 生成段的 token loss_mask 必须为 1(强化模型行为)。
4. fake_train_step 产出的统计里 trainable_tokens 等于 loss_mask 中 1 的个数。

零外部依赖, 秒级完成: python scripts/test_v0_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toy_rl.sample import Sample
from toy_rl.train_loop import build_hardcoded_sample, fake_train_step


def test_loss_mask_length_matches_tokens() -> None:
    """不变量1: loss_mask 与 tokens 等长。"""
    s = build_hardcoded_sample()
    assert len(s.loss_mask) == len(s.tokens), (
        f"loss_mask {len(s.loss_mask)} != tokens {len(s.tokens)}"
    )
    print("  loss_mask_length_matches_tokens OK")


def test_mismatched_mask_rejected() -> None:
    """不变量1 反向: 长度不一致时 Sample 必须报错。"""
    raised = False
    try:
        Sample(tokens=[1, 2, 3], loss_mask=[1, 0])  # 故意不等长
    except ValueError:
        raised = True
    assert raised, "长度不一致的 loss_mask 应触发 ValueError"
    print("  mismatched_mask_rejected OK")


def test_tool_tokens_not_trained() -> None:
    """不变量2: tool 返回段 loss_mask 为 0。

    build_hardcoded_sample 里 tool 段是 "[tool:5]", 用 response 定位它的位置,
    断言对应 loss_mask 为 0。这里直接检查: response 中所有非 agent 生成的部分不被训练。
    """
    s = build_hardcoded_sample()
    # tool token 是第 9 个 token(索引 8): prompt 5 个 + agent 3 个 + tool 1 个
    tool_idx = 8
    assert s.loss_mask[tool_idx] == 0, "tool 返回 token 不应参与训练"
    print("  tool_tokens_not_trained OK")


def test_agent_tokens_trained() -> None:
    """不变量3: agent 生成段 loss_mask 为 1。"""
    s = build_hardcoded_sample()
    # agent 段: 索引 5,6,7 (我/调用/calculator) 和 9,10,11 (答案/是/5)
    for idx in (5, 6, 7, 9, 10, 11):
        assert s.loss_mask[idx] == 1, f"agent token idx={idx} 应参与训练"
    # prompt 段(0-4) 不训练
    for idx in range(5):
        assert s.loss_mask[idx] == 0, f"prompt token idx={idx} 不应参与训练"
    print("  agent_tokens_trained OK")


def test_stats_trainable_count() -> None:
    """不变量4: fake_train_step 的 trainable_tokens == loss_mask 中 1 的个数。"""
    s = build_hardcoded_sample()
    stats = fake_train_step([s], rollout_id=0)
    assert stats["trainable_tokens"] == sum(s.loss_mask), "统计的 trainable_tokens 不符"
    assert stats["reward_mean"] == 1.0, "单样本 reward=1.0, 均值应为 1.0"
    assert stats["batch_size"] == 1
    print("  stats_trainable_count OK")


def main() -> int:
    tests = [
        test_loss_mask_length_matches_tokens,
        test_mismatched_mask_rejected,
        test_tool_tokens_not_trained,
        test_agent_tokens_trained,
        test_stats_trainable_count,
    ]
    print(f"Running {len(tests)} V0 contract tests...")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
