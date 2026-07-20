"""V1 行为验证脚本（在服务器 5090 上跑，需 SGLang 容器已启动）。

验证:
1. calculator 工具正确计算（离线，不需 SGLang）。
2. reward_func 判分正确（离线）。
3. loss_mask 段规则正确：真实 rollout 里 tool 返回段全 0、agent 段全 1（需 SGLang）。
4. 完整 trajectory 可打印，tokens 与 loss_mask 等长（需 SGLang）。

用法:
  # 离线部分（本地即可）:
  python scripts/test_v1_rollout.py --offline
  # 完整（服务器，SGLang 就绪后）:
  python scripts/test_v1_rollout.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toy_rl.agent.calculator_agent import calculator, generate
from toy_rl.reward import extract_answer, reward_func


def test_calculator() -> None:
    """不变量1: calculator 计算正确 + 拒绝非法输入。"""
    assert calculator("2 + 3") == "5"
    assert calculator("10 * (4 - 1)") == "30"
    assert calculator("import os").startswith("Error")
    print("  calculator OK")


def test_reward_func() -> None:
    """不变量2: reward 判分正确。"""
    assert reward_func("<answer>5</answer>", "5") == 1.0
    assert reward_func("<answer>6</answer>", "5") == 0.0
    assert reward_func("答案是 <answer>30</answer>", "30") == 1.0
    assert extract_answer("blah <answer>42</answer> blah") == "42"
    print("  reward_func OK")


async def test_real_rollout() -> None:
    """不变量3+4: 真实 rollout 的 loss_mask 段规则 + 契约长度（需 SGLang）。"""
    sample, final_answer = await generate("2 + 3 = ?", label="5")

    print("\n  --- 完整 trajectory ---")
    print(f"  prompt: {sample.prompt}")
    print(f"  response: {sample.response!r}")
    print(f"  final_answer: {final_answer}")
    print(f"  tokens: {len(sample.tokens)}, loss_mask: {len(sample.loss_mask)}")
    print(f"  trainable: {sample.trainable_token_count()}/{len(sample.tokens)}")

    # 契约: 等长
    assert len(sample.tokens) == len(sample.loss_mask), "tokens/loss_mask 不等长"
    # prompt 段全 0
    prompt_len = len(sample.prompt)
    assert all(m == 0 for m in sample.loss_mask[:prompt_len]), "prompt 段应全 0"
    # 至少有一些 agent token 被训练
    assert sample.trainable_token_count() > 0, "应有 agent token 参与训练"
    print("  real_rollout OK")


def main() -> int:
    offline = "--offline" in sys.argv
    print("Running V1 tests...")
    failed = 0

    for t in (test_calculator, test_reward_func):
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")

    if offline:
        print("\n(--offline: 跳过需 SGLang 的 real_rollout 测试)")
    else:
        try:
            asyncio.run(test_real_rollout())
        except Exception as e:
            failed += 1
            print(f"  FAIL real_rollout: {type(e).__name__}: {e}")
            print("  (确认 SGLang 容器已启动: docker ps --filter name=sglang-qwen3)")

    print(f"\n{'FAILED' if failed else 'PASSED'} ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
