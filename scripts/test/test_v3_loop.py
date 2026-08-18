"""V3 行为验证：mini_slime 最小闭环 rollout→train→update_weights。

跑 num_rollout 轮闭环（真 SGLang rollout + fake train/sync），断言（对齐 v3.md 验证清单）：
  1. 每轮 metrics 字段齐全（gen/train/sync_time、reward_mean、tokens_per_rollout 都在且类型对）。
  2. trainer.weight_version == num_rollout（每轮同步一次，权重版本正确递增）。
  3. rollout_data 契约：len(tokens)==len(loss_masks)==len(rewards)==batch_size；
     每条 len(tokens[i])==len(loss_masks[i])。
  4. reward_mean ∈ [0, 1]。

用法:
  python scripts/test_v3_loop.py --offline   # 本地: monkeypatch _chat, 不连 SGLang
  python scripts/test_v3_loop.py             # 服务器: 真 SGLang 端到端
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "toy_rl").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError(f"Could not locate project root from {_HERE}")

from mini_slime.args import Args
from mini_slime.rollout_manager import RolloutManager
from mini_slime.train import train
from mini_slime.trainer import Trainer


def _install_offline_stub() -> None:
    """离线打桩: 把 calculator_agent._chat 换成"直接算出答案"的假 chat。

    对齐 v3.md "本地离线: generate hook 可打桩(monkeypatch _chat)跑闭环, 不连 SGLang"。
    真 _chat 打 SGLang; 桩从 user prompt 里取算式、用 CalculatorTool 求值,
    直接返回 <answer>结果</answer>——让整条闭环(generate loop→convert→train→sync)在本地可跑。
    """
    import toy_rl.agent.calculator_agent as ca
    from toy_rl.agent.tools import CalculatorTool

    calc = CalculatorTool()

    def fake_chat(client, model_name, messages, max_tokens, temperature) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        expr = user.replace("=", "").replace("?", "").strip()  # "2 + 3 = ?" -> "2 + 3"
        result = calc.execute(expr)
        return f"<answer>{result}"  # loop 会补回被 stop 截掉的 </answer>

    ca._chat = fake_chat


def _assert_contract(rollout_data: dict, batch_size: int) -> None:
    """不变量3: rollout_data 契约(RolloutManager→Trainer 的 dict)。"""
    keys = ("tokens", "loss_masks", "rewards", "response_lengths")
    for k in keys:
        assert k in rollout_data, f"train_data 缺字段 {k}"
        assert len(rollout_data[k]) == batch_size, f"{k} 长度 {len(rollout_data[k])} != batch_size {batch_size}"
    for i in range(batch_size):
        assert len(rollout_data["tokens"][i]) == len(rollout_data["loss_masks"][i]), (
            f"第{i}条 tokens/loss_mask 不等长"
        )


async def test_convert_contract(args: Args) -> None:
    """不变量3: 单独跑一轮 generate, 校验 train_data 契约。"""
    rm = RolloutManager(args)
    rollout_data = await rm.generate(rollout_id=0)
    _assert_contract(rollout_data, args.batch_size)
    print("  convert_contract OK")


async def test_full_loop(args: Args) -> None:
    """不变量1/2/4: 跑完整闭环, 校验 metrics 字段/权重版本/reward 范围。"""
    metrics_log = await train(args)

    # 不变量1: 每轮 metrics 字段齐全且类型对
    assert len(metrics_log) == args.num_rollout, f"应有 {args.num_rollout} 轮 metrics"
    for m in metrics_log:
        for k in ("gen_time", "train_time", "sync_time", "reward_mean", "tokens_per_rollout"):
            assert k in m, f"metrics 缺字段 {k}"
        assert isinstance(m["gen_time"], float) and isinstance(m["train_time"], float)
        assert isinstance(m["sync_time"], float)
        assert isinstance(m["tokens_per_rollout"], int) and m["tokens_per_rollout"] > 0
        # 不变量4: reward_mean ∈ [0, 1]
        assert 0.0 <= m["reward_mean"] <= 1.0, f"reward_mean 越界: {m['reward_mean']}"

    # 不变量2: 每轮同步一次 → 权重版本 == num_rollout
    assert metrics_log[-1]["weight_version"] == args.num_rollout, (
        f"weight_version {metrics_log[-1]['weight_version']} != num_rollout {args.num_rollout}"
    )
    print("  full_loop OK")


def test_weight_version_increment() -> None:
    """不变量2(单元): Trainer.update_weights 每调一次版本 +1。"""
    t = Trainer(Args())
    assert t.weight_version == 0
    t.update_weights()
    t.update_weights()
    assert t.weight_version == 2, f"两次 update_weights 后应为 2, 得到 {t.weight_version}"
    print("  weight_version_increment OK")


def main() -> int:
    offline = "--offline" in sys.argv
    print(f"Running V3 tests ({'offline' if offline else 'server/SGLang'})...")
    if offline:
        _install_offline_stub()

    args = Args(num_rollout=2, batch_size=2)
    failed = 0

    try:
        test_weight_version_increment()
    except AssertionError as e:
        failed += 1
        print(f"  FAIL weight_version_increment: {e}")

    for coro in (test_convert_contract(args), test_full_loop(args)):
        try:
            asyncio.run(coro)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {e}")

    print(f"\n{'FAILED' if failed else 'PASSED'} ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
