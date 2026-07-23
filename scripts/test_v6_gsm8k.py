"""V6 GSM8K 真训练验证：calculator 工具、真 log_probs、GRPO 组归一、真训练一步。

  python scripts/test_v6_gsm8k.py --offline   # 确定性离线：calculator 真求值 + 闭环（无 GPU）
  python scripts/test_v6_gsm8k.py             # 服务器：真 log_probs + 真训练 + acc_after>acc_before

V6.0 只覆盖离线段（calculator/数据/闭环）；V6.1+ 逐步补服务器段。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime import train_ray
from mini_slime.args import Args
from toy_rl.agent.toolorchestra import gsm8k_data, gsm8k_stub_rollout, rollout


def _offline_args() -> Args:
    return Args(
        num_rollout=1,
        batch_size=2,
        gsm8k_num_train=4,
        data_source_path="toy_rl.agent.toolorchestra.gsm8k_data.load_data_source",
        custom_generate_function_path="toy_rl.agent.toolorchestra.gsm8k_stub_rollout.generate",
        custom_rm_path="toy_rl.agent.toolorchestra.rollout.reward_func",
    )


async def _calculator_roundtrip(args: Args) -> None:
    sample = gsm8k_data.load_data_source(args)[0]
    sample = await gsm8k_stub_rollout.generate(args, sample)
    turns = sample.metadata["turns"]
    events = sample.metadata["events"]

    assert len(turns) == 2, "stub 应走 calculator -> answer 两轮"
    assert [event["tool_name"] for event in events] == ["calculator", "answer"]
    assert all(turn["kind"] == "orchestrator" for turn in turns)

    # calculator 真求值：event output 必须是数字（非错误串），且进入下一轮 prompt。
    calc_event = events[0]
    assert calc_event["status"] == "ok", f"calculator 应成功求值，得到 {calc_event}"
    assert calc_event["output"].lstrip("-").isdigit(), f"calculator 输出应为数字，得到 {calc_event['output']!r}"
    assert "[TOOL name=calculator]" in turns[1]["prompt_text"], "calculator observation 必须进入下一轮 prompt"

    # 仅 orchestrator token 可训练（calculator/expert 不进 sample.tokens）。
    assert sum(sample.loss_mask) == sum(turn["response_length"] for turn in turns)

    reward = await rollout.reward_func(args, sample)
    assert reward["correctness"] == 1.0, f"stub expert 吐了 label，应判对，得到 {reward}"
    assert reward["tool_counts"].get("expert_fast") == 1


def test_calculator(args: Args) -> None:
    asyncio.run(_calculator_roundtrip(args))
    print("  calculator_roundtrip OK")


def test_closed_loop(args: Args) -> None:
    metrics = train_ray.train(args)
    assert len(metrics) == 1
    assert metrics[0]["tokens_per_rollout"] > 0
    assert 0.0 <= metrics[0]["reward_mean"] <= 1.0
    print("  closed_loop OK")


def main() -> int:
    offline = "--offline" in sys.argv
    if not offline:
        print("  V6 服务器段（真 log_probs/真训练）尚未实现，仅 --offline 可跑")
        return 0
    args = _offline_args()
    failed = 0
    for test in (test_calculator, test_closed_loop):
        try:
            test(args)
        except Exception as exc:
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    ray.shutdown()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
