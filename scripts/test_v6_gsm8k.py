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


def test_grpo_group_norm() -> None:
    """V6.2：同题多 rollout → 组内 min-max + GRPO 标准化产出有区分度的 reward。"""
    from mini_slime.custom_convert import _compute_preference_rewards, _grpo_normalize_and_filter

    pref = {"accuracy": 1.0, "cost": 0.1, "latency": 0.1, "expert_fast": 0.3, "expert_precise": 0.1}
    # 同题 4 rollout：#0 错、#1 对且便宜快(fast)、#2 对但贵慢(precise)、#3 对且便宜快。
    group = [
        {"correctness": 0.0, "total_cost": 0.0, "total_latency": 0.0, "tool_counts": {}},
        {"correctness": 1.0, "total_cost": 0.0005, "total_latency": 20.0, "tool_counts": {"expert_fast": 1}},
        {"correctness": 1.0, "total_cost": 0.005, "total_latency": 1500.0, "tool_counts": {"expert_precise": 1}},
        {"correctness": 1.0, "total_cost": 0.0006, "total_latency": 25.0, "tool_counts": {"expert_fast": 1}},
    ]
    pref_rewards = _compute_preference_rewards(group, [pref] * 4)
    assert pref_rewards[0] == 0.0, "错误 rollout preference reward 必须为 0"
    # 便宜快的 fast rollout 应比贵慢的 precise 高（cost/latency 归一后 fast 占优）。
    assert pref_rewards[1] > pref_rewards[2], f"fast 应优于 precise，得到 {pref_rewards}"

    normalized, keep = _grpo_normalize_and_filter(pref_rewards, n=4)
    assert all(keep), "组内有方差（部分对部分错），应保留学习信号"
    assert abs(sum(normalized)) < 1e-3, "GRPO 标准化后组内均值应约为 0"
    assert max(normalized) <= 3.0 and min(normalized) >= -3.0, "标准化 reward 须 clip 到 [-3,3]"

    # 无信号组（全对且完全一样）→ std<0.1 → 全 mask、reward=0。
    same = [{"correctness": 1.0, "total_cost": 0.001, "total_latency": 10.0, "tool_counts": {"expert_fast": 1}}] * 4
    same_rewards = _compute_preference_rewards(same, [pref] * 4)
    _, keep2 = _grpo_normalize_and_filter(same_rewards, n=4)
    assert not any(keep2), "无方差组应被 mask（std<0.1，无学习信号）"
    print("  grpo_group_norm OK")


def test_grpo_closed_loop() -> None:
    """V6.2：GRPO custom_convert 接入 train_ray 闭环（同题多 rollout → 拆 turn → fake-train）。"""
    args = Args(
        num_rollout=1,
        batch_size=1,
        n_samples_per_prompt=4,
        gsm8k_num_train=2,
        data_source_path="toy_rl.agent.toolorchestra.gsm8k_data.load_data_source",
        custom_generate_function_path="toy_rl.agent.toolorchestra.gsm8k_stub_rollout.generate",
        custom_rm_path="toy_rl.agent.toolorchestra.rollout.reward_func",
        custom_convert_path="mini_slime.custom_convert.custom_convert",
    )
    metrics = train_ray.train(args)
    assert len(metrics) == 1
    # stub 全答对 → 组内无方差 → GRPO 全 mask（reward 归一后 0）。这本身是对 GRPO 语义的正确验证：
    # 确定性 stub 产不出组内差异，真实模型的采样随机性才有。这里只断言闭环跑通、指标合法。
    assert metrics[0]["tokens_per_rollout"] > 0
    print("  grpo_closed_loop OK")


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
    # V6.2 GRPO 组归一（纯数值 + 闭环）。
    for test in (test_grpo_group_norm, test_grpo_closed_loop):
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    ray.shutdown()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
