"""V7.4 验证：infra learner ABI（learner_contract + learner_metrics）+ 版本戳，纯 CPU 离线。

  python scripts/test_v7.4_learner_abi.py

三块（对齐 infra scripts/07_learner_batch_contract.py + 11_learner_trace_contract.py 的不变量）：
  1. learner_contract：token→causal-target 右移与 repo `_pad_batch` 的 loss_mask[1:] 逐值一致；
     单一 rollout 版本校验；退化样本按训练器 keep 过滤跳过。
  2. learner_metrics：LearnerTrace 不变量（trainable<=model、policy_version_gap）、
     sum(target_mask)==sum(loss_mask) 分母恒等。
  3. 版本戳端到端：RolloutManager → _convert/custom_convert 带出 rollout_policy_versions 列，
     与其它列逐行对齐（custom_convert 拆 turn 后每行继承父样本版本）。

无 GPU、无 SGLang：走 gsm8k stub rollout（确定性），train_backend=fake。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "toy_rl").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError(f"Could not locate project root from {_HERE}")

import ray

from mini_slime import train_ray
from mini_slime.args import Args
from mini_slime.learner_contract import (
    LearnerSample,
    masked_mean,
    require_single_rollout_policy_version,
    token_layout_to_target_layout,
    validate_train_data,
)
from mini_slime.learner_metrics import LearnerTrace, summarize_learner_traces


def test_causal_target_alignment() -> None:
    """对齐 infra 07：token 布局 → causal-target(T-1) 布局，右移逐值正确。"""
    sample = LearnerSample.from_token_aligned(
        tokens=[101, 11, 12, 201, 202, 13],
        loss_mask=[0, 1, 1, 0, 0, 1],
        old_log_probs=[0.0, -1.2, -0.7, -0.1, -0.2, -0.8],
        rollout_policy_version=7,
    )
    assert sample.targets == (11, 12, 201, 202, 13)
    assert sample.target_mask == (1, 1, 0, 0, 1)
    assert sample.old_log_probs == (-1.2, -0.7, -0.1, -0.2, -0.8)
    assert sample.trainable_target_indices == (0, 1, 4)
    assert sample.trainable_tokens == 3
    assert masked_mean(sample.old_log_probs, sample.target_mask) == (-1.2 - 0.7 - 0.8) / 3
    assert require_single_rollout_policy_version([sample]) == 7
    # 右移与 repo _pad_batch 的 loss_mask[1:] 恒等。
    assert list(token_layout_to_target_layout([0, 1, 1, 0, 0, 1])) == [1, 1, 0, 0, 1]
    print("  causal_target_alignment OK")


def test_single_version_and_denominator() -> None:
    """单一版本校验拒绝混版本；sum(target_mask)==sum(loss_mask) 分母恒等（首 token 恒 mask=0）。"""
    loss_mask = [0, 1, 1, 0, 1]
    s = LearnerSample.from_token_aligned(
        tokens=[1, 2, 3, 4, 5], loss_mask=loss_mask, rollout_policy_version=3
    )
    assert s.trainable_tokens == sum(loss_mask)  # 分母恒等

    mixed = [
        LearnerSample.from_token_aligned(tokens=[1, 2, 3], loss_mask=[0, 1, 1], rollout_policy_version=1),
        LearnerSample.from_token_aligned(tokens=[1, 2, 3], loss_mask=[0, 1, 1], rollout_policy_version=2),
    ]
    try:
        require_single_rollout_policy_version(mixed)
        raise AssertionError("混版本 batch 应被拒绝")
    except ValueError:
        pass
    print("  single_version_and_denominator OK")


def test_learner_trace_invariants() -> None:
    """对齐 infra 11：LearnerTrace 不变量 + policy_version_gap + 聚合。"""
    trace = LearnerTrace(
        rollout_id=0,
        rollout_policy_version=2,
        trainer_policy_version=3,
        samples=4,
        model_tokens=100,
        trainable_tokens=40,
        batch_preparation_seconds=0.01,
        log_prob_seconds=0.0,          # repo 融合（偏离）
        forward_backward_seconds=1.5,
        optimizer_seconds=0.0,
        weight_publish_seconds=2.0,
    )
    d = trace.to_dict()
    assert d["policy_version_gap"] == 1  # trainer - rollout
    assert d["learner_update_seconds"] == 0.01 + 1.5
    assert d["learner_end_to_end_seconds"] == 0.01 + 1.5 + 2.0
    agg = summarize_learner_traces([trace, trace])
    assert agg["updates"] == 2 and agg["trainable_tokens"] == 80
    # 不变量：trainable<=model 应被 __post_init__ 守住。
    try:
        LearnerTrace(0, 0, 0, 1, 10, 20, 0, 0, 0, 0, 0)  # trainable>model
        raise AssertionError("trainable>model 应被拒绝")
    except ValueError:
        pass
    print("  learner_trace_invariants OK")


def _stub_args(**over) -> Args:
    base = dict(
        num_rollout=1,
        batch_size=2,
        gsm8k_num_train=4,
        data_source_path="toy_rl.agent.toolorchestra.gsm8k_data.load_data_source",
        custom_generate_function_path="toy_rl.agent.toolorchestra.gsm8k_stub_rollout.generate",
        custom_rm_path="toy_rl.agent.toolorchestra.rollout.reward_func",
    )
    base.update(over)
    return Args(**base)


def test_version_stamp_column() -> None:
    """端到端：RolloutManager.generate(version) → train_data 带出 rollout_policy_versions 列，
    与其它列逐行对齐；custom_convert 拆 turn 后每行继承父样本版本。"""
    import asyncio

    from mini_slime.rollout_manager import RolloutManager

    # 内置 _convert（无 custom_convert）：每样本一行，版本列 == 传入版本。
    rm = RolloutManager(_stub_args())
    data = asyncio.run(rm.generate(rollout_id=0, rollout_policy_version=5))
    pv = data["rollout_policy_versions"]
    assert len(pv) == len(data["tokens"]), "版本列须与 tokens 逐行对齐"
    assert all(v == 5 for v in pv), f"内置转换每行应戳版本 5，得到 {pv}"
    validate_train_data(data)  # 校验通过（已戳版本、右移一致）

    # custom_convert（GRPO 拆 turn）：版本列长度 == turn-row 数，每行继承父版本。
    rm2 = RolloutManager(_stub_args(
        n_samples_per_prompt=2, batch_size=1,
        custom_convert_path="mini_slime.custom_convert.custom_convert",
    ))
    data2 = asyncio.run(rm2.generate(rollout_id=0, rollout_policy_version=9))
    pv2 = data2["rollout_policy_versions"]
    assert len(pv2) == len(data2["tokens"]), "custom_convert 版本列须与 turn-row 逐行对齐"
    assert all(v == 9 for v in pv2), f"拆 turn 后每行应继承父版本 9，得到 {pv2}"
    print("  version_stamp_column OK")


def test_validate_rejects_unstamped() -> None:
    """未戳版本（None，V0-A3 路径）→ validate_train_data 报错（int 契约要求非空）。"""
    import asyncio

    from mini_slime.rollout_manager import RolloutManager

    rm = RolloutManager(_stub_args())
    data = asyncio.run(rm.generate(rollout_id=0))  # 不传版本 → None
    try:
        validate_train_data(data)
        raise AssertionError("未戳版本应被 validate 拒绝")
    except ValueError:
        pass
    print("  validate_rejects_unstamped OK")


def test_trace_closed_loop() -> None:
    """opt-in learner_trace 走 fake 后端闭环：不炸、metrics 带 policy_version_gap。"""
    args = _stub_args(learner_trace=True, learner_contract_validate=True)
    metrics = train_ray.train(args)
    assert len(metrics) == 1
    # fake 后端无 trained_samples/_trace？fake 分支也 _fill_trace，但版本已戳 → 有 gap。
    assert "policy_version_gap" in metrics[0], "opt-in trace 应产出 policy_version_gap"
    print(f"  trace_closed_loop OK (gap={metrics[0]['policy_version_gap']})")


def main() -> int:
    failed = 0
    tests = [
        test_causal_target_alignment,
        test_single_version_and_denominator,
        test_learner_trace_invariants,
        test_version_stamp_column,
        test_validate_rejects_unstamped,
        test_trace_closed_loop,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    ray.shutdown()
    print(f"[V7.4] {'PASSED' if failed == 0 else f'FAILED ({failed})'}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
