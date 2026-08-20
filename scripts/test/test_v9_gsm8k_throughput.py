"""CPU contract checks for the V9 GSM8K collect/replay workload."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from mini_slime.args import Args
from mini_slime.custom_convert import custom_convert
from scripts.bench.v9_gsm8k_throughput import (
    DEFAULT_PAPER_PATH,
    _drop_masked_rows,
    _validate_workload,
    load_paper,
    validate_collection_quality,
)
from toy_rl.agent.toolorchestra.gsm8k_throughput_rollout import _apply_output
from toy_rl.agent.toolorchestra.rollout import GenOutput
from toy_rl.sample import Sample


def _sample(correct: bool) -> Sample:
    sample = Sample(prompt="What is 1 + 1?", label="2")
    output = GenOutput(
        response="\\boxed{2}" if correct else "\\boxed{3}",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5],
        generated_log_probs=[-0.1, -0.2],
    )
    _apply_output(sample, "Question: What is 1 + 1?", output)
    sample.metadata["reward_features"] = {"correctness": float(correct)}
    sample.rollout_policy_version = 0
    return sample


def main() -> int:
    paper, paper_sha256 = load_paper(DEFAULT_PAPER_PATH)
    assert paper["indices"] == [0, 1]
    assert paper["samples_per_prompt"] == 4
    assert len(paper_sha256) == 64
    samples = [_sample(False), _sample(True), _sample(True), _sample(False)]
    data = custom_convert(Args(n_samples_per_prompt=4), samples)
    assert len(data["tokens"]) == 4
    assert all(len(t) == len(m) == len(lp) for t, m, lp in zip(
        data["tokens"], data["loss_masks"], data["rollout_log_probs"], strict=True
    ))
    assert data["raw_rewards"] == [0.0, 1.0, 1.0, 0.0]
    assert data["grpo_group_count"] == 1
    assert data["grpo_active_group_count"] == 1
    assert data["rewards"][0] < 0 < data["rewards"][1]

    masked = custom_convert(Args(n_samples_per_prompt=4), [_sample(True) for _ in range(4)])
    _validate_workload(masked, require_trainable_rows=False)
    assert _drop_masked_rows(masked) == 4
    assert not masked["tokens"]
    metadata = {"grpo_active_group_count": 8, "trainable_tokens": 4096}
    validate_collection_quality(metadata, min_active_groups=8, min_trainable_tokens=4096)
    try:
        validate_collection_quality(metadata, min_active_groups=9, min_trainable_tokens=4096)
    except RuntimeError as error:
        assert "active_grpo_groups=8 < 9" in str(error)
    else:
        raise AssertionError("quality gate accepted too few active groups")
    print("V9 GSM8K throughput workload contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
