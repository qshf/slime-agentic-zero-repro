"""V7.4: learner 阶段 trace —— infra `learner_metrics` 的 in-repo 忠实重写。

来源与定位：infra（agentic-rl-infra-lab/src/agentic_rl_infra_lab/learner_metrics.py）是规范，
本文件按同一语义在 repo 内重写，用它的不变量当验收。把 `Trainer` 原来扁平的 metric dict 升级成
**分相位计时 + 明确的 token/s 分母**，让「慢在哪个相位」可定位（各相位的补救手段不同）。

不变量（infra 已证，repo 直接对上）：
  - `trainable_tokens == sum(loss_mask)`：repo 现有 n_trainable。因首 token 恒为 prompt(mask=0)，
    `sum(target_mask)==sum(loss_mask)`，两侧分母恒等。
  - `trainable_tokens <= model_tokens`（model_tokens = 全序列长度之和）。
  - `policy_version_gap = trainer_policy_version - rollout_policy_version`（gap 1 版本戳的收益）。

偏离（登记见 docs/decisions/v7.4.md）：
  - **只填可测相位**：repo 把 log-prob 重算**融进同一次 forward**（torch_actor.py:108-110 /
    fsdp_trainer.py:168-176 的 log_softmax 紧跟 model(...)），无独立 no-grad log-prob pass →
    `log_prob_seconds=0.0`，登记为偏离（源用独立 log-prob 前向，nano 融合）。可测的：
    batch_preparation / forward_backward / optimizer / weight_publish。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _rate(numerator: int, seconds: float) -> float:
    return numerator / seconds if seconds > 0 else 0.0


@dataclass(frozen=True)
class LearnerTrace:
    """一次 learner 更新，按「补救手段不同」的相位切开。对齐 infra LearnerTrace。"""

    rollout_id: int
    rollout_policy_version: int
    trainer_policy_version: int
    samples: int
    model_tokens: int
    trainable_tokens: int
    batch_preparation_seconds: float
    log_prob_seconds: float
    forward_backward_seconds: float
    optimizer_seconds: float
    weight_publish_seconds: float

    def __post_init__(self) -> None:
        if min(self.rollout_id, self.rollout_policy_version, self.trainer_policy_version) < 0:
            raise ValueError("rollout and policy versions must be non-negative")
        if min(self.samples, self.model_tokens, self.trainable_tokens) < 0:
            raise ValueError("sample and token counts must be non-negative")
        if self.trainable_tokens > self.model_tokens:
            raise ValueError("trainable_tokens cannot exceed model_tokens")
        phase_values = (
            self.batch_preparation_seconds,
            self.log_prob_seconds,
            self.forward_backward_seconds,
            self.optimizer_seconds,
            self.weight_publish_seconds,
        )
        if any(seconds < 0 for seconds in phase_values):
            raise ValueError("phase durations must be non-negative")

    @property
    def learner_update_seconds(self) -> float:
        """消费算法输出到完成 learner optimizer 更新的耗时（不含发布）。"""
        return (
            self.batch_preparation_seconds
            + self.log_prob_seconds
            + self.forward_backward_seconds
            + self.optimizer_seconds
        )

    @property
    def learner_end_to_end_seconds(self) -> float:
        """含把新版本发布回引擎的同步 learner 路径。"""
        return self.learner_update_seconds + self.weight_publish_seconds

    def to_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        data.update(
            {
                "learner_update_seconds": self.learner_update_seconds,
                "learner_end_to_end_seconds": self.learner_end_to_end_seconds,
                "model_tokens_per_update_second": _rate(self.model_tokens, self.learner_update_seconds),
                "trainable_tokens_per_compute_second": _rate(
                    self.trainable_tokens, self.forward_backward_seconds + self.optimizer_seconds
                ),
                "policy_version_gap": self.trainer_policy_version - self.rollout_policy_version,
            }
        )
        return data


def summarize_learner_traces(traces: list[LearnerTrace]) -> dict[str, int | float]:
    """聚合多次更新，不对逐次 rate 求平均（分子分母各自求和再算）。对齐 infra 同名函数。"""
    if not traces:
        return {"updates": 0, "samples": 0, "model_tokens": 0, "trainable_tokens": 0}
    samples = sum(trace.samples for trace in traces)
    model_tokens = sum(trace.model_tokens for trace in traces)
    trainable_tokens = sum(trace.trainable_tokens for trace in traces)
    update_seconds = sum(trace.learner_update_seconds for trace in traces)
    end_to_end_seconds = sum(trace.learner_end_to_end_seconds for trace in traces)
    compute_seconds = sum(trace.forward_backward_seconds + trace.optimizer_seconds for trace in traces)
    return {
        "updates": len(traces),
        "samples": samples,
        "model_tokens": model_tokens,
        "trainable_tokens": trainable_tokens,
        "learner_update_seconds": update_seconds,
        "learner_end_to_end_seconds": end_to_end_seconds,
        "model_tokens_per_update_second": _rate(model_tokens, update_seconds),
        "trainable_tokens_per_compute_second": _rate(trainable_tokens, compute_seconds),
    }
