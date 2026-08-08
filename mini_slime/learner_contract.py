"""V7.4: learner 输入契约 —— infra `learner_contract` 的 in-repo 忠实重写。

来源与定位（见 docs/decisions/v7-onward-plan.md「infra as spec」）：
  - infra（agentic-rl-infra-lab/src/agentic_rl_infra_lab/learner_contract.py）是 learner
    基础设施的**规范**；其代码是参考，本文件按同一语义在 repo 内重写，用它的 CPU 不变量当验收。
  - 首要锚点仍是 slime-agentic：token→causal-target 的右移正是 `mini_slime/torch_actor.py:_pad_batch`
    里的 `loss_masks[i][1:]`、`fsdp_trainer.py:_microbatch_backward` 里的 `loss_mask[1:]`。
    本模块把「那一步右移」显式成一个可校验的坐标系（T token → T-1 causal target）。

**这是一层「校验视图」，不是数据变换**（关键设计，见 v7.4.md）：
  - 训练器（torch_actor/fsdp_trainer）内部**照旧**自己做 [1:] 右移消费 loss_mask。
  - 本模块只在 opt-in 时把同一批数据**过一遍 LearnerSample 校验**（右移一致、单一 rollout 版本、
    causal 对齐），失败即抛。不改训练器消费的字段 → 对 V0-V7.3 零回归。

偏离（登记见 docs/decisions/v7.4.md）：
  - advantages 传 None（gap 2 暂缓）：repo GRPO 是**逐序列标量** reward（rewards 列），训练器内部
    广播 [B,1]→[B,L-1]；不物化逐 token advantage。LearnerSample.advantages 字段保留、类型受校验，
    但本版不填——对齐 infra 文档「选项 b（sample-level 标量）」的推荐，逐 token 留将来 GAE/过程奖励。
  - rollout_policy_version 取单 int（repo 权威版本源 WeightUpdater.version）；源 Sample.weight_versions
    是 list[str]（从 SGLang meta 累积）。见 toy_rl/sample.py 字段注释。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TargetValues = tuple[float, ...]


def token_layout_to_target_layout(values: Sequence[int | float]) -> tuple[int | float, ...]:
    """丢掉 token 0——causal LM 永远不会「从前一个 token」预测第 0 个 token。

    对齐 infra 同名函数 + repo `_pad_batch` 的 `values[1:]`。
    """
    if len(values) < 2:
        raise ValueError("a causal-LM sequence needs at least two token-aligned values")
    return tuple(values[1:])


def _validate_target_values(name: str, values: TargetValues | None, target_length: int) -> None:
    if values is not None and len(values) != target_length:
        raise ValueError(f"{name} must align with causal targets: {len(values)} != {target_length}")


@dataclass(frozen=True)
class LearnerSample:
    """一条算法侧产出的样本，落在 learner 的 causal-target 坐标系里。

    `target_mask[i]` 及每个可选的 target 对齐字段，描述「用位置 i 的 logit 预测 tokens[i+1]」。
    对齐 infra LearnerSample（字段/校验逐一对应）。
    """

    tokens: tuple[int, ...]
    target_mask: tuple[int, ...]
    rollout_policy_version: int
    old_log_probs: TargetValues | None = None
    advantages: TargetValues | None = None
    returns: TargetValues | None = None

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise ValueError("a learner sample needs at least two tokens")
        target_length = len(self.tokens) - 1
        if len(self.target_mask) != target_length:
            raise ValueError("target_mask must have len(tokens) - 1 entries")
        if any(value not in (0, 1) for value in self.target_mask):
            raise ValueError("target_mask must contain only 0 or 1")
        if self.rollout_policy_version < 0:
            raise ValueError("rollout_policy_version must be non-negative")
        _validate_target_values("old_log_probs", self.old_log_probs, target_length)
        _validate_target_values("advantages", self.advantages, target_length)
        _validate_target_values("returns", self.returns, target_length)

    @classmethod
    def from_token_aligned(
        cls,
        *,
        tokens: Sequence[int],
        loss_mask: Sequence[int],
        rollout_policy_version: int,
        old_log_probs: Sequence[float] | None = None,
        advantages: Sequence[float] | None = None,
        returns: Sequence[float] | None = None,
    ) -> LearnerSample:
        """从 token 对齐布局（repo Sample/train_data 用的）适配到 causal-target 布局。

        对齐 infra from_token_aligned：`loss_mask[0]` 必须为 0（token 0 无 causal target）、
        `old_log_probs[0]` 必须是 0.0 占位——repo solver 拼接时 prompt 段前缀就是 0/0.0，天然满足。
        """
        if len(loss_mask) != len(tokens):
            raise ValueError("loss_mask must align with tokens before causal shifting")
        if len(tokens) < 2:
            raise ValueError("a causal-LM sequence needs at least two tokens")
        if loss_mask[0] != 0:
            raise ValueError("loss_mask[0] must be 0: token zero has no causal target")
        for name, values in (
            ("old_log_probs", old_log_probs),
            ("advantages", advantages),
            ("returns", returns),
        ):
            if values is not None and len(values) != len(tokens):
                raise ValueError(f"{name} must align with tokens before causal shifting")
        if old_log_probs is not None and old_log_probs[0] != 0.0:
            raise ValueError("old_log_probs[0] must be the zero placeholder for token zero")
        return cls(
            tokens=tuple(tokens),
            target_mask=tuple(token_layout_to_target_layout(loss_mask)),
            rollout_policy_version=rollout_policy_version,
            old_log_probs=(
                tuple(token_layout_to_target_layout(old_log_probs)) if old_log_probs is not None else None
            ),
            advantages=(tuple(token_layout_to_target_layout(advantages)) if advantages is not None else None),
            returns=tuple(token_layout_to_target_layout(returns)) if returns is not None else None,
        )

    @property
    def targets(self) -> tuple[int, ...]:
        """被对应 causal-LM logit 预测的 token。"""
        return self.tokens[1:]

    @property
    def trainable_target_indices(self) -> tuple[int, ...]:
        """target/target 对齐张量里 loss 生效的下标。"""
        return tuple(index for index, value in enumerate(self.target_mask) if value)

    @property
    def trainable_tokens(self) -> int:
        return sum(self.target_mask)


def require_single_rollout_policy_version(samples: Sequence[LearnerSample]) -> int:
    """返回共享的 behavior-policy 版本，或拒绝混版本的 learner batch。对齐 infra 同名函数。"""
    if not samples:
        raise ValueError("a learner batch cannot be empty")
    versions = {sample.rollout_policy_version for sample in samples}
    if len(versions) != 1:
        raise ValueError(f"learner batch mixes rollout policy versions: {sorted(versions)}")
    return versions.pop()


def masked_mean(values: Sequence[float], target_mask: Sequence[int]) -> float:
    """只在 learner 施加 loss 的位置对 target 对齐值求均值。对齐 infra 同名函数。"""
    if len(values) != len(target_mask):
        raise ValueError("values and target_mask must align")
    active_values = [value for value, mask in zip(values, target_mask, strict=True) if mask]
    return sum(active_values) / len(active_values) if active_values else 0.0


def _keep(tokens: Sequence[int], loss_mask: Sequence[int]) -> bool:
    """训练器实际会训练的样本谓词（对齐 torch_actor._pad_batch / fsdp_trainer 的 keep 过滤）：
    len>=2 且有可训 token。退化样本被训练器静默跳过，本校验也须先过滤，否则会对合法退化样本误报。
    """
    return len(tokens) >= 2 and sum(loss_mask) > 0


def validate_train_data(train_data: dict) -> int:
    """把 RolloutManager 产的列式 train_data 过一遍 LearnerSample 校验（opt-in 视图）。

    seam：逻辑上插在 `_convert_samples_to_train_data`/`custom_convert` 之后、训练器 `_pad_batch`/
    `train_batch` 之前。**不改** train_data（训练器仍消费原列），只做「右移一致 + 单一版本 + causal 对齐」
    的可判定校验，失败即抛。

    返回：这批数据的共享 rollout 版本（require_single_rollout_policy_version 的结果）。
    要求每条样本都戳了 rollout_policy_version（V7.4 真路径已戳；未戳=None 会在这里被 int 校验挡下）。
    """
    tokens = train_data["tokens"]
    loss_masks = train_data["loss_masks"]
    log_probs = train_data.get("rollout_log_probs") or [None] * len(tokens)
    versions = train_data.get("rollout_policy_versions") or [None] * len(tokens)

    learner_samples: list[LearnerSample] = []
    for i in range(len(tokens)):
        if not _keep(tokens[i], loss_masks[i]):
            continue  # 训练器会跳过的退化样本，校验也跳过（对齐 keep 过滤）
        if versions[i] is None:
            raise ValueError(
                f"sample {i} has no rollout_policy_version; learner_contract 校验要求已戳版本"
            )
        lp = log_probs[i] if log_probs[i] else None  # 空 [] → None（对齐 trainer 适配器）
        learner_samples.append(
            LearnerSample.from_token_aligned(
                tokens=tokens[i],
                loss_mask=loss_masks[i],
                rollout_policy_version=int(versions[i]),
                old_log_probs=lp,
                advantages=None,  # gap 2 暂缓：repo GRPO 用逐序列标量 reward，不物化逐 token adv
            )
        )
    return require_single_rollout_policy_version(learner_samples)
