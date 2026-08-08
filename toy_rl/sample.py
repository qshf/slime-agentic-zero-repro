"""V0: RL 训练样本的数据契约。

对齐源项目 slime/utils/types.py:8 的 Sample dataclass，但只取教学必需的字段子集。
源项目 Sample 有 30+ 字段(多模态/投机解码/前缀缓存/路由专家...)，这里砍到 6 个核心字段，
因为主线一只关心一个问题：**一条训练样本长什么样，哪些 token 参与训练。**
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sample:
    """一条 Agentic RL 训练样本 = 一整条 agent 交互轨迹 + 打分。

    这是连接「agent rollout」和「trainer」的唯一契约：
      generate() 负责把 prompt/response/tokens/loss_mask 填好；
      reward_func() 负责把 reward 填好；
      trainer 只认这个结构，不关心 agent 内部怎么交互的。
    """

    # --- 输入 ---
    prompt: str = ""            # 原始问题，如 "2 + 3 = ?"
    label: str | None = None    # 标准答案，用于 reward 判分，如 "5"

    # --- agent 产生的输出 ---
    response: str = ""          # agent 生成的完整文本(含工具调用与返回)

    # --- 训练用的 token 级数据(核心) ---
    tokens: list[int] = field(default_factory=list)      # 完整 token 序列(prompt + response)
    loss_mask: list[int] = field(default_factory=list)   # 与 tokens 等长: 1=参与训练, 0=不参与
    #   loss_mask 语义(主线一最重要的概念):
    #     prompt token       -> 0  (模型不该学习复述题目)
    #     agent 生成 token    -> 1  (这是我们要强化的行为)
    #     tool 返回 token     -> 0  (工具输出不是模型生成的, 学它等于学环境噪声)

    # --- rollout 引擎产生时的 log_probs（V6.1 起，对齐源 Sample.rollout_log_probs）---
    rollout_log_probs: list[float] = field(default_factory=list)
    #   与 tokens 等长：prompt 段补 0，response 段是 SGLang /generate 返回的真 log_prob。
    #   真训练一步用它作 old_log_probs 算 importance ratio = exp(new_log_prob - old_log_prob)。
    #   V0-A3 走 chat 端点拿不到，留空；V6.1 起 orchestrator 走 /generate 才填真值。

    # --- reward_func 产生的打分 ---
    reward: float | None = None   # 标量奖励; 源项目也支持 dict(多组件 reward), 主线二 A3 再引入

    # --- rollout 时的策略版本号（V7.4 起，infra 缺口 1）---
    rollout_policy_version: int | None = None
    #   产生这条 rollout 时推理引擎上的权重版本（= 当时 trainer/WeightUpdater 的 version int）。
    #   infra learner_contract 用它算 off-policy staleness（trainer_version - rollout_version）、
    #   并拒绝混版本 batch（require_single_rollout_policy_version）。
    #   偏离源：源 Sample.weight_versions 是 list[str]（从 SGLang meta_info["weight_version"] 累积）；
    #   nano 取单 int（repo 权威版本源是 WeightUpdater.version，单调 int），更贴 infra int 契约。
    #   V0-A3/离线不戳 → None（validate 前会短路过滤，不进 LearnerSample）。

    # --- 附带元信息 ---
    metadata: dict = field(default_factory=dict)
    #   对齐源项目 Sample.metadata: 放不进训练序列、但 reward/eval/日志需要的东西。
    #   V1 用它承载 {"final_output": 最终答案, "turns": 每轮结构}，
    #   对齐源项目 rollout.py 里 sample.metadata["final_output"] / train_metadata["turns"]。

    def trainable_token_count(self) -> int:
        """有多少 token 真正参与训练(loss_mask==1)。"""
        return sum(self.loss_mask)

    def __post_init__(self) -> None:
        # 契约不变量: loss_mask 必须与 tokens 等长, 否则训练时对不齐。
        if self.loss_mask and len(self.loss_mask) != len(self.tokens):
            raise ValueError(
                f"loss_mask 长度 {len(self.loss_mask)} != tokens 长度 {len(self.tokens)}"
            )
        if self.rollout_log_probs and len(self.rollout_log_probs) != len(self.tokens):
            raise ValueError(
                f"rollout_log_probs 长度 {len(self.rollout_log_probs)} != tokens 长度 {len(self.tokens)}"
            )
