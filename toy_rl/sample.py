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

    # --- reward_func 产生的打分 ---
    reward: float | None = None   # 标量奖励; 源项目也支持 dict(多组件 reward), 主线二 A3 再引入

    def trainable_token_count(self) -> int:
        """有多少 token 真正参与训练(loss_mask==1)。"""
        return sum(self.loss_mask)

    def __post_init__(self) -> None:
        # 契约不变量: loss_mask 必须与 tokens 等长, 否则训练时对不齐。
        if self.loss_mask and len(self.loss_mask) != len(self.tokens):
            raise ValueError(
                f"loss_mask 长度 {len(self.loss_mask)} != tokens 长度 {len(self.tokens)}"
            )
