"""V1: reward 函数——对比最终答案与标准答案。

对齐源项目: agentic/agentflow/rollout.py:209 reward_func
  源项目用 LLM-as-judge (Rewarder)；V1 先用规则匹配，教学够用。
  LLM-as-judge 在 A2（AgentFlow 主线二）引入。
"""

from __future__ import annotations

import re


def extract_answer(text: str) -> str | None:
    """从模型输出里提取 <answer>...</answer> 中的数字。"""
    m = re.search(r"<answer>\s*(.+?)\s*(?:</answer>|$)", text, re.DOTALL)
    if m:
        # 小模型常吐出 <answer>5</</answer> 这种脏尾巴，答案不含 '<'，取其前段最鲁棒
        return m.group(1).split("<")[0].strip()
    # 兜底：试着找最后一个独立数字
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def reward_func(response: str, label: str) -> float:
    """规则打分: 答案完全匹配得 1.0，否则 0.0。

    对齐源项目签名: async def reward_func(args, sample, **kwargs) -> dict
    V1 简化为同步，返回 float；V2 hook 化时对齐签名。
    """
    predicted = extract_answer(response)
    if predicted is None:
        return 0.0
    # 归一化比较（去空格，数值相等即可）
    try:
        return 1.0 if float(predicted) == float(label) else 0.0
    except ValueError:
        return 1.0 if predicted.strip() == label.strip() else 0.0
