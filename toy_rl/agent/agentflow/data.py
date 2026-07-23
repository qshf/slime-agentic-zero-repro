"""A2: AgentFlow 数据源 —— 镜像源 data_source_cls 的产物（nano 硬编码算术 QA）。

A2 的 executor 唯一工具是 calculator，故数据取**算术题**：question 是一道计算题，label 是数值答案
（可被 boxed 精确命中）。这逼 ReAct 循环真的走"planner 规划→executor 调 calculator→verifier STOP→
final_output 出 \\boxed{}"。

偏离（详见 docs/decisions/a2.md）：硬编码算术 QA 而非真数据集（源 prepare_data 生成）——结构等价
（prompt/label 二件套）。label 取整数字符串，boxed + is_equiv（去空格/小写）能稳定命中。
"""

from __future__ import annotations

from toy_rl.sample import Sample

_QA: list[dict] = [
    {"prompt": "What is 37 * 24?", "label": "888"},
    {"prompt": "What is 1234 + 5678?", "label": "6912"},
    {"prompt": "What is (144 / 12) + 7?", "label": "19"},
    {"prompt": "What is 256 - 89?", "label": "167"},
]


def load_data_source(args) -> list[Sample]:
    """对齐源 data_source_cls(args)：返回算术 QA 的 Sample 列表（AgentFlow 路径）。"""
    return [Sample(prompt=q["prompt"], label=q["label"]) for q in _QA]
