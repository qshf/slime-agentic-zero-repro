"""A2: AgentFlow 数据源 —— 镜像源 data_source_cls 的产物（nano 硬编码多步计算 QA）。

A2 的工具是 python_coder（生成并执行 Python），故数据取需要**多步计算**的题：逼出
plan→exec→verify 多轮，且答案适合"写段代码算出来"。label 是数值答案（可被 boxed 精确命中）。

偏离（详见 docs/decisions/a2.md）：硬编码 mini QA 而非真数据集（源 prepare_data 生成）——结构等价
（prompt/label 二件套）。label 取整数字符串，boxed + is_equiv（去空格/小写）能稳定命中。
"""

from __future__ import annotations

from toy_rl.sample import Sample

_QA: list[dict] = [
    {"prompt": "What is the sum of squares of the odd numbers from 1 to 10?", "label": "165"},
    {"prompt": "What is the factorial of 6?", "label": "720"},
    {"prompt": "What is the sum of the first 20 positive integers?", "label": "210"},
    {"prompt": "How many prime numbers are there below 30?", "label": "10"},
]


def load_data_source(args) -> list[Sample]:
    """对齐源 data_source_cls(args)：返回多步计算 QA 的 Sample 列表（AgentFlow 路径）。"""
    return [Sample(prompt=q["prompt"], label=q["label"]) for q in _QA]
