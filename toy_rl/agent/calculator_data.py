"""A1: calculator 数据源 loader —— 从 RolloutManager 抽出的硬编码题库。

对齐源 slime `data_source_cls = load_function(args.data_source_path)`：源的数据源是按路径动态
加载的**可切换**组件。主线一（V3-V5）时 nano 把 calculator 题库硬编码在 RolloutManager 里；A1 起
主线二要插入不同 agent（MemAgent 用带 context 的 QA），故把"数据从哪来"抽成 `data_source_path` hook。

本文件是 **calculator 路径的数据源**（V3-V5 默认），产出与旧 `_PROMPTS` **完全一致**的题目，保证
主线一回归零改动。MemAgent 的数据源见 toy_rl/agent/memagent/data.py。

偏离说明（详见 docs/decisions/a1.md）：
  - loader 直接返回 `list[Sample]`（源 data_source_cls(args) 是个从数据集流式取样的类）。nano 硬编码
    小集，返回 Sample 列表更直观；角色等价（都是 RolloutManager 的数据来源）。
"""

from __future__ import annotations

from toy_rl.sample import Sample

# 从 mini_slime/rollout_manager.py 原样搬来（V3 注释）：4B 心算能答对小算术会绕过工具，故用
# 大数乘除/多步表达式逼模型真去调 calculator。label 均由 eval 校验过。
_PROMPTS: list[tuple[str, str]] = [
    ("347 * 89 = ?", "30883"),
    ("638 * 47 = ?", "29986"),
    ("72 * 84 = ?", "6048"),
    ("(123 + 456) * 7 = ?", "4053"),
    ("1024 / 16 = ?", "64"),
    ("9876 - 5432 = ?", "4444"),
]


def load_data_source(args) -> list[Sample]:
    """对齐源 data_source_cls(args)：返回本轮训练可滚动取样的 Sample 列表（calculator 路径）。"""
    return [Sample(prompt=prompt, label=label) for prompt, label in _PROMPTS]
