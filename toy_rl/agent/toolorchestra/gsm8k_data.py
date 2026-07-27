"""V6 GSM8K 数据源：主线三真训练用。

对齐源 ToolOrchestra QA data 的 metadata 形态（tools / model_mapping / tool_pricing / pref_vec /
budget），但把源的 STEM/HLE + FAISS search 换成 **GSM8K + 确定性 calculator 工具**：

  ① 源怎么做：源 QA 数据是 STEM/HLE 网络知识题，search 工具调 FAISS 检索服务拿证据。
  ② 为何偏离：nano 教学复现要"调工具才答对 + 答案可严格验证 + base 有提升空间"，且不起检索服务。
     GSM8K 小学数学应用题答案是 `#### 数字`（可严格验证）、4B 裸答多步算术会错、天然需 calculator。
  ③ 语义是否等价：**结构等价**——calculator 同样是"非终止工具、结果回灌下一轮 prompt"（对齐源
     SearchRetrievalTool 的结构语义），answer 同样是"终止工具、路由到 expert 产最终答案"。数据集
     换成可验证数学题是为了 nano 的 RL 有效性实验，偏离登记见 docs/decisions/v6.md。

答案抽取：GSM8K 的 answer 字段形如 "...<<48/2=24>>... #### 72"，label 取 `#### ` 之后的数字。
"""

from __future__ import annotations

import re

from toy_rl.sample import Sample


_TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression (digits and + - * / ( ) . only). "
        "Use it to compute intermediate results before answering.",
        "parameters": {"expression": "e.g. (3+5)*2"},
    },
    {
        "name": "call_expert",
        "description": "Ask one expert to produce the final numeric answer, ending with \\boxed{}.",
        "parameters": {"expert": "expert_fast or expert_precise"},
    },
]


def _extract_gold(answer_field: str) -> str:
    """GSM8K answer 字段以 `#### 数字` 结尾，抽出该数字为 label（去逗号）。"""
    match = re.search(r"####\s*(.+)\s*$", answer_field.strip())
    gold = match.group(1) if match else answer_field.strip()
    return gold.replace(",", "").strip()


def _metadata() -> dict:
    """每条 GSM8K 样本共享的工具/专家/价格/偏好 metadata（对齐源 QA metadata 形态）。"""
    return {
        "category": "qa",
        "tools": _TOOLS,
        # calculator 是确定性工具（非 LLM），无 model 映射；answer 路由到两个逻辑专家。
        "model_mapping": {
            "expert_fast": "nano/fast",
            "expert_precise": "nano/precise",
        },
        # 单位是每 token 的虚拟价格，专门让 reward 的成本组件可观测（对齐源 tool_pricing 形态）。
        "tool_pricing": {
            "nano/fast": {"input": 0.000001, "output": 0.000002},
            "nano/precise": {"input": 0.000006, "output": 0.000012},
        },
        # accuracy 权重压倒性大：GSM8K 首要是答对，成本/延迟/路由偏好是次要区分项。
        "pref_vec": {
            "accuracy": 1.0,
            "cost": 0.1,
            "latency": 0.1,
            "expert_fast": 0.3,
            "expert_precise": 0.1,
        },
        "cost_budget": 0.02,
        "latency_budget_ms": 5_000.0,
    }


def _load_rows(split: str, n: int, local_dir: str):
    """加载 GSM8K 某 split 的前 N 行。

    服务器连不上 HF（CLAUDE.md 已记）：优先读本地 parquet（local_dir/<split>-00000-of-00001.parquet）；
    不存在时回退 datasets.load_dataset("openai/gsm8k")（本地开发有网）。返回 [{question, answer}, ...]。
    """
    import os

    parquet_path = os.path.join(local_dir, f"{split}-00000-of-00001.parquet")
    if os.path.isfile(parquet_path):
        import pandas as pd

        frame = pd.read_parquet(parquet_path).head(n)
        return frame.to_dict("records")

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [dataset[i] for i in range(min(n, len(dataset)))]


def _load_split(split: str, n: int, local_dir: str) -> list[Sample]:
    return [
        Sample(
            prompt=str(row["question"]),
            label=_extract_gold(str(row["answer"])),
            metadata=_metadata(),
        )
        for row in _load_rows(split, n, local_dir)
    ]


def load_data_source(args) -> list[Sample]:
    """训练数据源：GSM8K train 前 N 条（对齐源 data_source_cls 的可切换加载）。"""
    return _load_split("train", args.gsm8k_num_train, args.gsm8k_local_dir)


def load_eval_source(args) -> list[Sample]:
    """eval 数据源：GSM8K test 前 N 条，量 before/after 答对率（held-out，不参与训练）。"""
    return _load_split("test", args.gsm8k_num_eval, args.gsm8k_local_dir)
