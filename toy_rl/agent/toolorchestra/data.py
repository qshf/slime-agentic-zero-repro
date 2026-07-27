"""A3 QA 数据源：源 ToolOrchestra data 的最小 metadata 形态。"""

from __future__ import annotations

from toy_rl.sample import Sample


_TOOLS = [
    {
        "name": "search",
        "description": "Retrieve supplied evidence before answering.",
        "parameters": {"query": "short retrieval query"},
    },
    {
        "name": "call_expert",
        "description": "Ask one expert to produce the final answer.",
        "parameters": {"expert": "expert_fast or expert_precise"},
    },
]


def _metadata(search_context: str, pref_vec: dict[str, float]) -> dict:
    return {
        "category": "qa",
        "tools": _TOOLS,
        "search_context": search_context,
        "model_mapping": {
            "expert_fast": "nano/fast",
            "expert_precise": "nano/precise",
        },
        # 单位是每 token 的虚拟价格，专门让 reward 的成本组件可观测。
        "tool_pricing": {
            "nano/fast": {"input": 0.000001, "output": 0.000002},
            "nano/precise": {"input": 0.000006, "output": 0.000012},
        },
        "pref_vec": pref_vec,
        "cost_budget": 0.006,
        "latency_budget_ms": 2_000.0,
    }


def load_data_source(args) -> list[Sample]:
    """返回 QA 样本；每项有工具、专家映射、价格和偏好向量。"""
    return [
        Sample(
            prompt="What is the sum of squares of the odd numbers from 1 to 10?",
            label="165",
            metadata=_metadata(
                "Odd numbers from 1 to 10 are 1, 3, 5, 7, 9. Their squares sum to 165.",
                {"accuracy": 1.0, "cost": 0.35, "latency": 0.25, "expert_fast": 0.8, "expert_precise": 0.2},
            ),
        ),
        Sample(
            prompt="What is the factorial of 6?",
            label="720",
            metadata=_metadata(
                "The factorial 6! equals 6 * 5 * 4 * 3 * 2 * 1 = 720.",
                {"accuracy": 1.0, "cost": 0.2, "latency": 0.2, "expert_fast": 0.4, "expert_precise": 0.9},
            ),
        ),
    ]
