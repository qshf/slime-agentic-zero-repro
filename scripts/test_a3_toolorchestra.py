"""A3 ToolOrchestra QA 验证：messages 工具反馈、专家路由、多组件 reward、闭环。

  python scripts/test_a3_toolorchestra.py --offline  # 确定性离线 + 闭环
  python scripts/test_a3_toolorchestra.py            # 服务器 SGLang 真实 QA rollout
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime import train_ray
from mini_slime.args import Args
from toy_rl.agent.toolorchestra import data, rollout, stub_rollout
from toy_rl.agent.toolorchestra.solver import OrchestraSolver


def _offline_args() -> Args:
    return Args(
        num_rollout=1,
        batch_size=2,
        data_source_path="toy_rl.agent.toolorchestra.data.load_data_source",
        custom_generate_function_path="toy_rl.agent.toolorchestra.stub_rollout.generate",
        custom_rm_path="toy_rl.agent.toolorchestra.rollout.reward_func",
    )


def _server_args() -> Args:
    return Args(
        data_source_path="toy_rl.agent.toolorchestra.data.load_data_source",
        custom_generate_function_path="toy_rl.agent.toolorchestra.rollout.generate",
        custom_rm_path="toy_rl.agent.toolorchestra.rollout.reward_func",
    )


async def _single_sample(args: Args) -> None:
    sample = await stub_rollout.generate(args, data.load_data_source(args)[0])
    turns = sample.metadata["turns"]
    events = sample.metadata["events"]
    assert len(turns) == 2, "stub 应走 search -> answer 两轮"
    assert [event["tool_name"] for event in events] == ["search", "answer"]
    assert all(turn["kind"] == "orchestrator" for turn in turns)
    assert all(turn["loss_mask"] == [1] * turn["response_length"] for turn in turns)
    assert "[TOOL name=search]" in turns[1]["prompt_text"], "search observation 必须进入下一轮 messages"
    assert sum(sample.loss_mask) == sum(turn["response_length"] for turn in turns)
    reward = await rollout.reward_func(args, sample)
    assert reward["reward"] == 1.0  # 简化版只返回 correctness
    assert reward["pred"] == "165"


async def _error_reroute(args: Args) -> None:
    calls = 0

    async def orchestrator(prompt_text: str) -> str:
        nonlocal calls
        calls += 1
        if '"status": "error"' in prompt_text:
            return '<tool_call>{"name":"answer","arguments":{"expert":"expert_fast"}}</tool_call>'
        return '<tool_call>{"name":"answer","arguments":{"expert":"expert_precise"}}</tool_call>'

    async def expert(prompt_text: str) -> str:
        if "expert_precise" in prompt_text:
            raise RuntimeError("precise endpoint unavailable")
        return "Recovered by fast expert. \\boxed{165}"

    sample = data.load_data_source(args)[0]
    sample = await OrchestraSolver(orchestrator, expert, max_steps=3).solve(args, sample)
    assert calls == 2
    assert [event["status"] for event in sample.metadata["events"]] == ["error", "ok"]
    assert "precise endpoint unavailable" in sample.metadata["turns"][1]["prompt_text"]


def test_single_sample(args: Args) -> None:
    asyncio.run(_single_sample(args))
    asyncio.run(_error_reroute(args))
    print("  single_sample + error_reroute OK")


def test_closed_loop(args: Args) -> None:
    metrics = train_ray.train(args)
    assert len(metrics) == 1
    assert metrics[0]["tokens_per_rollout"] > 0
    assert 0.0 <= metrics[0]["reward_mean"] <= 1.0
    print("  closed_loop OK")


async def _server_sample(args: Args) -> None:
    sample = await rollout.generate(args, data.load_data_source(args)[0])
    turns = sample.metadata["turns"]
    assert turns, "真实 orchestrator 至少应产生一轮"
    assert all(turn["kind"] == "orchestrator" for turn in turns)
    assert len(sample.tokens) == len(sample.loss_mask)
    assert sum(sample.loss_mask) == sum(turn["response_length"] for turn in turns)
    reward = await rollout.reward_func(args, sample)
    assert 0.0 <= reward["reward"] <= 1.0
    print(f"  server turns={len(turns)} events={len(sample.metadata['events'])} reward={reward['reward']:.3f}")


def test_server_sample(args: Args) -> None:
    asyncio.run(_server_sample(args))


def main() -> int:
    offline = "--offline" in sys.argv
    args = _offline_args() if offline else _server_args()
    tests = (test_single_sample, test_closed_loop) if offline else (test_server_sample,)
    failed = 0
    for test in tests:
        try:
            test(args)
        except Exception as exc:
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    ray.shutdown()
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
