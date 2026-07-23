"""A2 行为验证：AgentFlow（Planner→Executor→Verifier 多角色 / executor token 不训练）插进主线一闭环。

断言（对齐 docs/decisions/a2.md 验证清单 = plan 里的关键不变量 1-6）：
  1. 多轮结构：turns == 1(plan) + K(next_step)，K = 到 STOP 的步数（≤ af_max_steps）。
  2. **executor token 不训练（A2 核心）**：每个 turn loss_mask 全 1（planner 全训练）；
     sum(loss_mask) == sum(response_length) —— executor/verifier/final_output 对可训练 token 数零贡献；
     tokens/loss_mask 等长。
  3. 执行痕迹在 response 不在 tokens：turns 里全是 kind=="planner"，无 executor/verifier turn。
  4. reward：离线 stub \\boxed{label} 命中 → reward=1.0；reward∈[0,1]。
  5. 闭环：agentflow 数据源+generate+reward 插进 train_ray 跑通，reward_mean/weight_version 正确。
  6. 零回归：靠 test_v0/v2/v3/v4/v5/a1（本脚本外单独跑）。

用法:
  python scripts/test_a2_agentflow.py --offline   # 本地: stub 求解循环(不连 SGLang) + 闭环
  python scripts/test_a2_agentflow.py             # 服务器: 真 SGLang(planner→4B@30001, 固定引擎默认同端点)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime.args import Args
from mini_slime import train_ray
from toy_rl.agent.agentflow import data as af_data
from toy_rl.agent.agentflow import rollout as af_rollout
from toy_rl.agent.agentflow import stub_rollout as af_stub


def _offline_args() -> Args:
    """离线：agentflow 数据源 + stub 求解循环（不连 SGLang）+ agentflow reward。"""
    return Args(
        num_rollout=1,
        batch_size=2,
        af_max_steps=3,
        data_source_path="toy_rl.agent.agentflow.data.load_data_source",
        custom_generate_function_path="toy_rl.agent.agentflow.stub_rollout.generate",
        custom_rm_path="toy_rl.agent.agentflow.rollout.reward_func",
    )


def _server_args() -> Args:
    """服务器：真 SGLang（planner→Qwen3.5-4B@30001；固定引擎默认同端点，可改 af_fixed_base_url 指 30000）。"""
    return Args(
        num_rollout=1,
        batch_size=2,
        af_max_steps=3,
        data_source_path="toy_rl.agent.agentflow.data.load_data_source",
        custom_generate_function_path="toy_rl.agent.agentflow.rollout.generate",
        custom_rm_path="toy_rl.agent.agentflow.rollout.reward_func",
    )


async def _one_sample(args: Args, generate) -> float:
    """不变量 1/2/3/4：跑一条 sample，逐轮查 loss_mask + 只有 planner 进 turns + boxed reward。"""
    sample = af_data.load_data_source(args)[0]
    sample = await generate(args, sample)

    turns = sample.metadata["turns"]
    print(f"  [{sample.prompt}] turns={len(turns)} tokens={len(sample.tokens)} "
          f"trainable={sum(sample.loss_mask)}")

    # 1) 多轮结构：1 plan + K next_step（K≥1，≤ max_steps）
    assert len(turns) >= 2, f"turns {len(turns)} 应 ≥ 2（plan + ≥1 next_step）"
    assert len(turns) <= 1 + args.af_max_steps, f"turns {len(turns)} 超过 1+max_steps"
    # 2) executor token 不训练：每轮 loss_mask 全 1、response_length>0
    for i, t in enumerate(turns):
        assert t["response_length"] > 0, f"turn {i} response 为空"
        assert t["loss_mask"] == [1] * t["response_length"], f"turn {i} loss_mask 非全 1"
    assert len(sample.tokens) == len(sample.loss_mask), "tokens/loss_mask 不等长"
    assert sum(sample.loss_mask) == sum(t["response_length"] for t in turns), \
        "可训练 token 数 != 各轮 response 之和（executor/verifier/final_output 不应贡献）"
    # 3) turns 里全是 planner，无 executor/verifier turn
    assert all(t["kind"] == "planner" for t in turns), "turns 里出现了非 planner 的 kind"
    # 执行痕迹应在 response（供日志/reward）
    assert "[executor]" in sample.response and "[verifier]" in sample.response, \
        "response 里应有 executor/verifier 痕迹"
    # python_coder 真跑 subprocess：执行结果（label 的 stdout）应出现在 response 里
    label = str(sample.label)
    assert label in sample.response, f"python_coder 执行结果 {label!r} 未出现在 response（子进程未真执行？）"
    # 4) reward
    r = await af_rollout.reward_func(args, sample)
    print(f"    final_output={sample.metadata['final_output']!r} pred={r['pred']!r} reward={r['reward']}")
    assert 0.0 <= r["reward"] <= 1.0, f"reward 越界 {r['reward']}"
    return r["reward"]


def test_single_sample(args: Args, offline: bool) -> None:
    generate = af_stub.generate if offline else af_rollout.generate
    reward = asyncio.run(_one_sample(args, generate))
    if offline:
        assert reward == 1.0, f"离线 stub 应命中 boxed(label) → reward=1.0，得 {reward}"
    print("  single_sample OK")


def test_closed_loop(args: Args, offline: bool) -> None:
    """不变量 5：agentflow 插进 mini_slime 闭环（train_ray）跑通。"""
    metrics_log = train_ray.train(args)
    assert len(metrics_log) == args.num_rollout
    for m in metrics_log:
        for k in ("gen_time", "train_time", "reward_mean", "tokens_per_rollout", "weight_version"):
            assert k in m, f"metrics 缺字段 {k}"
        assert m["tokens_per_rollout"] > 0
        assert 0.0 <= m["reward_mean"] <= 1.0
    assert metrics_log[-1]["weight_version"] == args.num_rollout + 1  # 含训练前初始 update_weights
    if offline:
        assert metrics_log[-1]["reward_mean"] == 1.0, "离线闭环 reward_mean 应为 1.0"
    print("  closed_loop OK")


def main() -> int:
    offline = "--offline" in sys.argv
    print(f"Running A2 AgentFlow tests ({'offline/stub' if offline else 'server/SGLang'})...")
    args = _offline_args() if offline else _server_args()

    failed = 0
    for t in (test_single_sample, test_closed_loop):
        try:
            t(args, offline)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")

    ray.shutdown()
    print(f"\n{'FAILED' if failed else 'PASSED'} ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
