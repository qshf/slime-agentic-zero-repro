"""A1 行为验证：MemAgent（单引擎 / 无工具 / loss_mask 全 1）插进主线一闭环。

断言（对齐 docs/decisions/a1.md 验证清单）：
  1. 多轮结构：turns == chunks + 1（N 个记忆更新轮 + 1 最终回答轮）。
  2. **loss_mask 全 1 不变量**（A1 核心）：每轮 response 段 loss_mask 全 1、无工具边界置 0；
     拼接后 sum(loss_mask) == sum(response_lengths)，tokens/loss_mask 等长。
  3. boxed reward：final_output 里 \\boxed{label} 被抽取、is_equiv 命中 → reward=1.0（离线 stub）。
  4. 闭环：memagent 数据源 + generate + reward 插进 mini_slime（train_ray）跑通，reward_mean/weight_version 正确。
  5. 主线一未破坏：calculator 默认数据源路径仍产出原题库（回归靠 test_v3/v4/v5）。

用法:
  python scripts/test_a1_memagent.py --offline   # 本地: stub 记忆循环(不连 SGLang) + 闭环
  python scripts/test_a1_memagent.py             # 服务器: 真 SGLang(Qwen3.5-4B@30001) 记忆循环
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "toy_rl").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError(f"Could not locate project root from {_HERE}")

import ray

from mini_slime.args import Args
from mini_slime import train_ray
from toy_rl.agent.memagent import data as mem_data
from toy_rl.agent.memagent import rollout as mem_rollout
from toy_rl.agent.memagent import stub_rollout as mem_stub


def _offline_args() -> Args:
    """离线：memagent 数据源 + stub 记忆循环（不连 SGLang）+ memagent reward。chunk 调小逼出多轮。"""
    return Args(
        num_rollout=1,
        batch_size=2,
        mem_chunk_chars=200,
        data_source_path="toy_rl.agent.memagent.data.load_data_source",
        custom_generate_function_path="toy_rl.agent.memagent.stub_rollout.generate",
        custom_rm_path="toy_rl.agent.memagent.rollout.reward_func",
    )


def _server_args() -> Args:
    """服务器：真 SGLang 记忆循环（默认 Qwen3.5-4B@30001）。"""
    return Args(
        num_rollout=1,
        batch_size=2,
        mem_chunk_chars=400,
        data_source_path="toy_rl.agent.memagent.data.load_data_source",
        custom_generate_function_path="toy_rl.agent.memagent.rollout.generate",
        custom_rm_path="toy_rl.agent.memagent.rollout.reward_func",
    )


async def _one_sample(args: Args, generate) -> None:
    """不变量 1/2/3：直接跑一条 sample，逐轮查 loss_mask 全 1 + boxed reward。"""
    sample = mem_data.load_data_source(args)[0]
    n_chunks = len(mem_rollout._split_chunks(
        sample.metadata["context"], args.mem_chunk_chars, args.mem_max_chunks))
    sample = await generate(args, sample)

    turns = sample.metadata["turns"]
    print(f"  [{sample.prompt[:40]}...] chunks={n_chunks} turns={len(turns)} "
          f"tokens={len(sample.tokens)} trainable={sum(sample.loss_mask)}")

    # 1) 多轮结构：N 记忆轮 + 1 最终轮
    assert len(turns) == n_chunks + 1, f"turns {len(turns)} != chunks+1 {n_chunks + 1}"
    # 2) loss_mask 全 1 不变量：每轮 response 段全 1、无 0；response_length>0
    for i, t in enumerate(turns):
        assert t["response_length"] > 0, f"turn {i} response 为空"
        assert t["loss_mask"] == [1] * t["response_length"], f"turn {i} loss_mask 非全 1（不该有工具边界）"
    assert len(sample.tokens) == len(sample.loss_mask), "tokens/loss_mask 不等长"
    assert sum(sample.loss_mask) == sum(t["response_length"] for t in turns), "可训练 token 数 != 各轮 response 之和"
    # 3) boxed reward
    r = await mem_rollout.reward_func(args, sample)
    print(f"    final_output={sample.response!r} pred={r['pred']!r} reward={r['reward']}")
    assert 0.0 <= r["reward"] <= 1.0, f"reward 越界 {r['reward']}"
    return r["reward"]


def test_single_sample(args: Args, offline: bool) -> None:
    generate = mem_stub.generate if offline else mem_rollout.generate
    reward = asyncio.run(_one_sample(args, generate))
    if offline:
        assert reward == 1.0, f"离线 stub 应命中 boxed(label) → reward=1.0，得 {reward}"
    print("  single_sample OK")


def test_closed_loop(args: Args, offline: bool) -> None:
    """不变量 4：memagent 插进 mini_slime 闭环（train_ray）跑通。"""
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
    print(f"Running A1 MemAgent tests ({'offline/stub' if offline else 'server/SGLang'})...")
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
