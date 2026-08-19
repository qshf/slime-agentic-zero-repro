#!/usr/bin/env python3
"""Megatron DP: synchronous vs asynchronous rollout orchestration.

This benchmark fixes TP=1 and uses Megatron data parallelism (default DP=2),
so both modes exercise the same distributed trainer and differ only in where
the generation future is synchronized.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


MODEL_PATH = "/home/ubuntu/models/Qwen/Qwen3-0.6B"
SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_GEN_URL = "http://localhost:30000/generate"
GEN_PATH = "toy_rl.agent.calculator_hooks.generate"
RM_PATH = "toy_rl.agent.calculator_hooks.reward_func"
DATA_PATH = "toy_rl.agent.calculator_data.load_data_source"


def make_args(dp_size: int, rounds: int):
    from mini_slime.args import Args

    return Args(
        train_backend="megatron",
        train_model_path=MODEL_PATH,
        num_rollout=rounds,
        batch_size=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        megatron_data_parallel_size=dp_size,
        megatron_qkv_format="thd",
        sglang_base_url=SGLANG_BASE_URL,
        sglang_generate_url=SGLANG_GEN_URL,
        custom_generate_function_path=GEN_PATH,
        custom_rm_path=RM_PATH,
        data_source_path=DATA_PATH,
        fake_train_seconds=0.0,
        fake_gen_seconds=0.0,
        update_weights_interval=1,
        learner_trace=True,
        learner_contract_validate=False,
    )


def run_mode(mode: str, dp_size: int, rounds: int) -> tuple[list[dict], float]:
    import ray

    if ray.is_initialized():
        ray.shutdown()
    args = make_args(dp_size, rounds)
    if mode == "sync":
        from mini_slime.train_ray import train
    else:
        from mini_slime.train_async import train

    t0 = time.time()
    metrics = train(args)
    wall = time.time() - t0
    ray.shutdown()
    return metrics, wall


def summarize(metrics: list[dict], wall: float) -> dict[str, object]:
    steady = metrics[1:] if len(metrics) > 1 else metrics
    wait = [m.get("wait_gen_time", m.get("gen_time", 0.0)) for m in steady]
    train = [m["train_time"] for m in steady]
    sync = [m.get("sync_time", 0.0) for m in steady]
    losses = [float(m["loss"]) for m in metrics if "loss" in m]
    total = sum(
        m.get("wait_gen_time", m.get("gen_time", 0.0))
        + m["train_time"]
        + m.get("sync_time", 0.0)
        for m in metrics
    )
    return {
        "wait_gen_median": float(np.median(wait)),
        "train_median": float(np.median(train)),
        "sync_median": float(np.median(sync)),
        "total": total,
        "wall": wall,
        "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "loss_delta": (losses[-1] - losses[0]) if len(losses) >= 2 else float("nan"),
        "losses": losses,
        "tokens": [m.get("tokens_per_rollout", 0) for m in metrics],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Megatron DP sync/async comparison")
    ap.add_argument("--dp-size", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out-dir", default="docs/decisions")
    args = ap.parse_args()
    if args.dp_size < 1 or args.rounds < 1:
        ap.error("--dp-size and --rounds must be positive")

    results = {}
    print(f"=== Megatron DP sync/async (TP=1, DP={args.dp_size}, THD) ===")
    for mode in ("sync", "async"):
        print(f"\n--- {mode} ---")
        metrics, wall = run_mode(mode, args.dp_size, args.rounds)
        result = summarize(metrics, wall)
        results[mode] = result
        print(
            f"{mode}: wait_gen={result['wait_gen_median']:.3f}s "
            f"train={result['train_median']:.3f}s "
            f"sync={result['sync_median']:.3f}s total={result['total']:.2f}s "
            f"wall={result['wall']:.1f}s"
        )
        print(
            f"  loss={result['losses']} mean={result['loss_mean']:.6f} "
            f"delta(first->last)={result['loss_delta']:.6f} tokens={result['tokens']}"
        )

    sync, async_ = results["sync"], results["async"]
    print("\n--- comparison ---")
    print(f"total async/sync = {async_['total'] / sync['total']:.3f}x")
    print(f"wall async/sync = {async_['wall'] / sync['wall']:.3f}x")
    print(f"loss mean async-sync = {async_['loss_mean'] - sync['loss_mean']:.6f}")
    print("Note: loss comparison is meaningful only when token batches match; inspect tokens above.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "v9_megatron_dp_sync_async.txt"
    path.write_text(repr(results) + "\n", encoding="utf-8")
    print(f"Result -> {path}")


if __name__ == "__main__":
    main()
