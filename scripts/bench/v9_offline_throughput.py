#!/usr/bin/env python3
"""V9 offline learner-throughput matrix over one immutable GSM8K workload.

This script never calls SGLang and never publishes weights. It measures only
learner work on saved token rows, masks, old log-probabilities, and GRPO
advantages. Therefore sync/async is deliberately absent: without online
generation there is no rollout that an async learner could overlap.

Create the workload once with ``v9_gsm8k_throughput.py --collect`` and the
versioned paper in ``scripts/bench/data``. Then use this script to run Torch
single-GPU and Megatron TP=2 against exactly the same rows.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bench.v9_gsm8k_throughput import (
    MODEL_PATH,
    load_workload,
    make_args,
    replay,
    validate_collection_quality,
)

DEFAULT_WORKLOAD = "/tmp/v9_gsm8k_offline_throughput.json"


def _gpu_snapshot() -> str:
    """Capture contention context without making the benchmark depend on it."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "nvidia-smi unavailable"


def _parse_configs(value: str) -> list[tuple[str, str, int, int]]:
    """Map public names to backend, TP, and DP without exposing Ray internals."""
    catalog = {
        "torch": ("torch", "torch", 1, 1),
        "megatron-tp1": ("megatron-tp1", "megatron", 1, 1),
        "megatron-tp2": ("megatron-tp2", "megatron", 2, 1),
        "megatron-tp1-dp2": ("megatron-tp1-dp2", "megatron", 1, 2),
    }
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--configs cannot be empty")
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(f"unknown offline config(s): {', '.join(unknown)}")
    return [catalog[name] for name in names]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "median_step_seconds",
        "trainable_tokens_per_second",
        "model_tokens_per_second",
        "median_loss",
        "median_tflops",
    )
    return {
        **{key: statistics.median(float(record[key]) for record in records) for key in keys},
        "runs": len(records),
        "per_run": records,
    }


def run_config(
    name: str,
    backend: str,
    tp_size: int,
    dp_size: int,
    workload: dict[str, Any],
    metadata: dict[str, Any],
    cli: argparse.Namespace,
) -> dict[str, Any]:
    prompts = int(metadata.get("prompts", 1))
    samples_per_prompt = int(metadata.get("samples_per_prompt", 2))
    # Data hooks are unused during replay. They are still configured so the
    # learner Args retain the same contract as the collection run.
    args = make_args(
        backend=backend,
        prompts=prompts,
        samples_per_prompt=samples_per_prompt,
        dp_size=dp_size,
        gsm8k_local_dir=cli.gsm8k_local_dir,
        model_path=cli.model_path,
        prompt_indices=tuple(range(prompts)),
    )
    if backend == "megatron":
        args.tensor_model_parallel_size = tp_size

    runs = []
    for run in range(cli.runs):
        print(f"[{name}] run {run + 1}/{cli.runs}", flush=True)
        runs.append(replay(args, workload, cli.updates, cli.warmup))
    return {
        "backend": backend,
        "tensor_parallel_size": tp_size,
        "data_parallel_size": dp_size,
        **_aggregate(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V9 offline GSM8K learner-throughput matrix")
    parser.add_argument("--workload", default=DEFAULT_WORKLOAD)
    parser.add_argument("--out", default="/tmp/v9_offline_throughput_result.json")
    parser.add_argument(
        "--configs",
        default="torch,megatron-tp2",
        help="comma-separated: torch, megatron-tp1, megatron-tp2, megatron-tp1-dp2",
    )
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-active-grpo-groups", type=int, default=8)
    parser.add_argument("--min-trainable-tokens", type=int, default=4096)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--gsm8k-local-dir", default=None)
    cli = parser.parse_args()
    if cli.updates < 2 or cli.warmup < 0 or cli.warmup >= cli.updates or cli.runs < 1:
        parser.error("require updates >= 2, 0 <= warmup < updates, and runs >= 1")

    configs = _parse_configs(cli.configs)
    workload_path = Path(cli.workload)
    workload, metadata = load_workload(workload_path)
    validate_collection_quality(
        metadata, cli.min_active_grpo_groups, cli.min_trainable_tokens
    )
    gpu_before = _gpu_snapshot()
    print(
        f"offline workload={workload_path} rows={len(workload['tokens'])} "
        f"trainable_tokens={metadata.get('trainable_tokens', 0)} "
        f"active_grpo={metadata.get('grpo_active_group_count', 0)}/"
        f"{metadata.get('grpo_group_count', 0)} updates={cli.updates} warmup={cli.warmup}",
        flush=True,
    )
    print(f"gpu_before={gpu_before}", flush=True)

    results = {}
    for name, backend, tp_size, dp_size in configs:
        results[name] = run_config(
            name, backend, tp_size, dp_size, workload, metadata, cli
        )
        summary = results[name]
        print(
            f"[{name}] step={summary['median_step_seconds']:.3f}s "
            f"trainable_tok/s={summary['trainable_tokens_per_second']:.1f} "
            f"model_tok/s={summary['model_tokens_per_second']:.1f} "
            f"loss={summary['median_loss']:.6f}",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "benchmark": "v9-offline-gsm8k-learner-throughput",
        "workload": str(workload_path),
        "workload_metadata": metadata,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_before": gpu_before,
        "gpu_after": _gpu_snapshot(),
        "updates": cli.updates,
        "warmup": cli.warmup,
        "runs": cli.runs,
        "results": results,
    }
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"result -> {out_path}")


if __name__ == "__main__":
    main()
