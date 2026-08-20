#!/usr/bin/env python3
"""V9 online end-to-end benchmark: GSM8K rollout, learner, and weight publish.

This script measures orchestration throughput, not a controlled learner-only
comparison. Use ``v9_gsm8k_throughput.py --replay`` for the latter: replay
holds token rows and advantages fixed. Here every cell runs real online
rollouts, so generation length, reward mix, and async policy staleness are
reported as workload quality gates instead of hidden sources of variance.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_PATH = "/home/ubuntu/models/Qwen/Qwen3-0.6B"
SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_GEN_URL = "http://localhost:30000/generate"

GSM8K_GEN_PATH = "toy_rl.agent.toolorchestra.gsm8k_throughput_rollout.generate"
GSM8K_RM_PATH = "toy_rl.agent.toolorchestra.gsm8k_throughput_rollout.reward_func"
GSM8K_DATA_PATH = "toy_rl.agent.toolorchestra.gsm8k_data.load_data_source"
GRPO_CONVERT_PATH = "mini_slime.custom_convert.custom_convert"

CALCULATOR_GEN_PATH = "toy_rl.agent.calculator_hooks.generate"
CALCULATOR_RM_PATH = "toy_rl.agent.calculator_hooks.reward_func"
CALCULATOR_DATA_PATH = "toy_rl.agent.calculator_data.load_data_source"

CELLS: list[dict] = [
    dict(id="A", backend="torch", mode="sync", label="no-infra sync"),
    dict(id="B", backend="torch", mode="async", label="no-infra async"),
    dict(id="C", backend="megatron", mode="sync", label="Megatron TP=2 sync"),
    dict(id="D", backend="megatron", mode="async", label="Megatron TP=2 async"),
]


class CellResult(NamedTuple):
    wait_gen_median: float
    train_median: float
    sync_median: float
    end_to_end_step_median: float
    phase_total: float
    raw_reward_mean: float
    active_grpo_rate: float
    trainable_tokens_per_second: float
    model_tokens_per_second: float
    tflops_median: float
    loss_median: float
    policy_version_gap: float


def workload_paths(name: str) -> tuple[str, str, str, str]:
    if name == "gsm8k":
        return GSM8K_GEN_PATH, GSM8K_RM_PATH, GSM8K_DATA_PATH, GRPO_CONVERT_PATH
    return CALCULATOR_GEN_PATH, CALCULATOR_RM_PATH, CALCULATOR_DATA_PATH, ""


def make_args(backend: str, cli: argparse.Namespace):
    from mini_slime.args import Args

    gen_path, rm_path, data_path, convert_path = workload_paths(cli.workload)
    args = Args(
        train_backend=backend,
        train_model_path=cli.model_path,
        num_rollout=cli.rounds,
        batch_size=cli.prompts,
        n_samples_per_prompt=cli.samples_per_prompt,
        gsm8k_num_train=max(64, cli.prompts * cli.rounds),
        sglang_base_url=SGLANG_BASE_URL,
        sglang_generate_url=SGLANG_GEN_URL,
        custom_generate_function_path=cli.gen_path or gen_path,
        custom_rm_path=cli.rm_path or rm_path,
        data_source_path=cli.data_path or data_path,
        custom_convert_path=convert_path,
        rollout_temperature=cli.temperature,
        orchestra_max_tokens=cli.max_new_tokens,
        fake_train_seconds=0.0,
        fake_gen_seconds=0.0,
        update_weights_interval=1,
        use_tensor_weight_sync=(cli.weight_sync == "tensor"),
        megatron_qkv_format="thd",
        learner_trace=True,
        # A no-signal GRPO round has no train rows by definition. The benchmark
        # reports it through active_grpo_rate instead of rejecting the whole run.
        learner_contract_validate=False,
    )
    if cli.gsm8k_local_dir:
        args.gsm8k_local_dir = cli.gsm8k_local_dir
    if backend == "megatron":
        args.tensor_model_parallel_size = 2
    return args


def run_cell(backend: str, mode: str, cli: argparse.Namespace) -> list[dict]:
    import ray

    if ray.is_initialized():
        ray.shutdown()
    args = make_args(backend, cli)
    if mode == "sync":
        from mini_slime.train_ray import train
    else:
        from mini_slime.train_async import train
    metrics = train(args)
    ray.shutdown()
    return metrics


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _extract_metrics(metrics_log: list[dict], warmup: int) -> CellResult:
    steady = metrics_log[warmup:] if len(metrics_log) > warmup else metrics_log
    wait = [float(m.get("wait_gen_time", m.get("gen_time", 0.0))) for m in steady]
    train = [float(m.get("train_time", 0.0)) for m in steady]
    sync = [float(m.get("sync_time", 0.0)) for m in steady]
    step = [a + b + c for a, b, c in zip(wait, train, sync, strict=True)]
    train_seconds = sum(train)
    trainable_tokens = sum(int(m.get("trainable_tokens", 0)) for m in steady)
    model_tokens = sum(int(m.get("model_tokens", 0)) for m in steady)
    group_count = sum(int(m.get("grpo_group_count", 0)) for m in steady)
    active_groups = sum(int(m.get("grpo_active_group_count", 0)) for m in steady)

    return CellResult(
        wait_gen_median=_median(wait),
        train_median=_median(train),
        sync_median=_median(sync),
        end_to_end_step_median=_median(step),
        phase_total=sum(step),
        raw_reward_mean=float(np.mean([m.get("raw_reward_mean", m["reward_mean"]) for m in steady])),
        active_grpo_rate=(active_groups / group_count) if group_count else 0.0,
        trainable_tokens_per_second=(trainable_tokens / train_seconds) if train_seconds else 0.0,
        model_tokens_per_second=(model_tokens / train_seconds) if train_seconds else 0.0,
        tflops_median=_median([float(m.get("tflops", 0.0)) for m in steady]),
        loss_median=_median([float(m.get("loss", 0.0)) for m in steady]),
        policy_version_gap=_median([float(m.get("policy_version_gap", 0.0)) for m in steady]),
    )


def print_table(results: dict[str, CellResult], workload: str, min_active_rate: float) -> None:
    header = (
        f"{'cell':22s} {'wait(s)':>8} {'train(s)':>9} {'sync(s)':>8} {'e2e(s)':>8} "
        f"{'raw_r':>7} {'active':>8} {'train tok/s':>12} {'model tok/s':>12} "
        f"{'TFLOPs':>8} {'loss':>9} {'gap':>5}"
    )
    print("\n" + header)
    print("-" * len(header))
    for label, result in results.items():
        quality = "pass" if workload != "gsm8k" or result.active_grpo_rate >= min_active_rate else "FAIL"
        print(
            f"{label:22s} {result.wait_gen_median:8.3f} {result.train_median:9.3f} "
            f"{result.sync_median:8.3f} {result.end_to_end_step_median:8.3f} "
            f"{result.raw_reward_mean:7.3f} {result.active_grpo_rate:7.1%} "
            f"{result.trainable_tokens_per_second:12.1f} {result.model_tokens_per_second:12.1f} "
            f"{result.tflops_median:8.3f} {result.loss_median:9.5f} "
            f"{result.policy_version_gap:5.1f} {quality}"
        )


def save_csv(results: dict[str, CellResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *CellResult._fields])
        for label, result in results.items():
            writer.writerow([label, *result])
    print(f"CSV -> {path}")


def plot_2x2(results: dict[str, CellResult], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot")
        return

    labels = list(results)
    wait = [results[label].wait_gen_median for label in labels]
    train = [results[label].train_median for label in labels]
    sync = [results[label].sync_median for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, wait, label="wait/generate", color="#4c8cbf")
    ax.bar(x, train, bottom=wait, label="train", color="#e07b39")
    ax.bar(x, sync, bottom=np.array(wait) + np.array(train), label="weight publish", color="#5a9b69")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("median seconds per steady update")
    ax.set_title("V9 online GSM8K end-to-end phases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V9 online end-to-end A/B/C/D benchmark")
    parser.add_argument("--cell", choices=[cell["id"] for cell in CELLS], default=None)
    parser.add_argument("--workload", choices=("gsm8k", "calculator"), default="gsm8k")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--weight-sync", choices=("tensor", "disk"), default="tensor")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--gsm8k-local-dir", default=None)
    parser.add_argument("--min-active-grpo-rate", type=float, default=0.25)
    parser.add_argument("--fail-on-quality-gate", action="store_true")
    parser.add_argument("--gen-path", default=None, help="override the workload generate hook")
    parser.add_argument("--rm-path", default=None, help="override the workload reward hook")
    parser.add_argument("--data-path", default=None, help="override the workload data hook")
    parser.add_argument("--out-dir", default="docs/decisions")
    cli = parser.parse_args()
    if cli.rounds < 1 or cli.warmup_rounds < 0 or cli.repeats < 1:
        parser.error("rounds, warmup rounds, and repeats must be valid positive counts")
    if cli.prompts < 1 or cli.samples_per_prompt < 2:
        parser.error("GSM8K GRPO requires at least one prompt and two samples per prompt")
    if not 0.0 <= cli.min_active_grpo_rate <= 1.0:
        parser.error("--min-active-grpo-rate must be in [0, 1]")

    print("=== V9 online end-to-end benchmark ===")
    print(
        f"workload={cli.workload} rounds={cli.rounds} warmup={cli.warmup_rounds} "
        f"prompts/round={cli.prompts} samples/prompt={cli.samples_per_prompt} "
        f"temperature={cli.temperature} weight_sync={cli.weight_sync}"
    )
    if cli.workload == "gsm8k":
        print(
            "Quality gate: active GRPO groups must reach "
            f"{cli.min_active_grpo_rate:.0%}; raw reward and normalized loss are reported separately."
        )

    cells = [cell for cell in CELLS if cli.cell is None or cell["id"] == cli.cell]
    results: dict[str, CellResult] = {}
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        label = f"{cell['id']} {cell['label']}"
        print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
        runs: list[CellResult] = []
        for repeat in range(cli.repeats):
            print(f"-- repeat {repeat + 1}/{cli.repeats} --")
            t0 = time.perf_counter()
            metrics = run_cell(cell["backend"], cell["mode"], cli)
            result = _extract_metrics(metrics, cli.warmup_rounds)
            runs.append(result)
            print(
                f"wall={time.perf_counter() - t0:.1f}s e2e_step={result.end_to_end_step_median:.3f}s "
                f"trainable_tok/s={result.trainable_tokens_per_second:.1f} "
                f"active_grpo={result.active_grpo_rate:.1%} raw_reward={result.raw_reward_mean:.3f}"
            )
        results[label] = CellResult(*(
            _median([getattr(run, field) for run in runs]) for field in CellResult._fields
        ))

    print_table(results, cli.workload, cli.min_active_grpo_rate)
    save_csv(results, out_dir / "v9_end_to_end.csv")
    plot_2x2(results, out_dir / "v9_end_to_end.png")
    quality_failed = (
        cli.workload == "gsm8k"
        and any(result.active_grpo_rate < cli.min_active_grpo_rate for result in results.values())
    )
    if quality_failed:
        print("WARNING: one or more cells failed the GRPO quality gate; do not interpret their loss or learner rate.")
        if cli.fail_on_quality_gate:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
