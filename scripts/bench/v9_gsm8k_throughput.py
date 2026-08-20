#!/usr/bin/env python3
"""V9 GSM8K learner-throughput benchmark with a collect/replay boundary.

``collect`` obtains one fixed GRPO workload from SGLang and writes exact
tokens, masks, old log-probabilities, and normalized advantages to JSON.
``replay`` feeds that immutable workload repeatedly to one learner backend.
This is the valid trainer-throughput comparison: every configuration sees the
same THD segments and the same trainable-token denominator.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_PATH = "/home/ubuntu/models/Qwen/Qwen3-0.6B"
SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_GENERATE_URL = "http://localhost:30000/generate"
DIRECT_DATA_PATH = "toy_rl.agent.toolorchestra.gsm8k_data.load_data_source"
DIRECT_GENERATE_PATH = "toy_rl.agent.toolorchestra.gsm8k_throughput_rollout.generate"
DIRECT_REWARD_PATH = "toy_rl.agent.toolorchestra.gsm8k_throughput_rollout.reward_func"
GRPO_CONVERT_PATH = "mini_slime.custom_convert.custom_convert"
DEFAULT_PAPER_PATH = Path(__file__).with_name("data") / "v9_gsm8k_paper.json"


def load_paper(path: Path) -> tuple[dict[str, Any], str]:
    """Read and validate the versioned GSM8K prompt paper used for collection."""
    raw = path.read_bytes()
    paper = json.loads(raw)
    if paper.get("dataset") != "openai/gsm8k" or paper.get("config") != "main":
        raise ValueError("V9 paper must identify openai/gsm8k main")
    if paper.get("split") != "train":
        raise ValueError("V9 throughput paper must use the GSM8K train split")
    indices = paper.get("indices")
    if not isinstance(indices, list) or not indices or any(
        not isinstance(index, int) or index < 0 for index in indices
    ):
        raise ValueError("V9 paper requires non-negative integer indices")
    if len(set(indices)) != len(indices):
        raise ValueError("V9 paper indices must be unique")
    if not isinstance(paper.get("samples_per_prompt"), int) or paper["samples_per_prompt"] < 2:
        raise ValueError("V9 paper samples_per_prompt must be at least two")
    return paper, hashlib.sha256(raw).hexdigest()


def make_args(
    backend: str,
    prompts: int,
    samples_per_prompt: int,
    dp_size: int,
    gsm8k_local_dir: str | None,
    model_path: str,
    prompt_indices: tuple[int, ...],
):
    from mini_slime.args import Args

    args = Args(
        train_backend=backend,
        train_model_path=model_path,
        sglang_base_url=SGLANG_BASE_URL,
        sglang_generate_url=SGLANG_GENERATE_URL,
        orchestra_orchestrator_base_url=SGLANG_BASE_URL,
        orchestra_orchestrator_model="Qwen/Qwen3-0.6B",
        data_source_path=DIRECT_DATA_PATH,
        custom_generate_function_path=DIRECT_GENERATE_PATH,
        custom_rm_path=DIRECT_REWARD_PATH,
        custom_convert_path=GRPO_CONVERT_PATH,
        batch_size=prompts,
        n_samples_per_prompt=samples_per_prompt,
        gsm8k_num_train=len(prompt_indices),
        gsm8k_train_indices=prompt_indices,
        rollout_temperature=0.7,
        orchestra_max_tokens=192,
        megatron_qkv_format="thd",
        learner_contract_validate=True,
        learner_trace=True,
        update_weights_interval=0,
    )
    if gsm8k_local_dir:
        args.gsm8k_local_dir = gsm8k_local_dir
    if backend == "megatron":
        args.tensor_model_parallel_size = 1
        args.megatron_data_parallel_size = dp_size
    return args


def _validate_workload(data: dict[str, Any], *, require_trainable_rows: bool = True) -> None:
    required = ("tokens", "loss_masks", "rewards", "response_lengths", "rollout_log_probs")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"workload missing fields: {missing}")
    rows = len(data["tokens"])
    if rows == 0 or any(len(data[name]) != rows for name in required):
        raise ValueError("workload columns must be non-empty and aligned")
    for index, (tokens, mask, log_probs) in enumerate(
        zip(data["tokens"], data["loss_masks"], data["rollout_log_probs"], strict=True)
    ):
        if len(tokens) < 2 or len(mask) != len(tokens) or len(log_probs) != len(tokens):
            raise ValueError(f"invalid token-aligned row {index}")
        if require_trainable_rows and not any(mask):
            raise ValueError(f"workload row {index} has no trainable tokens")


def _drop_masked_rows(workload: dict[str, Any]) -> int:
    """Remove GRPO groups that custom_convert correctly marked as no-signal."""
    keep = [index for index, mask in enumerate(workload["loss_masks"]) if any(mask)]
    dropped = len(workload["tokens"]) - len(keep)
    row_columns = (
        "tokens",
        "loss_masks",
        "rewards",
        "response_lengths",
        "rollout_log_probs",
        "rollout_policy_versions",
    )
    for name in row_columns:
        if name in workload:
            workload[name] = [workload[name][index] for index in keep]
    return dropped


async def _collect_batch_request(manager, args, rollout_id: int, rollout_policy_version: int) -> dict[str, Any]:
    """Generate one full GRPO rollout batch through one SGLang /generate request.

    The regular RolloutManager contract is deliberately per-sample so that
    multi-turn agents can await tool calls. This throughput collector is a
    single-turn GSM8K-only path: batching its formatted prompts is semantically
    equivalent and lets SGLang schedule the whole batch together.
    """
    import asyncio
    import requests

    from toy_rl.agent.toolorchestra.gsm8k_throughput_rollout import _apply_output, _prompt
    from toy_rl.agent.toolorchestra.rollout import GenOutput, _strip_think, _tokenizer

    samples = manager._next_batch(rollout_id)
    tokenizer = _tokenizer(args.train_model_path)
    prompt_texts = [_prompt(str(sample.prompt)) for sample in samples]
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt_text in prompt_texts
    ]
    prompt_token_ids = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in formatted_prompts
    ]
    payload = {
        "text": formatted_prompts,
        "sampling_params": {
            "max_new_tokens": args.orchestra_max_tokens,
            "temperature": args.rollout_temperature,
        },
        "return_logprob": True,
    }

    def request_batch() -> list[dict[str, Any]]:
        response = requests.post(args.sglang_generate_url, json=payload, timeout=360)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, list) or len(result) != len(samples):
            raise ValueError("SGLang batch /generate response must align with request rows")
        return result

    outputs = await asyncio.get_running_loop().run_in_executor(None, request_batch)
    for sample, prompt_text, token_ids, output in zip(
        samples, prompt_texts, prompt_token_ids, outputs, strict=True
    ):
        pairs = output.get("meta_info", {}).get("output_token_logprobs", [])
        generated_ids = [int(pair[1]) for pair in pairs]
        generated_log_probs = [float(pair[0]) for pair in pairs]
        _apply_output(
            sample,
            prompt_text,
            GenOutput(
                response=_strip_think(str(output.get("text", ""))),
                prompt_token_ids=list(token_ids),
                generated_token_ids=generated_ids,
                generated_log_probs=generated_log_probs,
            ),
        )
        reward_result = await manager.reward_func(args, sample)
        sample.reward = reward_result["reward"]
        sample.rollout_policy_version = rollout_policy_version
        sample.metadata["reward_features"] = {"correctness": sample.reward}
    return manager.custom_convert(args, samples)


async def collect_workload(args, warmup_batches: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    from mini_slime.rollout_manager import RolloutManager

    manager = RolloutManager(args)
    for warmup_id in range(warmup_batches):
        await _collect_batch_request(manager, args, warmup_id, rollout_policy_version=0)
    # With a fixed paper, warmup IDs may wrap around the prompt list. Their data
    # is deliberately discarded; rollout_id only selects the source window.
    workload = await _collect_batch_request(manager, args, warmup_batches, rollout_policy_version=0)
    _validate_workload(workload, require_trainable_rows=False)
    dropped_rows = _drop_masked_rows(workload)
    _validate_workload(workload)
    metadata = {
        "schema_version": 1,
        "paper": getattr(args, "v9_paper_metadata", {}),
        "model_path": args.train_model_path,
        "prompts": args.batch_size,
        "samples_per_prompt": args.n_samples_per_prompt,
        "rollout_temperature": args.rollout_temperature,
        "warmup_batches": warmup_batches,
        "rows": len(workload["tokens"]),
        "masked_rows_dropped": dropped_rows,
        "model_tokens": sum(len(tokens) for tokens in workload["tokens"]),
        "trainable_tokens": sum(sum(mask) for mask in workload["loss_masks"]),
        "raw_reward_mean": (
            sum(workload.get("raw_rewards", [])) / len(workload.get("raw_rewards", []))
            if workload.get("raw_rewards") else 0.0
        ),
        "grpo_group_count": workload.get("grpo_group_count", 0),
        "grpo_active_group_count": workload.get("grpo_active_group_count", 0),
    }
    return workload, metadata


def write_workload(path: Path, workload: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "workload": workload}, indent=2), encoding="utf-8")


def load_workload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    workload = payload["workload"]
    _validate_workload(workload)
    return workload, payload.get("metadata", {})


def validate_collection_quality(
    metadata: dict[str, Any], min_active_groups: int, min_trainable_tokens: int
) -> None:
    """Reject a saved rollout that is too small or has too little GRPO signal.

    These are workload-quality gates, not learner correctness checks. Replaying a
    no-signal group is valid structurally but measures Python/Ray overhead rather
    than useful policy-gradient throughput.
    """
    if min_active_groups < 0 or min_trainable_tokens < 0:
        raise ValueError("collection quality thresholds must be non-negative")
    active_groups = int(metadata.get("grpo_active_group_count", 0))
    trainable_tokens = int(metadata.get("trainable_tokens", 0))
    failures = []
    if active_groups < min_active_groups:
        failures.append(f"active_grpo_groups={active_groups} < {min_active_groups}")
    if trainable_tokens < min_trainable_tokens:
        failures.append(f"trainable_tokens={trainable_tokens} < {min_trainable_tokens}")
    if failures:
        raise RuntimeError("collected workload failed quality gate: " + ", ".join(failures))


def _fixed_batches(workload: dict[str, Any], train_batch_size: int) -> list[dict[str, Any]]:
    """Split an immutable row pool into equal learner batches without reshuffling.

    A large collected workload provides length and reward diversity. Feeding all
    rows to one padded Torch forward can make the `[B, L, vocab]` logits exceed
    device memory, so the throughput benchmark cycles fixed equal-size batches.
    Dropping a final short tail keeps every timed update comparable.
    """
    if train_batch_size < 1:
        return [workload]
    rows = len(workload["tokens"])
    if train_batch_size > rows:
        raise ValueError(f"train_batch_size={train_batch_size} exceeds workload rows={rows}")
    usable_rows = rows - rows % train_batch_size
    if usable_rows == 0:
        raise ValueError("workload has no full learner batch")
    columns = (
        "tokens",
        "loss_masks",
        "rewards",
        "response_lengths",
        "rollout_log_probs",
        "rollout_policy_versions",
    )
    batches = []
    for start in range(0, usable_rows, train_batch_size):
        batch = {name: workload[name][start : start + train_batch_size] for name in columns if name in workload}
        batches.append(batch)
    return batches


def replay(
    args, workload: dict[str, Any], updates: int, warmup: int, train_batch_size: int = 0
) -> dict[str, Any]:
    import ray

    from mini_slime.learner_contract import validate_train_data
    from mini_slime.ray.placement_group import create_placement_groups, create_training_models

    batches = _fixed_batches(workload, train_batch_size)
    for batch in batches:
        validate_train_data(batch)
    create_placement_groups(args)
    group, _ = create_training_models(args)
    group.async_init(args)
    records = []
    try:
        for update in range(updates):
            batch = batches[update % len(batches)]
            t0 = time.perf_counter()
            metrics = ray.get(group.async_train(update, batch))[0]
            elapsed = time.perf_counter() - t0
            records.append({
                "update": update,
                "wall_seconds": elapsed,
                "model_tokens": sum(len(tokens) for tokens in batch["tokens"]),
                "trainable_tokens": sum(sum(mask) for mask in batch["loss_masks"]),
                **metrics,
            })
    finally:
        ray.shutdown()

    steady = records[warmup:] if len(records) > warmup else records
    steady_seconds = sum(record["wall_seconds"] for record in steady)
    steady_trainable_tokens = sum(record["trainable_tokens"] for record in steady)
    steady_model_tokens = sum(record["model_tokens"] for record in steady)
    return {
        "backend": args.train_backend,
        "megatron_dp": args.megatron_data_parallel_size if args.train_backend == "megatron" else 1,
        "updates": updates,
        "warmup_updates": warmup,
        "steady_updates": len(steady),
        "workload_rows": len(workload["tokens"]),
        "train_batch_size": train_batch_size or len(workload["tokens"]),
        "batch_count": len(batches),
        "model_tokens_per_update_median": statistics.median(record["model_tokens"] for record in steady),
        "trainable_tokens_per_update_median": statistics.median(
            record["trainable_tokens"] for record in steady
        ),
        "median_step_seconds": statistics.median(record["wall_seconds"] for record in steady),
        "trainable_tokens_per_second": steady_trainable_tokens / steady_seconds if steady_seconds else 0.0,
        "model_tokens_per_second": steady_model_tokens / steady_seconds if steady_seconds else 0.0,
        "median_loss": statistics.median(record.get("loss", 0.0) for record in steady),
        "median_tflops": statistics.median(record.get("tflops", 0.0) for record in steady),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V9 GSM8K fixed-workload learner throughput")
    parser.add_argument("--collect", action="store_true", help="collect one real SGLang rollout workload")
    parser.add_argument("--replay", action="store_true", help="replay the saved workload through a learner")
    parser.add_argument("--workload", default="/tmp/v9_gsm8k_throughput.json")
    parser.add_argument("--result", default="/tmp/v9_gsm8k_throughput_result.json")
    parser.add_argument("--backend", choices=("torch", "megatron"), default="megatron")
    parser.add_argument("--megatron-dp", type=int, default=1)
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="local model path; use /models/Qwen3-0.6B inside v9-dev",
    )
    parser.add_argument(
        "--paper",
        default=str(DEFAULT_PAPER_PATH),
        help="versioned GSM8K prompt paper; its indices and sampling count define collect",
    )
    parser.add_argument("--prompts", type=int, default=None, help="must match --paper when supplied")
    parser.add_argument(
        "--samples-per-prompt", type=int, default=None, help="must match --paper when supplied"
    )
    parser.add_argument(
        "--gsm8k-local-dir",
        default=None,
        help="directory containing train-00000-of-00001.parquet; required inside v9-dev unless mounted",
    )
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=0,
        help="rows per learner update during --replay; 0 replays the full workload",
    )
    parser.add_argument("--require-active-grpo", action="store_true")
    parser.add_argument(
        "--collect-warmup-batches",
        type=int,
        default=1,
        help="discard this many full SGLang batch requests before writing --workload",
    )
    parser.add_argument(
        "--min-active-grpo-groups",
        type=int,
        default=0,
        help="fail collection if fewer groups have non-zero GRPO variance",
    )
    parser.add_argument(
        "--min-trainable-tokens",
        type=int,
        default=0,
        help="fail collection if saved rows contain fewer trainable tokens",
    )
    cli = parser.parse_args()
    if not cli.collect and not cli.replay:
        parser.error("choose --collect, --replay, or both")
    paper_path = Path(cli.paper)
    paper, paper_sha256 = load_paper(paper_path)
    prompts = len(paper["indices"])
    samples_per_prompt = paper["samples_per_prompt"]
    if cli.prompts is not None and cli.prompts != prompts:
        parser.error("--prompts must match the fixed paper")
    if cli.samples_per_prompt is not None and cli.samples_per_prompt != samples_per_prompt:
        parser.error("--samples-per-prompt must match the fixed paper")
    if (
        cli.updates < 1
        or cli.warmup < 0
        or cli.train_batch_size < 0
        or cli.collect_warmup_batches < 0
    ):
        parser.error("invalid workload or update size")

    args = make_args(
        cli.backend,
        prompts,
        samples_per_prompt,
        cli.megatron_dp,
        cli.gsm8k_local_dir,
        cli.model_path,
        tuple(paper["indices"]),
    )
    args.v9_paper_metadata = {
        "paper_id": paper.get("paper_id", paper_path.name),
        "path": str(paper_path),
        "sha256": paper_sha256,
        "indices": paper["indices"],
    }
    path = Path(cli.workload)
    workload = None
    metadata: dict[str, Any] = {}
    if cli.collect:
        workload, metadata = asyncio.run(collect_workload(args, cli.collect_warmup_batches))
        write_workload(path, workload, metadata)
        print(f"workload={path} rows={metadata['rows']} model_tokens={metadata['model_tokens']} "
              f"trainable_tokens={metadata['trainable_tokens']} raw_reward={metadata['raw_reward_mean']:.3f} "
              f"active_grpo={metadata['grpo_active_group_count']}/{metadata['grpo_group_count']} "
              f"request_batch_rows={prompts * samples_per_prompt} warmup_batches={metadata['warmup_batches']}")
        if cli.require_active_grpo and metadata["grpo_active_group_count"] == 0:
            raise RuntimeError("collected workload has no active GRPO group; adjust temperature or prompt mix")
        validate_collection_quality(
            metadata, cli.min_active_grpo_groups, cli.min_trainable_tokens
        )
    if cli.replay:
        if workload is None:
            workload, metadata = load_workload(path)
        result = replay(args, workload, cli.updates, cli.warmup, cli.train_batch_size)
        result["workload"] = str(path)
        result["workload_metadata"] = metadata
        result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        Path(cli.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
