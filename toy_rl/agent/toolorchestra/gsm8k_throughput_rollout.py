"""Single-turn GSM8K rollout used to build reproducible throughput workloads.

Unlike the ToolOrchestra agent loop, this hook makes exactly one SGLang
``/generate`` request. It keeps the same token/log-prob contract as the real
agent rollout but removes tool and expert latency from learner throughput.
"""

from __future__ import annotations

import asyncio

import requests

from mini_slime.args import Args
from toy_rl.agent.memagent.rollout import _is_equiv
from toy_rl.sample import Sample

from .rollout import GenOutput, _prediction, _sglang_gen_fn, _strip_think, _tokenizer


def _prompt(question: str) -> str:
    return (
        "Solve the following GSM8K math word problem. Show concise reasoning and end "
        "with the final numeric answer in \\boxed{}.\n\n"
        f"Question: {question}"
    )


def _apply_output(sample: Sample, prompt_text: str, output: GenOutput) -> Sample:
    """Populate the standard rollout fields from one SGLang generation result."""
    prompt_len = len(output.prompt_token_ids)
    generated_len = len(output.generated_token_ids)
    if generated_len == 0 or generated_len != len(output.generated_log_probs):
        raise ValueError("SGLang generation must return aligned non-empty token log-probs")

    sample.response = output.response
    sample.tokens = list(output.prompt_token_ids) + list(output.generated_token_ids)
    sample.loss_mask = [0] * prompt_len + [1] * generated_len
    sample.rollout_log_probs = [0.0] * prompt_len + list(output.generated_log_probs)
    sample.metadata["final_output"] = output.response
    # custom_convert produces one training row per turn. Keep a single direct
    # turn, so the collector and the full agent share that output contract.
    sample.metadata["turns"] = [{
        "kind": "direct",
        "full_token_ids": list(sample.tokens),
        "generated_length": generated_len,
        "generated_loss_mask": [1] * generated_len,
        "generated_log_probs": list(output.generated_log_probs),
        "prompt_text": prompt_text,
        "response": output.response,
    }]
    return sample


async def generate(args: Args, sample: Sample) -> Sample:
    """Generate one answer and retain exact behavior-policy log-probabilities."""
    prompt_text = _prompt(str(sample.prompt))
    output = await _sglang_gen_fn(args)(prompt_text)
    return _apply_output(sample, prompt_text, output)


async def generate_batch(args: Args, samples: list[Sample]) -> list[Sample]:
    """Generate a single-turn GSM8K batch in one SGLang ``/generate`` call."""
    if not samples:
        return []
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

    def request_batch() -> list[dict]:
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
        _apply_output(
            sample,
            prompt_text,
            GenOutput(
                response=_strip_think(str(output.get("text", ""))),
                prompt_token_ids=list(token_ids),
                generated_token_ids=[int(pair[1]) for pair in pairs],
                generated_log_probs=[float(pair[0]) for pair in pairs],
            ),
        )
    return samples


async def reward_func(args: Args, sample: Sample) -> dict:
    """Strict GSM8K correctness reward, matching the ToolOrchestra contract."""
    del args
    prediction = _prediction(str(sample.metadata.get("final_output", "")))
    label = str(sample.label) if sample.label is not None else ""
    correctness = float(bool(prediction and label and _is_equiv(prediction, label)))
    return {"reward": correctness, "pred": prediction}
