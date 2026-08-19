"""Single-turn GSM8K rollout used to build reproducible throughput workloads.

Unlike the ToolOrchestra agent loop, this hook makes exactly one SGLang
``/generate`` request. It keeps the same token/log-prob contract as the real
agent rollout but removes tool and expert latency from learner throughput.
"""

from __future__ import annotations

from mini_slime.args import Args
from toy_rl.agent.memagent.rollout import _is_equiv
from toy_rl.sample import Sample

from .rollout import GenOutput, _prediction, _sglang_gen_fn


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


async def reward_func(args: Args, sample: Sample) -> dict:
    """Strict GSM8K correctness reward, matching the ToolOrchestra contract."""
    del args
    prediction = _prediction(str(sample.metadata.get("final_output", "")))
    label = str(sample.label) if sample.label is not None else ""
    correctness = float(bool(prediction and label and _is_equiv(prediction, label)))
    return {"reward": correctness, "pred": prediction}
