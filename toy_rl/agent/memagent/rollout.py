"""A1: MemAgent rollout —— 镜像源 agentic/memagent/rollout.py（1:1 对齐结构）。

MemAgent 的 chunk-by-chunk 记忆更新流水线（源 rollout.py:4-9）：

    for chunk in split(context):
        memory = LLM(problem, memory, chunk)   # 记忆更新轮
    answer = LLM(problem, memory)              # 最终回答轮（\\boxed{}）

**每一轮是一条独立的训练序列**（context-independent multi-conversation，源 rollout.py:10）：memory 以
**文本**跨轮传递，不是让 token 上下文一直增长。这与主线一 calculator 的"一条增长对话 + 工具返回拼回"
根本不同，也是 MemAgent 单引擎/无工具的由来。

**A1 核心教学点：loss_mask 全 1**（源 rollout.py:134/148 `[1]*len(token_ids)`）——每轮 response 的每个
token 都参与训练，**没有工具返回段要置 0**。这正好和 A2 AgentFlow 的"executor token=0"（工具边界精华课）
形成对照：MemAgent 是最简 loss_mask，AgentFlow 才引入边界。

范式对齐（铁律）：源 memagent 的 chunk 循环**就写在 generate 里**（不像 agentflow 委托 solver，memagent
没有 solver）。故 A1 忠实地把记忆循环放在本文件的 `_run_memory_loop` 里、generate 只做薄适配——按**对应源
结构**对齐，而非套用 calculator 的"loop 独立模块"（那是因 agentflow 有 solver）。real / stub 两个 generate
只在"LLM 调用"这一处不同（注入 chat_fn），循环本身只有一份。

偏离说明（详见 docs/decisions/a1.md 偏离表）：
  - **char 级分 chunk**：源用真 tokenizer 按 token 切（MEM_CHUNK_TOKENS）；nano 无真 tokenizer（沿用
    V1-V5），按**字符** args.mem_chunk_chars 切。语义等价（都把长 context 切段喂记忆循环），真 tokenizer 留 V6。
  - **假 token / 无真 log_probs**：源走 /generate 拿真 token_ids+log_probs；nano 走 chat 端点 + 字符级假
    token（V1-V5 已定）。loss_mask 程序化构造等价。
  - **reward 归一化取最简**：源 _strip_string 有 10+ LaTeX/分数替换；nano 按路线图只做主流程（去空格/换行/
    小写），边界 case 不追平。
  - **reward 不做 turn 均摊**：源 custom_convert 把 reward 均摊到各 turn（Multi-Conv RL）；nano fake trainer
    不算真梯度，均摊不影响闭环验证，记录设计、真训练 credit 分配留主线三。
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from openai import OpenAI

from mini_slime.args import Args  # noqa: F401  (re-export，保持 `from ...rollout import Args` 可用)
from toy_rl.sample import Sample

# loss_mask 段类型（与 V0/V1 概念对齐）——MemAgent 只用到 PROMPT/RESPONSE 两种，无 TOOL 段。
_MASK_PROMPT = 0     # 每轮的 prompt（problem+memory+chunk）不训练
_MASK_RESPONSE = 1   # 每轮模型生成的 memory / answer 全训练（A1 核心：无工具边界，全 1）

# ── Prompt 模板（从源 rollout.py:46-74 搬来，保持文本对齐）───────────────────────────
_MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

_FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

_NO_MEMORY = "No previous memory"

# chat_fn 契约：给定一轮的 prompt 文本 + 是否最终轮，返回模型文本。real 打 SGLang，stub 造假答案。
ChatFn = Callable[[str, bool], Awaitable[str]]


def _tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """字符级假 tokenizer（沿用 V1-V5 约定；真 tokenizer 留 V6）。跨轮共享一个 vocab。"""
    return [vocab.setdefault(ch, len(vocab)) for ch in text]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """去掉 Qwen3 混合推理模型的 <think>…</think> 段。

    Qwen3.5-4B 是混合推理模型，默认每轮先吐一大段思考。若把它当"更新后的记忆"传给下一轮，
    记忆会被思考噪声污染、最终轮也答不出 \\boxed{}。故对 chat 输出剥掉 think 段（配合下面
    enable_thinking=False 双保险）。对齐源 memagent `_strip_stop_tokens` 的同类"清洗生成文本"角色。
    """
    return _THINK_RE.sub("", text).strip()


def _split_chunks(context: str, chunk_chars: int, max_chunks: int) -> list[str]:
    """按**字符**把 context 切成若干 chunk（对齐源 rollout.py:115-118 按 token 切，nano 无真 tokenizer）。"""
    return [context[i: i + chunk_chars] for i in range(0, len(context), chunk_chars)][:max_chunks]


async def _run_memory_loop(args: Args, sample: Sample, chat_fn: ChatFn) -> Sample:
    """MemAgent 记忆更新循环的**唯一实现处**（real / stub 都调它，只换 chat_fn）。

    对齐源 rollout.py:120-179：逐 chunk 更新记忆 → 最终回答轮 → 每轮建独立训练序列 → 拼接进 sample。
    """
    question = sample.prompt
    context = sample.metadata.get("context", "") if isinstance(sample.metadata, dict) else ""

    chunks = _split_chunks(context, args.mem_chunk_chars, args.mem_max_chunks)
    vocab: dict[str, int] = {}
    turns: list[dict] = []
    memory = _NO_MEMORY

    async def _emit_turn(prompt_text: str, response_text: str) -> None:
        """建一轮独立训练序列：prompt 段 loss_mask=0，response 段 **全 1**（对齐源 rollout.py:131-136）。"""
        ptoks = _tokenize(prompt_text, vocab)
        rtoks = _tokenize(response_text, vocab)
        turns.append({
            "tokens": ptoks + rtoks,
            "response_length": len(rtoks),
            "loss_mask": [_MASK_RESPONSE] * len(rtoks),  # 只存 response 段的 mask，拼接时再补 prompt 段的 0
        })

    # --- 记忆更新轮 ---
    for chunk in chunks:
        prompt_text = _MEMORY_TEMPLATE.format(prompt=question, memory=memory, chunk=chunk)
        resp = await chat_fn(prompt_text, False)
        memory = resp.strip() or memory
        await _emit_turn(prompt_text, resp)

    # --- 最终回答轮（\\boxed{}）---
    final_prompt = _FINAL_TEMPLATE.format(prompt=question, memory=memory)
    final_resp = await chat_fn(final_prompt, True)
    await _emit_turn(final_prompt, final_resp)

    # --- 拼接兼容字段（对齐源 rollout.py:165-179）：所有 turn 首尾相接进 sample ---
    cat_tokens: list[int] = []
    cat_loss_mask: list[int] = []
    for t in turns:
        p_len = len(t["tokens"]) - t["response_length"]
        cat_tokens += t["tokens"]
        cat_loss_mask += [_MASK_PROMPT] * p_len + t["loss_mask"]

    sample.response = final_resp
    sample.tokens = cat_tokens
    sample.loss_mask = cat_loss_mask
    sample.metadata["final_output"] = final_resp
    sample.metadata["turns"] = turns   # 供教学/调试：几个记忆轮 + 1 最终轮
    return sample


def _sglang_chat_fn(args: Args) -> ChatFn:
    """real chat_fn：打 SGLang chat completion（无 tool stop，记忆/答案自然结束）。"""
    client = OpenAI(base_url=args.sglang_base_url, api_key="EMPTY")
    loop = asyncio.get_event_loop()

    def _call(prompt_text: str, max_tokens: int) -> str:
        resp = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            temperature=args.temperature,
            # 关掉 Qwen3 思考段：off-the-shelf 混合推理模型会先吐一大段 think 把 token 预算耗尽、
            # 甚至对着模板 meta-rambling 而不做任务（实测 reward=0 的根因）。源 MemAgent 用的是为记忆
            # 任务微调过的模型、无此问题；nano 接现成模型故显式关思考，让它直接产记忆/答案。
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""

    async def chat_fn(prompt_text: str, is_final: bool) -> str:
        max_tokens = args.mem_max_final if is_final else args.mem_max_memory
        # 同步 OpenAI 客户端丢线程池，别阻塞事件循环（同 calculator_agent）
        text = await loop.run_in_executor(None, _call, prompt_text, max_tokens)
        return _strip_think(text)  # 双保险：剥掉可能残留的 <think> 段，避免污染记忆/答案

    return chat_fn


async def generate(args: Args, sample: Sample) -> Sample:
    """对齐源 memagent/rollout.py:93 的 generate 签名（薄适配器：注入真 SGLang chat_fn 后调共享循环）。"""
    return await _run_memory_loop(args, sample, _sglang_chat_fn(args))


# ── reward_func（从源 rollout.py:197-297 搬来，归一化取最简版）─────────────────────────

def _last_boxed_only_string(string: str) -> str | None:
    """取最后一个 \\boxed{...}（支持嵌套花括号）。源 rollout.py:197 的 nano 版。"""
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i, right_brace_idx, num_left = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_left += 1
        if string[i] == "}":
            num_left -= 1
            if num_left == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx: right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def _strip_string(s: str) -> str:
    """归一化——nano 只做主流程（去换行/空格 + 小写）；源 _strip_string 的 LaTeX/分数替换按路线图不追平。"""
    return s.replace("\n", "").replace(" ", "").lower()


def _is_equiv(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return _strip_string(a) == _strip_string(b)
    except Exception:
        return a == b


async def reward_func(args: Args, sample: Sample) -> dict:
    """计算 reward：只看正确性，0.0 或 1.0。

    args 参数保留用于接口统一，但本函数不使用。
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    final_output = metadata.get("final_output", "") or sample.response or ""
    label = str(sample.label) if sample.label is not None else ""

    boxed = _last_boxed_only_string(final_output)
    pred = _remove_boxed(boxed).strip() if boxed is not None else ""
    score = 1.0 if (label and _is_equiv(pred, label)) else 0.0
    return {"reward": score, "pred": pred}
