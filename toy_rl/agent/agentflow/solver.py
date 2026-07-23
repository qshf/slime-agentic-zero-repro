"""A2: AgentFlow 编排器 —— 本版的星，镜像源 agentic/agentflow/core/solver.py:Solver.solve。

**A2 教学核心全在这里：只有 planner 的两类调用被打成 training turns（loss_mask=1），
executor / verifier / final_output 的生成完全不进 turns。** 这就是"executor token 不训练"在系统层
的落地——不训练的角色交给固定引擎跑（对 loss 零贡献），训练引擎只优化 planner 这个真正的策略。

ReAct 循环（对齐源 solver.py:65-212）：
    analysis = planner.plan(question)                        # 训练序列 #0  → _emit_turn
    for step in range(max_steps):
        next_step = planner.generate_next_step(...)          # 训练序列 #1、#2…  → _emit_turn
        ctx, sub_goal, tool = extract_context_subgoal_and_tool(next_step)
        cmd_query = executor.generate_tool_command(...)      # 固定引擎，不进 turns
        result   = executor.execute_command(tool, cmd_query) # 按 tool_name 真分发到工具，不进 turns
        memory.add_action(...)
        conclusion = verifier.verificate_context(...)        # 固定引擎，不进 turns
        if conclusion == "STOP": break
    final = planner.generate_final_output(...)               # 固定引擎，不进 turns（reward 从它算）

范式对齐（铁律）：
  - 结构 1:1 对齐源 solve()：turns 只在 plan/next_step 处 append；拼接段与源 solver.py:186-200 一致
    （首轮 response 起头，后续轮补 prompt 段的 0 + response 段的 mask）。
  - **偏离①（formatters 折进本文件）**：源 `extract_context_subgoal_and_tool` 独立在 core/formatters.py；
    仅 solver 用，nano 折成模块级 helper（微整合，语义等价——同一函数）。
  - **偏离②（字符级假 token）**：源 turn 存真 token_ids + log_probs（来自 SGLangEngine）；nano 无真
    tokenizer/log_probs（沿用 V1-V5/A1），turn 存字符级假 token、无 log_probs。loss_mask 构造等价。
  - **偏离③（Solver 收 chat_fn 而非 engine_map）**：源 Solver 收 `engine_map` dict（6 键→2 引擎）并
    在内部构造 Planner/Executor/Verifier；nano 直接收造好的 planner/executor/verifier/rewarder 与
    fixed_chat_fn（供 final_output），engine_map 的"双引擎"语义落在 rollout.py 造 chat_fn 处。等价。
"""

from __future__ import annotations

import re

from mini_slime.args import Args
from toy_rl.sample import Sample

from .executor import Executor
from .memory import Memory
from .planner import ChatFn, Planner
from .verifier import Verifier

# loss_mask 段类型（与 V0/V1/A1 概念对齐）
_MASK_PROMPT = 0     # 每个 turn 的 prompt 前缀不训练
_MASK_RESPONSE = 1   # planner 生成的 token 全训练（executor/verifier/final_output 根本不进 turns）


def _tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """字符级假 tokenizer（沿用 V1-V5/A1；真 tokenizer 留 V6）。跨 turn 共享一个 vocab。"""
    return [vocab.setdefault(ch, len(vocab)) for ch in text]


def extract_context_subgoal_and_tool(raw_response: str) -> tuple[str, str, str]:
    """从 next_step 响应抽 Context / Sub-Goal / Tool Name（源 core/formatters.py 折进本文件）。"""
    text = raw_response
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("**", "")
    pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return "", "", ""
    context, sub_goal, tool_name = matches[-1]
    return context.strip(), sub_goal.strip(), tool_name.strip()


class Solver:
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        verifier: Verifier,
        final_output_chat_fn: ChatFn,
        max_steps: int,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.final_output_chat_fn = final_output_chat_fn   # 固定引擎（final_output 不训练）
        self.max_steps = max_steps

    async def solve(self, args: Args, sample: Sample) -> Sample:
        """编排 ReAct 循环、只把 planner 两类调用建成 training turns，回填 sample 字段。"""
        question = sample.prompt
        memory = Memory()
        vocab: dict[str, int] = {}
        turns: list[dict] = []
        response_text = ""

        def _emit_turn(prompt_text: str, response: str) -> None:
            """建一条独立训练序列：prompt 段 loss_mask=0、planner response 段 **全 1**
            （对齐源 solver.py:80-85/102-107）。只有 planner 调用会走到这里。"""
            ptoks = _tokenize(prompt_text, vocab)
            rtoks = _tokenize(response, vocab)
            turns.append({
                "kind": "planner",
                "tokens": ptoks + rtoks,
                "response_length": len(rtoks),
                "loss_mask": [_MASK_RESPONSE] * len(rtoks),  # 拼接时再补 prompt 段的 0
            })

        # ── Plan（训练序列 #0）──
        plan_prompt, analysis = await self.planner.plan(question)
        _emit_turn(plan_prompt, analysis)
        response_text += f"===== [plan] =====\n{analysis}"

        # ── ReAct 循环 ──
        for step_count in range(self.max_steps):
            # next_step（训练序列 #1、#2…）—— 唯一进 turns 的循环内调用
            ns_prompt, next_step = await self.planner.generate_next_step(question, memory, analysis)
            _emit_turn(ns_prompt, next_step)
            response_text += f"\n\n===== [next_step {step_count}] =====\n{next_step}"

            context, sub_goal, tool_name = extract_context_subgoal_and_tool(next_step)
            if not tool_name:
                tool_name = self.planner.available_tools[0] if self.planner.available_tools else ""

            # executor 生成命令 + 按 tool_name 真分发执行（固定引擎，不进 turns；
            # python_coder→内部 coder 模型写代码→子进程 / base_generator→固定引擎直接答）
            cmd_query = await self.executor.generate_tool_command(question, context, sub_goal, tool_name)
            execution_result = await self.executor.execute_command(tool_name, cmd_query)
            memory.add_action(step_count, tool_name, sub_goal, cmd_query, execution_result)
            response_text += (
                f"\n===== [executor] tool={tool_name} query={cmd_query!r} =====\n{execution_result}"
            )

            # verifier 判 STOP/CONTINUE（固定引擎，不进 turns）
            conclusion = await self.verifier.verificate_context(question, analysis, memory)
            response_text += f"\n===== [verifier] conclusion={conclusion} ====="
            if conclusion == "STOP":
                break

        # ── final_output（固定引擎，不进 turns；reward 从它算）──
        final_prompt, final_output = await self.planner.generate_final_output(
            analysis, question, memory, self.final_output_chat_fn
        )
        response_text += f"\n\n===== [final_output] =====\n{final_output}"

        # ── 拼接兼容字段（对齐源 solver.py:186-200）：只有 planner turns 进 tokens/loss_mask ──
        cat_tokens: list[int] = []
        cat_loss_mask: list[int] = []
        for t in turns:
            p_len = len(t["tokens"]) - t["response_length"]
            cat_tokens += t["tokens"]
            cat_loss_mask += [_MASK_PROMPT] * p_len + t["loss_mask"]

        sample.response = response_text
        sample.tokens = cat_tokens
        sample.loss_mask = cat_loss_mask
        sample.metadata["final_output"] = final_output
        sample.metadata["turns"] = turns   # 供教学/断言：全是 planner turn，无 executor/verifier
        return sample
