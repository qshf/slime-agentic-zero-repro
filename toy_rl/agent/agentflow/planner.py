"""A2: AgentFlow 规划器 —— 镜像源 agentic/agentflow/core/planner.py:Planner。

**Planner 是 A2 里唯一被训练的角色**：`plan()`（第 0 轮）+ `generate_next_step()`（第 1、2… 轮）
的生成 token 会被 solver 打成 training turns（loss_mask=1）。它跑在**训练引擎**上（权重每轮更新，
就是要优化的策略）。`generate_final_output()` 虽然也定义在 Planner 上（对齐源），但**用固定引擎**跑、
不进 turns——最终答案由不变权重的基座模型产出，保证 reward 信号稳定。

范式对齐（铁律）：
  - prompt 模板文本从源 planner.py:80-169 逐字搬来（保持与源可比）。
  - **偏离**：源 Planner 持有真 `SGLangEngine`（`llm_engine.generate(messages)` 返 GenerationOutput，
    自带 prompt_token_ids/token_ids/log_probs）；nano 沿用 V1-V5/A1 的**字符级假 tokenizer**、走 chat
    端点，故 Planner 只持有一个 `chat_fn(prompt_text)->text`，**tokenize/建 turn 交给 solver**（对齐
    源 solver.solve 里"turns.append(...tokenize...)"的分工）。方法因此返回 `(prompt_text, response)`
    二元组而非 GenerationOutput——语义等价（solver 拿到 prompt 文本 + 生成文本即可建训练序列）。
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .memory import Memory

# chat_fn 契约：给定一段 prompt 文本，返回模型文本。real 打 SGLang，stub 造假答案。
ChatFn = Callable[[str], Awaitable[str]]


class Planner:
    def __init__(self, chat_fn: ChatFn, available_tools: list[str], toolbox_metadata: dict) -> None:
        self.chat_fn = chat_fn                        # 训练引擎（planner 唯一策略）
        self.available_tools = available_tools
        self.toolbox_metadata = toolbox_metadata

    # ── Plan（训练序列 #0）────────────────────────────────────────────────────
    def build_plan_prompt(self, question: str) -> str:
        return f"""Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

"""

    async def plan(self, question: str) -> tuple[str, str]:
        prompt_text = self.build_plan_prompt(question)
        response = await self.chat_fn(prompt_text)
        return prompt_text, response

    # ── next_step（训练序列 #1、#2…）────────────────────────────────────────────
    def build_next_step_prompt(self, query: str, query_analysis: str, memory: Memory) -> str:
        return f"""
Task: Determine the optimal next step to address the query using available tools and previous steps.
Context:
- **Query:** {query}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.

"""

    async def generate_next_step(self, query: str, memory: Memory, query_analysis: str) -> tuple[str, str]:
        prompt_text = self.build_next_step_prompt(query, query_analysis, memory)
        response = await self.chat_fn(prompt_text)
        return prompt_text, response

    # ── final_output（用固定引擎跑、不进 turns）──────────────────────────────────
    def build_final_output_prompt(self, query_analysis: str, question: str, memory: Memory) -> str:
        return f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Initial Analysis:** {query_analysis}
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.
3. You MUST end your response with the final answer enclosed in \\boxed{{}}. For example: \\boxed{{42}}.
"""

    async def generate_final_output(
        self, query_analysis: str, question: str, memory: Memory, chat_fn: ChatFn
    ) -> tuple[str, str]:
        """对齐源 planner.generate_final_output(..., llm_engine=fixed)：显式收固定引擎 chat_fn，
        **不**用 self.chat_fn（训练引擎），确保最终答案由不变权重基座模型产出。"""
        prompt_text = self.build_final_output_prompt(query_analysis, question, memory)
        response = await chat_fn(prompt_text)
        return prompt_text, response
