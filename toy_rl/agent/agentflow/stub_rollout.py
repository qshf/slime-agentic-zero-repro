"""A2 离线测试用：no-SGLang / no-DeepSeek 的 AgentFlow stub generate。

同 V4/V5/A1 的 stub —— rollout 在 RolloutManager actor 的独立进程里，主进程 monkeypatch 不到，
故离线跑闭环时把 `custom_generate_function_path` 指到这里。它复用 rollout.py 里**同一个**
`_build_solver` + `Solver.solve`（agent loop 只有一份），只注入三个"造假"chat_fn：

  - **planner chat_fn**（训练引擎位，SGLang 打桩）：plan 轮产一句规划；next_step 轮产可被 formatters
    解析的 `Context / Sub-Goal / Tool Name: Python_Code_Generator_Tool`。
  - **fixed chat_fn**（固定 base 引擎位，SGLang 打桩）：executor 轮产 `tool.execute(query="…")`；
    verifier 轮首步即 `Conclusion: STOP`；final_output 轮产 `\\boxed{label}`。
  - **coder chat_fn**（python_coder 内部模型位，DeepSeek 打桩）：**不调 DeepSeek**，返回一段确定的
    Python 代码块 `print(<label>)`——但 python_coder **真跑 subprocess** 执行它。这样离线就验证了
    "工具边界 + python_coder 真执行"，不需要 GPU / 外部 API。

离线 coder 打桩说明（偏离）：真 DeepSeek 会把自然语言 query 翻成计算代码；stub coder 直接 print 已知
label（它在 stub 作用域可见），只为让 subprocess 真跑、stdout 确定。这不削弱工具边界断言（executor/
coder 的 token 本就不进 turns），只把"NL→code 翻译"这一步替换成确定输出。真翻译在服务器 e2e 验证。

reward_func 直接复用 rollout.py 的（re-export 供 hook 加载）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mini_slime.args import Args
from toy_rl.agent.agentflow.rollout import _build_solver, reward_func  # noqa: F401  (reward re-export)
from toy_rl.sample import Sample


async def generate(args: Args, sample: Sample) -> Sample:
    """stub：注入三个造假 chat_fn 后调共享的 _build_solver/Solver.solve（不连 SGLang / 不调 DeepSeek）。"""
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["original_question"] = sample.prompt

    label = str(sample.label) if sample.label is not None else ""
    tool_name = "Python_Code_Generator_Tool"

    async def planner_chat_fn(prompt_text: str) -> str:
        if "Determine the optimal next step" in prompt_text:  # next_step 轮
            return (
                "Justification: The query needs a multi-step computation; the code tool solves it.\n"
                f"Context: {sample.prompt}\n"
                "Sub-Goal: Compute the requested value with a short Python script.\n"
                f"Tool Name: {tool_name}\n"
            )
        # plan 轮（或其它）——产一句规划即可（不影响解析）
        return "Analysis: this is a computational query; plan to use the Python code tool."

    async def fixed_chat_fn(prompt_text: str) -> str:
        if "Generate a precise command" in prompt_text:       # executor 轮：产自然语言命令
            return (
                "Generated Command:\n```python\n"
                f'execution = tool.execute(query="""{sample.prompt}""")\n```'
            )
        if "Evaluate if the current memory" in prompt_text:   # verifier 轮
            return "The computed result is now in memory. Conclusion: STOP"
        if "Generate the final output" in prompt_text:        # final_output 轮
            return f"The computation is complete. \\boxed{{{label}}}"
        return "VERDICT: True"                                # judge 轮（reward 回退兜底）

    async def coder_chat_fn(system_prompt: str, query: str) -> str:
        """python_coder 内部模型打桩：返回确定 Python 代码块（真 subprocess 执行）。"""
        return f"```python\nprint({label})\n```"

    solver = _build_solver(args, planner_chat_fn, fixed_chat_fn, coder_chat_fn)
    return await solver.solve(args, sample)
