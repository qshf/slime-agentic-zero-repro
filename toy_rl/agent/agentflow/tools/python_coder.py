"""A2: python_coder 工具 —— 镜像源 agentic/agentflow/tools/python_coder/tool.py。

**源 AgentFlow 的第三个模型**在这里：executor 出自然语言 query → 本工具**内部调独立 coder 模型**
把 query 翻成 Python 代码 → sanitize → subprocess 执行取 stdout。整条链是**纯环境行为**——不进
rollout trajectory、不训练（对齐源 rollout.py:129-134 的 `coder_engine`@30001，独立于 policy）。

三模型分工（对齐源 engine_map，rollout.py:136-144）：
  - planner（plan/next_step）= 正在训练的 policy（唯一训练目标）；
  - executor/verifier/final_output = 固定 base 模型（纯环境）；
  - **python_coder 内部 coder 模型 = 又一个独立模型**（纯环境）。

nano 忠实保留"两层结构 + coder 用独立模型"（不把写代码塞进 executor——那会把"规划"和"写代码"
两个本可独立训练的关注点耦合掉，篡改训练目标，违铁律）。nano 的 coder 模型 = **外部 DeepSeek API**
（OpenAI 兼容 SDK），密钥/base_url/model 走环境变量、不进 Args/不进 git。离线测试打桩 coder（不调
DeepSeek）但**真跑 subprocess**，以验证工具边界。

偏离（详见 docs/decisions/a2.md）：
  - coder = 外部 DeepSeek API（非本地第二个 SGLang 实例）：nano 不起第二个 SGLang；coder 是独立于
    policy 的纯环境模型，用外部 API 当它最贴源三分。语义等价（policy 之外的固定模型帮写代码，不进
    trajectory）。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Awaitable, Callable

_EXEC_TIMEOUT = 30           # 秒（对齐源 _EXEC_TIMEOUT）
_MAX_OUTPUT_LENGTH = 4000    # 对齐源 _MAX_OUTPUT_LENGTH
_DANGEROUS_CALLS = ["exit", "quit", "sys.exit", "os._exit"]

TOOL_NAME = "Python_Code_Generator_Tool"   # 对齐源外部工具名
TOOL_DESCRIPTION = (
    "A tool that generates and executes Python code snippets for calculations and "
    "math-related problems. It returns the printed output from the executed code."
)

# coder 的 system prompt（从源 execute() 搬来，去掉 LIMITATION/BEST_PRACTICE 长文，保主流程）
_CODER_SYSTEM_PROMPT = (
    "You are a Python code generator. "
    "Write a self-contained Python script that solves the problem and prints the final result. "
    "Always end the generated code with a print() statement so the result is captured. "
    "Return ONLY a single Python code block wrapped in ```python ... ```."
)

# coder chat_fn 契约：给定 (system, user) 返回模型文本。real 打 DeepSeek，stub 返回确定代码。
CoderChatFn = Callable[[str, str], Awaitable[str]]


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 30
    return text[:half] + "\n... (truncated) ...\n" + text[-half:]


def _sanitize_code(code: str) -> str:
    """去掉会杀进程的危险调用（对齐源 _sanitize_code）。"""
    sanitized = code
    for func in _DANGEROUS_CALLS:
        sanitized = re.sub(rf"{re.escape(func)}\s*\([^)]*\)", "pass", sanitized)
    return sanitized


def preprocess_code(text: str) -> str:
    """从 coder 响应抽第一个 Python 代码块（对齐源 preprocess_code 的三级回退）。"""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _run_code_in_subprocess(code: str, timeout: int) -> str:
    """在子进程跑代码、取 stdout（对齐源 _run_code_in_subprocess，超时可干净 kill）。"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Execution error: code timed out after {timeout}s"

    output = stdout.decode(errors="replace").strip()
    err_output = stderr.decode(errors="replace").strip()
    if proc.returncode != 0:
        return f"Execution error: {err_output or f'exit code {proc.returncode}'}"
    if not output:
        return f"(no stdout, stderr: {_truncate(err_output, _MAX_OUTPUT_LENGTH)})" if err_output else "(no output)"
    return _truncate(output, _MAX_OUTPUT_LENGTH)


class PythonCoderTool:
    """镜像源 Python_Coder_Tool：内部 coder LLM 写代码 → 子进程执行。永不抛异常（对齐源 execute）。"""

    tool_name = TOOL_NAME
    tool_description = TOOL_DESCRIPTION

    def __init__(self, coder_chat_fn: CoderChatFn) -> None:
        self.coder_chat_fn = coder_chat_fn   # 独立 coder 模型（DeepSeek / stub），纯环境、不训练

    async def execute(self, query: str) -> str:
        # Step 1: 独立 coder 模型把 query 翻成 Python（源 llm_engine.generate）
        try:
            response = await asyncio.wait_for(
                self.coder_chat_fn(_CODER_SYSTEM_PROMPT, query), timeout=120.0
            )
        except asyncio.TimeoutError:
            return "Code generation error: coder LLM timed out after 120s"
        except Exception as exc:
            return f"Code generation error: {exc}"

        # Step 2: 抽代码 + sanitize（源 preprocess_code → _sanitize_code）
        code = _sanitize_code(preprocess_code(response))

        # Step 3: 子进程执行（源 _run_code_in_subprocess）
        try:
            return await _run_code_in_subprocess(code, _EXEC_TIMEOUT)
        except Exception as exc:
            return f"Execution error: {exc}"


def deepseek_coder_chat_fn() -> CoderChatFn:
    """real coder chat_fn：打外部 DeepSeek API（OpenAI 兼容）。密钥/base_url/model 走环境变量。

    - DEEPSEEK_API_KEY（必填）
    - CODER_BASE_URL（默认 https://api.deepseek.com/v1）
    - CODER_MODEL（默认 deepseek-v4-flash）
    DeepSeek v4 是推理模型：只取 message.content（忽略 reasoning_content）。
    """
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("CODER_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("CODER_MODEL", "deepseek-v4-flash")
    client = OpenAI(api_key=api_key, base_url=base_url)
    loop = asyncio.get_event_loop()

    def _call(system_prompt: str, query: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=1024,   # 推理模型：reasoning 也吃 token，给足以吐出代码块
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    async def coder_chat_fn(system_prompt: str, query: str) -> str:
        return await loop.run_in_executor(None, _call, system_prompt, query)

    return coder_chat_fn
