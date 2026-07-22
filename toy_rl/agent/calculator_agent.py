"""V1: 极简 calculator agent + Qwen3-0.6B 真推理。

从 V0 的"写死单样本"升级到"真实多轮交互生成样本"。
Qwen3-0.6B 通过 SGLang 的 OpenAI 兼容接口运行，这和源项目的架构形态一致:
  源项目: agentic/agentflow/rollout.py:104 generate() → core/solver.py:65 Solver.solve()
  V1:    calculator_agent.py: generate() → run_agent_loop()

V1 的核心教学点: 看清 tokens/loss_mask 如何从真实多轮对话里程序化生成——
  不再是手工填写，而是每轮交互结束后，按"谁生成的这段文字"来打 loss_mask。

范式对齐（铁律）:
  源项目把 agent loop 放在 Solver（一处），rollout.py 只是薄适配器调 solver.solve()。
  这里同样把 loop 收在 run_agent_loop（一处）:
    - V1 入口 generate(prompt, label)     —— 本文件下方，测试用
    - V2 hook generate(args, sample)      —— calculator_hooks.py，薄适配器
  两个入口都复用同一个 run_agent_loop，绝不各抄一份。
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from toy_rl.agent.tools import CalculatorTool, ToolRegistry
from toy_rl.sample import Sample

# SGLang 的 OpenAI 兼容接口，对齐源项目 agentic/agentflow/rollout.py 里的默认端口
SGLANG_BASE_URL = "http://localhost:30000/v1"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# loss_mask 的三种段类型（与 V0 保持概念对齐）
MASK_PROMPT = 0   # 题目/context，不训练
MASK_AGENT  = 1   # 模型生成，训练
MASK_TOOL   = 0   # 工具返回，不训练


# ── 工具层 ──────────────────────────────────────────────────────────────────
# 工具定义/注册/分发已抽到 toy_rl/agent/tools.py（对齐源项目 tools/ 与 core/executor 的分离）。
# 这里只建一个 registry 单例；加工具 = 往列表里加一个 BaseTool 子类，loop 一行不改。
# 源项目 Solver 经 Executor.execute_command 动态分发；nano 用 _REGISTRY.execute_command 对齐这个角色。

_REGISTRY = ToolRegistry([CalculatorTool()])

# 兼容旧测试：test_v1_rollout.py 直接断言 calculator("2 + 3") == "5"。
# 保留模块级 calculator 名，指向工具的 execute（纯内部重构，不改测试）。
calculator = CalculatorTool().execute


def _tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """简单的字符级 tokenizer（V1 教学用）。

    源项目用 SGLang 的真 tokenizer；V1 用字符级假 tokenizer，
    目的是让 tokens/loss_mask 的对应关系在 print 里肉眼可读。
    真 tokenizer 在 V6 接 SGLang RolloutManager 时自然引入。
    """
    return [vocab.setdefault(ch, len(vocab)) for ch in text]


# ── 轨迹累加器 ─────────────────────────────────────────────────────────────
# 把"追加一段文字 → 按 mask 类型打标 → 记进 turns"这三件事收进一个小对象，
# 免得 loop 里反复对 tokens / loss_mask / response 三个 list 手工 append（V1 旧代码的糙点）。

class _Trajectory:
    def __init__(self, prompt: str) -> None:
        self.vocab: dict[str, int] = {}
        self.tokens: list[int] = _tokenize(prompt, self.vocab)
        self.loss_mask: list[int] = [MASK_PROMPT] * len(self.tokens)
        self.response_chunks: list[str] = []
        # turns: 每轮一条结构化记录，对齐源项目 solver.py 的 turns[]（放进 metadata 供教学/调试）
        self.turns: list[dict] = []

    def emit(self, text: str, mask: int, *, kind: str, tool_result: str | None = None) -> None:
        toks = _tokenize(text, self.vocab)
        self.tokens.extend(toks)
        self.loss_mask.extend([mask] * len(toks))
        # 工具返回段不算模型生成的一"轮"，只作为上下文；只有 agent 段才记进 turns
        if mask == MASK_AGENT:
            self.response_chunks.append(text)
            turn = {"kind": kind, "text": text, "token_count": len(toks)}
            if tool_result is not None:
                turn["tool_result"] = tool_result
            self.turns.append(turn)
        else:
            self.response_chunks.append(text)

    @property
    def response(self) -> str:
        return "".join(self.response_chunks)


def _chat(client: OpenAI, model_name: str, messages: list[dict],
          max_tokens: int, temperature: float) -> str:
    """一次 SGLang chat completion，返回文本。stop 在工具/答案闭合标签处截断。"""
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</tool>", "</answer>"],
    )
    return resp.choices[0].message.content or ""


_SYSTEM_PROMPT = (
    "你是一个数学助手。如果需要计算，用 <tool>工具名: <参数></tool> 调用工具。"
    "得到工具结果后，给出最终答案: <answer>数字</answer>。"
    "注意: 只输出工具调用或最终答案，不要多余解释。\n"
    "可用工具:\n"
    # 工具清单/示例由 registry 从工具元数据动态渲染，对齐源项目"元数据注入 planner prompt"。
    + _REGISTRY.render_for_prompt()
)


async def run_agent_loop(
    client: OpenAI,
    model_name: str,
    prompt: str,
    *,
    max_turns: int = 5,
    max_tokens: int = 512,  # Qwen3 think 段较长，128 会截断 → 废轮；见 mini_slime/args.py 注释
    temperature: float = 0.0,
) -> tuple[_Trajectory, str | None]:
    """多轮 calculator agent loop——本项目里 agent 交互的唯一实现处。

    对齐源项目 core/solver.py:65 Solver.solve()：持有循环本身。
    V1 入口 generate() 与 V2 hook 都调它，绝不各写一份。

    每轮：
      模型判断要不要调工具
        -> 调工具: 生成 <tool>calculator: <expr></tool>，_REGISTRY.execute_command 执行，结果拼回上下文
        -> 出答案: 生成 <answer>...</answer>，结束

    返回 (trajectory, final_answer)。trajectory 里已备好 tokens/loss_mask/response/turns。
    """
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    traj = _Trajectory(prompt)
    final_answer: str | None = None
    loop = asyncio.get_event_loop()

    for turn in range(max_turns):
        # 同步 OpenAI 客户端丢到线程池，别阻塞事件循环
        agent_text = await loop.run_in_executor(
            None, _chat, client, model_name, messages, max_tokens, temperature
        )

        tool_match = re.search(r"<tool>calculator:\s*(.+?)(?:</tool>|$)", agent_text, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.+?)(?:</answer>|$)", agent_text, re.DOTALL)

        if tool_match:
            # --- 工具调用轮 ---
            agent_call = agent_text + "</tool>"          # 补回被 stop 截掉的闭合标签
            expr = tool_match.group(1).strip()
            tool_result = _REGISTRY.execute_command("calculator", expr)
            tool_text = f"\n[calculator结果: {tool_result}]\n"

            traj.emit(agent_call, MASK_AGENT, kind="tool_call", tool_result=tool_result)
            traj.emit(tool_text, MASK_TOOL, kind="tool_return")

            messages.append({"role": "assistant", "content": agent_call})
            messages.append({"role": "tool", "content": tool_text, "tool_call_id": f"t{turn}"})

        elif answer_match:
            # --- 最终答案轮 ---
            traj.emit(agent_text + "</answer>", MASK_AGENT, kind="answer")
            # 0.6B 小模型格式不稳，可能吐 <answer>5</</answer>；答案不含 '<'，截到第一个 '<' 前最鲁棒
            final_answer = answer_match.group(1).split("<", 1)[0].strip()
            break

        else:
            # 模型输出了别的，当普通 agent 文本继续
            traj.emit(agent_text, MASK_AGENT, kind="agent")
            messages.append({"role": "assistant", "content": agent_text})

    return traj, final_answer


async def generate(prompt: str, label: str, max_turns: int = 5) -> Sample:
    """V1 便捷入口（测试用）：跑一遍 loop，组装成 Sample。

    对齐源项目 rollout.py:104 generate() 的"薄"定位：不含 loop 逻辑，只调 run_agent_loop
    再填 Sample。final_answer 放进 metadata["final_output"]，对齐源项目
    sample.metadata["final_output"]（不再像旧代码返回 (Sample, final_answer) 元组）。
    """
    client = OpenAI(base_url=SGLANG_BASE_URL, api_key="EMPTY")
    traj, final_answer = await run_agent_loop(
        client, MODEL_NAME, prompt, max_turns=max_turns
    )
    return Sample(
        prompt=prompt,
        label=label,
        response=traj.response,
        tokens=traj.tokens,
        loss_mask=traj.loss_mask,
        reward=None,  # reward 由 reward_func 单独计算，V1 先留 None
        metadata={"final_output": final_answer, "turns": traj.turns},
    )
