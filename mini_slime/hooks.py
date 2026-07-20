"""V2: custom generate / reward hook 的动态加载。

对齐源项目 slime/utils/misc.py:9 的 load_function —— slime 框架本身不认识任何具体
agent，它只通过命令行参数拿到一个字符串路径(如 "agentic.agentflow.rollout.generate")，
用 importlib 动态加载成函数对象。这就是 --custom-generate-function-path 等 hook 的实现原理。

V2 的痛点解决:
  V1 里主循环直接 `from calculator_agent import generate` —— 换 agent 要改代码。
  V2 后主循环只认签名，agent 通过配置路径注入，换 agent 零改代码。
"""

from __future__ import annotations

import importlib
from typing import Callable


def load_function(path: str) -> Callable:
    """按 "module.submodule.func" 路径动态加载函数对象。

    与源项目 slime/utils/misc.py:9 逐行对齐:
      >>> load_function("toy_rl.agent.calculator_agent.generate")
      <function generate at 0x...>
    """
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"路径必须是 '模块.函数' 形式，得到: {path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# --- hook 契约签名(仅作文档/类型提示，运行时由 load_function 加载真身) ---
#
# generate hook  : async def generate(args, sample) -> Sample
#   对齐源项目   : agentic/agentflow/rollout.py:104
#   职责         : 把 sample 的 response/tokens/loss_mask 填好
#
# reward hook    : async def reward_func(args, sample) -> dict
#   对齐源项目   : agentic/agentflow/rollout.py:209
#   职责         : 计算 reward，返回 {"reward": float, ...}
