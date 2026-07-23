"""A2 AgentFlow —— 主线二第二个真实 agent。

镜像源 agentic/agentflow/：Planner→Executor→Verifier 多角色 ReAct 循环。
教学核心 = "executor token 不训练"：只有 Planner 的生成进 training turns（loss_mask=1），
Executor/Verifier/final_output 交给固定引擎跑、完全不进 turns（对 loss 零贡献）。
详见 docs/decisions/a2.md。
"""
