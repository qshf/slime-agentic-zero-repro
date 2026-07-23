"""A2 AgentFlow 工具层 —— 镜像源 agentic/agentflow/tools/。

目前只含 python_coder（对齐源 tools/python_coder/tool.py）：executor 出自然语言命令 → 本工具
**内部调独立 coder 模型（DeepSeek）** 把 query 翻成 Python → subprocess 执行取 stdout。这一层的
LLM 调用是**纯环境行为**（不进 rollout trajectory、不训练），是源"三模型分工"的第三个模型。
"""
