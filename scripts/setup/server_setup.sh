#!/bin/bash
# V1 环境初始化脚本（在服务器 5090 上运行一次）
# 用法: ssh 5090 'bash -s' < scripts/server_setup.sh
# 或:   ssh 5090 'cd /home/ubuntu/zj && bash slime-agentic-zero-repro/scripts/server_setup.sh'

set -e

REPO_DIR="/home/ubuntu/zj/slime-agentic-zero-repro"
VENV_DIR="$REPO_DIR/.venv"

echo "=== 1. 拉取/克隆仓库 ==="
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR" && git pull
else
    mkdir -p /home/ubuntu/zj
    cd /home/ubuntu/zj
    git clone git@github.com:qshf/-slime-agentic-zero-repro.git
fi

echo "=== 2. 创建 Python 虚拟环境 ==="
cd "$REPO_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# V1 只需要 openai client（调 SGLang 的 OpenAI 兼容接口）
pip install -q openai

echo "=== 3. 验证 V0 契约在服务器上跑通 ==="
python scripts/test_v0_contract.py

echo ""
echo "=== 服务器环境初始化完成 ==="
echo "下一步: 启动 SGLang 容器，运行 scripts/start_sglang.sh"
