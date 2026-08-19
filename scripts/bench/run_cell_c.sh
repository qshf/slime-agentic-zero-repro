#!/bin/bash
# V9.2 Cell C 一键运行脚本（服务器上直接执行，避免 SSH 超时）
set -e

echo "=== V9.2 Cell C: Megatron TP=2 + 同步 ==="
echo ""

# 1. 清理所有旧的 v9_end_to_end 进程
echo "[1/4] 清理旧进程..."
pkill -9 -f "v9_end_to_end.py" 2>/dev/null || true
sleep 2
remaining=$(ps aux | grep "v9_end_to_end.py" | grep -v grep | wc -l)
echo "      残留进程: $remaining"

# 2. 确认 GPU 状态
echo "[2/4] 检查 GPU..."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo ""

# 3. 确认代码版本（检查关键修复是否都在）
cd /home/ubuntu/slime-agentic-zero-repro
current_commit=$(git rev-parse --short HEAD)
echo "[3/4] 当前代码: $current_commit"
# 检查 NCCL 后端修复是否在（d91316e 的关键字）
if ! git log --oneline -5 | grep -q "MegatronTrainer 多卡自动选 NCCL"; then
    echo "      缺少 NCCL 后端修复，先 git fetch && git reset --hard origin/v9"
    exit 1
fi
echo "      关键修复已就位（placement_group GPU + NCCL 后端）"

# 4. 跑 Cell C（GPU 2+3，NCCL 后端已在代码里自动选）
echo "[4/4] 运行 Cell C (CUDA_VISIBLE_DEVICES=2,3)..."
echo ""

CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH=/home/ubuntu/slime-agentic-zero-repro \
.venv/bin/python -u scripts/bench/v9_end_to_end.py --cell C --repeats 1

echo ""
echo "=== Cell C 完成 ==="
