#!/usr/bin/env bash
# V9 吞吐实验专用启动脚本
# 用法：
#   ./scripts/run_v9_labs.sh [gpu_devices] [script_path] [script_args...]
# 示例：
#   ./scripts/run_v9_labs.sh "device=2,3" scripts/test_v9_throughput.py --tp 2
#   ./scripts/run_v9_labs.sh all scripts/test_v9_packing.py --layers 28
set -euo pipefail

GPUS="${1:-all}"
shift
SCRIPT="${1:?Usage: $0 [gpu_devices] script_path [args...]}"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="agentic-rl-infra-lab:fa2-mcore"
MODEL_PATH="${NANO_MODEL_PATH:-/home/ubuntu/models/Qwen/Qwen3-0.6B}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  echo "Set NANO_MODEL_PATH or place model at default location" >&2
  exit 1
fi

# V9 方法学纪律：
# 1. 验证前必拍 nvidia-smi 快照（共享 GPU 上的吞吐数字不可比）
# 2. 必须独占 GPU（其它容器占用会导致 OOM 或吞吐失真）
echo "=== GPU状态快照（验证前） ===" >&2
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >&2
echo >&2

docker run --rm --gpus "$GPUS" --network host --shm-size=4g \
  -v "$ROOT:/workspace" \
  -v "$MODEL_PATH:/models/Qwen3-0.6B:ro" \
  -v /tmp:/tmp \
  -w /workspace \
  -e NANO_MODEL_PATH=/models/Qwen3-0.6B \
  "$IMAGE" \
  bash -c "python $SCRIPT $*"
