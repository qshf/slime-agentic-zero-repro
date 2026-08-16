#!/usr/bin/env bash
# 启动 V9 开发常驻容器
# 用法：./scripts/start_v9_dev_container.sh [container_name] [gpus]
set -euo pipefail

CONTAINER_NAME="${1:-v9-dev}"
GPUS="${2:-all}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="agentic-rl-infra-lab:fa2-mcore"
MODEL_PATH="${NANO_MODEL_PATH:-/home/ubuntu/models/Qwen/Qwen3-0.6B}"

# 检查是否已存在同名容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "容器 ${CONTAINER_NAME} 已存在" >&2
  docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Status}}\t{{.Names}}"
  read -p "删除并重建？(y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rm -f "${CONTAINER_NAME}"
  else
    exit 0
  fi
fi

echo "=== 启动前 GPU 快照 ===" >&2
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo >&2

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"'"${GPUS}"'"' \
  --network host \
  --shm-size=4g \
  -v "$ROOT:/workspace" \
  -v "$MODEL_PATH:/models/Qwen3-0.6B:ro" \
  -v /tmp:/tmp \
  -w /workspace \
  -e NANO_MODEL_PATH=/models/Qwen3-0.6B \
  "${IMAGE}" \
  tail -f /dev/null

echo "容器 ${CONTAINER_NAME} 已启动" >&2
echo "进入容器：docker exec -it ${CONTAINER_NAME} bash" >&2
echo "停止容器：docker stop ${CONTAINER_NAME}" >&2
echo "删除容器：docker rm -f ${CONTAINER_NAME}" >&2
