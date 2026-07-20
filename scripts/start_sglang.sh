#!/bin/bash
# 启动 SGLang + Qwen3-0.6B Docker 容器（在服务器 5090 上运行）
# 用法: bash scripts/start_sglang.sh
#
# 对齐源项目: slime-agentic 的推理引擎用 SGLang（requirements.txt: sglang-router>=0.2.3）
# 源项目里 rollout engine 是一个独立的推理服务，trainer 通过 HTTP 调它——这个脚本就是那个服务。
# V6 会把这个容器换成真正的 slime RolloutManager；V1 只是让 calculator agent 能跑起来。

set -e

MODEL="Qwen/Qwen3-0.6B"
CONTAINER_NAME="sglang-qwen3"
HOST_PORT=30000        # SGLang 默认端口，与源项目 agentic/agentflow/rollout.py 对齐
GPUS="device=0"        # 单卡，V1 用 0.6B 够了

# 如果容器已存在就停掉
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "=== 启动 SGLang + $MODEL ==="
echo "    端口: $HOST_PORT"
echo "    GPU:  $GPUS"

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus "$GPUS" \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    lmsysorg/sglang:latest \
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$HOST_PORT" \
        --host 0.0.0.0 \
        --trust-remote-code

echo ""
echo "=== 等待 SGLang 就绪（约 30-60s）==="
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$HOST_PORT/health" > /dev/null 2>&1; then
        echo "SGLang 已就绪! (${i}s)"
        break
    fi
    sleep 1
    printf "."
done

echo ""
echo "=== 验活 ==="
curl -s http://localhost:$HOST_PORT/v1/models | python3 -m json.tool | grep -E '"id"|"object"'
echo ""
echo "SGLang 容器: $CONTAINER_NAME 已启动"
echo "接口: http://localhost:$HOST_PORT/v1  (OpenAI 兼容)"
echo "查看日志: docker logs -f $CONTAINER_NAME"
