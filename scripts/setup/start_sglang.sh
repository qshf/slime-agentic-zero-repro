#!/bin/bash
# 启动 SGLang + Qwen3-0.6B Docker 容器（在服务器 5090 上运行）
# 用法: bash scripts/start_sglang.sh
#
# 对齐源项目: slime-agentic 的推理引擎用 SGLang（requirements.txt: sglang-router>=0.2.3）
# 源项目里 rollout engine 是一个独立的推理服务，trainer 通过 HTTP 调它——这个脚本就是那个服务。
# V6 会把这个容器换成真正的 slime RolloutManager；V1 只是让 calculator agent 能跑起来。

set -e

# 服务器 5090 连不到 HuggingFace，模型用 modelscope 预下载到本地，挂载进容器。
# 本地路径 -> 容器内路径。--model-path 用容器内路径，SGLang 不会再去 HF 拉。
HOST_MODEL_DIR="/home/ubuntu/models/Qwen/Qwen3-0.6B"
CONTAINER_MODEL_DIR="/models/Qwen3-0.6B"
CONTAINER_NAME="sglang-qwen3"
HOST_PORT=30000        # SGLang 默认端口，与源项目 agentic/agentflow/rollout.py 对齐
GPUS="device=0"        # 单卡，V1 用 0.6B 够了

if [ ! -d "$HOST_MODEL_DIR" ]; then
    echo "ERROR: 本地模型目录不存在: $HOST_MODEL_DIR"
    echo "先用 modelscope 下载 Qwen3-0.6B 到该路径。"
    exit 1
fi

# 如果容器已存在就停掉
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "=== 启动 SGLang + Qwen3-0.6B（本地挂载）==="
echo "    模型: $HOST_MODEL_DIR -> $CONTAINER_MODEL_DIR"
echo "    端口: $HOST_PORT"
echo "    GPU:  $GPUS"

# `update_weights_from_tensor` serializes tensor storages through Python's
# multiprocessing resource-sharer socket. The trainer runs in `v9-dev`,
# which already bind-mounts host `/tmp`; SGLang must see that same socket
# path for the disk-free HTTP path to work across the two containers.
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus "$GPUS" \
    --network host \
    -v "$HOST_MODEL_DIR":"$CONTAINER_MODEL_DIR":ro \
    -v /tmp:/tmp \
    lmsysorg/sglang:latest \
    python -m sglang.launch_server \
        --model-path "$CONTAINER_MODEL_DIR" \
        --served-model-name "Qwen/Qwen3-0.6B" \
        --port "$HOST_PORT" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --mem-fraction-static 0.62 \
        --context-length 4096
# 踩坑: 4 张卡都被别的进程占了（各只剩 ~2.8G free），默认 mem-fraction=0.874 会 OOM。
# SGLang 会按当前 free 显存标定最小可行值（本机提示 >=0.573），设 0.62 稳妥；
# 0.6B 权重仅 1.5G，KV cache 用 context-length 4096 压住，够 calculator 任务用。

echo ""
echo "=== 等待 SGLang 就绪（首启编译 flashinfer + 加载权重，约 2-4 分钟）==="
# 踩坑: /health 在能正常服务后仍可能返回 503，别拿它当就绪信号。
# /v1/models 返回 200 才是真就绪。首启慢是因为要编译 CUDA kernel。
for i in $(seq 1 300); do
    if curl -sf "http://localhost:$HOST_PORT/v1/models" > /dev/null 2>&1; then
        echo ""
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
