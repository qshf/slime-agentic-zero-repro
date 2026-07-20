#!/bin/bash
# 看门狗: 反复 docker pull(超时即重启,已下层会被缓存跳过),扛过镜像源限速/断流。
# 在服务器 5090 上后台跑: nohup bash pull_sglang_watchdog.sh > /tmp/pull_watchdog.log 2>&1 &
set +e

IMAGE="lmsysorg/sglang:latest"
LAYER_LOG="/tmp/sglang_layers.log"

for attempt in $(seq 1 60); do
  if docker images | grep -q sglang; then
    echo "[$(date +%H:%M:%S)] READY after $attempt attempts"
    exit 0
  fi
  echo "[$(date +%H:%M:%S)] attempt $attempt: pulling (timeout 120s)..."
  timeout 120 docker pull "$IMAGE" >> "$LAYER_LOG" 2>&1
  done_layers=$(grep -c "Download complete" "$LAYER_LOG" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] attempt $attempt done, layers Download complete: $done_layers"
  sleep 3
done

echo "GAVE UP after 60 attempts"
exit 1
