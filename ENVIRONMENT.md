# 环境配置

本项目分为两套环境：本地用于开发和离线测试，5090 服务器用于 Qwen3-0.6B/SGLang/Ray 的端到端验证。

## 目录约定

| 位置 | 项目目录 | 用途 |
| --- | --- | --- |
| 本地 | `/Users/qshf/my-project/slime-agentic-zero-repro` | 开发、提交、V0 离线测试 |
| 服务器 5090 | `/home/ubuntu/slime-agentic-zero-repro` | V1 及后续 GPU 验证 |

远端 Git 仓库的名称本身以 `-` 开头：`git@github.com:qshf/-slime-agentic-zero-repro.git`。它不是服务器上的工作目录名；服务器统一使用不带 `-` 的 `slime-agentic-zero-repro`。

## 本地开发

前置条件：Python 3.10 或更高版本，以及 [uv](https://docs.astral.sh/uv/)。

```bash
cd /Users/qshf/my-project/slime-agentic-zero-repro
uv sync --locked
uv run python scripts/test_v0_contract.py
```

需要进入虚拟环境时使用：

```bash
source .venv/bin/activate
```

日常运行优先使用 `uv run ...`，这样不会依赖当前 shell 是否已经激活 `.venv`。

## 同步到服务器

服务器不能稳定访问 GitHub，因此本地完成修改后通过 `rsync` 同步代码。该命令会保留服务器端的 `.venv` 和 Git 元数据。

```bash
rsync -az -e 'ssh -o ClearAllForwardings=yes' --exclude '.venv' --exclude '.git' \
  /Users/qshf/my-project/slime-agentic-zero-repro/ \
  5090:/home/ubuntu/slime-agentic-zero-repro/
```

首次部署前，先在服务器上删除误建的带 `-` 目录，并确保正确目录存在：

```bash
ssh -o ClearAllForwardings=yes 5090 \
  'rm -rf /home/ubuntu/-slime-agentic-zero-repro'
```

删除前可先用下面命令确认位置：

```bash
ssh -o ClearAllForwardings=yes 5090 \
  'ls -ld /home/ubuntu/-slime-agentic-zero-repro 2>/dev/null'
```

## 服务器 Python 环境

代码同步完成后，在服务器运行一次初始化脚本：

```bash
ssh -o ClearAllForwardings=yes 5090 \
  'cd /home/ubuntu/slime-agentic-zero-repro && bash scripts/server_setup.sh'
```

脚本会创建项目内 `.venv`、按 `uv.lock` 安装 `openai` 和 `ray`，并运行 V0 契约测试。后续依赖变更后重复执行即可。

## SGLang 推理服务

服务器需要满足以下条件：Docker 可用、GPU 0 可用，且模型已经位于 `/home/ubuntu/models/Qwen/Qwen3-0.6B`。启动服务：

```bash
ssh -o ClearAllForwardings=yes 5090 \
  'cd /home/ubuntu/slime-agentic-zero-repro && bash scripts/start_sglang.sh'
```

服务的 OpenAI 兼容地址是 `http://127.0.0.1:30000/v1`。脚本使用 `--mem-fraction-static 0.62`、`--context-length 4096` 和 `--disable-cuda-graph`，以适配当前服务器紧张的可用显存。

确认服务就绪：

```bash
ssh -o ClearAllForwardings=yes 5090 'curl -sf http://127.0.0.1:30000/v1/models'
```

## 验证命令

```bash
# 本地 V0
uv run python scripts/test_v0_contract.py

# 服务器 V1
ssh -o ClearAllForwardings=yes 5090 \
  'cd /home/ubuntu/slime-agentic-zero-repro && uv run python scripts/test_v1_rollout.py'

# 服务器 V4（离线桩）
ssh -o ClearAllForwardings=yes 5090 \
  'cd /home/ubuntu/slime-agentic-zero-repro && uv run python scripts/test_v4_ray.py --offline'
```
