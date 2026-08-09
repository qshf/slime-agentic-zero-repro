# 租机环境速查（bench5090 / GPU 服务器）

> 目的：**下次开工不用再探测工作目录、不用重读环境**。租的 GPU 机 IP/端口经常变、显卡型号也可能换——**变的部分我每次口头给你**；**不变的部分（镜像、工作目录、挂载、模型路径）全记在这**。
> 最近一次核对：2026-08-09，对 `root@223.109.239.36 -p 15016` 实测确认（见 §5「上次核对快照」）。

---

## 0. 每次开工，你只需给我这三样

| 每次可能变 | 形式 | 我拿到后怎么用 |
|---|---|---|
| **IP + 端口** | `ssh root@<IP> -p <PORT>` | 拼进下面所有命令的 `SSH` 前缀 |
| **显卡型号/数量**（若换机） | 一句话，如"还是单张 5090"/"换成 2×A100" | 决定走 FA2 双机通用路径 / 是否涉及 5090 专属项 |
| **哪张卡空闲**（若与 SGLang 争卡） | 如"GPU0 给 SGLang，训练用别的" | 设 `CUDA_VISIBLE_DEVICES` |

其余一切（用户 root、工作目录、镜像 tag、容器挂载、模型路径、FA2 wheel 位置）**默认按本文，不用问、不用探测**。若某次机器是全新的、这些不成立，你明确告诉我"新机，环境要重建"。

---

## 1. 不变的事实（记牢，别再探测）

| 项 | 值 |
|---|---|
| **登录用户** | `root`（区别于旧 `5090` 别名机的 `ubuntu`） |
| **家目录** | `/root` |
| **slime 工作目录** | `/root/slime-agentic-zero-repro` （本 repo；**注意：机器上这份不是 git 仓库**，见 §3） |
| **infra 工作目录** | `/root/agentic-rl-infra-lab` （规范/spike 参考） |
| **模型** | `/root/models/Qwen/Qwen3-0.6B` |
| **常驻容器名** | `bench5090` |
| **容器镜像** | `agentic-rl-infra-lab:te-cudnn-system-spike` |
| **容器挂载** | `/root/agentic-rl-infra-lab` → `/workspace/agentic-rl-infra-lab`（bind mount） |
| **FA2 wheel（宿主机）** | `/root/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` |
| **镜像继承链** | `lmsysorg/sglang:latest` → `te-spike` → `te-cudnn-system-spike` / `fa4-spike` |

> 镜像层次、各后端可用性、冲突矩阵的完整版在 infra `docs/ops/docker-env-a100-vs-5090.md`，本文不重复，只记"这台机怎么连、怎么跑"。

---

## 2. SSH 前缀（把 IP/端口换成当次的）

```bash
# 约定：下文用 $SSH 代指这段前缀。每次开工把 <IP>/<PORT> 换成你给我的值。
SSH="ssh -o ConnectTimeout=12 -o ClearAllForwardings=yes -p <PORT> root@<IP>"

# 最近一次（2026-08-09）：
# SSH="ssh -o ConnectTimeout=12 -o ClearAllForwardings=yes -p 15016 root@223.109.239.36"
```

- `-o ClearAllForwardings=yes`：沿用旧机踩坑约定，避免代理端口转发失败导致连接直接被拒。
- 首次连新 IP 会问 host key，加 `-o StrictHostKeyChecking=accept-new` 自动接受。

**连通性 + 环境自检一条龙**（换机后先跑这个，确认本文事实仍成立）：
```bash
$SSH 'whoami; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;
      docker ps --format "{{.Names}} <- {{.Image}}";
      ls -d /root/slime-agentic-zero-repro /root/agentic-rl-infra-lab /root/models/Qwen/Qwen3-0.6B'
```

---

## 3. slime repo 在这台机上的状态（重要区别）

**机器上的 `/root/slime-agentic-zero-repro` 不是 git 仓库**（无 `.git`），但**有 `.venv`**。这跟旧 `5090` 别名机（git 管理、`git reset --hard` 恢复）不同——本文这台是**手动同步**的一份代码副本。

所以往这台机同步代码用 **rsync**（保留 `.venv`，不碰 `.git`——因为本来就没有）：
```bash
rsync -az -e "ssh -o ClearAllForwardings=yes -p <PORT>" --exclude '.venv' --exclude '.git' \
  /Users/qshf/my-project/slime-agentic-zero-repro/ \
  root@<IP>:/root/slime-agentic-zero-repro/
```

> 若哪天把这台也纳入 git 管理，改用 `ENVIRONMENT.md` 里的 `git fetch + reset --hard` 流程并更新本行。

---

## 4. 常用运行姿势（按用途选）

### 4.1 slime repo 的 GPU 脚本（FSDP / packing 等，用宿主机 .venv）

slime 的 `.venv` 只有 dev + ray/openai，**没有 torch**；GPU 脚本需要在**有 torch 的环境**里跑。这台机上有 torch 的是 `bench5090` 容器（挂的是 infra 目录）。两种办法：

- **A. 需要 torch 的 slime 脚本** → 走容器的系统 python。容器只挂了 infra 目录，slime 目录没进容器，需额外挂载。**注意容器缺 `accelerate`**（`FSDPTrainer` 的 tie 分支依赖它），跑 FSDP 脚本前必须补装：
  ```bash
  $SSH 'docker run --rm --gpus all \
    -v /root/slime-agentic-zero-repro:/workspace/slime \
    -v /root/models/Qwen/Qwen3-0.6B:/models/Qwen3-0.6B:ro \
    -w /workspace/slime agentic-rl-infra-lab:te-cudnn-system-spike \
    bash -lc "pip install -q accelerate && \
      NANO_MODEL_PATH=/models/Qwen3-0.6B \
      torchrun --nproc_per_node=1 scripts/test_v7.5_packing.py"'
  ```
- **B. 纯 CPU 契约测试** → slime `.venv` 直接跑：
  ```bash
  $SSH 'cd /root/slime-agentic-zero-repro && uv run python scripts/test_v0_contract.py'
  ```

### 4.2 infra spike / attention 后端脚本（用 bench5090 常驻容器）

```bash
# TE cuDNN（NVTE_FLASH_ATTN=0，仅 forward 对照——backward 有 bug，见 v8-5090-followup）
$SSH 'docker exec bench5090 bash -c "cd /workspace/agentic-rl-infra-lab && NVTE_FLASH_ATTN=0 python scripts/06_attn_backend_throughput.py --backend te ..."'
```

**FA2 varlen（2026-08-09 实测配方）**——三个坑：①容器里 `flash_attn` namespace 被 **FA4 b15** 占（`import flash_attn` 成功但无 `__version__`），必须先卸载；②缺 `accelerate`；③**wheel 不能单文件 bind mount**（挂载改名破坏 wheel 命名规则 → `Invalid wheel filename`），要挂目录。**装完 FA2 会让 TE import 崩**（覆盖 namespace），故务必用 `--rm` 临时容器，别在 `bench5090` 常驻容器里装：

```bash
$SSH 'docker run --rm --gpus all -v /root:/host:ro \
  -v /root/slime-agentic-zero-repro:/workspace/slime \
  -w /workspace/slime agentic-rl-infra-lab:te-cudnn-system-spike bash -lc "
    pip uninstall -y flash-attn-4 -q
    pip install --no-deps -q /host/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    pip install -q accelerate
    NANO_MODEL_PATH=/host/models/Qwen/Qwen3-0.6B python ..."'
```
实测：`FA2 2.8.3` / `is_flash_attn_2_available: True` / `accelerate 1.14.0` / varlen forward+backward OK。
**FA2 拒绝 fp32**（`only support fp16 and bf16`）——影响验收门设计，见 `docs/decisions/v7-onward-plan.md` §2.3。

### 4.3 SGLang（V1+ 真 rollout）

模型在 `/root/models/Qwen/Qwen3-0.6B`。起独立容器 `--network host`、端口 30000（沿用 infra `sync-5090.md` 的姿势，注意 bridge 容器访问宿主机走网关 `172.17.0.1`）。

---

## 5. 上次核对快照（2026-08-09，仅当次有效）

> 这些是**当次**的值，仅供参考——机器一换/一重启就可能失效，别当常量。

| 项 | 当次值 |
|---|---|
| 连接 | `ssh root@223.109.239.36 -p 15016` |
| GPU | 1× RTX 5090，32607 MiB |
| driver | 595.84 |
| 容器内 torch | 2.11.0+cu130（CUDA 13.0） |
| TE | 2.17.0 |
| cuDNN（torch 报） | 9.19.0 |
| 常驻容器 | `bench5090`（Up 3h）| 
| 镜像 | `agentic-rl-infra-lab:{te-spike, te-cudnn-system-spike, fa4-spike}` + `lmsysorg/sglang:latest` |

---

## 6. 与其他环境文档的关系

| 文档 | 讲什么 | 区别 |
|---|---|---|
| 本文 | **这台租机**怎么连、目录/镜像/挂载在哪 | IP 会变，事实不变部分记死 |
| `ENVIRONMENT.md` | 旧 `5090` 别名机（ubuntu 用户、git 管理、rsync 同步） | **不同机器**，别混 |
| infra `docs/ops/docker-env-a100-vs-5090.md` | 镜像层次 + 后端可用性 + 冲突矩阵 | 后端细节看它 |
| infra `docs/ops/sync-5090.md` | infra repo 的 git 同步 + SGLang 网络坑 | 面向 infra repo |
| `docs/decisions/v8-5090-followup.md` | 5090 专属后端结论（FA2 首选、FA4 备选、TE bug） | 决策，非环境 |
