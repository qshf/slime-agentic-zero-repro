# Scripts 目录

按功能划分的脚本组织结构。

## 目录结构

```
scripts/
├── setup/              # 环境搭建脚本
├── container/          # 容器管理脚本
├── test/              # 测试脚本（按版本）
├── demo/              # 演示与学习脚本
└── README.md          # 本文件
```

---

## setup/ — 环境搭建

### `server_setup.sh`
初始化服务器环境（V1 时期）。
- 安装依赖：Python 3、uv、OpenAI SDK
- 创建虚拟环境
- 已过时，现在用 Docker 容器

### `start_sglang.sh`
启动 SGLang 容器（V1-V5，非分布式推理）。
- 单卡 Qwen3-0.6B
- 端口 30000
- V6 起改用 `/generate` 拿真 token/log_probs

---

## container/ — 容器管理

### `start_v9_dev_container.sh` ⭐
**启动 V9 常驻开发容器**（推荐）。
```bash
./scripts/container/start_v9_dev_container.sh [container_name] [gpus]
# 默认：v9-dev, all GPUs
```
- 容器名：`v9-dev`（默认）
- 镜像：`agentic-rl-infra-lab:fa2-mcore` (TE 2.18)
- 挂载：项目根 `/workspace`、模型 `/models/Qwen3-0.6B:ro`、`/tmp`
- 用法：
  ```bash
  docker exec -it v9-dev bash              # 进入容器
  docker exec v9-dev bash -c "command"     # 容器外执行
  docker stop v9-dev                       # 停止
  docker rm -f v9-dev                      # 删除
  ```

### `run_v9_labs.sh`
**一次性容器运行脚本**（快速测试用）。
```bash
./scripts/container/run_v9_labs.sh [gpus] [script] [args...]
# 例：./scripts/container/run_v9_labs.sh "device=2" scripts/test/test_v8.2_parallel.py --layers 4
```
- 自动检测 `torchrun`（识别 `test_v8.2_parallel` 或响应 `NPROC` 环境变量）
- 自动拍 GPU 快照
- `--rm` 运行后删除

---

## test/ — 测试脚本

按版本号排序，每个测试对应一版的验收门。

### 主线一（V0-V5）：最小训练流水线

- **`test_v0_contract.py`** — V0 硬编码单样本 + fake trainer
  - 离线测试，5 个契约
  - 词级假 token，loss_mask 肉眼可读

- **`test_v2_hooks.py`** — V2 custom generate/reward hook 化
  - 离线测试，4 个契约
  - 验证 hook 可插拔

- **`test_v1_rollout.py`** — V1 极简 calculator agent + Qwen3-0.6B
  - 需 SGLang@30000
  - 4 个契约，reward=1.0

- **`test_v3_loop.py`** — V3 mini_slime 最小闭环
  - `--offline` 本地测试（3/3）
  - 服务器端到端：rollout→train→update_weights，reward=1.0

- **`test_v4_ray.py`** — V4 Ray 化（分进程）
  - `--offline` 本地测试（3/3）
  - 服务器端到端：rollout/trainer 独立 PID，reward=1.0

- **`test_v5_async.py`** — V5 同步 vs 异步
  - `--offline` 本地测试：async < sync 耗时
  - 服务器端到端：真 SGLang gen 50-70s，saving ≈ 15.6s

### 主线二（A1-A3）：三个真实 Agent

- **`test_a1_memagent.py`** — A1 MemAgent（单引擎/无工具/loss_mask 全 1）
  - `--offline`：3 chunk→4 turn、trainable=167、reward=1.0
  - 服务器：single reward=1.0、闭环 reward_mean=0.5

- **`test_a2_agentflow.py`** — A2 AgentFlow（executor token 不训练）
  - `--offline`：turns=2、loss_mask 仅 planner、python_coder 真跑、reward=1.0
  - 服务器：真 4B planner + 真 DeepSeek coder、trainable=3018、reward=1.0

- **`test_a3_toolorchestra.py`** — A3 ToolOrchestra（多专家路由 + 多组件 reward）
  - `--offline`：search→answer 循环、多组件 reward、闭环 reward_mean=0.991
  - 服务器：turns=2、reward=0.880（非满分正是成本/延迟组件起作用）

### 主线三（V6-V8.2）：分布式后端

- **`test_v6_gsm8k.py`** — V6 真训练闭环（GSM8K）
  - 真 token+log_probs、GRPO 组归一、torch 真训一步、disk reload 权重同步
  - 服务器：real_logprobs OK、真 backward、weight_v 递增、acc 变化证明权重真被改

- **`test_v7.0_fsdp_single.py`** — V7.0 FSDP 基础（2 卡）
  - 真分片（q_proj local shard = 全量的一半）
  - 忠实 tie 处理（不单独 wrap、全 rank load 全量）
  - loss=-2.0（数学精确）、grad_norm=308

- **`test_v7.2_grad_accum.py`** — V7.2 梯度累积
  - 累积等价性：max_diff=0.000e+00
  - 演示「loss 抵消≠梯度抵消」

- **`test_v7.3_fsdp_ray.py`** — V7.3 Ray+FSDP+权重同步
  - `--world 1/2`：真 dist 组、DP-split、集体 save/rank0 POST
  - weight_v 递增、真 backward、未 hang

- **`test_v7.4_learner_abi.py`** — V7.4 learner_contract + trace + 版本戳
  - **纯 CPU 测试**（6/6），无需 GPU
  - causal 右移、单一版本、LearnerTrace 不变量、gap=1 真 staleness

- **`test_v7.5_packing.py`** — V7.5 sequence packing（退休偏离 #1）
  - 5090 world=1：fp32 精确证明（loss Δ=2.4e-7、cosine=0.99999999）
  - 块对角 mask 隔离无泄漏

- **`test_v7.6_fa2_packing.py`** — V7.6 FA2 varlen packing（退休偏离 #7）
  - 5090 world=1：varlen 真 dispatch 28 次、cu_seqlens 一致
  - negative control 判别力 8721×

- **`test_v8.0_megatron_tp.py`** — V8.0 Megatron TP（单卡 torchrun）
  - fp32 精确门：loss Δ=0、grad rel_L2=1.9e-5、cosine=1.0
  - bf16 主门：loss Δ=5.1e-6、cosine=0.99909
  - 全 28 层复跑同样过

- **`test_v8.1_megatron_to_hf.py`** — V8.1 Megatron→HF 权重转换
  - 往返 max_abs_diff=0.000e+00（无容差硬门）
  - 名字集合双向比对、GLU 重排、GQA 拆分

- **`test_v8.2_parallel.py`** ⭐ — V8.2 多卡 Megatron（TP×PP×CP）
  - **5090 4 卡**，`--layers 4`
  - G1 NCCL / G2 PP / G4 TP×PP：fp32 loss Δ=0、cosine=1.0
  - G3a THD packing：fp32 Δ=0（TE spec + unfused）
  - G3b CP=2：bf16 cosine=0.99921（flash 后端）
  - G3c negative control：cosine=0.588（度量有判别力）
  - G5 拓扑、G6 往转换往返 max_abs_diff=0
  - 用法：
    ```bash
    # 容器内执行
    torchrun --nproc-per-node 1 scripts/test/test_v8.2_parallel.py --layers 4 --fp32
    torchrun --nproc-per-node 2 scripts/test/test_v8.2_parallel.py --layers 4 --tp 2 --backend nccl --fp32
    
    # 容器外执行
    docker exec v9-dev bash -c "torchrun --nproc-per-node 1 scripts/test/test_v8.2_parallel.py --layers 4 --fp32"
    ```

---

## demo/ — 演示与学习

### `collective_ops_demo.py`
集体通信操作演示（all-reduce, all-gather, reduce-scatter 等）。
- 教学用，不在主线路径
- 需 torchrun 多进程

### 其他 demo（根目录未整理）
- `demo_v7_3_communication.py` — V7.3 FSDP 通信模式演示
- `probe_v8.2_cp_packing.py` — V8.2 CP + packing 组合探针
- `ray_debug_demo.py` / `ray_debug_demo2.py` — Ray 调试演示

---

## 使用指南

### 快速验证环境（烟雾测试）
```bash
# 启动常驻容器
./scripts/container/start_v9_dev_container.sh

# 单卡基线
docker exec v9-dev bash -c "torchrun --nproc-per-node 1 scripts/test/test_v8.2_parallel.py --layers 4 --fp32 --dump /tmp/smoke.pt"

# 双卡 TP=2
docker exec v9-dev bash -c "torchrun --nproc-per-node 2 scripts/test/test_v8.2_parallel.py --layers 4 --tp 2 --backend nccl --fp32 --compare /tmp/smoke.pt"
```

### V9 开发工作流
1. **本地修改**：编辑代码 → `git commit` → `git push origin v9`
2. **服务器拉取**：`ssh 5090 'cd /home/ubuntu/slime-agentic-zero-repro && git pull origin v9'`
3. **容器内测试**：`docker exec v9-dev bash -c "your command"`
4. 容器已挂载 `/workspace`，代码更新立即生效，无需重启

---

## 版本对照

| 版本 | 测试脚本 | 关键学点 | 状态 |
|------|---------|---------|------|
| V0 | test_v0_contract.py | 硬编码单样本 + fake trainer | ✅ |
| V1 | test_v1_rollout.py | calculator agent + Qwen3-0.6B | ✅ |
| V2 | test_v2_hooks.py | custom generate/reward hook 化 | ✅ |
| V3 | test_v3_loop.py | mini_slime 最小闭环 | ✅ |
| V4 | test_v4_ray.py | Ray 化（分进程） | ✅ |
| V5 | test_v5_async.py | 同步 vs 异步 | ✅ |
| A1 | test_a1_memagent.py | MemAgent（loss_mask 全 1） | ✅ |
| A2 | test_a2_agentflow.py | AgentFlow（executor token=0） | ✅ |
| A3 | test_a3_toolorchestra.py | ToolOrchestra（多专家 + 多组件 reward） | ✅ |
| V6 | test_v6_gsm8k.py | 真训练闭环（真 log_probs + GRPO + torch） | ✅ |
| V7.0 | test_v7.0_fsdp_single.py | FSDP 基础（分片 + tie） | ✅ |
| V7.2 | test_v7.2_grad_accum.py | 梯度累积 | ✅ |
| V7.3 | test_v7.3_fsdp_ray.py | Ray+FSDP+权重同步 | ✅ |
| V7.4 | test_v7.4_learner_abi.py | learner_contract + trace + 版本戳 | ✅ |
| V7.5 | test_v7.5_packing.py | sequence packing（块对角 mask） | ✅ |
| V7.6 | test_v7.6_fa2_packing.py | FA2 varlen packing | ✅ |
| V8.0 | test_v8.0_megatron_tp.py | Megatron TP（单卡） | ✅ |
| V8.1 | test_v8.1_megatron_to_hf.py | Megatron→HF 权重转换 | ✅ |
| V8.2 | test_v8.2_parallel.py | 多卡 Megatron（TP×PP×CP） | ✅ |
| **V9** | test_v9_throughput.py | 吞吐实验（Timer/FLOPs/MFU） | 📋 待实施 |

---

## 注意事项

1. **V8.2 起需 TE 2.18 镜像**：`agentic-rl-infra-lab:fa2-mcore` (IMAGE_ID `57ca81f86c44`)
2. **多卡测试需 NCCL 后端**：单卡多 rank 用 gloo，多卡用 nccl
3. **验证前必拍 GPU 快照**：确保独占，否则吞吐数字不可比
4. **fp32 vs bf16 分层 gate**：fp32 才是精确证明，bf16 只到舍入
5. **negative control 验收范式**：正例与反例度量必须隔 ≥3 个数量级
