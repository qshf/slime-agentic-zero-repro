# V9 GSM8K 吞吐实验结果

测试日期：2026-08-20。模型为 Qwen3-0.6B；训练和 Ray 运行在 `5090` 宿主机项目的 `.venv`，SGLang 在独立容器的 GPU 0 提供生成与权重更新服务。训练使用 GPU 2（单卡）或 GPU 2、3（双卡）；GPU 2-3 为同 NUMA 的 PCIe/Host Bridge 路径，没有 NVLink。所有 Megatron 实验使用 THD packed sequence。

本文的两部分回答不同问题：端到端测试衡量在线采样、训练和权重发布的整体耗时；离线测试固定训练数据，只比较 learner 吞吐。两者的工作负载和计时范围不同，不能直接横向比较数值。

## 1. 在线端到端测试

### 配置

比较四种编排：

| 格 | learner | 调度 |
|---|---|---|
| A | Torch 单卡 | 同步 |
| B | Torch 单卡 | 异步 |
| C | Megatron TP=2 | 同步 |
| D | Megatron TP=2 | 异步 |

每格独立运行 3 次；每次 6 个 rollout，前 2 步预热、后 4 步统计。每步向 SGLang 发送一个 batch `/generate` 请求，包含 8 道 GSM8K 题、每题 4 个采样，共 32 条 rollout；`temperature=1.0`，`max_new_tokens=192`。Torch 训练 activation microbatch 为 8 并累积为完整 global batch mean；Megatron THD microbatch 为 8。

有效训练更新通过 `POST /update_weights_from_tensor` 从宿主 `.venv` 发布到容器 SGLang。Megatron 导出时所有 TP rank 都参加 collective，只有 rank 0 发送 HTTP 请求；该路径已在本次运行中逐次确认 HTTP 200。

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/ubuntu/slime-agentic-zero-repro \
  .venv/bin/python -u scripts/bench/v9_end_to_end.py \
  --rounds 6 --warmup-rounds 2 --repeats 3 --prompts 8 --samples-per-prompt 4 \
  --temperature 1.0 --max-new-tokens 192 --gsm8k-local-dir /home/ubuntu/data/gsm8k \
  --megatron-microbatch-size 8 --torch-microbatch-size 8 --weight-sync tensor
```

### 结果

下表每项是 3 次重复的中位数。`e2e` 为完整步骤的实际 wall clock。异步列中的 `wait=0` 表示下一批生成已与当前训练重叠完成，不表示生成没有成本；生成等待会在权重发布前被回收，已包含在 e2e 中。

| 格 | wait/gen s | train s | tensor sync s | e2e s | raw reward | 有效 GRPO 组率 | trainable tok/s | model tok/s | loss | version gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A Torch sync | 5.992 | 0.511 | 9.462 | 15.968 | 0.438 | 43.8% | 3,859.1 | 16,482.6 | 0.00009 | 0 |
| B Torch async | 0.000 | 0.411 | 13.995 | 14.512 | 0.477 | 40.6% | 4,574.5 | 18,430.6 | 0.00054 | 1 |
| C Megatron TP=2 sync | 6.559 | 1.702 | 6.851 | 15.162 | 0.461 | 53.1% | 1,526.2 | 4,726.5 | 0.00040 | 0 |
| D Megatron TP=2 async | 0.000 | 1.248 | 10.641 | 12.324 | 0.445 | 40.6% | 1,538.5 | 5,398.4 | 0.00009 | 1 |

- Torch 异步 B 相对同步 A：`15.968 -> 14.512s`，端到端降低 **9.1%**。
- Megatron 异步 D 相对同步 C：`15.162 -> 12.324s`，端到端降低 **18.7%**。
- C 与 A 的端到端时间接近，但原因不是 TP=2 learner 更快，而是在线时间主要由生成和权重发布构成；Megatron 更长的训练段提供了更大的异步重叠窗口。
- 四格有效 GRPO 组率都超过 25% 门槛，存在训练信号。loss 仅用于数值健康检查：在线采样得到的题目、响应长度和优势不同，不能用 loss 大小比较后端或模型质量。

异步采用“一代预取”：训练 batch N 时生成 batch N+1，发布权重前等待该生成结束，避免生成中途切换权重。因此异步 `policy_version_gap=1`、同步为 0 是设计语义，而不是计量错误。

原始结果：`/tmp/v9_e2e_online_20260820/v9_end_to_end.csv` 和 `/tmp/v9_e2e_online_20260820/v9_end_to_end_repeats.csv`。

## 2. 离线训练吞吐测试

### 配置

离线测试先冻结 GSM8K rollout 数据，再让各后端重复消费相同 token、旧 logprob、优势和 loss mask。冻结题卷包含 16 道题、每题 8 个 rollout；过滤无方差 GRPO 组后得到 88 条可用样本、11 个有效组、14,223 个 trainable tokens。这样避免在线采样波动干扰 learner 吞吐比较。

所有结果前 10 个 update 预热，统计后 50 个 update 的中位数。表中 loss 只检查数值有限且量级一致；DP=2 的 TFLOPs 为本地 rank 诊断口径，不与 DP=1 作严格全局算力比较。

### 统一 global batch 8、microbatch 8

| 后端 | GPU/并行 | step s | trainable tok/s | model tok/s | loss | TFLOPs |
|---|---|---:|---:|---:|---:|---:|
| TorchActor | GPU 2，DP=1 | 0.277 | 4,616.0 | 7,494.6 | 0.007559 | 26.66 |
| Megatron | TP=1，GPU 2 | 0.413 | 2,992.9 | 4,859.3 | 0.007032 | 19.57 |
| Megatron | TP=2，GPU 2、3 | 0.488 | 2,558.5 | 4,154.1 | 0.006651 | 15.84 |
| Megatron | TP=1、DP=2，GPU 2、3 | 0.803 | 1,582.2 | 2,568.8 | 0.006873 | 4.72 |

同一 batch 下，Torch 单卡约为 Megatron TP=1 的 1.54 倍。TP=2 比 TP=1 慢约 14.5%：0.6B 模型、batch 8 和 PCIe 通信下，TP 固定开销高于分摊到的计算收益。DP=2 每卡仅 4 条样本，梯度规约与双进程开销无法摊薄，因此不适合以 batch 8 评估 DP 扩展。

### DP 扩展

以下只比较 Megatron TP=1 的 DP=1 和 DP=2，microbatch 等于 global batch。

| global batch | 并行 | step s | trainable tok/s | model tok/s | loss |
|---:|---|---:|---:|---:|---:|
| 16 | DP=1 | 0.488 | 4,958.2 | 8,021.8 | 0.005829 |
| 16 | DP=2 | 0.833 | 2,990.0 | 4,837.5 | 0.006059 |
| 32 | DP=1 | 0.906 | 5,408.5 | 8,678.5 | 0.003262 |
| 32 | DP=2 | 0.857 | 5,795.2 | 9,299.0 | 0.003487 |

global batch 16 时，每个 DP rank 只有 8 条样本，DP=2 吞吐反而低约 39.7%。global batch 32 时，每卡 16 条样本足以开始摊薄规约开销，DP=2 的 trainable tokens/s 比 DP=1 高 **7.1%**。该冻结题卷仅有 88 行，batch 32 实际循环两个完整 batch；若要获得稳定的 DP scaling 曲线，应采集数百条有效 rollout，并继续测试 global batch 32/64/128。

复现入口：`scripts/bench/v9_offline_throughput.py`。本次离线题卷为 `/tmp/v9_gsm8k_offline_throughput_batched.json`；在线端到端入口为 `scripts/bench/v9_end_to_end.py`。
