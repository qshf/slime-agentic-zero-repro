# V9 GSM8K 吞吐实验结果

> 测试日期：2026-08-20  
> 状态：固定工作负载的训练吞吐已完成；在线端到端仅完成 A 格冒烟。B/C/D 的完整对比尚未执行，不能从本文推出 async 的最终加速比。  
> 历史参考：[`v9-results.md`](v9-results.md) 是 2026-08-19 的 calculator 原型记录。其中 async 总耗时漏计权重发布、`policy_version_gap` 接线错误，保留作排障历史，不作为本报告结论。

## 1. 目的和判定边界

本轮把两个问题拆开测量：

| 测试 | 回答的问题 | 输入和指标 |
|---|---|---|
| 固定回放（`v9_gsm8k_throughput.py --replay`） | 同一批有效 GRPO 数据下，Torch、Megatron DP=1、Megatron DP=2 的纯训练吞吐差异 | 固定 token IDs、旧 logprob、优势和 mask；`step time`、trainable/model tokens/s、loss、TFLOPs |
| 在线端到端（`v9_end_to_end.py`） | rollout、训练、发布权重叠加后，sync/async 调度的真实耗时 | GSM8K 在线采样；`wait/gen`、train、sync、e2e、GRPO 信号质量、loss |

固定回放用于训练后端的因果比较；在线测试用于调度和权重发布路径的比较。两类数值的 batch 大小、序列长度和计时范围不同，不能直接横向相除。

所有 Megatron 实验使用 `THD` packed sequence，未使用 BSHD padding。

## 2. 运行环境

| 项目 | 配置 |
|---|---|
| 训练仓库与 Ray | 宿主机 `5090`，`/home/ubuntu/slime-agentic-zero-repro`，宿主 `.venv` |
| 训练 GPU | 固定回放：GPU 2（DP=1）或 GPU 2、3（DP=2）；在线 A 冒烟：GPU 2 |
| 推理 GPU | GPU 0 上独立运行的 SGLang 容器，服务为 `http://localhost:30000` |
| 模型 | `Qwen3-0.6B`，宿主 `/home/ubuntu/models/Qwen/Qwen3-0.6B`，容器 `/models/Qwen3-0.6B` |
| 数据 | 本地 GSM8K，宿主 `/home/ubuntu/data/gsm8k`；采样容器使用挂载后的 `/tmp/gsm8k` |
| 权重发布 | 在线 A 冒烟使用 `disk`。宿主训练环境没有 `sglang` Python client，不能从宿主进程调用 tensor sync。 |

此前 tensor sync 在 `v9-dev` 容器内单独验证成功，发布约 5.33 s，随后生成可见新 weight version。该结果与宿主 disk 发布约 24 s 处在不同运行拓扑，说明接口可用，不能当作严格的端到端速度比。

## 3. 工作负载和训练信号

固定回放先通过 SGLang 真实采样 `2` 道 GSM8K 题、每题 `4` 个样本（temperature `0.7`，max new tokens `192`），再冻结为 JSON 工作负载。

| 采样项 | 数值 |
|---|---:|
| 原始 rollout 数 | 8 |
| 可训练 rollout 数 | 4 |
| 被 mask 的 rollout 数 | 4 |
| GRPO 分组数 / 有效分组数 | 2 / 1 |
| 原始奖励均值 | 0.75 |
| 每 update model tokens | 686 |
| 每 update trainable tokens | 402 |

GRPO 只在同一题的多个样本奖励存在差异时产生有效优势。无方差组会使 mask 为零：这不是训练器故障，但该轮没有有效梯度。当前 sync/async 编排已在 `trainable_tokens == 0` 时跳过权重发布和版本递增，避免把无训练信号的空更新误记成策略更新。

## 4. 固定回放：训练吞吐

每个后端运行 5 个 update，前 1 个为 warmup，统计后 4 个的中位数。三组实验复用上节完全相同的工作负载。

| 后端 | 并行 | median step time | trainable tokens/s | model tokens/s | median loss | 报告 TFLOPs |
|---|---|---:|---:|---:|---:|---:|
| TorchActor | 单卡 GPU 2 | 0.300 s | 1,334.2 | 2,276.7 | -0.00576 | 当时未记录 |
| Megatron | TP=1, DP=1，GPU 2 | 1.348 s | 297.7 | 508.0 | -0.00534 | 1.812 |
| Megatron | TP=1, DP=2，GPU 2、3 | 1.117 s | 359.8 | 614.0 | -0.00361 | 1.190* |

结论：

- DP=2 相对 Megatron DP=1 的 trainable tokens/s 提升 **20.9%**，远低于 2 倍。0.6B、小 batch 下，进程与梯度规约开销占比高。
- 在这个小模型和小工作负载上，TorchActor 单卡比 Megatron DP=1 快约 **4.48x**，比 Megatron DP=2 快约 **3.71x**（按 trainable tokens/s）。这描述的是当前 benchmark 的工程开销，不代表大模型或大 batch 的一般结论。
- 三组 loss 均为小的有限值，且同一固定数据下数值量级一致，说明训练路径没有出现 NaN 或奖励接线错误。它不是收敛性证明：仅 5 个 update 且只有 1 个有效 GRPO 组。

\* DP=2 的 TFLOPs 当前由每个 rank 的本地计时/计数上报，DP=1 和 DP=2 的口径尚未统一，不能把 `1.812` 与 `1.190` 解读为全局算力退化。后续应统一为全局有效 tokens 和全局训练 wall time 后再比较 TFLOPs。

## 5. 在线端到端：A 格 GSM8K 冒烟

命令使用 `--cell A --rounds 3 --warmup-rounds 1 --prompts 2 --samples-per-prompt 4 --weight-sync disk`。首轮 warmup 后取两轮中位数：

| 指标 | 结果 |
|---|---:|
| rollout generation | 5.418 s |
| learner train | 0.321 s |
| disk weight sync | 24.022 s |
| end-to-end step | 29.760 s |
| 原始奖励均值 | 0.50 |
| 有效 GRPO group rate | 50% |
| trainable tokens/s（仅 train 段） | 2,198.6 |
| model tokens/s（仅 train 段） | 6,099.0 |
| TFLOPs（当前训练计时口径） | 21.696 |
| loss | 0.01321 |

时间构成接近 `5.418 + 0.321 + 24.022 = 29.761 s`：generation 约 **18.2%**、训练约 **1.1%**、disk 发布约 **80.7%**。因此当前瓶颈是训练后把权重发布给独立 SGLang 容器，而不是 learner 前反向计算。

`reward_mean=0`（标准化后的组内优势均值）不等于任务没有奖励；本次原始 reward 均值为 `0.50`，且 50% 分组有有效方差，达到当前 25% 的质量门槛。冒烟中曾打印 `policy_version_gap=1`，该值来自发布后才取 trainer version 的旧接线，已在后续提交中改为发布前取版本；该旧值不纳入结果。

## 6. 异步回归与未完成项

基础异步状态机已在宿主 `.venv` 运行 `scripts/test/test_v5_async.py --offline`，结果为 `PASSED (0 failures)`；一次对照中 sync `5.102 s`、async `4.294 s`。这是 fake backend 回归，证明生命周期没有因新增指标而中断，不是 GSM8K A/B/C/D 的性能结论。

完整四格应使用当前脚本的默认规模（12 rounds，2 warmup，4 prompts x 4 samples/group），每格至少重复 3 次，并同时记录质量门槛：

```bash
# A/B: 单卡 Torch；C/D: GPU 2、3 上的 Megatron TP=2
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/ubuntu/slime-agentic-zero-repro \
  .venv/bin/python -u scripts/bench/v9_end_to_end.py \
  --rounds 12 --warmup-rounds 2 --prompts 4 --samples-per-prompt 4 \
  --temperature 0.7 --max-new-tokens 192 --gsm8k-local-dir /tmp/gsm8k \
  --weight-sync disk --repeats 3
```

执行时分别传入 `--cell A`、`B`、`C`、`D`。在 B/D 结果产生前，不能宣称异步缩短了完整 e2e step；它最多能隐藏 generation，不能隐藏本轮的 disk weight sync。要比较 tensor 发布，需要先让宿主训练环境具备 SGLang Python client，或在同时具备 Ray 和 SGLang client 的一致容器环境运行训练器。

## 7. 复现入口

| 用途 | 文件 |
|---|---|
| 固定工作负载采集与回放 | `scripts/bench/v9_gsm8k_throughput.py` |
| 大样本离线训练吞吐矩阵 | `scripts/bench/v9_offline_throughput.py` |
| 离线吞吐题卷（16 题 x 8 个采样） | `scripts/bench/data/v9_gsm8k_offline_throughput_paper.json` |
| 在线 A/B/C/D | `scripts/bench/v9_end_to_end.py` |
| GSM8K 直接采样和 reward | `toy_rl/agent/toolorchestra/gsm8k_throughput_rollout.py` |
| 固定回放契约测试 | `scripts/test/test_v9_gsm8k_throughput.py` |

本轮固定工作负载文件和在线冒烟 CSV 位于服务器临时目录：`/tmp/v9_gsm8k_throughput.json`、`/tmp/v9_e2e_smoke/v9_end_to_end.csv`。如需跨机器长期保留，应把 workload JSON 与完整四格 CSV 复制到版本化的实验产物目录。

离线 workload 采集使用一次完整的 SGLang batch `/generate` 请求（`text` 为所有 prompt 的数组）；预热 batch 丢弃，随后才冻结 JSON。响应逐行恢复到原始输入顺序，因此每题的多个采样仍在连续 GRPO 分组中。
