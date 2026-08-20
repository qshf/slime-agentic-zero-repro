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

多卡结果必须记录 `nvidia-smi topo -m`：TP rank 需要同一台主机，且应优先选同一 NUMA 域的 GPU 对。本机 GPU `2-3` 为 `NODE`（同 NUMA 的 PCIe Host Bridge 路径）；GPU `1-2` 为 `SYS`（跨 NUMA），不用于 TP=2 对照。该机未报告 `NV#`，即没有可用 NVLink 链路。

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

冻结的 workload 是数据池，不等于一次 learner update。离线矩阵以 `--train-batch-size 8` 循环消费连续的有效 GRPO 组，防止 Torch padding 后的全词表 logits 以 88 行同时驻留而 OOM；每种后端接受相同的 batch 顺序。

## 8. 2026-08-20 节点内纯训练矩阵

本轮在 `5090` 上重新执行四个后端，避免把跨 NUMA 的 GPU 对误当成 TP 对照：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/ubuntu/slime-agentic-zero-repro \
  .venv/bin/python -u scripts/bench/v9_offline_throughput.py \
  --workload /tmp/v9_gsm8k_offline_throughput_batched.json \
  --configs torch,megatron-tp1,megatron-tp2,megatron-tp1-dp2 \
  --train-batch-size 8 --updates 60 --warmup 10 --runs 1
```

固定 workload 为 16 道题、每题 8 个 rollout；采样预热 1 个 batch 后冻结，过滤无训练信号的组，留下 88 行、11 个有效 GRPO 组和 14,223 个 trainable tokens。训练阶段每次使用 8 行（一个 GRPO 组），循环 60 update，前 10 个 warmup，后 50 个计入中位数。所有 Megatron 路径为 THD packed sequence。

| 后端 | GPU/并行 | median step time | trainable tokens/s | model tokens/s | loss | TFLOPs |
|---|---|---:|---:|---:|---:|---:|
| TorchActor | GPU 2，DP=1 | 0.255 s | 5,052.0 | 8,202.6 | 0.007889 | 29.06 |
| Megatron | TP=1，GPU 2 | 2.772 s | 465.3 | 755.4 | 0.007417 | 2.68 |
| Megatron | TP=2，GPU 2、3 | 2.905 s | 436.2 | 708.1 | 0.007077 | 2.51 |
| Megatron | TP=1、DP=2，GPU 2、3 | 1.814 s | 724.8 | 1,176.8 | 0.006966 | 2.07* |

这里的 TP=2 两个 rank 实际分别落在物理 GPU 2 和 3。`nvidia-smi topo -m` 显示 GPU 2-3 为同 NUMA 的 `NODE` 路径；GPU 1-2 为跨 NUMA 的 `SYS`，未用于该实验。机器没有 `NV#` 链路，因此这是同节点 PCIe/Host Bridge 通信，不是 NVLink 通信。

相对 Megatron TP=1，TP=2 的 trainable tokens/s 下降约 6.3%；在当前 0.6B、小 batch、单节点无 NVLink 的工作负载上，TP 通信和 Megatron 固定开销大于张量并行带来的计算收益。DP=2 的吞吐提升约 55.8%，但未达到线性 2 倍，主要受梯度规约、Ray/进程和小 batch 开销影响。Torch 单卡约为 Megatron TP=1 的 10.9 倍；该结论只代表本 benchmark 的小模型和 batch 配置，不能外推到大模型或高算力占用场景。

四项 loss 均为有限值且量级接近，可作为训练路径健康检查；它们来自各自重新初始化模型后的短吞吐运行，不是收敛性或模型质量比较。DP=2 的 TFLOPs 仍按当前 rank 本地计时口径记录，和 DP=1 的全局口径未完全统一，不能单独据此判断算力效率。

## 9. 统一 microbatch 粒度的补充矩阵

上一节的 Megatron 默认 `microbatch_size=1`，一个 8 行 learner batch 会拆成 8 次 forward/backward 调度；这会放大小 batch 下的调度开销。为验证这一点，新增 `--megatron-microbatch-size` 参数，并在同一 workload、同一 GPU 2-3 拓扑下使用 `8`，使 TP=1/TP=2 各自把一个 learner batch 作为单个 THD microbatch 执行：

| 后端 | median step time | trainable tokens/s | model tokens/s | loss | TFLOPs |
|---|---:|---:|---:|---:|---:|
| TorchActor | 0.277 s | 4,616.0 | 7,494.6 | 0.007559 | 26.66 |
| Megatron TP=1 | 0.413 s | 2,992.9 | 4,859.3 | 0.007032 | 19.57 |
| Megatron TP=2 | 0.488 s | 2,558.5 | 4,154.1 | 0.006651 | 15.84 |
| Megatron TP=1、DP=2 | 0.803 s | 1,582.2 | 2,568.8 | 0.006873 | 4.72* |

对比默认 microbatch=1 的结果，Megatron TP=1 吞吐提升约 **6.43 倍**，TP=2 提升约 **5.87 倍**，证明之前 Torch 的优势很大一部分来自执行粒度差异，而不是单纯的模型计算速度。统一微批后 Torch 仍快于 Megatron TP=1，约为 **1.54 倍**；剩余差异来自 Megatron schedule、THD/Transformer Engine 路径和分布式框架固定开销。

TP=2 在统一微批后仍比 TP=1 慢约 **14.5%**，这才更接近本机无 NVLink、PCIe 同 NUMA 通信的真实代价。DP=2 在本配置反而低于 DP=1：全局 batch 仍为 8，但每个 DP rank 只拿到 4 条样本，无法充分填满单卡；同时增加梯度规约和双进程同步，因此这个 batch 太小，不适合证明 DP 扩展性。后续 DP=2 应使用至少 16 或 32 的 global batch，并令每个 rank 有足够的本地 microbatch。

该补充矩阵仍是短序列、0.6B 模型的吞吐诊断，不代表大模型生产配置。`--megatron-microbatch-size` 默认仍为 `1`，以保持旧实验和接口兼容。
