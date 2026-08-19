# V9 实验结果记录

> 测试日期：2026-08-19  
> 测试机器：5090，GPU 2、3 参与 Megatron 实验  
> 模型：`Qwen3-0.6B`  
> 推理服务：SGLang，`http://localhost:30000`  
> 数据与奖励：calculator toy agent，3 个 rollout，首轮作为 warmup，steady 统计取后两轮中位数  
> Megatron 序列格式：THD，未使用 BSHD padding

## 1. 原始四格对比

脚本：`scripts/bench/v9_end_to_end.py --repeats 1`

| 格 | Trainer | 编排 | 并行 | gen/wait_gen 中位数 | train 中位数 | 报告 total | reward | 报告 gap | wait ratio |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A | TorchActor | 同步 | 单卡 | 2.798 s | 0.284 s | 86.55 s | 1.000 | 1.0 | 0.908 |
| B | TorchActor | 异步 | 单卡 | 0.000 s | 0.276 s | 8.48 s | 1.000 | 0.0 | 0.000 |
| C | Megatron | 同步 | TP=2 | 1.397 s | 0.480 s | 85.64 s | 1.000 | 1.0 | 0.744 |
| D | Megatron | 异步 | TP=2 | 0.000 s | 0.480 s | 9.53 s | 1.000 | 0.0 | 0.000 |

### 每轮原始记录

**A：TorchActor + 同步**

```text
rollout 0: gen=5.368s train=2.5621s sync=24.7280s tokens=1880
rollout 1: gen=4.063s train=0.2638s sync=23.7876s tokens=1768
rollout 2: gen=1.534s train=0.3036s sync=23.9423s tokens=652
```

**B：TorchActor + 异步**

```text
rollout 0: wait_gen=5.374s train=2.554s tokens=1880
rollout 1: wait_gen=0.000s train=0.278s tokens=1768
rollout 2: wait_gen=0.000s train=0.273s tokens=652
```

**C：Megatron TP=2 + 同步**

```text
rollout 0: gen=5.327s train=10.3925s sync=25.7962s tokens=1880
rollout 1: gen=2.002s train=0.4848s sync=25.0056s tokens=1768
rollout 2: gen=0.792s train=0.4753s sync=15.3596s tokens=652
```

**D：Megatron TP=2 + 异步**

```text
rollout 0: wait_gen=2.649s train=5.922s tokens=1880
rollout 1: wait_gen=0.000s train=0.472s tokens=1768
rollout 2: wait_gen=0.000s train=0.488s tokens=652
```

### 结果解释

- 异步把后续 rollout 的生成放到训练期间，因此 steady 轮的 `wait_gen` 接近 0。
- A/B、C/D 的原始报告分别显示异步为同步的 `0.098x`、`0.111x`。但这次四格运行时，异步指标尚未把权重同步耗时纳入 `total`，所以这两个比值只能说明生成等待被隐藏，不能作为完整 wall-clock 加速比。
- C 的 TP=2 训练中位数约为 A 的 `1.69x`，没有体现线性加速；0.6B 模型较小，TP 通信和权重发布占比很高。
- 所有 rollout 的 `reward_mean=1.0`，说明这次 calculator 任务奖励路径正常，但不能据此证明策略质量提升。
- 原始四格中的 `policy_version_gap` 与预期定义不一致，属于版本标记接线问题，不能据此判断同步/异步 staleness。

## 2. Megatron DP=2 对比

脚本：`scripts/bench/v9_megatron_dp.py --dp-size 2 --rounds 3`

配置固定为 `TP=1, PP=1, DP=2, THD`。这才是真正的数据并行实验：每个 rank 使用一张 GPU，Megatron 在 DP 组内规约梯度和 loss。

| 模式 | gen/wait_gen 中位数 | train 中位数 | sync 中位数 | 阶段合计 | 进程 wall | pre-update PPO loss |
|---|---:|---:|---:|---:|---:|---|
| 同步 | 1.393 s | 1.762 s | 25.165 s | 95.88 s | 150.9 s | `[-1,-1,-1]`，均值 `-1.0` |
| 异步 | 0.000 s | 3.062 s | 29.473 s | 107.12 s | 166.8 s | `[-1,-1,-1]`，均值 `-1.0` |

对应的 token 序列完全一致：`[1880, 1768, 652]`。异步/同步比值为：

```text
阶段合计：1.117x
进程 wall：1.105x
loss mean 差值：0.000000
```

### DP 结果解释

- DP=2 的分布式训练和跨 rank loss 规约正常完成，没有 shape mismatch 或 collective hang。
- 在这组硬件和模型规模下，异步没有带来收益，反而慢约 10%。主要原因是每轮 `update_weights` 约 25--29 秒，已经超过训练和可重叠的生成时间。
- `loss=-1` 是真实的 GRPO policy loss，不是占位值：首轮是 on-policy，`ratio=1`；奖励为 1 时，`-min(ratio * advantage, clipped)` 为 `-1`。
- 在线 rollout 的 pre-update loss 主要用于检查数值路径是否正常。同步和异步的训练效果不应只用这个值判断；需要固定同一批 held-out rollout，在更新后统一评估 loss/reward。

## 3. 结论与后续

1. 当前主要瓶颈不是 Megatron 的前向/反向，而是训练后把权重重新发布到 SGLang 的 disk reload 路径。
2. async overlap 对生成较慢、权重同步较快的场景才有收益；当前 0.6B、2 卡 DP 配置不满足这个条件。
3. 后续若要继续优化，应优先单独测量或替换权重同步路径，再重新比较 sync/async；同时增加固定 rollout 的 post-update loss/reward 评估。

原始日志位于服务器：`/tmp/v9_all_cells.log`、`/tmp/v9_megatron_dp.log`。DP 结果文件由脚本写入 `docs/decisions/v9_megatron_dp_sync_async.txt`。
