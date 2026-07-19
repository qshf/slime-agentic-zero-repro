# slime-agentic 从 0 复现学习计划

## 目标

本项目用于从 0 复现一个最小 Agentic RL 训练系统，并在复现过程中学习 `slime-agentic` 的系统设计。

最终目标不是一开始就读懂 Megatron、Ray、SGLang 的全部细节，而是逐步建立下面这条链路：

```text
Prompt 数据
  -> Agent Rollout
  -> Reward
  -> tokens / loss_mask / rewards
  -> Trainer
  -> update_weights
  -> Rollout Engine
  -> 下一轮 Rollout
```

需要重点掌握的问题：

```text
1. Agent 多轮交互如何变成 RL 训练样本？
2. loss_mask 如何区分模型输出和工具返回？
3. reward 如何基于完整轨迹计算？
4. rollout 和 train 如何组成闭环？
5. 训练后的权重如何同步给推理引擎？
6. 为什么 Agentic RL 的瓶颈常常在系统吞吐率和 GPU 利用率？
```

## 推荐项目结构

```text
slime-agentic-zero-repro/
  README.md
  notes/
    00_project_map.md
    01_rl_data_contract.md
    02_agent_rollout.md
    03_ray_sglang.md
    04_training_acceleration.md
    05_experiments.md
  toy_rl/
    data/
    agent/
    train_loop.py
    reward.py
    sample.py
  mini_slime/
    rollout_manager.py
    trainer.py
    weight_sync.py
  experiments/
    logs/
    configs/
```

## 阶段 0：建立项目地图

时间：1-2 天

先不运行复杂训练，不改代码，只画系统图。

重点阅读原项目：

```text
/Users/qshf/my-project/slime-agentic/README.md
/Users/qshf/my-project/slime-agentic/train.py
/Users/qshf/my-project/slime-agentic/train_async.py
/Users/qshf/my-project/slime-agentic/agentic/
/Users/qshf/my-project/slime-agentic/slime/ray/
/Users/qshf/my-project/slime-agentic/slime/rollout/
/Users/qshf/my-project/slime-agentic/slime/backends/
```

本阶段产物：

```text
notes/00_project_map.md
```

需要写清楚：

```text
1. train.py 的主循环是什么？
2. RolloutManager 负责什么？
3. Actor Trainer 负责什么？
4. SGLang engine 负责什么？
5. Agentic 方法代码和 slime 框架代码如何连接？
```

验收标准：

能够画出：

```text
Dataset -> RolloutManager -> custom generate -> custom reward -> Trainer -> update_weights -> Rollout Engine
```

## 阶段 1：手写最小 RL 数据契约

时间：3-5 天

不要使用 Ray、SGLang、Megatron。先自己写一个 toy 版本，理解 Agentic RL 和训练系统之间到底靠什么连接。

需要实现的数据结构：

```python
sample = {
    "prompt": "...",
    "response": "...",
    "tokens": [...],
    "loss_mask": [...],
    "reward": 1.0,
}
```

推荐 toy 任务：

```text
数学题 + calculator 工具
```

示例轨迹：

```text
Prompt: 2 + 3 = ?
Agent: 我需要调用 calculator
Tool: 5
Agent: 最终答案是 5
Reward: 正确为 1，错误为 0
```

loss_mask 规则：

```text
prompt token: 0
agent 生成 token: 1
tool 返回 token: 0
final answer token: 可根据训练目标设为 0 或 1
```

本阶段产物：

```text
toy_rl/sample.py
toy_rl/reward.py
toy_rl/train_loop.py
notes/01_rl_data_contract.md
```

验收标准：

运行 toy loop 后，可以打印出完整样本：

```text
prompt
response
tokens
loss_mask
reward
```

并能解释哪些 token 被训练，哪些 token 不被训练。

## 阶段 2：复现 custom generate / reward

时间：3-5 天

开始对齐 slime-agentic 的方法接口。

重点理解三个 hook：

```bash
--custom-generate-function-path
--custom-rm-path
--custom-eval-rollout-log-function-path
```

学习任务：

```text
1. 找到 agentic/ 中某个方法的 generate 函数
2. 找到它的 reward_func
3. 看 generate 如何构造完整 agent 轨迹
4. 看 loss_mask 如何构造
5. 看 reward 是最终答案打分还是过程打分
```

建议优先阅读：

```text
/Users/qshf/my-project/slime-agentic/agentic/agentflow/
```

本阶段产物：

```text
toy_rl/agent/
notes/02_agent_rollout.md
```

验收标准：

自己实现一个异步 `generate(sample)`：

```python
async def generate(sample):
    ...
    return sample
```

返回的 sample 至少包含：

```text
response
tokens
loss_mask
reward 或 reward 计算所需信息
```

## 阶段 3：实现最小训练闭环

时间：1 周

仍然不接入复杂框架，先实现一个本地版本：

```text
读取 prompts
-> generate rollout
-> reward_func
-> convert to rollout batch
-> fake trainer step
-> fake update_weights
-> 下一轮
```

这里的 trainer 可以先不真的训练大模型，只模拟：

```text
计算 loss
打印 batch 统计
记录 reward 均值
记录 token 数量
```

本阶段产物：

```text
mini_slime/rollout_manager.py
mini_slime/trainer.py
mini_slime/weight_sync.py
experiments/logs/stage3_toy_loop.log
```

验收标准：

能够运行一个完整闭环：

```text
rollout 0 -> train 0 -> update_weights 0
rollout 1 -> train 1 -> update_weights 1
```

并输出：

```text
rollout_time
train_time
update_weights_time
reward_mean
tokens_per_rollout
```

## 阶段 4：学习 Ray 调度层

时间：3-5 天

只学项目用到的 Ray 概念：

```text
@ray.remote
.remote()
ray.get()
Ray Actor
ObjectRef
placement_group
```

对照原项目阅读：

```text
/Users/qshf/my-project/slime-agentic/slime/ray/placement_group.py
/Users/qshf/my-project/slime-agentic/slime/ray/actor_group.py
/Users/qshf/my-project/slime-agentic/slime/ray/rollout.py
```

本阶段任务：

```text
1. 将 toy RolloutManager 改成 Ray actor
2. 将 toy Trainer 改成 Ray actor
3. 用 ray.get 控制同步点
4. 观察哪些地方可以异步
```

本阶段产物：

```text
mini_slime/ray_rollout_manager.py
mini_slime/ray_trainer.py
notes/03_ray_sglang.md
```

验收标准：

能够解释：

```text
1. rollout 进程和 train 进程如何分开
2. ray.get 为什么会造成等待
3. train_async.py 为什么能提前启动下一轮 rollout
```

## 阶段 5：学习 SGLang rollout 角色

时间：1 周

先不研究 kernel，只理解 SGLang 在系统里的位置。

重点问题：

```text
1. 为什么 rollout 要用独立推理服务？
2. 多个 SGLang engine 如何并发？
3. rollout-num-gpus-per-engine 和 tp_size 是什么关系？
4. Prefill/Decode 分离为什么适合多轮 agent？
5. update_weights 如何把训练权重同步到 rollout engine？
```

重点阅读：

```text
/Users/qshf/my-project/slime-agentic/slime/ray/rollout.py
/Users/qshf/my-project/slime-agentic/slime/backends/sglang_utils/
/Users/qshf/my-project/slime-agentic/docs/zh/advanced/pd-disaggregation.md
/Users/qshf/my-project/slime-agentic/docs/zh/developer_guide/profiling.md
```

本阶段产物：

```text
notes/03_ray_sglang.md
```

验收标准：

能够说清：

```text
训练模型和 rollout 模型为什么是两套运行形态
为什么训练后需要 update_weights
为什么多轮 agent rollout 会让推理吞吐变复杂
```

## 阶段 6：学习训练加速

时间：2-3 周

学习顺序：

```text
FSDP
-> data packing / dynamic batch
-> Megatron 基础概念
-> slime 如何调用 Megatron
```

先看 FSDP：

```text
/Users/qshf/my-project/slime-agentic/slime/backends/fsdp_utils/actor.py
```

重点：

```text
FSDP 包模型
CPU offload
gradient checkpointing
dynamic batch size
max_tokens_per_gpu
sequence packing
optimizer step
```

再看 Megatron：

```text
/Users/qshf/my-project/slime-agentic/slime/backends/megatron_utils/actor.py
/Users/qshf/my-project/slime-agentic/slime/backends/megatron_utils/model.py
```

只先理解概念：

```text
TP: tensor parallel
PP: pipeline parallel
CP: context parallel
EP: expert parallel
sequence parallel
recompute
async save
```

本阶段产物：

```text
notes/04_training_acceleration.md
```

验收标准：

能够解释：

```text
1. dynamic batch size 为什么能提升训练效率
2. data packing 解决了什么浪费
3. offload_train / offload_rollout 解决了什么显存问题
4. Megatron 比普通 DDP 多解决了哪些并行问题
```

## 阶段 7：复现同步与异步训练循环

时间：3-5 天

对比原项目：

```text
/Users/qshf/my-project/slime-agentic/train.py
/Users/qshf/my-project/slime-agentic/train_async.py
```

实现两个 toy 版本：

同步版：

```text
rollout N
-> train N
-> update_weights
-> rollout N+1
```

异步版：

```text
rollout N
-> 提前启动 rollout N+1
-> train N
-> 定期 update_weights
```

本阶段产物：

```text
mini_slime/train_sync.py
mini_slime/train_async.py
notes/05_experiments.md
```

验收标准：

能够通过日志比较：

```text
同步版总耗时
异步版总耗时
rollout 等待 train 的时间
train 等待 rollout 的时间
```

## 阶段 8：实现自己的 mini Agentic RL 任务

时间：1-2 周

实现一个真正有 agent 交互的 toy 任务：

```text
数学题 + calculator 工具
```

agent loop：

```text
模型生成工具调用
-> Python calculator 执行
-> 工具结果拼回上下文
-> 模型继续生成
-> 最终答案
-> reward_func 对比标准答案
```

必须实现：

```text
custom generate
custom reward
loss_mask
rollout batch
训练循环
实验日志
```

本阶段产物：

```text
toy_rl/agent/calculator_agent.py
toy_rl/reward.py
mini_slime/train_sync.py
experiments/logs/mini_agentic_rl.log
```

验收标准：

能够清楚展示：

```text
1. 每个样本的完整 trajectory
2. 哪些 token 参与训练
3. reward 如何计算
4. 每轮 rollout/train/update_weights 的耗时
```

## 阶段 9：做吞吐率实验

时间：1 周

系统修改参数并记录结果。

实验参数：

```text
rollout_batch_size
n_samples_per_prompt
global_batch_size
max_tokens_per_gpu
rollout_num_gpus
rollout_num_gpus_per_engine
update_weights_interval
colocate / non-colocate
sync / async
```

观察指标：

```text
rollout_time
train_time
update_weights_time
tokens/s
samples/s
GPU 利用率
显存占用
reward_mean
```

本阶段产物：

```text
experiments/configs/
experiments/logs/
notes/05_experiments.md
```

验收标准：

能够回答：

```text
1. 当前瓶颈在 rollout 还是 train？
2. GPU 是否有明显等待？
3. 异步训练是否提升吞吐？
4. update_weights 是否成为瓶颈？
5. 多轮 agent 是否拉低 rollout 吞吐？
```

## 学习节奏

如果每天投入 2-3 小时，建议节奏：

```text
第 1 周：项目地图 + toy RL 数据契约
第 2 周：custom generate / reward / loss_mask
第 3 周：最小训练闭环
第 4 周：Ray + SGLang rollout
第 5 周：dynamic batch / data packing / 权重同步
第 6 周：FSDP + Megatron 接入方式
第 7 周：自己的 mini Agentic RL 任务
第 8 周：吞吐率和 GPU 利用率实验
```

## 避坑原则

不要从这里开始：

```text
Megatron 源码
SGLang kernel
大模型多机训练脚本
MoE 大模型配置
论文复现效果
```

应该从这里开始：

```text
train.py 主循环
Sample 数据结构
custom generate
custom reward
loss_mask
rollout / train / update_weights
```

最重要的原则：

```text
先复现数据如何流动，再学习系统如何加速。
```

