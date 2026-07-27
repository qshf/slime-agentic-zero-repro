# GRPO 公式 ↔ 代码对照（nano vs minimind）

> 为什么写这份：nano 的 GRPO 实现（[mini_slime/custom_convert.py](../mini_slime/custom_convert.py) +
> [mini_slime/torch_actor.py](../mini_slime/torch_actor.py)）和你熟悉的
> `minimind-new/trainer/train_grpo.py` 写法很不一样，看不懂。本文把 **GRPO 数学公式**逐条对上
> **两边的代码**，证明 nano 没算错，只是跟了不同的锚点、用了不同的代码风格。
>
> **一句话结论**：核心 policy-gradient 数学**两边等价**；差异全在「reward 怎么算」「要不要 KL」
> 「多轮拆 turn」「向量化 vs for 循环」这些**外围**，不在 GRPO 本身。

---

## 0. 先说清两份代码的"身份"

| | minimind `train_grpo.py` | nano（slime-agentic 复现） |
|---|---|---|
| 锚点 | 教科书单轮 GRPO（DeepSeek GRPO 原版） | slime-agentic 的 **ToolOrchestra** 变体 |
| 任务 | 单轮问答（prompt → 一段 response） | **多轮 agentic**（orchestrator 多次调工具） |
| reward | 一个标量（规则分 + reward model 打分） | **偏好加权多组件**（正确性 + 成本 + 延迟 + 路由） |
| 归一 | **一层**：组内标准化 `(r-mean)/std` | **两层**：先 min-max 偏好归一，再组内标准化 |
| KL 项 | **有**（`beta=0.1`，k3 估计器，带 ref model） | **无**（V6 有意关掉，取最纯 policy gradient） |
| 代码形态 | 向量化张量 `[B·num_gen, R]` 一把算 | Python **逐样本 for 循环** |
| 低方差组 | 不显式处理（靠 `std+1e-4`，adv≈0 自然无信号） | **显式 mask**（`std<0.1` 的组 loss_mask 置 0） |

> 所以你觉得"写法不一致"是对的——它们本来就是**同一算法的两个变体**。下面证明重叠的部分等价。

---

## 1. GRPO 的数学（这是两边共同遵守的）

GRPO = 把一个 prompt 采样 **n 个 rollout** 成一组，用**组内相对好坏**当 advantage，套 PPO 的 clip 目标。
没有 critic、没有 GAE，advantage 就是"这条 rollout 比同组平均好多少个标准差"。

**（a）组内相对 advantage**（每条 rollout 一个标量，广播到它的每个 token）：

```
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

**（b）重要性采样比**（当前策略 vs 采样时的旧策略，逐 token）：

```
ratio_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t) = exp( logp_cur_t − logp_old_t )
```

**（c）PPO clip 目标**（逐 token 取悲观值，A_i 是上面那个标量）：

```
L_t = − min( ratio_t · A_i ,  clip(ratio_t, 1−ε, 1+ε) · A_i )   [ + β·KL_t ]
```

**（d）聚合**（只在 response 的 loss_mask=1 处；先序列内平均，再样本间平均）：

```
loss = mean_over_samples(  Σ_t (L_t · mask_t) / Σ_t mask_t  )
```

---

## 2. 逐条对上代码

### (a) 组内 advantage `A_i = (R_i − mean) / (std + eps)`

**minimind**（一层归一，直接对标量 reward 做）——[train_grpo.py:121-124](../../minimind-new/trainer/train_grpo.py#L121)：

```python
grouped_rewards = rewards.view(-1, args.num_generations)          # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_gen)   # 组均值
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_gen)
advantages = (rewards - mean_r) / (std_r + 1e-4)                  # A_i
```

**nano**（两层：先偏好 min-max，再组内标准化）——[custom_convert.py:72-90](../mini_slime/custom_convert.py#L72)：

```python
for g in range(num_examples):
    group = rewards[g*n : (g+1)*n]
    mean = sum(group) / len(group)
    std  = (sum((r-mean)**2 for r in group) / len(group)) ** 0.5   # unbiased=False，和 minimind 一致
    has_signal = std > MIN_STD_THRESHOLD                            # 0.1；minimind 无此步
    for r in group:
        nr = clip((r - mean) / (std + 1e-6), -3, 3) if has_signal else 0.0   # A_i，多了 clip[-3,3]
```

对应关系：
- `mean` ↔ `mean_r`，`std` ↔ `std_r`（都用 **有偏方差** `/len`，即 minimind 的 `unbiased=False`）。
- **差异 1**：nano 的 `R_i` 不是原始标量，而是先经 `_compute_preference_rewards` 做了组内 min-max
  偏好加权（正确性 + 成本 + 延迟 + 工具路由）；minimind 的 `R_i` 就是 `calculate_rewards` 的标量。
  → 这是 **ToolOrchestra 多组件 reward** 的产物，不是 GRPO 本身。
- **差异 2**：nano 把 `A_i` clip 到 `[-3,3]`，minimind 不 clip。防离群值，语义无害。
- **差异 3**：nano 对 `std<0.1` 的组显式 `A_i=0` + mask；minimind 靠 `std+1e-4` 让 `A_i≈0`。
  **效果同向**（低方差组不学），nano 是显式版。

> ⚠️ 这层就是 nano V6"acc 变了但不稳定涨"的根源：小规模下多数组要么全对要么全错 → `std<0.1` →
> 被 mask → 几乎不训练。**这是规模/方差问题，不是算法 bug**（minimind 大 batch + 连续 reward
> 天然有方差，不撞这个坑）。

### (b) 重要性比 `ratio = exp(cur − old)`

**minimind**——[train_grpo.py:134](../../minimind-new/trainer/train_grpo.py#L134)：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)   # exp(cur - old)
```

**nano**——[torch_actor.py:84-85](../mini_slime/torch_actor.py#L84)：

```python
ppo_kl = old_log_probs - cur_log_probs   # = old - cur
ratio  = (-ppo_kl).exp()                 # = exp(cur - old)  ← 完全一样，只是绕了个变量名
```

`(-ppo_kl).exp() = exp(-(old-cur)) = exp(cur-old)`。**恒等**。nano 绕这一步是为了对齐 slime 源
`ppo_kl` 命名（源里 `ppo_kl` 还用于 KL 日志），不是算法差异。

### (c) clip 目标 `L_t = −min(ratio·A, clip(ratio)·A)`

**minimind**——[train_grpo.py:139-142](../../minimind-new/trainer/train_grpo.py#L139)：

```python
clipped_ratio   = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
per_token_loss1 = ratio * advantages.unsqueeze(1)
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
per_token_loss  = -(torch.min(per_token_loss1, per_token_loss2) - beta * per_token_kl)
#                 = -min(ratio·A, clip·A) + beta·KL
```

**nano**——[torch_actor.py:87-89](../mini_slime/torch_actor.py#L87)：

```python
pg_losses1 = -ratio * adv
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * adv
pg_loss    = torch.maximum(pg_losses1, pg_losses2)
#          = max(-ratio·A, -clip·A)
```

**关键等式**（你最可能卡在这里）：

```
max(-ratio·A, -clip·A)  ==  -min(ratio·A, clip·A)
```

因为对任意 x,y 恒有 `max(-x, -y) = -min(x, y)`。所以 nano 的 `pg_loss` 和 minimind 的
`-min(...)` 项**逐元素相等**。

**唯一实质差异**：minimind 有 `+ beta * per_token_kl`（KL 惩罚），nano 没有。
- minimind KL 用 k3 估计器：`per_token_kl = exp(ref-cur) - (ref-cur) - 1`（[train_grpo.py:132-133](../../minimind-new/trainer/train_grpo.py#L132)），需要 **ref model**。
- nano V6 **有意关掉 KL**（`beta=0`，无 ref model），取"最纯 policy gradient"，见
  [torch_actor.py 顶部偏离表](../mini_slime/torch_actor.py#L13) 和 docs/decisions/v6.md。
  → 这是**简化，不是错**：β=0 时两式完全一致。补 KL 是后续（需再起一个 ref 前向）。

另注：minimind 的 `epsilon_high` 默认 5.0（宽上界，配 cispo），nano 默认 `eps_clip_high=0.2`（对称 clip）；
这只是超参不同，公式位置一致。

### (d) 聚合：序列内 masked mean → 样本间 mean

**minimind**——[train_grpo.py:143](../../minimind-new/trainer/train_grpo.py#L143)：

```python
policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1).clamp(min=1)).mean()
#             每条序列: Σ(L·mask)/Σmask  → 再对 batch 求 mean
```

**nano**——[torch_actor.py:92-99](../mini_slime/torch_actor.py#L92)：

```python
sample_loss = (pg_loss * mask).sum() / torch.clamp_min(mask.sum(), 1.0)  # 每条: Σ(L·mask)/Σmask
total_loss  = total_loss + sample_loss                                    # 累加
...
loss = total_loss / n_samples                                             # 再除以样本数 = mean
```

**同一个聚合**：都是"每条序列在 mask 上求均值，再对所有序列求均值"。minimind 用张量
`.sum(dim=1)/....mean()` 一步到位，nano 用 for 循环累加再除，结果相同。

---

## 3. advantage 是标量还是逐 token？——都对，且一致

你可能疑惑 nano [torch_actor.py:86](../mini_slime/torch_actor.py#L86) `adv = float(rewards[i])` 是个**标量**，
而 minimind `advantages.unsqueeze(1)` 是 `[B·num_gen, 1]` **广播**到 `[B·num_gen, R]`。

这俩语义相同：**GRPO 的 advantage 是"每条 rollout 一个标量"，广播到该 rollout 的每个 response token**。
- minimind：`advantages`（每 rollout 一个数）`.unsqueeze(1)` 后靠 broadcast 铺满 R 个 token。
- nano：逐样本循环，`adv` 是 Python float，`-ratio * adv` 里 `ratio` 是 `[seq-1]` 张量，`adv` 标量
  自动广播到每个 token。

这也和 slime 源一致——[actor.py:502-505](../../slime-agentic/slime/backends/fsdp_utils/actor.py#L502) 就是
`torch.tensor([rewards[i]] * response_lengths[i])`（把标量复制成整段），nano 的标量广播是它的等价简写。

---

## 4. log_prob 的位置对齐（最容易出 off-by-one 的地方）

GRPO 要 `ratio_t = exp(logp_cur_t − logp_old_t)`，前提是 `cur` 和 `old` 指的是**同一个 token 位置**。

**minimind** 用 `logp_pos` 精确 gather 出 completion 段的 logp（[train_grpo.py:93,101](../../minimind-new/trainer/train_grpo.py#L93)）：
`logits[:, :-1]` 预测 `outputs[:, 1:]`，即位置 t 的 logits 预测 token t+1。

**nano** [torch_actor.py:41-53](../mini_slime/torch_actor.py#L41) 同样 `shifted=logits[:-1]` 对 `targets=tokens[1:]`，
返回长度 `len(tokens)-1` 的向量，第 i 个 = `logP(tokens[i+1])`。然后 [torch_actor.py:75,79](../mini_slime/torch_actor.py#L75)
把 `loss_mask` 和 `old_log_probs` 都取 `[1:]` 右移一位，使三者（cur / old / mask）在**同一 token**对齐。

两边都遵守"logits[t] 预测 token[t+1]"这条铁律，只是 nano 单序列、minimind batch gather。**无 off-by-one**。

---

## 5. 差异清单（供快速核对，全部是"外围/简化"非"算法错"）

| 环节 | minimind | nano | 是不是 bug |
|---|---|---|---|
| reward 来源 | 标量（规则+RM） | 偏好加权多组件（min-max 后） | 否，ToolOrchestra 变体 |
| advantage 公式 | `(r-mean)/(std+1e-4)` | 同 + clip[-3,3] + 低方差 mask | 否，等价+防离群 |
| ratio | `exp(cur-old)` | `exp(-(old-cur))` | 否，恒等 |
| clip 目标 | `-min(ratio·A, clip·A)` | `max(-ratio·A, -clip·A)` | 否，恒等 |
| KL 项 | 有（β=0.1, k3, ref model） | 无（β=0，V6 有意关） | 否，β=0 特例 |
| 聚合 | 张量 masked-mean→batch-mean | for 循环 masked-mean→样本-mean | 否，同一聚合 |
| 多轮 | 单序列 | 按 turn 拆独立训练样本 | 否，agentic 需要 |
| 代码形态 | 向量化 | 逐样本 Python | 否，风格差异 |

---

## 6. 如果想让 nano 写法更像 minimind（可选，非必须）

nano 当前风格是"贴 slime-agentic 源"（铁律要求）。若纯为可读性想更接近 minimind：

1. **补 KL 项**：起一个 `ref_model` 前向，加 `beta * (exp(ref-cur) - (ref-cur) - 1)`——但这会偏离
   V6"最纯 GRPO"的刻意设计，且多一份显存。
2. **向量化聚合**：把逐样本 for 换成 padding 成 batch 张量——但 nano 刻意逐样本是为了教学清晰
   （[torch_actor.py 偏离表](../mini_slime/torch_actor.py#L15) 写了 packing 留 V7）。

两条都**不改变数学**，只改风格，且都与"对齐源 + 教学清晰"的取舍相悖，故默认不动。

---

## 附：为什么 nano V6 acc 不稳定涨（和本算法无关）

`std<0.1 → mask` 在小规模下几乎把所有组 mask 掉（4 题 × n=4 × 小模型 → 组内多为全对/全错）。
这不是 GRPO 写错，minimind 靠大 batch + 连续 reward 天然有方差绕过了。要在 nano 复现"稳定涨"，
调 rollout 采样温度制造方差 / 换适中难度题 / 加大 n 和轮数即可（见 docs/decisions/v6.md 尾部）。
