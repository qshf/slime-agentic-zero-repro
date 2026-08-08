# V7 后续迭代方案 —— A100 + FA2 适配版

> 基于：`docs/decisions/v7-onward-plan.md`（5090 版本）
> 差异前提：A100 sm_80 有 FA2 预编译 wheel 且已验证 PASSED；`agentic-rl-infra-lab` 吞吐测试显示 FA2 比 TE cuDNN 快 **4.7×–40×**（序列越长差距越大）。
> 意义：**V7 的两大偏离（#1 packing 用块对角 mask、#7 O(T²) 隔离）在 A100 上都可以诚实退休**——直接用 FA2 varlen kernel 替代显式 mask。

---

## 0. 5090 版与 A100 版的核心差异

| 维度 | 5090 方案 | A100 方案 |
|---|---|---|
| FA2 可用性 | ❌ sm_120 源码编译产物丢失，未知 | ✅ 预编译 wheel，PASSED |
| packing 隔离 | 显式块对角 4D mask（O(T²)，偏离 #7）| FA2 varlen（O(N) HBM，真正的 IO-aware kernel）|
| attention 后端 | TE cuDNN（唯一可用）| **FA2（首选）**，TE cuDNN 仅作对照基线 |
| V7.5 退休内容 | 退休偏离 #1（packing 暂缓），保留偏离 #7 | **同时退休偏离 #1 + #7** |

---

## 1. 落地原则（与 5090 版相同，不重复）

- 源锚点仍是 slime-agentic；infra 提供 CPU 不变量验收
- 每处落地退休或新登记偏离（铁律三要素）
- 每子版本独立可验证、独立回归

---

## 2. 迭代顺序

### V7.4 · 接零风险 infra（与 5090 版相同）

learner_contract + learner_metrics + 版本戳，纯 CPU，A100/5090 无差异。详见 5090 版 §V7.4，不重复。

---

### V7.5-a100 · 退休偏离 #1 + #7：FA2 varlen packing

**5090 版 V7.5 的方案**：用显式块对角 4D mask 替代 FA2 varlen（因 5090 无 FA2），消除 padding 浪费但付出 O(T²) mask 开销，登记为新偏离 #7。

**A100 版的变化**：FA2 已可用，直接用 varlen 路径，无需 #7 这个折中。

#### 实施

`data_packing.py`：`pack_sequences` 不变（flat 1D + cu_seqlens），删去 `build_block_diagonal_causal_mask`。

`FSDPTrainer._packed_backward`：

```python
from flash_attn import flash_attn_varlen_func

# 替换 4D mask 路径：直接传 cu_seqlens，attention_mask=None
outputs = model(
    input_ids=packed_tokens,
    attention_mask=None,          # FA2 varlen 靠 cu_seqlens 隔离，不传 4D mask
    position_ids=position_ids,    # reset-position-ids，每段从 0 开始
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
    use_flash_attn=True,
)
```

或直接在 attention forward 里调 `flash_attn_varlen_func`，取决于 transformers 版本的集成方式（5.x 已内置 FA2 dispatch，传 `use_flash_attention_2=True` 即可）。

**关键正确性要求**：
1. `position_ids` 按段 reset（每段从 0 开始），否则位置编码跨段混淆
2. `cu_seqlens` 精确对齐 `loss_mask`（pack 后 loss_mask 仍需正确对应每段 response）
3. 验收断言：backend 必须是 FA2（打印 `flash_attn.__version__`，不能静默回退到 eager）

#### 验收

与 5090 版 V7.5 相同的等价断言，但 precision 门槛更严（FA2 varlen vs padding 的 loss 差应 ≤ FA2 本身的数值误差，不含 5090 版引入的 O(T²) mask 舍入）：

```
fp32: loss diff < 1e-5，grad cosine > 0.9999
bf16: loss diff < 5e-4，grad cosine > 0.99
```

#### 偏离处理

- **退休偏离 #1**（packing 暂缓）：✅ FA2 varlen 上线
- **不引入偏离 #7**（5090 版的块对角 mask）：A100 直接用 varlen，不存在此偏离
- **新登记偏离（若有）**：transformers 5.x FA2 dispatch 的集成方式若与源 `data_packing.pack_sequences` + `attention_mask=None` 路径有细节偏差，在此登记

---

### V7.6-a100 · 切换 attention 后端为 FA2（全模型）

V7.5-a100 只在 packing 路径用 FA2 varlen；V7.6 把整个训练前向（含非 packing 的单样本路径）也切到 FA2。

**动机**：`06-attn-backend-throughput-a100-2026-08-09.md` 数据——RL rollout 典型序列 1k–4k tokens，FA2 在此区间比 TE cuDNN 快 **19×–40×**，端到端训练吞吐提升显著。

**实施**：

```bash
# 安装（容器内，--no-deps 避免破坏 TE 依赖）
pip install --no-deps flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

```python
# FSDPTrainer 初始化时
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path)
config._attn_implementation = "flash_attention_2"
model = AutoModelForCausalLM.from_config(config, ...)
```

**注意**：FA2 安装后 TE import 崩溃（见 `06-attn-backend-throughput` 文档）——若同一容器还需要跑 TE 相关脚本，先 `pip uninstall flash-attn -y`，overlayfs 会恢复镜像层的 FA4，TE 重新可用。

**验收**：
- `config._attn_implementation` 确认是 `flash_attention_2`（不是 eager/sdpa）
- loss 与 eager 路径数值一致（parity < bf16 误差范围）
- 端到端训练步时间（含 attention 层）明显下降

---

## 3. TE cuDNN 的定位（A100 上）

A100 上 TE cuDNN **不再是首选**，降级为：
- **infra 对照基线**（`04_megatron_te_thd_spike.py`，已 PASSED）
- **FA4 不可用时的保底**（A100 上 FA4 架构拒绝，TE 是唯一非 FA2 选项）
- **镜像层保留**（te-cudnn-system-spike 仍是训练镜像基底，TE 是其中一层）

FA2 装进后 TE import 崩溃，两者在同一进程里互斥。需要 TE 时 `pip uninstall flash-attn`，需要 FA2 时重装。若频繁切换，考虑分别做两个镜像 tag。

---

## 4. 与 5090 版迭代的关系

| 子版本 | 5090 | A100 |
|---|---|---|
| V7.4 | 相同 | 相同 |
| V7.5 | 块对角 mask，登记偏离 #7 | FA2 varlen，退休 #1+#7 |
| V7.6 | TE cuDNN（唯一可用）| **切 FA2 全模型**，TE 降为对照 |
| V8+ | Megatron 多维并行 | 相同，但 attention 后端已是 FA2 |

**分支策略**：A100 和 5090 各开独立分支，不共用同一分支。
- `v8-a100`：从 `v7` 末端切出，FA2 varlen 路径，退休偏离 #1+#7
- `v8-5090`：从 `v7` 末端切出，块对角 mask 路径，保留偏离 #7

两分支代码会有差异（attention 后端不同），不强行合并——等 5090 的 FA2/FA4 路径打通后再收拢。

---

## 5. 待决策

- [ ] transformers 5.x `_attn_implementation="flash_attention_2"` 与手动调 `flash_attn_varlen_func` 哪个更贴源？（前者更简洁，后者控制更精确）
- [ ] FA2 + TE 两镜像 tag 是否值得固化（`te-cudnn-system-spike` + `fa2-system-spike`），避免频繁 pip uninstall/install
- [ ] V7.6 上线后补一次端到端 attention 层吞吐对比（full model，不只 attention kernel）
