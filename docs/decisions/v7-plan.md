# V7 FSDP 真训一步 — 实施计划

> 状态：规划中（2026-07-28）  
> 分支：`v7`（从 `v6` 末端切出）  
> 验证目标：5090 服务器多卡端到端，FSDP 分片 + packing + 真训练一步 + 权重同步

---

## 1. 核心目标

**从 V6 的痛点出发**：
- V6 使用单卡纯 torch，无分布式并行
- 逐样本训练无 packing（浪费算力）
- 无参数 offload（大模型放不下）
- 权重同步用 disk reload（低效）

**V7 要解决**：
把 V6 的 `toy_rl/trainer/torch_trainer.py` 换成 **FSDP 后端**，对齐源项目 `slime/backends/fsdp_utils/actor.py` 的核心机制：

1. **FSDP 包装模型** — 参数分片到多卡（1D data parallelism）
2. **Sequence packing** — 多条样本拼成一个 batch 提高利用率
3. **真训练一步** — 保持 V6 已有的真 log_probs + GRPO + backward，但运行在 FSDP 上
4. **DTensor 权重同步** — 使用 distributed checkpoint + broadcast 替代 disk reload

**不做（Phase 2 或更晚）**：
- CPU offload（先单卡能跑通，显存不够时再加）
- Reference model（V6 已关 `--no-ref`）
- OPSM masking / KL loss（V6 已关）
- Multimodal inputs（无视觉）
- 多维并行（TP/PP/CP，留 V8）

---

## 2. 源项目对照（FSDP 核心流程）

基于 explore agent 报告，核心流程：

```
init()
  ├─ _setup_device_mesh() → 1D (dp,) mesh
  ├─ Load model (ranks sequentially to avoid cache race)
  ├─ apply_fsdp2(model, mesh, cpu_offload=False, mp_policy)
  │   └─ fully_shard() on decoder layers + embeddings
  ├─ _fsdp2_load_full_state_dict() → broadcast from rank 0
  ├─ Create optimizer (AdamW)
  └─ Load checkpoint (optional)

train(rollout_data)
  ├─ _packed_data() → pack_sequences() + compute num_microbatches
  ├─ For each packed_batch:
  │   └─ _train_step()
  │       ├─ Forward pass → logits
  │       ├─ get_logprob_and_entropy(logits, target_tokens)
  │       ├─ Unpack sequences
  │       ├─ Compute loss (GRPO policy loss + entropy)
  │       ├─ loss.backward()
  │       └─ If (mbs_id+1) in grad_accum:
  │           ├─ clip_grad_norm_()
  │           ├─ optimizer.step()
  │           └─ all_gather_object() → metrics
  └─ Return aggregated metrics

update_weights()
  ├─ Get state_dict (DTensor → redistribute + to_local)
  └─ dist.broadcast() each param to rollout engines
```

**关键文件与行号**（源项目 `/Users/qshf/my-project/slime-agentic`）：
- `slime/backends/fsdp_utils/actor.py:48-154` — init
- `slime/backends/fsdp_utils/actor.py:933-984` — apply_fsdp2
- `slime/backends/fsdp_utils/data_packing.py:10-121` — pack_sequences
- `slime/backends/fsdp_utils/actor.py:551-722` — _train_step
- `slime/backends/fsdp_utils/update_weight_utils.py:32-82` — update_weights

---

## 3. nano V7 架构设计

### 3.1 目录结构

```
toy_rl/
├── trainer/
│   ├── torch_trainer.py         # V6 单卡版本（保留用于对比）
│   ├── fsdp_trainer.py          # ← V7 新增：FSDP 后端
│   └── data_packing.py          # ← V7 新增：sequence packing
└── utils/
    └── fsdp_utils.py            # ← V7 新增：apply_fsdp2, device_mesh

mini_slime/
├── train_ray.py                 # V6 版本（保留）
└── train_fsdp.py                # ← V7 新增：Ray + FSDP 编排
```

### 3.2 模块职责

**`toy_rl/utils/fsdp_utils.py`**（对齐源 `actor.py:933-984`）
- `setup_device_mesh(world_size)` → 1D mesh for data parallelism
- `apply_fsdp2(model, mesh, args)` → wrap with `fully_shard()`

**`toy_rl/trainer/data_packing.py`**（对齐源 `data_packing.py`）
- `pack_sequences(tokens, loss_masks, rewards, ...)` → list of packed batches
- `unpack_sequences(packed_batch)` → per-sample dicts for loss computation

**`toy_rl/trainer/fsdp_trainer.py`**（对齐源 `actor.py`）
- `FSDPTrainer.__init__()` → setup mesh, load model, apply FSDP, create optimizer
- `FSDPTrainer.train(samples)` → pack → train_step → aggregate metrics
- `FSDPTrainer._train_step(packed_batch)` → forward, loss, backward, optimizer step
- `FSDPTrainer.get_state_dict()` → DTensor redistribution
- `FSDPTrainer.load_state_dict()` → distributed checkpoint load

**`mini_slime/train_fsdp.py`**（对齐源 `train.py` + Ray 编排）
- 主循环：rollout → train → update_weights
- TrainRayActor 用 `FSDPTrainer` 替代 `TorchTrainer`
- update_weights 通过 `dist.broadcast()` 同步到 rollout engines

---

## 4. 实施路径（四个子版本）

### V7.0 FSDP 包装 + 单样本验证

**目标**：建立 FSDP 基础设施，单样本前向+后向通过。

**实施**：
1. 新建 `toy_rl/utils/fsdp_utils.py`
   - `setup_device_mesh(world_size)` → 1D mesh
   - `apply_fsdp2(model, mesh, args)` → 按 `_no_split_modules` wrap layers
2. 新建 `toy_rl/trainer/fsdp_trainer.py`（骨架）
   - `__init__()` → load model, apply FSDP, create optimizer
   - `train(samples)` → 先硬编码单样本、不 pack
   - `_train_step()` → forward, loss, backward（复用 V6 loss 计算）
3. 测试脚本 `scripts/test_v7.0_fsdp_single.py`
   - 单进程初始化 FSDP trainer
   - 传入一条 V6 GSM8K 样本
   - 断言：loss 有值、backward 成功、optimizer step 执行

**验证**：本地单卡（或服务器单卡）离线通过。

**对齐源**：`actor.py:48-154` (init), `actor.py:551-678` (train_step)

**偏离**：暂无 packing（单样本）、暂无 Ray（单进程）、暂无权重同步。

---

### V7.1 Sequence Packing

**目标**：把多条样本 pack 成 batch，提高 GPU 利用率。

**实施**：
1. 新建 `toy_rl/trainer/data_packing.py`
   - `pack_sequences(tokens, loss_masks, rewards, ...)` → 按 `num_packs` 分区
   - `unpack_sequences(packed_batch)` → 按 `cu_seqlens` 切回 per-sample
2. 修改 `FSDPTrainer.train(samples)`
   - 调用 `pack_sequences()`，返回 `packed_batches`
   - 循环 `for packed_batch in packed_batches: _train_step(packed_batch)`
3. 修改 `FSDPTrainer._train_step(packed_batch)`
   - 输入改为 flattened tokens + `cu_seqlens` + `position_ids`
   - unpack 后再计算 per-sample loss
   - loss scaling: `loss * dp_size / global_batch_size`
4. 测试脚本 `scripts/test_v7.1_packing.py`
   - 传入 4 条样本（长度不同）
   - 断言：packed_batches 长度 = `num_packs`
   - 断言：unpacked 后 loss 维度正确

**验证**：本地单卡离线通过（monkeypatch SGLang）。

**对齐源**：`data_packing.py:10-121` (pack), `actor.py:566-579` (unpack), `actor.py:676` (loss scaling)

**偏离**：char 级假 token（非真 tokenizer），无 multimodal_train_inputs。

---

### V7.2 Gradient Accumulation

**目标**：支持 microbatch 梯度累积，模拟大 batch。

**实施**：
1. 修改 `FSDPTrainer.train(samples)`
   - 根据 `num_packs` 计算 `grad_accum` 累积点（`list(accumulate(num_microbatches))`）
   - 只在 `(mbs_id+1) in grad_accum` 时调用 `optimizer.step()` + `zero_grad()`
2. 修改 `FSDPTrainer._train_step(packed_batch, mbs_id, grad_accum)`
   - 收集 metrics 到 `reported_accum[key].append(value)`
   - 只在累积点 all-gather metrics、平均
3. 测试脚本 `scripts/test_v7.2_grad_accum.py`
   - 传入 8 条样本，`num_packs=4`, `grad_accum=[2,4]`（每 2 个 microbatch 一次 step）
   - 断言：`optimizer.step()` 调用次数 = 2
   - 断言：final metrics 是 all_gather 后的平均值

**验证**：本地单卡离线通过。

**对齐源**：`actor.py:684-722` (optimizer step + all_gather)

**偏离**：单进程模拟 dp_size（无真多卡 all_gather，用假 group）。

---

### V7.3 Ray + FSDP + 权重同步

**目标**：Ray 编排 + FSDP trainer + DTensor 权重同步到 rollout engines。

**实施**：
1. 新建 `mini_slime/train_fsdp.py`
   - 复制 `train_ray.py` 结构
   - TrainRayActor 内部用 `FSDPTrainer` 替代 `TorchTrainer`
   - `update_weights()` 改用 FSDP 的 DTensor redistribution
2. 修改 `toy_rl/trainer/fsdp_trainer.py`
   - `get_state_dict()` → 对每个 param，如果是 DTensor，调用 `.redistribute([Replicate()], async_op=True).to_local()`
   - 返回 replicated state_dict（可用于 broadcast）
3. 权重同步机制（两种方案二选一）：
   - **方案 A（disk reload，对齐 V6）**：trainer save → rollout engines load from disk
   - **方案 B（broadcast，对齐源）**：创建临时 NCCL group，`dist.broadcast()` 每个 param
   - **V7.3 先用方案 A**（复用 V6 逻辑，风险低），方案 B 留 V7.4 优化
4. 测试脚本 `scripts/test_v7.3_fsdp_ray.py`
   - 起 Ray，创建 RolloutManager + TrainRayActor（FSDP）
   - 跑 2 轮 rollout → train → update_weights
   - 断言：weight_version 递增、rollout engines reload 成功

**验证**：5090 服务器端到端（单卡或双卡）。

**对齐源**：`update_weight_utils.py:48-74` (DTensor redistribution), `train.py` (编排)

**偏离**：权重同步用 disk reload（方案 A），非源的 NCCL broadcast（方案 B）；单卡或双卡（非满配 4 卡）。

---

### V7.4 分布式权重广播（优化，可选）

**目标**：用 NCCL broadcast 替代 disk reload，降低 I/O 延迟。

**实施**：
1. 新建 `toy_rl/utils/weight_sync.py`
   - `create_weight_sync_group(trainer_ranks, rollout_ranks)` → 临时 NCCL group
   - `broadcast_weights(state_dict, group)` → 对每个 param 调用 `dist.broadcast(async_op=True)`
2. 修改 `mini_slime/train_fsdp.py`
   - `update_weights()` 改用 `broadcast_weights()`
3. 对比测试 `scripts/test_v7.4_broadcast_vs_disk.py`
   - 分别测量 disk reload 和 NCCL broadcast 的耗时
   - 断言：broadcast 耗时 < disk reload（在多卡环境下）

**验证**：5090 服务器多卡端到端。

**对齐源**：`update_weight_utils.py:195-286` (UpdateWeightFromDistributed)

**偏离**：无（完全对齐源）。

---

## 5. 验证契约（每个子版本必须通过）

### V7.0 单样本
- ✅ FSDP model 初始化成功（`isinstance(model.module, torch.nn.Module)`）
- ✅ Forward pass 返回 logits（shape 正确）
- ✅ Loss 计算成功（scalar tensor）
- ✅ Backward 不报错
- ✅ Optimizer step 执行（检查 `lr_scheduler.get_last_lr()`）

### V7.1 Packing
- ✅ `pack_sequences()` 返回 list of dicts
- ✅ 每个 packed_batch 包含 `tokens`, `loss_masks`, `cu_seqlens`, `position_ids`
- ✅ `unpack_sequences()` 返回 per-sample dicts
- ✅ Unpacked 后 loss 维度 = `sum(response_lengths)`
- ✅ Loss scaling 正确（`loss * dp_size / global_batch_size`）

### V7.2 Gradient Accumulation
- ✅ `grad_accum` 计算正确（cumulative indices）
- ✅ Optimizer step 只在累积点调用（计数验证）
- ✅ Metrics all-gather 后平均（模拟多卡用单进程 mock）

### V7.3 Ray + FSDP
- ✅ Ray actor 启动成功（不同进程）
- ✅ FSDP trainer 在 Ray actor 内初始化成功
- ✅ 闭环通过：rollout → train → update_weights（weight_version 递增）
- ✅ Rollout engines reload 权重成功（disk reload）
- ✅ Reward mean 与 V6 同数量级（机制正确性）

### V7.4 Broadcast（可选）
- ✅ NCCL group 创建成功
- ✅ Broadcast 耗时 < disk reload（多卡环境）
- ✅ Broadcast 后 rollout engines 权重一致性（hash 验证）

---

## 6. 源项目对齐检查清单

**必须对齐**：
- [x] Device mesh 用 1D (dp,) — `torch.distributed.init_device_mesh("cuda", (world_size,))`
- [x] FSDP wrapping 按 `_no_split_modules` — `model._no_split_modules` 或 Qwen2 默认 layer class
- [x] Mixed precision policy — `param_dtype=bf16, reduce_dtype=fp32`
- [x] Pack sequences 用 `cu_seqlens` — 累积序列长度数组
- [x] Unpack sequences 用 slicing — `tokens[cu_seqlens[i]:cu_seqlens[i+1]]`
- [x] Loss 计算复用 V6 GRPO — `compute_policy_loss(ppo_kl, advantages, eps_clip)`
- [x] Loss scaling — `loss * dp_size / global_batch_size`
- [x] Gradient accumulation — `grad_accum = list(accumulate(num_microbatches))`
- [x] DTensor redistribution — `param.redistribute([Replicate()], async_op=True).to_local()`

**允许偏离（登记在案）**：
- [ ] CPU offload — V7 不做，留 Phase 2（显存够用时）
- [ ] Reference model — V6 已关 `--no-ref`，V7 继承
- [ ] OPSM / KL loss — V6 已关，V7 继承
- [ ] Multimodal inputs — 无视觉，V7 skip
- [ ] 权重同步方式 — V7.3 用 disk reload（方案 A），V7.4 改 NCCL broadcast（方案 B）
- [ ] Char 级假 token — V6 继承，V7 不改（真 tokenizer 留后续）

---

## 7. 环境依赖

**新增依赖**（加入 `pyproject.toml [train]`）：
```toml
[project.optional-dependencies]
train = [
    "torch>=2.5.0",              # V6 已有
    "transformers>=4.46.0",      # V6 已有
    # FSDP2 需要 PyTorch 2.5+ 原生支持，无需额外包
]
```

**服务器环境确认**：
- PyTorch 版本 >= 2.5.0（支持 FSDP2 API）
- CUDA >= 12.1（与 PyTorch 兼容）
- NCCL 后端可用（`torch.distributed.is_nccl_available()`）

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| FSDP wrapping 报错（找不到 `_no_split_modules`） | 阻塞 V7.0 | 硬编码 Qwen2DecoderLayer 类名，或用 `transformers.AutoConfig` 获取 |
| Packing 后 position_ids 错误 | Loss 错误 | 参考源 `data_packing.py:91-95` 逐 sample 重置 position_ids |
| Ray + FSDP 初始化竞争（多卡同时 load model） | OOM / 卡死 | 源 `actor.py:101-117` 用 barrier 顺序加载 |
| DTensor redistribution 卡死 | 阻塞 V7.3 | 确认 `async_op=True` + `.wait()` 在正确位置 |
| 权重 broadcast 跨 Ray actor 不通 | 阻塞 V7.4 | V7.3 先用 disk reload，V7.4 再调 NCCL group |

---

## 9. 时间估算

| 子版本 | 预估耗时 | 风险等级 |
|--------|---------|---------|
| V7.0 FSDP 包装 + 单样本 | 4-6 小时 | 中（FSDP API 不熟） |
| V7.1 Packing | 3-4 小时 | 低（逻辑清晰） |
| V7.2 Gradient Accumulation | 2-3 小时 | 低（V6 已有类似逻辑） |
| V7.3 Ray + FSDP | 4-6 小时 | 高（Ray + FSDP 交互，权重同步） |
| V7.4 Broadcast（可选） | 3-4 小时 | 中（NCCL group 配置） |
| **总计** | **16-23 小时** | — |

---

## 10. 成功标准（V7 收官）

**功能**：
- ✅ V7.3 在 5090 服务器端到端跑通（单卡或双卡）
- ✅ FSDP 分片工作（检查 `model.parameters()` 是 DTensor）
- ✅ Packing 提高吞吐（对比 V6 单样本 tokens/s）
- ✅ 权重同步成功（weight_version 递增，rollout engines reload）
- ✅ Reward mean 与 V6 同数量级（机制正确性）

**文档**：
- ✅ `docs/decisions/v7.md` 记录：对齐源 / 偏离 / 踩坑 / 性能对比
- ✅ 更新 `CLAUDE.md` 进度表：V7 ✅ 完成

**回归**：
- ✅ V0/V2/V3/V4/V5/A1/A2/A3/V6 回归全绿（用 `--offline` 或 stub hooks）

**性能对比**（V6 vs V7）：
- 显存占用（单样本 vs packed batch）
- 训练吞吐（tokens/s）
- 权重同步耗时（disk reload vs broadcast，如果 V7.4 完成）

---

## 11. 下一版引出（V8）

**V7 暴露的痛点**：
- 单一 1D data parallelism，大模型（如 Qwen2-7B）单卡放不下
- 无 Tensor Parallelism (TP) / Pipeline Parallelism (PP)

**V8 Megatron 并行**：
- TP：切分 attention heads / FFN，跨卡并行计算
- PP：切分 layers，流水线执行
- CP (Context Parallelism)：切分 sequence，Ring Attention
- EP (Expert Parallelism)：MoE 专家并行

---

## 附录：关键代码片段速查

### A. FSDP Wrapping（源 `actor.py:933-984`）
```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def apply_fsdp2(model, mesh, args):
    param_dtype = torch.float16 if args.fp16 else torch.bfloat16
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    
    layer_cls_to_wrap = model._no_split_modules  # e.g., ["Qwen2DecoderLayer"]
    modules = [m for name, m in model.named_modules() 
               if m.__class__.__name__ in layer_cls_to_wrap]
    
    for module in modules:
        fully_shard(module, mp_policy=mp_policy, mesh=mesh)
    
    fully_shard(model, mp_policy=mp_policy, mesh=mesh)
    return model
```

### B. Pack Sequences（源 `data_packing.py:10-60`）
```python
def pack_sequences(tokens, loss_masks, rewards, num_packs):
    seq_lengths = [len(t) for t in tokens]
    partitions = get_seqlen_balanced_partitions(seq_lengths, num_packs)
    
    packed_batches = []
    for indices in partitions:
        cu_seqlens = [0]
        flat_tokens, flat_masks = [], []
        
        for i in indices:
            flat_tokens.extend(tokens[i])
            flat_masks.extend(loss_masks[i])
            cu_seqlens.append(cu_seqlens[-1] + len(tokens[i]))
        
        packed_batches.append({
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "position_ids": _make_position_ids(cu_seqlens),
            # ... other fields
        })
    
    return packed_batches
```

### C. Unpack（源 `actor.py:566-579`）
```python
def unpack_sequences(packed_batch):
    cu_seqlens = packed_batch["cu_seqlens"]
    response_lengths = packed_batch["response_lengths"]
    
    unpacked = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        unpacked.append({
            "cur_log_probs": log_probs[start:end-1],  # -1 for shifted logits
            "loss_mask": packed_batch["loss_masks"][start:end-1],
            "response_length": response_lengths[i],
            # ... other fields
        })
    
    return unpacked
```

### D. Loss Scaling + Backward（源 `actor.py:676-678`）
```python
loss = pg_loss - entropy_coef * entropy_loss
loss = loss * dp_size / global_batch_size  # Scale for gradient accumulation
loss.backward()
```

### E. Gradient Accumulation（源 `actor.py:684-722`）
```python
if (mbs_id + 1) in grad_accum:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    
    # All-gather metrics
    reduced_aggregated = [None] * dp_size
    dist.all_gather_object(reduced_aggregated, aggregated, group=dp_group)
    
    for k in reported_accum.keys():
        aggregated[k] = sum([r[k] for r in reduced_aggregated]) / global_batch_size
```

### F. DTensor Redistribution（源 `update_weight_utils.py:48-74`）
```python
def get_state_dict(self):
    state_dict = {}
    for name, param in self.model.state_dict().items():
        if isinstance(param, DTensor):
            param = param.redistribute([Replicate()], async_op=True).to_local()
        state_dict[name] = param
    return state_dict
```

---

**下一步**：创建 `v7` 分支，开始实施 V7.0。
