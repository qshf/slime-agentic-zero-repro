# V9 权重同步优化 - 交接文档

> 文档创建日期：2026-08-20  
> 最后更新：2026-08-20  
> 状态：已在 5090 完成端到端验证

## 背景

V9.2 实验中发现权重同步（disk reload）是主要性能瓶颈：
- **当前方案**：训练后 `save_pretrained` 落盘 → SGLang `/update_weights_from_disk` 重载
- **实测耗时**：~25 秒
- **瓶颈占比**：同步路径（A/C）总耗时 85 秒中的 29%，是单点最大开销

## 优化目标

实现**免落盘权重同步**，预期耗时 2-5 秒，加速比 **5-12×**。

## 技术方案

### 方案选择

源项目提供两种免落盘方案：

1. **UpdateWeightFromDistributed (NCCL 广播)**
   - 预期耗时：1-3 秒（最快）
   - 要求：SGLang engine 作为 Ray actor（支持 `init_weights_update_group`）
   - 状态：❌ **不可行** - 当前 SGLang 运行在独立进程，不在 Ray 管理下

2. **UpdateWeightFromTensor (HTTP POST tensor)** ✅ **已实施**
   - 预期耗时：2-5 秒
   - 要求：SGLang 支持 `/update_weights_from_tensor` 接口
   - 状态：✅ **代码与端到端验证均已完成**

### 实施方案

**UpdateWeightFromTensor** 工作流：

1. 训练器获取 HF state_dict（Megatron 自动转换）
2. 按 dtype 分组打包成 `FlattenedTensorBucket`（序列化 CPU 传输）
3. HTTP POST 到 SGLang 的 `/update_weights_from_tensor` 端点
4. SGLang 反序列化并热加载权重（免落盘）

## 实施进展

### ✅ 已完成的工作

#### 1. 核心实现

- **[toy_rl/trainer/update_weight_from_distributed.py](../toy_rl/trainer/update_weight_from_distributed.py)**
  - `UpdateWeightFromTensor` 类完整实现
  - 支持 Megatron/FSDP/TorchActor 三种训练器
  - 自动处理 HF state_dict 转换
  - 按 dtype 分组打包和序列化
  - HTTP POST 到 SGLang 的 `/update_weights_from_tensor` 端点

#### 2. 框架集成

- **[mini_slime/weight_sync.py](../mini_slime/weight_sync.py)**
  - `WeightUpdater` 新增 `use_tensor_http_sync` 参数
  - 优先走 HTTP tensor 路径（rank0 发送）
  - 保留 disk reload 作为后备方案

- **[mini_slime/args.py](../mini_slime/args.py)**
  - 新增配置开关：`use_tensor_weight_sync: bool = False`
  - 默认关闭（零回归），opt-in 启用

- **[mini_slime/trainer.py](../mini_slime/trainer.py)**
  - 传入 `sglang_url` 给 WeightUpdater
  - 根据 `use_tensor_weight_sync` 选择同步路径

#### 3. 代码提交

所有代码已推送到 **v9 分支**（commit `5f4861d`）：

```bash
git log --oneline -5
# 5f4861d fix(v9): rewrite with pure English to avoid all Chinese punctuation
# 95d820c fix(v9): replace all Chinese punctuation with ASCII
# 5d761a6 fix(v9): replace Chinese full-width parentheses with ASCII
# 890a3f0 fix(v9): add local_files_only=True for AutoConfig/AutoTokenizer
# 575002a feat(v9): implement UpdateWeightFromTensor (HTTP POST免落盘权重同步)
```

### ✅ 验证成功的部分

测试脚本：[scripts/bench/test_tensor_sync.py](../scripts/bench/test_tensor_sync.py)

验证结果（5090 服务器，v9-dev 容器）：

1. ✅ **MegatronTrainer 初始化** - 成功加载 Qwen3-0.6B 并初始化 TP=1 配置
2. ✅ **UpdateWeightFromTensor 配置** - 成功创建权重更新器
3. ✅ **权重提取和转换** - 成功从 Megatron 格式转换到 HF state_dict
4. ✅ **序列化和打包** - 成功按 dtype 分组并打包成 FlattenedTensorBucket
5. ✅ **HTTP 连接** - 成功连接到 SGLang 服务（端口 30000）并发送请求
6. ✅ **SGLang 接口存在** - 确认 SGLang 0.5.15.post1 包含 `/update_weights_from_tensor` 接口

### ✅ 2026-08-20 5090 端到端验证

测试拓扑：SGLang 在 GPU 0，单卡 MegatronTrainer 在 `v9-dev` 容器的 GPU 1；`TP=1`，不执行 rollout 或训练，只验证一次完整权重同步。

```text
Weight sync completed
Time: 5.33s
New version: 1
POST /update_weights_from_tensor -> 200 OK
POST /v1/chat/completions -> 200 OK
response metadata.weight_version = "1"
```

`5.33s` 低于原 disk reload 的约 25 秒，约为 `4.7x` 加速；略高于原先 2--5 秒的估计，首次 Megatron 到 HF state_dict 转换也包含在该测量内。服务在更新后仍可正常生成，说明端到端链路可用。

### 已解决：历史 500 错误

**SGLang 服务端返回 500 Internal Server Error**

测试日志：
```
=== Executing weight sync ===
[UpdateWeightFromTensor] Send failed (dtype=torch.bfloat16): 
  500 Server Error: Internal Server Error for url: 
  http://localhost:30000/update_weights_from_tensor
```

**问题分析**：

已由 SGLang 服务日志确认：这个端点虽然用 HTTP 接收控制请求，tensor 数据实际仍由 Python `multiprocessing.resource_sharer` 通过 Unix socket 和文件描述符传递。训练容器 `v9-dev` 与 SGLang 容器原先有两个隔离问题：

1. SGLang 容器未挂载主机 `/tmp`，无法找到发送方 resource-sharer socket，报 `FileNotFoundError`。
2. 两个容器各自生成不同的 multiprocessing `authkey`；共享 `/tmp` 后连接虽能建立，但报 `AuthenticationError: digest received was wrong`。

修复：`ff72cfe` 为 SGLang 容器挂载 `/tmp`；`c2625cb` 在启动时生成权限为 `0600` 的 `/tmp/sglang_tensor_sync_authkey`，并让 SGLang scheduler 与训练侧同步前使用该 key。不需要额外的 SGLang post-training 启动参数，也不需要改变现有 `flattened_bucket` payload。

1. **接口存在性**：✅ 已确认 SGLang 0.5.15.post1 包含该接口
   - 路径：`sglang/multimodal_gen/runtime/entrypoints/post_training/weights_api.py`
   - 相关实现：`gpu_worker_post_training_mixin.py`、`weights_updater.py`、`scheduler_post_training_mixin.py`

2. **旧假设已排除**：无需 `--enable-post-training` 等额外启动参数；`flattened_bucket` payload 与本服务版本兼容。

3. **服务日志可用**：`docker logs sglang-qwen3` 提供了上述服务端堆栈和成功请求记录。

## 使用方法

### 启用 HTTP tensor 同步

在实验脚本中设置：

```python
from mini_slime.args import Args

args = Args(
    train_backend="megatron",
    use_tensor_weight_sync=True,  # ← 启用 HTTP tensor 同步
    sglang_generate_url="http://localhost:30000/generate",
    train_model_path="/models/Qwen3-0.6B",  # 容器内路径
    # ... 其他配置
)
```

### 在 V9.2+ 实验中使用

修改 [scripts/bench/v9_end_to_end.py](../scripts/bench/v9_end_to_end.py)：

```python
args = Args(
    train_backend="megatron",
    use_tensor_weight_sync=True,  # ← 添加这一行
    tensor_model_parallel_size=2,
    # ... 其他配置保持不变
)
```

### 验证 SGLang 支持

```bash
# 检查 SGLang 版本
docker exec v9-dev python3 -c "import sglang; print(sglang.__version__)"
# 输出：0.5.15.post1

# 验证接口存在
docker exec v9-dev python3 -c "
import os
weights_api = '/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/post_training/weights_api.py'
print('✓' if os.path.exists(weights_api) else '✗')
"
# 输出：✓
```

## 预期收益

| 阶段 | disk reload | HTTP tensor | 加速比 |
|------|------------|-------------|--------|
| save_pretrained | 15-20 秒 | **0 秒**（免落盘） | ∞ |
| 传输 + 加载 | 5-10 秒 | **2-5 秒**（HTTP + 反序列化） | 2-5× |
| **总耗时** | **25 秒** | **2-5 秒** | **5-12×** |

**V9.2 实验预期改善**：

- 同步路径（A/C）：总时间从 85s 降到约 **62-65s**（提升 23-27%）
- 异步路径（B/D）：权重同步不再是瓶颈，overlap 收益显现

## 待办事项

### 高优先级（前两项已完成）

1. ✅ **重启 SGLang 服务并验证**
   - 确保 SGLang 服务运行在端口 30000
   - 检查是否需要特定启动参数启用 post-training 功能
   - 重新运行 `test_tensor_sync.py` 验证完整链路

2. ✅ **分析 500 错误根因**
   - 查看 SGLang 服务日志获取详细错误堆栈
   - 确认 payload 格式是否与 SGLang 0.5.15.post1 接口匹配
   - 必要时调整 payload 格式或参数

3. **量化 disk reload 耗时构成**
   - 在 [mini_slime/weight_sync.py:87](../mini_slime/weight_sync.py#L87) 添加计时
   ```python
   import time
   t0 = time.time()
   self.torch_actor.save_pretrained(self.save_path)
   t1 = time.time()
   requests.post(url, json={"model_path": self.save_path}, timeout=120)
   t2 = time.time()
   print(f"save: {t1-t0:.2f}s, reload: {t2-t1:.2f}s")
   ```
   - 如果 `reload` 占大头，HTTP tensor 方案收益最大
   - 如果 `save` 占大头，可先改用 `/dev/shm` tmpfs 作为快速缓解

### 中优先级

4. **完整端到端测试**
   - 一旦 SGLang 问题解决，在 V9.2 实验中启用 `use_tensor_weight_sync=True`
   - 运行完整的 4 个 cell（A/B/C/D）benchmark
   - 验证实际耗时是否符合预期（2-5 秒）

5. **性能对比测试**
   - 对比 disk reload vs HTTP tensor 的实际耗时
   - 测量端到端 latency 和 throughput 改善
   - 记录到 [docs/decisions/v9-results.md](v9-results.md)

### 低优先级

6. **代码优化**
   - 添加重试机制（网络超时、临时错误）
   - 添加进度反馈（大模型权重传输可能需要几秒）
   - 考虑分批传输大模型权重（避免单次 HTTP 请求过大）

7. **文档完善**
   - 更新 [docs/decisions/v9-方案-最终版.md](v9-方案-最终版.md) 记录权重同步优化
   - 添加性能测试结果和对比分析
   - 补充故障排查指南（常见错误和解决方案）

## 技术细节

### 关键代码路径

#### 训练侧发送权重

[toy_rl/trainer/update_weight_from_distributed.py:48-94](../toy_rl/trainer/update_weight_from_distributed.py#L48-L94)

```python
@torch.no_grad()
def update_weights(self):
    """Disk-free weight sync via HTTP POST."""
    self.weight_version += 1

    # 1. Get HF state_dict
    if hasattr(self.trainer, "to_hf_state_dict"):
        hf_state_dict = self.trainer.to_hf_state_dict()
    elif hasattr(self.trainer, "model"):
        hf_state_dict = {
            k: v.cpu() if hasattr(v, "cpu") else v
            for k, v in self.trainer.model.state_dict().items()
        }
    else:
        raise ValueError("trainer must have to_hf_state_dict or model attribute")

    # 2. Group by dtype
    named_tensors_by_dtype = {}
    for name, tensor in hf_state_dict.items():
        if hasattr(tensor, "cuda") and tensor.is_cuda:
            tensor = tensor.cpu()

        dtype = tensor.dtype
        if dtype not in named_tensors_by_dtype:
            named_tensors_by_dtype[dtype] = []
        named_tensors_by_dtype[dtype].append((name, tensor))

    # 3. Pack and send each dtype
    for dtype, named_tensors in named_tensors_by_dtype.items():
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_data = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }

        serialized = MultiprocessingSerializer.serialize(flattened_data, output_str=True)

        payload = {
            "serialized_named_tensors": [serialized],
            "load_format": "flattened_bucket",
            "weight_version": str(self.weight_version),
        }

        try:
            response = requests.post(self.sglang_url, json=payload, timeout=300)
            response.raise_for_status()
        except Exception as e:
            print(f"[UpdateWeightFromTensor] Send failed (dtype={dtype}): {e}")
            raise
```

#### 框架侧调用

[mini_slime/weight_sync.py:60-69](../mini_slime/weight_sync.py#L60-L69)

```python
def update_weights(self, state_dict=None) -> int:
    """把训练后权重推回推理引擎。"""
    if self._tensor_updater:
        # V9+: HTTP tensor 同步路径（只 rank0 发送）
        if self.rank == 0:
            self._tensor_updater.update_weights()
            self.version = self._tensor_updater.weight_version
        else:
            # 非 rank0：只递增本地 version（与 rank0 保持同步）
            self.version += 1
    elif self.torch_actor is not None and self.save_path:
        # 旧路径：disk reload
        # ...
```

### SGLang 接口定义

根据验证结果，SGLang 0.5.15.post1 的接口定义在：

- `sglang/multimodal_gen/runtime/entrypoints/post_training/weights_api.py`

预期的 HTTP 端点：

```
POST /update_weights_from_tensor

Request body:
{
  "serialized_named_tensors": ["<serialized_data>"],  // list of serialized FlattenedTensorBucket
  "load_format": "flattened_bucket",                  // 格式标识
  "weight_version": "1"                                // 版本号（字符串）
}

Response:
200 OK - 权重加载成功
500 Internal Server Error - 服务端处理失败
```

### 依赖项

```python
# 训练侧
from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
from sglang.srt.utils import MultiprocessingSerializer
import requests
import torch

# SGLang 服务端
# 需要 SGLang 0.5.15+ 支持 post-training 权重更新功能
```

## 故障排查

### 问题 1：Connection refused (端口 30000)

**现象**：
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**原因**：SGLang 服务未运行或监听其他端口

**解决**：
```bash
# 检查 SGLang 服务状态
netstat -tlnp | grep 30000

# 检查 SGLang 进程
ps aux | grep sglang | grep -v grep

# 重启 SGLang 服务（根据实际部署方式）
systemctl restart sglang
# 或
docker restart <sglang-container>
```

### 问题 2：500 Internal Server Error

**现象**：
```
500 Server Error: Internal Server Error for url: http://localhost:30000/update_weights_from_tensor
```

**原因**（待验证）：
- SGLang 启动时未启用 post-training 功能
- Payload 格式不匹配
- SGLang 服务端处理请求时出错

**解决**：
1. 查看 SGLang 服务日志获取详细错误信息
2. 检查 SGLang 启动参数，确认是否需要 `--enable-post-training` 等参数
3. 验证 payload 格式与 SGLang 0.5.15.post1 接口匹配

### 问题 3：ImportError: No module named 'sglang'

**现象**：
```
ImportError: Need sglang installed for UpdateWeightFromTensor
```

**原因**：训练环境未安装 SGLang

**解决**：
```bash
pip install sglang>=0.5.15
```

## 参考资料

### 源项目实现

- [slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py](../../slime-agentic/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py)
- [slime/backends/sglang_utils/sglang_engine.py](../../slime-agentic/slime/backends/sglang_utils/sglang_engine.py)

### 相关文档

- [V9 方案（最终版）](v9-方案-最终版.md) - 完整的 V9 离线+在线对比方案
- [V9 实验结果](v9-results.md) - V9.0-V9.2 实验数据和分析
- [V9 方案对比](v9-方案对比.md) - 不同权重同步方案的性能对比（计划中）

### Git 提交历史

```bash
# 查看相关提交
git log --oneline --grep="UpdateWeightFromTensor" v9

# 查看代码变更
git show 575002a  # feat(v9): implement UpdateWeightFromTensor
git show 890a3f0  # fix(v9): add local_files_only=True
git show 5f4861d  # fix(v9): rewrite with pure English
```

## 总结

### 已完成

✅ **UpdateWeightFromTensor 完整实现** - 代码已推送到 v9 分支  
✅ **框架集成和配置** - 支持 opt-in 启用  
✅ **部分功能验证** - 训练侧链路已通过测试  
✅ **SGLang 接口确认** - 确认 0.5.15.post1 包含该接口  

### 待验证

⚠️ **端到端测试** - 等待 SGLang 服务端问题解决  
⚠️ **性能测试** - 等待完整链路打通后测量实际耗时  
⚠️ **V9.2 集成测试** - 在完整实验中验证效果  

### 下一步

1. **解决 SGLang 500 错误** - 查看日志、调整启动参数或 payload 格式
2. **完成端到端验证** - 确认 2-5 秒的预期耗时
3. **集成到 V9.2 实验** - 测量实际性能改善
4. **记录最终结果** - 更新 v9-results.md 和方案文档

---

**联系方式**：如有问题，请查看本文档或相关代码注释。

**最后更新**：2026-08-20
