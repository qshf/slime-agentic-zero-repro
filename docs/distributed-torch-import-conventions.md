# PyTorch 与分布式库函数用法速查

本文只解释第三方库的 API，不描述本项目自己的类或实现。示例默认 Python 3.10+、CUDA 多卡训练；函数签名以 PyTorch 2.12/2.13 官方文档为准。

## 1. 常用导入

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
```

| 导入 | 主要用途 |
| --- | --- |
| `torch` | Tensor 创建、设备迁移、自动求导、CUDA 状态。 |
| `torch.nn as nn` | `Module`、`Linear`、`Embedding` 等可训练层。 |
| `torch.nn.functional as F` | 无状态函数，如 `F.cross_entropy`、`F.softmax`。 |
| `torch.optim.AdamW` | 优化器；接收 `model.parameters()` 并在 `step()` 时更新参数。 |
| `torch.distributed as dist` | 进程组、rank 查询和跨进程集合通信。 |

不要写 `from torch import *`。`nn` 放层和模块，`F` 放无状态计算：例如 `nn.CrossEntropyLoss()` 可保存配置并复用，`F.cross_entropy()` 适合一次性计算。

## 2. `torch`：张量、设备与训练一步

### `torch.tensor`、`torch.zeros`、`torch.randn`

```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cuda")
mask = torch.zeros((2, 4), dtype=torch.bool, device=x.device)
noise = torch.randn((2, 4), device=x.device)
```

- `torch.tensor(data, ...)`：从 Python 数据复制创建张量；适合输入小型常量或外部数据。
- `torch.zeros(shape, ...)`：创建全零张量；`dtype=torch.bool` 常用于 mask。
- `torch.randn(shape, ...)`：从标准正态分布采样；常用于初始化或噪声。
- `device` 决定张量所在设备。参与同一运算的张量必须在相同设备上。

已有 Tensor 要迁移或转换精度时使用 `.to()`，它可同时指定设备和 dtype：

```python
x = x.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
```

`non_blocking=True` 只有在源内存为 pinned memory 等条件满足时才可能异步，不能把它当作必然提速开关。

### `nn.Module`、`model.train()` 与 `model.eval()`

```python
class Classifier(nn.Module):
    def __init__(self, width: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


model = Classifier(128, 10).to("cuda")
model.train()
logits = model(torch.randn(8, 128, device="cuda"))
```

- `nn.Module` 是模型和子模块的基类；赋值给 `self` 的 `nn.Module`/`nn.Parameter` 会被自动注册。
- `model.train()` 切到训练模式，影响 Dropout、BatchNorm 等层；它**不会**自动开启梯度。
- `model.eval()` 切到推理模式；它**不会**自动关闭梯度。
- 调用模型时使用 `model(x)`，不要直接调用 `model.forward(x)`。前者会经过 PyTorch 的 hooks，也是 FSDP2 等包装器触发通信的入口。

纯推理同时需要 `eval()` 和关闭梯度：

```python
model.eval()
with torch.inference_mode():
    logits = model(x)
```

`torch.inference_mode()` 比 `torch.no_grad()` 更严格，适用于不需要 autograd 元数据的纯推理；若后续仍需对结果参与梯度计算，使用 `torch.no_grad()` 或不要包裹该段。

### 反向传播与优化器

```python
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = F.cross_entropy(logits, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

| API | 含义与要点 |
| --- | --- |
| `optimizer.zero_grad(set_to_none=True)` | 清理上一步的 `.grad`。默认反向传播会累积梯度；`set_to_none=True` 通常更省内存，也能发现未参与反传的参数。 |
| `loss.backward()` | 从标量 loss 反向计算，将梯度累加到每个叶子参数的 `.grad`。 |
| `clip_grad_norm_` | 原地按全局梯度范数裁剪，返回裁剪前的总范数。应放在 `backward()` 后、`step()` 前。 |
| `optimizer.step()` | 按优化器规则原地更新参数。 |

梯度累积时，不在每个 microbatch 后调用 `step()` 或 `zero_grad()`，并把 loss 除以累积次数：

```python
optimizer.zero_grad(set_to_none=True)
for batch in microbatches:
    (compute_loss(model, batch) / len(microbatches)).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

## 3. `torch.distributed`：进程组与 rank

### 必备概念

- **process**：一个 Python 进程，通常独占一张 GPU。
- **rank**：进程在整个训练任务中的全局编号，范围是 `0 .. WORLD_SIZE - 1`。
- **world size**：进程总数。
- **process group**：一组可互相通信的 rank。默认进程组通常包含所有 rank。
- **local rank**：进程在本机的编号，用于选择本机 GPU；它不等于全局 `rank`，多机时尤其不能混用。

GPU 训练常用启动方式：

```bash
torchrun --standalone --nproc-per-node=4 train.py
```

`torchrun` 会为每个子进程设置 `LOCAL_RANK`、`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量；训练脚本不应手工覆盖它们。

### `dist.init_process_group`

```python
import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank
```

`dist.init_process_group(backend=..., init_method=..., rank=..., world_size=...)` 初始化默认进程组。使用 `torchrun` 时，最简写法只传 `backend="nccl"`，其余参数从环境变量的 `env://` rendezvous 读取。

- `backend="nccl"`：GPU/CUDA 训练的标准选择。
- `backend="gloo"`：CPU 通信与小型调试常用；不作为 GPU 高性能训练的默认后端。
- 若函数可能在已经初始化的环境中被调用，先用 `dist.is_initialized()` 判断，避免重复初始化。

```python
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
```

### 查询与销毁

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

dist.barrier()                 # 所有 rank 到达这里后才继续
dist.destroy_process_group()   # 脚本收尾时释放默认进程组
```

`dist.barrier()` 是同步屏障，不传递业务数据。它不能修复不同 rank 走了不同训练分支的问题；所有 rank 必须以相同顺序调用集合通信，否则任务会挂起。

## 4. 常用集合通信 API

以下 API 都要求通信组内的每个 rank 参加同一次调用。

### `dist.broadcast(tensor, src=0)`

```python
value = torch.tensor([42 if dist.get_rank() == 0 else 0], device="cuda")
dist.broadcast(value, src=0)
# 调用后每个 rank 的 value 都是 [42]
```

将 `src` rank 的 Tensor 原地复制到其他 rank。所有 rank 上的 tensor 必须具有兼容的 shape、dtype 和 device。适合发布初始化权重、随机种子或少量控制数据。

### `dist.all_reduce(tensor, op=dist.ReduceOp.SUM)`

```python
metric = torch.tensor([local_loss_sum, local_count], device="cuda")
dist.all_reduce(metric, op=dist.ReduceOp.SUM)
global_mean = metric[0] / metric[1].clamp_min(1)
```

对每个 rank 的同形状 Tensor 做规约，并把结果写回**每个** rank。常用 `SUM`、`AVG`、`MAX`、`MIN`。它适合同步指标；DDP/FSDP 的梯度同步由框架 hooks 自动完成，正常训练循环不应再手写同一份梯度的 `all_reduce`。

### `dist.all_gather_into_tensor(output, input)`

```python
local = torch.arange(2, device="cuda") + 10 * dist.get_rank()
gathered = torch.empty(2 * dist.get_world_size(), dtype=local.dtype, device="cuda")
dist.all_gather_into_tensor(gathered, local)
```

收集每个 rank 的 `input` 到每个 rank 的 `output`。`output` 的大小需能容纳全部输入，通常首维是 `world_size` 倍。用于需要每个 rank 都看到全局数据的场景；大张量会复制到每张卡，需注意显存。

### `dist.reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM)`

先规约，再把结果分片给不同 rank。它是全分片训练减少显存和通信开销的基础原语，通常由 FSDP 自动调用；业务训练代码一般不直接使用。

## 5. DDP：复制模型、同步梯度

```python
from torch.nn.parallel import DistributedDataParallel as DDP


rank, world_size, local_rank = setup_distributed()
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optimizer = AdamW(model.parameters(), lr=3e-4)
```

`DDP(module, ...)` 在每个 rank 复制完整模型，并在 `loss.backward()` 时自动同步梯度。它不负责切分输入数据，数据加载应配合 `DistributedSampler`：

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
loader = DataLoader(dataset, batch_size=per_rank_batch_size, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # shuffle=True 时每轮必须调用
    for batch in loader:
        ...
```

关键限制：模型应在包进 DDP 后保持参数集合不变；训练中新增/替换参数会使 DDP 已注册的梯度规约 hooks 与模型不一致。`find_unused_parameters=True` 只在确有条件分支导致部分参数不参与反向传播时使用，常规静态网络保持默认 `False`。

## 6. `DeviceMesh` 与 `DTensor`

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=("dp",))
```

`init_device_mesh(device_type, mesh_shape, mesh_dim_names=...)` 建立设备拓扑并管理对应的 process groups。`mesh_shape=(world_size,)` 是一维数据并行 mesh；多维 mesh 才用于 DP/TP/PP 等组合并行。

所有 rank 必须以相同的 `mesh_shape` 和 `mesh_dim_names` 调用它，否则可能挂起。该函数可在未初始化默认进程组时初始化它，但在训练脚本中仍建议显式调用 `init_process_group()`，让 rendezvous、后端和 rank 来源清晰可控。

```python
from torch.distributed.tensor import DTensor

if isinstance(parameter, DTensor):
    local_shard = parameter.to_local()
    full_tensor = parameter.full_tensor()  # 触发跨 rank 聚合
```

`DTensor` 表示逻辑上的全局 Tensor 与实际的本地 shard。`to_local()` 取得本 rank 的局部分片；`full_tensor()` 聚合完整 Tensor，适合验证与偶发保存，不应放到每步训练热路径。

## 7. FSDP2：`fully_shard`

FSDP2 的常用导入：

```python
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
```

最小 Transformer 风格包装示例：

```python
mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp_policy)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)

optimizer = AdamW(model.parameters(), lr=3e-4)
loss = compute_loss(model, batch)  # 必须调用 model(...)，而不是 model.forward(...)
loss.backward()
optimizer.step()
```

`fully_shard(module, *, mesh=None, reshard_after_forward=None, mp_policy=..., offload_policy=..., ...)` 原地为模块启用全分片数据并行：参数、梯度和优化器状态按数据并行 rank 分片，以通信换显存。

固定要点：

- **自底向上调用**：先 `fully_shard` 每个 Transformer block，再处理根模型。根模块不会重复管理已分片子模块的参数。
- **后建优化器**：`fully_shard` 后模型参数会成为 `DTensor`；优化器必须拿到包装后的 `model.parameters()`。
- **用 `model(...)` 前向**：FSDP2 的 pre/post-forward hooks 负责 all-gather 和 reshard，直接调用 `.forward()` 会绕开该机制。
- `MixedPrecisionPolicy` 设定参数、规约和输出精度。常见训练配置是参数 `bfloat16`、梯度规约 `float32`；具体精度取决于 GPU 与数值稳定性要求。
- FSDP2 是较新的 `fully_shard` API；不要与旧式 `FullyShardedDataParallel`（FSDP1）混用在同一模型上。

FSDP 训练或保存中，所有 rank 都必须进入涉及状态字典、前向、反向的集合操作。只允许 rank 0 执行的是上传 checkpoint、写单份日志等**外部副作用**，不是 FSDP 本身的通信步骤。

## 8. Ray：远程任务与 actor

Ray 不替代 `torch.distributed`。Ray 负责启动任务/actor、分配 CPU/GPU 资源；多卡梯度通信仍由 DDP 或 FSDP 的 `torch.distributed` 负责。

```python
import ray

ray.init()


@ray.remote(num_cpus=1, num_gpus=1)
class Worker:
    def square(self, x: int) -> int:
        return x * x


worker = Worker.remote()
ref = worker.square.remote(12)
result = ray.get(ref)  # 144
```

| API | 用法 |
| --- | --- |
| `ray.init()` | 初始化本地 Ray runtime，或连接已有集群。一般由 driver 调用一次。 |
| `ray.remote(fn_or_class)` | 将函数转为远程 task，或将类转为有状态 actor。可传 `num_cpus`、`num_gpus` 等资源要求。 |
| `remote_fn.remote(...)` | 提交 task，立即返回 `ObjectRef`。 |
| `Actor.remote(...)` | 创建 actor，返回 actor handle。actor 的普通方法通过 `actor.method.remote(...)` 提交。 |
| `ray.get(ref)` | 阻塞直到结果可用，并将值或远程异常带回 driver。传列表时返回结果列表。 |
| `ray.wait(refs, num_returns=1)` | 等待部分任务完成，返回 `(ready, remaining)`；适合流式消费，避免一次性阻塞所有任务。 |
| `ray.shutdown()` | 本地脚本或测试结束时关闭本进程创建的 Ray runtime。 |

`ray.get()` 是同步点。要并发提交多个任务，应先收集引用、后一次性获取：

```python
refs = [worker.square.remote(i) for i in range(8)]
results = ray.get(refs)
```

每个 actor 默认顺序执行普通同步方法；想让多个任务真正并发，需要多个 actor、async actor，或明确配置并发组。对于多 rank FSDP 初始化，先提交全部 actor 的初始化 RPC，再 `ray.get(refs)`，让所有进程都能同时进入 `init_process_group()`。

## 9. 最小选型表

| 需求 | 优先 API | 模型与内存形态 |
| --- | --- | --- |
| 单卡训练 | 普通 `nn.Module` + `AdamW` | 一份完整模型。 |
| 多卡、模型能放下一张卡 | `DDP` + `DistributedSampler` | 每张卡一份完整模型，自动同步梯度。 |
| 多卡、模型或优化器状态放不下一张卡 | FSDP2 `fully_shard` | 参数、梯度、优化器状态分片。 |
| 管理 rollout、训练等不同进程及资源 | Ray actor/task | 调度层；训练内部仍用 DDP/FSDP。 |

## 官方参考

- [PyTorch `torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch `torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray `remote`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) 与 [Ray `get`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html)
