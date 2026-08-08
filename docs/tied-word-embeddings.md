# Tied Word Embeddings：判断与 FSDP 初始化说明

本文记录本仓库在 FSDP2 训练中如何识别、验证和处理共享词嵌入（tied word embeddings）。
对应实现：

- `toy_rl/trainer/fsdp_trainer.py`：读取配置、建模、FSDP 包装和权重加载。
- `toy_rl/utils/fsdp_utils.py`：初始化上下文、FSDP 包装和 rank 0 广播。

## 1. 什么是共享词嵌入

因果语言模型通常有两处与词表有关的权重：

```text
token id
  -> input embedding [vocab_size, hidden_size]
  -> decoder layers
  -> lm head weight [vocab_size, hidden_size]
  -> token logits
```

非共享模型中，输入 embedding 和 `lm_head` 是两份独立参数。共享模型中，二者复用同一个
`torch.nn.Parameter`：

```python
model.get_input_embeddings().weight is model.get_output_embeddings().weight
```

共享的收益是减少一份大词表矩阵的参数和内存；约束是涉及参数加载、替换或分片时，必须始终保持
这两个模块引用同一份权重。

## 2. 如何判断模型是否共享

### 2.1 加载前：检查配置

本仓库的初始化分支以 `tie_word_embeddings` 为准。只读取配置即可检查，不需要先加载完整模型：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
is_tied = bool(getattr(config, "tie_word_embeddings", False))
print(is_tied)
```

也可以查看模型目录的 `config.json`：

```json
{
  "tie_word_embeddings": true
}
```

配置值表示模型作者声明的架构契约；它也是 `FSDPTrainer` 选择初始化策略的输入。当前仓库中
Qwen3-0.6B 的该值为 `True`。

### 2.2 加载后、FSDP 包装前：验证实际参数

配置不足以排除自定义模型实现与配置不一致的情况。模型已在 CPU 或 CUDA 上加载后，可做如下检查：

```python
input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings()

assert input_embeddings is not None
assert output_embeddings is not None

input_weight = input_embeddings.weight
output_weight = output_embeddings.weight

same_parameter = input_weight is output_weight
print(f"same Parameter object: {same_parameter}")
```

`is` 为 `True` 是最强的结论：两个模块确实共享同一个 `Parameter` 对象。对某些特殊实现，两个
不同的 `Parameter` 也可能复用同一块底层存储；可在真实 CPU/CUDA tensor 上额外检查：

```python
same_storage = (
    input_weight.untyped_storage().data_ptr()
    == output_weight.untyped_storage().data_ptr()
)
print(f"same storage: {same_storage}")
```

不要对 meta tensor 使用 `data_ptr()` 作为判断依据：meta tensor 没有真实数据存储。也不要在
`apply_fsdp2()` 之后用上述对象身份检查替代包装前检查，因为 FSDP 会管理、替换或展平参数表示。

若模型没有实现 `get_output_embeddings()`，需要按该模型的具体模块名检查，例如常见的：

```python
model.model.embed_tokens.weight is model.lm_head.weight
```

## 3. 为什么初始化要按 tie 分支

每个分布式 rank 都需要一个模型对象，但多卡进程不应都在启动时占有一份完整的真实权重。对于非共享
模型，本仓库使用下面的初始化策略：

| 条件 | rank 0 | 其他 rank | 目的 |
| --- | --- | --- | --- |
| `tie_word_embeddings=False` | CPU 上加载真实权重 | `accelerate.init_empty_weights()` 在 meta 上只建模型结构 | 避免每个 rank 都在 CPU 上暂存完整模型 |
| `tie_word_embeddings=True` | CPU 上加载真实权重 | CPU 上加载真实权重 | 规避 meta + tied weights 的已知挂起路径 |

`meta` tensor 只记录形状、dtype 等元信息，不为权重数值分配存储。非共享模型中，rank 0 的完整
state dict 会在 FSDP 包装后通过 `set_model_state_dict(..., broadcast_from_rank0=True)` 广播并分片给各
rank；其他 rank 因此可以先只持有 meta 模型骨架。

共享模型的输入 embedding 与输出 `lm_head` 必须保留同一参数引用。上游 `slime-agentic` 的 FSDP
加载路径已观察到：让这类模型在 meta 上初始化，再进入后续 broadcast/加载流程会挂起。因此
`get_init_weight_context()` 对 `tie_word_embeddings=True` 统一返回 CPU 上下文，让所有 rank 先构造
正常的共享参数关系。

这是一条针对本项目所对齐的 `Transformers + Accelerate + FSDP` 加载路径的兼容性策略，不是“任何
共享参数都绝不能在 meta 设备创建”的通用定理。代价是 tied 模型在初始化时每个 rank 都会暂存完整
CPU 权重，启动内存峰值更高；收益是避免 hang，且能可靠保持权重共享关系。

对应代码等价于：

```python
tie = self.config.tie_word_embeddings
init_context = get_init_weight_context(tie, self.rank)
with init_context():
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
```

其中 `cpu_init()` 返回 `torch.device("cpu")`，而 `init_empty_weights` 返回 Accelerate 的上下文管理器。
因此两者都可以通过统一的 `with init_context():` 调用。

## 4. 为什么 FSDP 不能单独分片 tied embedding

`apply_fsdp2()` 总会分别 `fully_shard()` decoder layer；但只在非共享模型中把
`torch.nn.Embedding` 也作为单独包装单元：

```python
isinstance(module, torch.nn.Embedding) and not tie
```

若输入 embedding 与输出层共享参数，单独包装 embedding 可能让同一参数跨越不兼容的 FSDP 管理边界，
从而破坏权重共享或造成分片加载错误。因此 tied 模型的 embedding 不单独包装，而由最终对顶层模型的
`fully_shard(model, ...)` 一并管理。

## 5. 排查顺序

出现加载卡住、权重共享失效或 FSDP state-dict 加载异常时，按下面顺序检查：

1. 用 `AutoConfig` 确认 `tie_word_embeddings` 的配置值。
2. 在 FSDP 包装前、真实 CPU/CUDA 权重已加载后，用 `input_weight is output_weight` 验证实际共享关系。
3. 确认 tied 模型没有走 `init_empty_weights()` 分支。
4. 确认 `apply_fsdp2()` 没有将 tied embedding 作为独立 `fully_shard()` 模块。
5. 确认所有 rank 都进入 `load_full_state_dict_fsdp()`；rank 0 提供真权重，其他 rank 接收广播后的分片。

本仓库的 V7 双卡验证已经覆盖 Qwen3-0.6B 的 tied 分支：模型未在初始化/广播时 hang，且普通 decoder
权重按两卡完成分片。背景和验证记录见 `docs/decisions/v7.md`。
