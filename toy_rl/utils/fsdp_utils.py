"""V7: FSDP2 工具函数 —— 对齐源 slime/backends/fsdp_utils/actor.py。

三个函数各对齐源的一段：
  - setup_device_mesh        ← actor.py:186-202 (_setup_device_mesh)：1D dp mesh
  - apply_fsdp2              ← actor.py:933-984：按 _no_split_modules wrap decoder 层
  - get_init_weight_context  ← actor.py:204-227 (_get_init_weight_context_manager)：tie 分支
  - load_full_state_dict_fsdp← actor.py:229-266 (_fsdp2_load_full_state_dict)：rank0 广播

偏离源项目（V7 全局）：
  - 源 apply_fsdp2 收 args 对象读 args.fp16；nano 收独立 fp16/cpu_offload 参数（更清晰，语义等价）。
  - 只 1D dp mesh，无 TP/PP/CP（源支持多维，留 V8 Megatron）。
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


def setup_device_mesh(world_size: int):
    """建 1D data-parallelism device mesh。对齐源 actor.py:198。

    源: init_device_mesh("cuda", mesh_shape=(dp_size,), mesh_dim_names=("dp",))
    nano 简化：无 TP/PP，只有 DP 一维。
    """
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    logger.info(f"Device mesh (1D): dp_size={world_size}")
    return mesh


def get_init_weight_context(tie_word_embeddings: bool, rank: int):
    """返回一个"建模型用的设备上下文管理器"工厂。对齐源 actor.py:204-227。

    源核心洞察（actor.py:212）：**tie_word_embeddings=True 时 meta-device 初始化会 hang**
    （tied 权重在 meta tensor 上无法正确共享）。故：
      - tie=False：rank0 在 CPU 建（有真权重）、其余 rank 用 accelerate.init_empty_weights()
        在 meta device 建（不分配内存，省显存；权重稍后从 rank0 广播过来）。
      - tie=True ：**所有 rank 都在 CPU 建全量**（避免 meta hang，对齐源 actor.py:226）。

    Qwen3-0.6B 是 tied 模型 → 走后者（全 rank CPU 全量）。
    """
    from accelerate import init_empty_weights

    def cpu_init():
        return torch.device("cpu")

    if not tie_word_embeddings:
        # 非 tie：rank0 CPU、其余 meta（省显存）。
        return cpu_init if rank == 0 else init_empty_weights

    # tie：所有 rank CPU 全量（meta 会 hang）。
    logger.info(f"[Rank {rank}] tie_word_embeddings=True, loading full model to CPU on all ranks")
    return cpu_init


def apply_fsdp2(model, mesh, cpu_offload: bool = False, fp16: bool = False):
    """按 _no_split_modules 把 decoder 层逐个 fully_shard，再 shard 顶层。对齐源 actor.py:933-984。

    Args:
        model: 待包装模型
        mesh: 1D dp DeviceMesh
        cpu_offload: True 则参数/梯度/优化器 offload 到 CPU（源默认 False）
        fp16: True 用 fp16 参数，否则 bf16（源默认 bf16）

    对齐要点：
      - 选 wrap 的模块 = _no_split_modules 里的类（decoder 层）**加上** Embedding——但
        **仅当 not tie_word_embeddings**（源 actor.py:956）。tied embedding 不单独 shard，
        跟着顶层 fully_shard 走（否则 tied 权重被拆两处，sharding 出错）。
      - MixedPrecision: param_dtype=bf16/fp16, reduce_dtype=fp32（梯度规约始终 fp32 求稳，源 961）。
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # 对齐源 actor.py:949-950：从模型拿要 wrap 的层类名（如 ["Qwen3DecoderLayer"]）。
    layer_cls_to_wrap = model._no_split_modules
    assert layer_cls_to_wrap and layer_cls_to_wrap[0] is not None, "模型缺 _no_split_modules"

    tie = model.config.tie_word_embeddings
    # 对齐源 actor.py:952-957：decoder 层总 wrap；Embedding 仅在 **非 tie** 时单独 wrap。
    modules = [
        module
        for _name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not tie)
    ]
    logger.info(f"FSDP wrapping {len(modules)} modules (layers={layer_cls_to_wrap}, tie={tie})")

    # 对齐源 actor.py:959-966：param bf16/fp16，reduce 恒 fp32。
    param_dtype = torch.float16 if fp16 else torch.bfloat16
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    logger.info(f"FSDP MixedPrecision: param_dtype={param_dtype}, reduce_dtype=float32")

    fsdp_kwargs = {"mp_policy": mp_policy, "offload_policy": offload_policy, "mesh": mesh}

    # 对齐源 actor.py:978-982：先逐层 shard，再 shard 顶层模型。
    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    logger.info("FSDP wrapping completed")
    return model


def load_full_state_dict_fsdp(model, full_state, cpu_offload: bool = False):
    """把全量 state_dict 从 rank0 广播进 FSDP 模型。对齐源 actor.py:229-266。

    核心（源 actor.py:246-259）：
      - rank0 把（含真权重的）模型 .to(cuda)；其余 rank 用 **to_empty(cuda)**（只分配不初始化，
        因它们的权重是 meta 或即将被覆盖）。
      - set_model_state_dict(broadcast_from_rank0=True) 从 rank0 把权重广播 + 分片到各 rank。
      - buffer 不被 set_model_state_dict 广播 → 手动逐个 dist.broadcast(src=0)。
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    rank = dist.get_rank()
    if rank == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        # to_empty：在设备上分配空张量（不初始化内存），权重马上从 rank0 广播过来。
        model = model.to_empty(device=torch.cuda.current_device())

    options = StateDictOptions(
        full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True
    )
    set_model_state_dict(model, full_state, options=options)

    # 对齐源 actor.py:257-259：buffer 需手动广播。
    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    logger.info(f"[Rank {rank}] full state dict loaded via rank0 broadcast")
    return model
