"""V8.1: Megatron 分片权重 → HF 格式（+ 反向 HF → Megatron 载入）。

**这是 slime-agentic 相对纯 Megatron 教程的真正增量，也是 nano 之前完全没有的东西。**
FSDP 的 `save_pretrained` 天然吐 HF 格式，disk reload 就能喂 SGLang（V6/V7.3 就这么做）；
Megatron 的参数**名和形状都不是 HF 的**——`decoder.layers.0.self_attention.linear_qkv.weight`
是 qkv 融合、还按 TP 切开的。要同步回推理引擎必须先 **TP-gather + 拆 qkv/gate + 改名**。
源项目为此写了整个 `megatron_to_hf/` 目录（10 个模型各一份）+ `update_weight/`（2232 行）。

对齐源：
  - `megatron_to_hf/qwen2.py:convert_qwen2_to_hf`（Qwen3 走这个分支，见 `__init__.py:46`
    `elif "qwen2" in model_name or "qwen3" in model_name`）—— 改名 + qkv 拆 + fc1 chunk(2)
  - `megatron_to_hf/processors/padding_remover.py:remove_padding` —— 去 vocab padding
  - `megatron_to_hf/__init__.py:22 convert_to_hf` —— 先 remove_padding 再改名的顺序
  - `update_weight/common.py:15 all_gather_param` —— 按 `param.partition_dim` TP-gather
    （含 **linear_fc1 的 GLU 重排** 这个最易错的点，见 `all_gather_param` docstring）

偏离源项目（登记见 docs/decisions/v8.md）：
  - **只做 Qwen3 dense 一种**（源 10 个模型 + MoE/MLA/quant 分支）：nano 只跑 Qwen3-0.6B，
    多模型分发是同一模式的重复。语义等价（同一条 qwen2 分支）。
  - **反向 `load_hf_to_megatron` 是 nano 自有**：源用离线 checkpoint 转换工具链（`mcore` 格式）
    把 HF ckpt 转成 Megatron ckpt，不在训练进程里做。nano 为了拿到 §3.2 的**逐值精确硬门**
    （HF → Megatron → 转回 HF，与原始比）必须有反向映射，故在进程内实现。
    它是**正向映射的严格逆**，两者互为验证——这正是硬门的判别力来源。
  - **无 quantize / MoE / q_lora 分支**：Qwen3-0.6B 用不上。
"""

from __future__ import annotations

import re
from typing import Iterator, Optional

import torch
import torch.distributed as dist

# 源 misc_utils.py:1 —— Megatron 的 DDP wrapper 会套 "module." 前缀（可能多层）。
# nano 的 GPTModel 是裸的（无 wrapper），故这里剥完前缀再匹配，比源写死
# "module.module.embedding..." 更通用；对源那种名字也照样成立（朝更清晰偏离）。
def strip_param_name_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


_VOCAB_PARAMS = {"embedding.word_embeddings.weight", "output_layer.weight"}


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """去 vocab padding（对齐源 padding_remover.py:6）。

    Megatron 会把 vocab 补齐到 `make_vocab_size_divisible_by × tp_size` 的整数倍，
    多出来的行是垃圾，转 HF 前必须切掉。Qwen3-0.6B 的 151936 恰好整除故通常是 no-op，
    但按源保留这一步——换模型/换 tp 就不再是 no-op。
    """
    if strip_param_name_prefix(name) in _VOCAB_PARAMS:
        return param[:vocab_size]
    return param


# ---- TP-gather（对齐源 update_weight/common.py:15 all_gather_param）------------


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """把 TP 分片的参数 all-gather 回全量张量。非 TP 参数原样返回。

    对齐源 `all_gather_param`。**最易错的点是 `linear_fc1` 的 GLU 重排**：
    Megatron 的 fc1 是 `[gate; up]` 沿 dim0 拼接后**整体**按 TP 切，
    所以 rank r 手里是 `[gate_r; up_r]`。朴素 `cat` 会拼成
    `[gate_0, up_0, gate_1, up_1]` —— **错的**，正确顺序是 `[gate_0, gate_1, up_0, up_1]`。
    源用「各分片先 chunk(2) 再按 gate 全体、up 全体重排」解决，nano 逐字照搬。
    （源还有 MoE `linear_fc2` 的 partition_dim 修正，nano 无 MoE 故不需要，见模块 docstring。）
    """
    if not getattr(param, "tensor_model_parallel", False) or (
        getattr(param, "parallel_mode", None) == "duplicated"
    ):
        return param.data

    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    if tp_size == 1:
        return param.data

    partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(partitions, param.data.contiguous(), group=tp_group)

    partition_dim = param.partition_dim
    is_glu_fc1 = "linear_fc1.weight" in name

    # **partition_stride 的处理是 nano 相对源的一处实质修正（登记偏离）**：
    #   源 `all_gather_param` 无条件 `assert param.partition_stride == 1`，然后**再**按名字
    #   给 linear_fc1 做 GLU 重排。但实测（megatron-core 0.18.2 + local spec）`linear_fc1.weight`
    #   的 `partition_stride` **就是 2** —— 照搬源的断言会在 TP>1 时直接炸。
    #   原因：stride=2 正是 Megatron 用来表达「[gate; up] 交错切分」的元数据，而源那段
    #   `chunk(2) → [gates..., ups...]` 重排**恰恰就是 stride=2 的解包逻辑**。也就是说源的
    #   断言与它自己的重排分支语义冲突（源用 TE spec 时 fc1 可能不带该 stride，故未暴露）。
    #   nano 的处理：**只对已被重排逻辑覆盖的 GLU fc1 放行 stride=2，其余仍严格要求 stride=1**
    #   —— 语义等价于源的意图（重排后就是全量 [gate; up]），且不放过真正没处理的 stride 情形。
    if is_glu_fc1:
        assert param.partition_stride in (1, 2), (
            f"{name}: 未支持的 partition_stride={param.partition_stride}"
        )
    else:
        assert param.partition_stride == 1, (
            f"{name}: partition_stride={param.partition_stride} != 1 is not supported"
        )

    if is_glu_fc1:
        chunked = [p.chunk(2, dim=0) for p in partitions]
        partitions = [c[0] for c in chunked] + [c[1] for c in chunked]

    return torch.cat(partitions, dim=partition_dim)


# ---- Megatron → HF（对齐源 megatron_to_hf/qwen2.py）----------------------------

_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.(.+)")


def convert_qwen3_to_hf(hf_config, name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """单个 **已 gather 成全量** 的 Megatron 参数 → 一个或多个 HF 参数（名, 张量）。

    对齐源 `convert_qwen2_to_hf`。一对多的两处正是 Megatron 的融合布局：
      - `linear_qkv.weight` → q/k/v 三份（按 **num_query_groups** 拆 GQA，不是简单三等分！）
      - `linear_fc1.weight` → gate/up 两份（`chunk(2, dim=0)`）
    """
    name = strip_param_name_prefix(name)

    if name == "embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    hidden_size = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", None) or (
        hidden_size // hf_config.num_attention_heads
    )
    num_query_groups = hf_config.num_key_value_heads
    value_num_per_group = hf_config.num_attention_heads // num_query_groups

    match = _LAYER_RE.match(name)
    if match:
        idx, rest = match.groups()
        p = f"model.layers.{idx}"
        if rest == "self_attention.linear_proj.weight":
            return [(f"{p}.self_attn.o_proj.weight", param)]
        if rest == "self_attention.linear_qkv.weight":
            # GQA 融合布局：[num_query_groups, value_num_per_group + 1(k) + 1(v), head_dim, hidden]
            # —— 每个 KV group 的 q 头与它的 k/v 相邻，故**不能**沿 dim0 三等分。
            v = param.view(num_query_groups, -1, head_dim, hidden_size)
            q, k, val = torch.split(v, [value_num_per_group, 1, 1], dim=1)
            return [
                (f"{p}.self_attn.q_proj.weight", q.reshape(-1, hidden_size)),
                (f"{p}.self_attn.k_proj.weight", k.reshape(-1, hidden_size)),
                (f"{p}.self_attn.v_proj.weight", val.reshape(-1, hidden_size)),
            ]
        if rest == "mlp.linear_fc1.weight":
            gate, up = param.chunk(2, dim=0)
            return [
                (f"{p}.mlp.gate_proj.weight", gate),
                (f"{p}.mlp.up_proj.weight", up),
            ]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{p}.mlp.down_proj.weight", param)]
        # ---- layernorm 的两种命名（**本处是 nano 相对源的一个实质差异**）----
        # 源用 TE spec（`get_gpt_layer_with_transformer_engine_spec`），layernorm 被**融进**
        # 后继 linear 的 `layer_norm_weight`（故源 qwen2.py:60-63 只认这两个名字）。
        # nano 用 `get_gpt_layer_local_spec`（容器装 FA2 后 TE import 会崩，见 v7.6.md 环境三坑），
        # 它把 layernorm 保留为**独立模块** `input_layernorm` / `pre_mlp_layernorm`。
        # 两者数学完全等价（同一个 RMSNorm，只是挂载位置不同），故这里**两种名字都认**：
        # 保留源的名字 = 忠实；加上 local spec 的名字 = nano 实际用的。
        if rest in ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"):
            return [(f"{p}.input_layernorm.weight", param)]
        if rest in ("mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"):
            return [(f"{p}.post_attention_layernorm.weight", param)]
        if rest == "self_attention.q_layernorm.weight":
            return [(f"{p}.self_attn.q_norm.weight", param)]
        if rest == "self_attention.k_layernorm.weight":
            return [(f"{p}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


def megatron_to_hf_state_dict(model, hf_config, tie: bool) -> dict[str, torch.Tensor]:
    """整个 Megatron 模型 → HF state_dict（TP-gather + remove_padding + 改名）。

    对齐源 `convert_to_hf`（`__init__.py:22`）：**先 remove_padding 再改名**。
    tie 时源不产出 `lm_head.weight`（HF 侧由 embed_tokens 共享）——nano 同样跳过。
    """
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        clean = strip_param_name_prefix(name)
        if tie and clean == "output_layer.weight":
            continue  # tie：HF 侧无独立 lm_head，由 embed_tokens 共享
        full = all_gather_param(clean, param)
        full = remove_padding(clean, full, hf_config.vocab_size)
        for hf_name, hf_param in convert_qwen3_to_hf(hf_config, clean, full):
            out[hf_name] = hf_param.detach().clone()
    return out


# ---- HF → Megatron（nano 自有的反向映射，为硬门服务）---------------------------


def _tp_shard(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size == 1:
        return tensor
    return tensor.chunk(tp_size, dim=dim)[tp_rank].contiguous()


def hf_to_megatron_state_dict(
    hf_state: dict[str, torch.Tensor],
    hf_config,
    num_layers: int,
    tp_rank: int,
    tp_size: int,
    tie: bool,
    padded_vocab_size: Optional[int] = None,
    fused_layernorm: bool = False,
) -> dict[str, torch.Tensor]:
    """HF state_dict → 本 rank 的 Megatron 分片 state_dict。**正向映射的严格逆**。

    nano 自有（源在训练进程外用 checkpoint 工具链做，见模块 docstring 偏离）。写它是为了
    §3.2 硬门：HF → Megatron → 转回 HF，与原始 HF 逐值比对，`max_abs_diff` 应恒为 0
    （全程只有 reshape/split/cat/rename，无浮点运算）。**正反两个方向互为验证**——
    单向转换写错很难发现，往返比对能抓住"拆 qkv 拆反了""fc1 gate/up 顺序错了"这类静默 bug。

    正向映射对 layernorm 认两种名字（TE 融合式 / local 独立式，见 `convert_qwen3_to_hf`），
    反向必须**产出模型实际用的那一种**，故有 `fused_layernorm` 开关（由 `load_hf_into_megatron`
    按模型真实参数名自动判定，不靠猜）。

    分片规则与 `all_gather_param` 的 gather 规则一一对应：
      - column-parallel（qkv/fc1/embedding）：沿 dim0 切；**fc1 必须 gate、up 分别切再拼**
      - row-parallel（o_proj/fc2）：沿 dim1 切
    """
    hidden_size = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", None) or (
        hidden_size // hf_config.num_attention_heads
    )
    num_query_groups = hf_config.num_key_value_heads
    value_num_per_group = hf_config.num_attention_heads // num_query_groups
    vocab_size = hf_config.vocab_size

    out: dict[str, torch.Tensor] = {}

    embed = hf_state["model.embed_tokens.weight"]
    if padded_vocab_size is not None and padded_vocab_size > vocab_size:
        pad = torch.zeros(
            padded_vocab_size - vocab_size, embed.shape[1], dtype=embed.dtype, device=embed.device
        )
        embed = torch.cat([embed, pad], dim=0)
    out["embedding.word_embeddings.weight"] = _tp_shard(embed, 0, tp_rank, tp_size)
    out["decoder.final_layernorm.weight"] = hf_state["model.norm.weight"]
    if not tie:
        lm_head = hf_state["lm_head.weight"]
        if padded_vocab_size is not None and padded_vocab_size > vocab_size:
            pad = torch.zeros(
                padded_vocab_size - vocab_size,
                lm_head.shape[1],
                dtype=lm_head.dtype,
                device=lm_head.device,
            )
            lm_head = torch.cat([lm_head, pad], dim=0)
        out["output_layer.weight"] = _tp_shard(lm_head, 0, tp_rank, tp_size)

    for i in range(num_layers):
        p, m = f"model.layers.{i}", f"decoder.layers.{i}"

        # qkv 融合：按 KV group 交错回 [groups, vpg+2, head_dim, hidden]，与正向的 view 逆着来。
        q = hf_state[f"{p}.self_attn.q_proj.weight"].view(
            num_query_groups, value_num_per_group, head_dim, hidden_size
        )
        k = hf_state[f"{p}.self_attn.k_proj.weight"].view(num_query_groups, 1, head_dim, hidden_size)
        v = hf_state[f"{p}.self_attn.v_proj.weight"].view(num_query_groups, 1, head_dim, hidden_size)
        qkv = torch.cat([q, k, v], dim=1).reshape(-1, hidden_size)
        # column-parallel 沿 dim0 切：因 group 在 dim0 连续排列，等价于「按 group 分给各 rank」。
        out[f"{m}.self_attention.linear_qkv.weight"] = _tp_shard(qkv, 0, tp_rank, tp_size)

        out[f"{m}.self_attention.linear_proj.weight"] = _tp_shard(
            hf_state[f"{p}.self_attn.o_proj.weight"], 1, tp_rank, tp_size  # row-parallel
        )

        # fc1：**gate 与 up 各自先切，再在本 rank 内拼成 [gate_r; up_r]**
        # （与 all_gather_param 的 GLU 重排互逆；直接切整块会错位）。
        gate = _tp_shard(hf_state[f"{p}.mlp.gate_proj.weight"], 0, tp_rank, tp_size)
        up = _tp_shard(hf_state[f"{p}.mlp.up_proj.weight"], 0, tp_rank, tp_size)
        out[f"{m}.mlp.linear_fc1.weight"] = torch.cat([gate, up], dim=0)

        out[f"{m}.mlp.linear_fc2.weight"] = _tp_shard(
            hf_state[f"{p}.mlp.down_proj.weight"], 1, tp_rank, tp_size  # row-parallel
        )

        ln_qkv = (
            f"{m}.self_attention.linear_qkv.layer_norm_weight"
            if fused_layernorm
            else f"{m}.input_layernorm.weight"
        )
        ln_mlp = (
            f"{m}.mlp.linear_fc1.layer_norm_weight"
            if fused_layernorm
            else f"{m}.pre_mlp_layernorm.weight"
        )
        out[ln_qkv] = hf_state[f"{p}.input_layernorm.weight"]
        out[ln_mlp] = hf_state[f"{p}.post_attention_layernorm.weight"]
        out[f"{m}.self_attention.q_layernorm.weight"] = hf_state[f"{p}.self_attn.q_norm.weight"]
        out[f"{m}.self_attention.k_layernorm.weight"] = hf_state[f"{p}.self_attn.k_norm.weight"]

    return out


def load_hf_into_megatron(model, hf_state: dict[str, torch.Tensor], hf_config, tie: bool) -> None:
    """把 HF state_dict 载入已建好的 Megatron 模型（本 rank 分片）。"""
    from megatron.core import mpu

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    # 从模型自身读 padded vocab（megatron.core 不加 padding，但换配置时这里就有值）。
    padded_vocab = model.embedding.word_embeddings.weight.shape[0] * tp_size

    num_layers = len(model.decoder.layers)
    own = dict(model.named_parameters())
    # 按模型**实际**参数名判定 layernorm 命名风格（TE 融合式 vs local 独立式），不靠猜。
    fused_ln = "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight" in {
        strip_param_name_prefix(n) for n in own
    }
    mega_state = hf_to_megatron_state_dict(
        hf_state,
        hf_config,
        num_layers,
        tp_rank,
        tp_size,
        tie,
        padded_vocab_size=padded_vocab,
        fused_layernorm=fused_ln,
    )

    own = {strip_param_name_prefix(n): p for n, p in own.items()}
    with torch.no_grad():
        for name, tensor in mega_state.items():
            param = own.get(name)
            if param is None:
                raise KeyError(f"Megatron 模型没有参数 {name}（反向映射与模型结构不一致）")
            if tuple(param.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"{name} 形状不匹配：模型 {tuple(param.shape)} vs 转换出 {tuple(tensor.shape)}"
                )
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))


def load_hf_weights(model_path: str) -> dict[str, torch.Tensor]:
    """从 HF checkpoint 目录读全量 state_dict（safetensors）。"""
    import glob
    import os

    from safetensors.torch import load_file

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{model_path} 下没有 safetensors 文件")
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_file(f))
    return state
