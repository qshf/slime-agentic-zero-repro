"""UpdateWeightFromTensor: 实际可行的免落盘权重同步(Gloo + Ray RPC)。

对齐源:slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py

工作流:
  1. 训练进程从 Megatron/HF 格式转到 HF state_dict
  2. 按 dtype 分组打包成 FlattenedTensorBucket(序列化 CPU 传输)
  3. 通过 HTTP POST 到 SGLang 的 /update_weights_from_tensor 端点
  4. SGLang 反序列化并热加载权重

为何不用 NCCL(UpdateWeightFromDistributed):
  - 需要 SGLang engine 作为 Ray actor(支持 init_weights_update_group)
  - 当前 SGLang 是独立服务,不在 Ray 管理下
  - UpdateWeightFromTensor 是唯一立即可用的免落盘方案

预期性能:2-5 秒(比 disk reload 25 秒快 5-12×)

from __future__ import annotations

import requests
import torch

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
    from sglang.srt.utils import MultiprocessingSerializer
except ImportError:
    # Fallback: 开发环境可能没装 SGLang
    FlattenedTensorBucket = None
    MultiprocessingSerializer = None


class UpdateWeightFromTensor:
    """通过 HTTP 发送序列化 tensor 到 SGLang (免落盘,实际可行)。"""

    def __init__(self, trainer, sglang_url: str):
        """
        trainer: 训练器实例(MegatronTrainer / FSDPTrainer / TorchActor)
        sglang_url: SGLang /generate 端点(替换为 /update_weights_from_tensor)
        """
        if FlattenedTensorBucket is None:
            raise ImportError("需要安装 sglang 才能使用 UpdateWeightFromTensor")

        self.trainer = trainer
        self.sglang_url = sglang_url.replace("/generate", "/update_weights_from_tensor")
        self.weight_version = 0


    @torch.no_grad()
    def update_weights(self):
        """免落盘同步权重(HTTP POST 序列化 tensor)。"""
        self.weight_version += 1

        # 1. 获取 HF state_dict
        if hasattr(self.trainer, "to_hf_state_dict"):
            # Megatron: 需要转换参数名
            hf_state_dict = self.trainer.to_hf_state_dict()
        elif hasattr(self.trainer, "model"):
            # TorchActor / FSDPTrainer: 直接用 HF 模型
            hf_state_dict = {
                k: v.cpu() if hasattr(v, "cpu") else v
                for k, v in self.trainer.model.state_dict().items()
            }
        else:
            raise ValueError("trainer 必须有 to_hf_state_dict 或 model 属性")

        # 2. 按 dtype 分组打包
        named_tensors_by_dtype = {}
        for name, tensor in hf_state_dict.items():
            # 移到 CPU(序列化传输)
            if hasattr(tensor, "cuda") and tensor.is_cuda:
                tensor = tensor.cpu()

            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtype:
                named_tensors_by_dtype[dtype] = []
            named_tensors_by_dtype[dtype].append((name, tensor))

        # 3. 每个 dtype 打一个 FlattenedTensorBucket
        for dtype, named_tensors in named_tensors_by_dtype.items():
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_data = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }

            # 序列化
            serialized = MultiprocessingSerializer.serialize(flattened_data, output_str=True)

            # 4. HTTP POST 到 SGLang
            payload = {
                "serialized_named_tensors": [serialized],  # SGLang 期望 list(支持多 TP rank)
                "load_format": "flattened_bucket",
                "weight_version": str(self.weight_version),
            }

            try:
                response = requests.post(self.sglang_url, json=payload, timeout=300)
                response.raise_for_status()
            except Exception as e:
                print(f"[UpdateWeightFromTensor] 发送失败 (dtype={dtype}): {e}")
                raise
