"""UpdateWeightFromTensor: fastest viable disk-free weight sync via HTTP POST.

Aligned with: slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py

Workflow:
  1. Get HF state_dict from trainer (Megatron auto-converts)
  2. Group by dtype, pack into FlattenedTensorBucket
  3. HTTP POST to SGLang /update_weights_from_tensor endpoint
  4. SGLang deserializes and hot-loads weights (disk-free)

Why not NCCL (UpdateWeightFromDistributed):
  - Requires SGLang as Ray actor (supporting init_weights_update_group)
  - Current SGLang runs standalone, not Ray-managed
  - UpdateWeightFromTensor is only immediately viable disk-free option

Expected: 2-5 sec (vs disk reload 25 sec, 5-12x faster)
"""

from __future__ import annotations

import multiprocessing as mp
import base64
import io
from pathlib import Path
from types import SimpleNamespace

import requests
import torch

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
    from sglang.srt.utils import MultiprocessingSerializer
except ImportError:
    # The learner runs in the host .venv (Ray is installed there), while the
    # rollout server runs in a separate SGLang container. The wire protocol is
    # deliberately simple: ForkingPickler plus a flat uint8 tensor and metadata
    # objects with these six attributes. The server reconstructs its own
    # FlattenedTensorMetadata instances, so importing all of SGLang here is not
    # required.
    from multiprocessing.reduction import ForkingPickler

    class FlattenedTensorBucket:
        def __init__(self, named_tensors):
            if not named_tensors:
                raise ValueError("Cannot create an empty tensor bucket")
            parts = []
            metadata = []
            offset = 0
            for name, tensor in named_tensors:
                flattened = tensor.flatten().view(torch.uint8)
                size = flattened.numel()
                parts.append(flattened)
                metadata.append(
                    SimpleNamespace(
                        name=name,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        start_idx=offset,
                        end_idx=offset + size,
                        numel=size,
                    )
                )
                offset += size
            self._flattened_tensor = torch.cat(parts, dim=0)
            self._metadata = metadata

        def get_flattened_tensor(self):
            return self._flattened_tensor

        def get_metadata(self):
            return self._metadata

    class MultiprocessingSerializer:
        @staticmethod
        def serialize(obj, output_str: bool = False):
            buffer = io.BytesIO()
            ForkingPickler(buffer).dump(obj)
            payload = buffer.getvalue()
            return base64.b64encode(payload).decode("utf-8") if output_str else payload


class UpdateWeightFromTensor:
    """Send serialized tensors to SGLang via HTTP (disk-free)."""

    def __init__(self, trainer, sglang_url: str):
        """
        trainer: Trainer instance (MegatronTrainer / FSDPTrainer / TorchActor)
        sglang_url: SGLang /generate endpoint (replaced with /update_weights_from_tensor)
        """
        self.trainer = trainer
        self.sglang_url = sglang_url.replace("/generate", "/update_weights_from_tensor")
        self.weight_version = 0

    @torch.no_grad()
    def update_weights(self):
        """Disk-free weight sync via HTTP POST."""
        self._configure_resource_sharer_authkey()
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

    @staticmethod
    def _configure_resource_sharer_authkey() -> None:
        """Use SGLang's shared resource-sharer key when the engine is in Docker."""
        path = Path("/tmp/sglang_tensor_sync_authkey")
        if path.is_file():
            mp.current_process().authkey = path.read_bytes().strip()
