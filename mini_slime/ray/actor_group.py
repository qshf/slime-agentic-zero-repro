"""V4: RayTrainGroup + TrainRayActor —— 对齐源 slime/ray/{actor_group,train_actor}.py。

角色分工（Explore 已核对源码）：
  - TrainRayActor（train_actor.py:28，`@ray.remote(num_gpus=1)`）：**每个 rank 一个进程**，
    持真训练 backend（FSDP/Megatron）。nano fake：壳里放 V3 的 Trainer，train/update_weights 直接
    委托，训练逻辑一行不重写。
  - RayTrainGroup（actor_group.py:10，**普通本地类，非 ray.remote**）：持 N 个 worker handle，
    把 async_train / update_weights **fan-out** 到每个 rank。async_train 返回 list[ObjectRef]
    （每 rank 一个），这是 DP 的形状。

偏离（见 docs/decisions/v4.md 偏离表）：
  - world_size=1（DP=1）：源 num_nodes*num_gpus_per_node 个 rank + update_weights 广播 rank0→其余；
    nano fake 无真并行必要，先看清"group→workers dispatch"形状。list[ObjectRef] 结构已保留。
  - TrainRayActor 用 num_gpus=0（源 num_gpus=1）：fake trainer 不占 GPU。
  - fake 训练/权重同步沿用 V3：真 backend 留主线三 V6-V9。
"""

from __future__ import annotations

import os

import ray

from mini_slime.args import Args
from mini_slime.trainer import Trainer


@ray.remote(num_cpus=1, num_gpus=0)
class TrainRayActor:
    """单个训练 rank 的 actor 进程。对齐源 train_actor.py:28 TrainRayActor（壳内放 fake Trainer）。"""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self._trainer: Trainer | None = None  # 真身在 init() 里建（对齐源 __init__ 只记 rank，init 才建模型）

    def init(self, args: Args) -> None:
        """对齐源 TrainRayActor.init：真实现会 torch.distributed.init + 加载模型；nano 只建 fake Trainer。"""
        self._trainer = Trainer(args)

    def train(self, rollout_id: int, rollout_data: dict) -> dict:
        """对齐源 TrainRayActor.train(rollout_id, rollout_data_ref)：本 rank 训一步。"""
        return self._trainer.train(rollout_id, rollout_data)

    def update_weights(self) -> int:
        """对齐源 TrainRayActor.update_weights：训后同步权重。返回新版本号（供 group 收集）。"""
        self._trainer.update_weights()
        return self._trainer.weight_version

    def get_weight_version(self) -> int:
        return self._trainer.weight_version

    def pid(self) -> int:
        """当前进程 PID，测试用它确认各 rank 在独立进程。"""
        return os.getpid()


class RayTrainGroup:
    """一组训练 actor 的本地包装。对齐源 actor_group.py:10 RayTrainGroup（非 ray.remote）。"""

    def __init__(self, args: Args, num_nodes: int = 1, num_gpus_per_node: int = 1) -> None:
        self.args = args
        world_size = num_nodes * num_gpus_per_node  # V4 = 1（DP=1）
        # 对齐源 _allocate_gpus_for_actor：每 rank 起一个 @ray.remote worker。
        # 源用 PlacementGroupSchedulingStrategy 钉 GPU bundle；nano fake 无 GPU，直接 .remote()。
        self._actor_handlers = [
            TrainRayActor.remote(rank, world_size) for rank in range(world_size)
        ]

    def async_init(self, args: Args) -> list:
        """对齐源 async_init：让每个 rank 初始化。这里内部 ray.get 等全部就绪（fake init 很快）。"""
        return ray.get([a.init.remote(args) for a in self._actor_handlers])

    def async_train(self, rollout_id: int, rollout_data) -> list:
        """对齐源 async_train：fan-out 到每 rank，返回 list[ObjectRef]（**不** ray.get，交给主循环控同步点）。"""
        return [a.train.remote(rollout_id, rollout_data) for a in self._actor_handlers]

    def update_weights(self) -> int:
        """对齐源 update_weights：内部 ray.get 同步所有 rank。返回权重版本号（各 rank 一致，取 rank0）。"""
        versions = ray.get([a.update_weights.remote() for a in self._actor_handlers])
        return versions[0]

    def weight_version(self) -> int:
        return ray.get(self._actor_handlers[0].get_weight_version.remote())

    def pids(self) -> list[int]:
        """各 rank 进程 PID，测试用它确认分进程。"""
        return ray.get([a.pid.remote() for a in self._actor_handlers])
