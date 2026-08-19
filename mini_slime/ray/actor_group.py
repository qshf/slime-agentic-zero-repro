"""V4/V7.3: RayTrainGroup + TrainRayActor —— 对齐源 slime/ray/{actor_group,train_actor}.py。

角色分工（Explore 已核对源码）：
  - TrainRayActor（train_actor.py:28，`@ray.remote(num_gpus=1)`）：**每个 rank 一个进程**，
    持真训练 backend（FSDP/Megatron）。V4/V6 fake/torch：壳里放 Trainer，train/update_weights 直接委托。
  - RayTrainGroup（actor_group.py:10，**普通本地类，非 ray.remote**）：持 N 个 worker handle，
    把 async_train / update_weights **fan-out** 到每个 rank。async_train 返回 list[ObjectRef]
    （每 rank 一个），这是 DP 的形状。

V7.3 增量（把 FSDPTrainer 搬进 Ray actor，形成真 torch.distributed 进程组）：
  - 对齐源 train_actor.py:29-48 + actor_group.py:94-106 的 Ray+FSDP 接法——
    actor `__init__` 里设 MASTER_ADDR/PORT/WORLD_SIZE/RANK/LOCAL_RANK env；rank0 先起、
    暴露自由端口，RayTrainGroup 把它广播给后续 rank；`dist.init_process_group` 在 init()
    阶段（FSDPTrainer.__init__ 内）跑，等所有 actor 建好后集体 rendezvous。
  - **LOCAL_RANK 恒设 0**（对齐源 get_local_gpu_id，train_actor.py:20-25）：Ray 给每个 actor
    独占 CUDA_VISIBLE_DEVICES，actor 视角里可见设备恒为索引 0。若用 rank，rank1 会去 cuda:1
    （它只看得到一张卡）→ invalid device。
  - 类装饰器改**运行时包装** ray.remote(num_gpus=N)(cls)（对齐源 actor_group.py:90）：
    fake/torch 选 num_gpus=0（CPU/单卡 docker 外），fsdp 选 num_gpus=1（每 rank 一张卡）。

偏离（见 docs/decisions/{v4,v7}.md 偏离表）：
  - 单节点：master_addr 取 localhost（源用 ray._private.services.get_node_ip_address 支持多节点）。
  - 无 placement group（fake/torch 无 GPU 调度；fsdp 直接 num_gpus=1 让 Ray 分卡，不钉 bundle）。
  - fake/torch 沿用 V3/V6 后端；fsdp 是本版新增的真分布式后端。
"""

from __future__ import annotations

import os
import socket

import ray

from mini_slime.args import Args
from mini_slime.trainer import Trainer


def _get_free_port() -> int:
    """绑定端口 0 让内核分配一个空闲端口（标准 trick），供 rank0 当 MASTER_PORT。

    对齐源 get_free_port（misc.py:65）的角色；nano 单节点用 socket 绑定取代逐端口探测。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TrainRayActor:
    """单个训练 rank 的 actor 进程。对齐源 train_actor.py:28-48 TrainRayActor。

    __init__ 只做两件事（对齐源）：记 rank/world_size、设分布式 env（fsdp 后端）。
    真训练器（Trainer→FSDPTrainer/TorchActor）在 init() 里建——那时所有 actor 的 env 已就绪，
    FSDPTrainer.__init__ 内的 dist.init_process_group 才能集体 rendezvous。
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self._trainer: Trainer | None = None  # 真身在 init() 里建（对齐源 __init__ 只记 rank/env）

        # master_addr 非 None ⇒ fsdp 分布式路径：设 torch.distributed rendezvous env。
        #   rank0 传入 master_addr=None → 自己选 localhost + 自由端口，供 RayTrainGroup 广播给 rank>0。
        #   fake/torch 路径 RayTrainGroup 传 master_addr=None 且 world_size=1、不会读这些 env（不 init_pg）。
        if master_addr is not None or rank == 0:
            self.master_addr = master_addr or "127.0.0.1"
            self.master_port = master_port or _get_free_port()
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(rank)
            # LOCAL_RANK 恒 0：Ray 给每 actor 独占 CUDA_VISIBLE_DEVICES，可见设备恒索引 0
            # （对齐源 get_local_gpu_id，train_actor.py:20-25）。
            os.environ["LOCAL_RANK"] = "0"
        else:
            self.master_addr, self.master_port = None, None

    def get_master_addr_and_port(self) -> tuple[str, int]:
        """rank0 暴露自己选的 master addr/port，供 RayTrainGroup 广播给 rank>0（对齐源 ray_actor.py:9）。"""
        return self.master_addr, self.master_port

    def init(self, args: Args) -> None:
        """对齐源 TrainRayActor.init：建训练器。fsdp 后端下 FSDPTrainer.__init__ 会读上面的 env
        并 dist.init_process_group（各 rank 集体 rendezvous）。fake/torch 无分布式，直接建。"""
        self._trainer = Trainer(args, rank=self.rank, world_size=self.world_size)

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
        # fsdp/megatron 后端：每 rank 一张卡，Ray 调度 GPU；fake/torch：CPU（不归 Ray 调度）。
        # megatron world_size = TP × PP × DP；TP/PP/DP 组由 Megatron mpu 建立。
        if args.train_backend == "fsdp":
            world_size = args.fsdp_world_size
            num_gpus = 1  # 每 rank 一张卡（对齐源 actor_group.py:90 num_gpus=1）。
        elif args.train_backend == "megatron":
            world_size = (
                args.tensor_model_parallel_size
                * args.pipeline_model_parallel_size
                * args.megatron_data_parallel_size
            )
            num_gpus = 1  # 每 rank 一张卡，与 fsdp 路径对称。
        else:
            world_size = num_nodes * num_gpus_per_node
            num_gpus = 0  # fake=CPU、torch=单卡 docker 外；不归 Ray 调度 GPU。
        self.world_size = world_size

        # 运行时包装（对齐源 actor_group.py:90）：同一个类按 backend 选 num_gpus。
        actor_cls = ray.remote(num_cpus=1, num_gpus=num_gpus)(TrainRayActor)

        # rank0 先起、拿到 master addr/port，再广播给 rank>0（对齐源 actor_group.py:94-106）。
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = actor_cls.remote(rank, world_size, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args: Args) -> list:
        """对齐源 async_init：**并发**发起每 rank 的 init.remote 再统一 ray.get。

        并发发起是关键（不可循环里逐个 ray.get）：fsdp 后端各 rank 的 init 内含集体
        init_process_group，必须所有 rank 都进 rendezvous 才返回——串行等第一个会死锁。
        """
        return ray.get([a.init.remote(args) for a in self._actor_handlers])

    def async_train(self, rollout_id: int, rollout_data) -> list:
        """对齐源 async_train：fan-out 到每 rank，返回 list[ObjectRef]（**不** ray.get，交给主循环控同步点）。"""
        return [a.train.remote(rollout_id, rollout_data) for a in self._actor_handlers]

    def update_weights(self) -> int:
        """对齐源 update_weights：fan-out 到所有 rank（fsdp 的 save 是集体操作，须每 rank 都进），
        内部 ray.get 同步。返回权重版本号（rank0 权威值）。"""
        versions = ray.get([a.update_weights.remote() for a in self._actor_handlers])
        return versions[0]

    def weight_version(self) -> int:
        return ray.get(self._actor_handlers[0].get_weight_version.remote())

    def pids(self) -> list[int]:
        """各 rank 进程 PID，测试用它确认分进程。"""
        return ray.get([a.pid.remote() for a in self._actor_handlers])
