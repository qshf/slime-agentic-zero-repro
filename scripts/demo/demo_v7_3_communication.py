#!/usr/bin/env python3
"""用 CPU 演示 V7.3 的三层通信，不加载模型、不需要 GPU 或 SGLang。

运行：

    uv sync --extra train
    uv run python scripts/demo_v7_3_communication.py --world-size 2

本脚本保留 V7.3 的通信形状：

1. Ray driver 创建 rank 0；rank 0 选 MASTER_ADDR/MASTER_PORT，driver 再传给其余 rank。
2. driver 并发调用所有 actor 的 init，所有 rank 用 torch.distributed 的 env:// rendezvous。
3. 用显式 broadcast / all_reduce 演示 FSDP 初始化和反向传播中会发生的集体通信。
4. 每个 rank 对同一批数据做 DP 跨步切分，并把各自的“权重分片” gather 到 rank 0。
5. 所有 rank 完成集体保存后，只有 rank 0 执行模拟的 SGLang reload 调用。

真实 V7.3 使用 NCCL + FSDP2；这里改用 CPU Gloo，目的是让通信顺序可在普通机器上观察。FSDP
在 forward/backward 中隐式发起参数 all-gather 与梯度 reduce-scatter；本脚本用可见的 all_reduce
代替它来展示“所有 rank 必须同时参与集体操作”的规则。

这是单机演示，MASTER_ADDR 固定为 127.0.0.1。多机场景应使用 rank 0 所在节点的可路由 IP。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import timedelta
from typing import Any

try:
    import ray
    import torch
    import torch.distributed as dist
except ImportError as exc:  # 让脚本在尚未安装训练依赖的环境里给出可操作提示。
    ray = None
    torch = None
    dist = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


def _get_free_port() -> int:
    """让内核分配一个本机可用端口，供 rank 0 作为 rendezvous 地址。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class CommunicationWorker:
    """一个 Ray actor = 一个 torch.distributed rank。

    该类故意与 mini_slime.ray.actor_group.TrainRayActor 的启动顺序相同：构造时只准备 env，
    在 initialize() 中才调用 init_process_group()。这样 driver 能先把所有 rank 建好，再让它们
    同时进入 rendezvous。
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
        self.weight_version = 0

        if rank == 0:
            self.master_addr = master_addr or "127.0.0.1"
            self.master_port = master_port or _get_free_port()
        else:
            if master_addr is None or master_port is None:
                raise ValueError("non-root rank must receive rank 0's MASTER_ADDR and MASTER_PORT")
            self.master_addr = master_addr
            self.master_port = master_port

        # 与 V7.3 相同的 env:// rendezvous 配置。CPU demo 没有 Ray 的 GPU 可见性问题，
        # 仍保留 LOCAL_RANK=0，表达“每个 actor 只使用自己的本地设备索引 0”。
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"

    def master_info(self) -> tuple[str, int]:
        """供 driver 从 rank 0 取 rendezvous 地址，再传给其他 actor。"""
        return self.master_addr, self.master_port

    def initialize(self) -> dict[str, Any]:
        """所有 rank 必须并发调用；Gloo 在此建立单机进程组。"""
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "backend": dist.get_backend(),
            "master": f"{self.master_addr}:{self.master_port}",
        }

    def train_collectives(self, samples: list[str]) -> dict[str, Any]:
        """演示 DP 切分、rank0 broadcast 与反向阶段的 all-reduce 形状。"""
        local_samples = samples[self.rank :: self.world_size]

        # V7.3 初始化权重时由 rank 0 提供真 state_dict；这里用一个标量让结果肉眼可见。
        initial_weight = torch.tensor([42 if self.rank == 0 else -1], dtype=torch.int64)
        dist.broadcast(initial_weight, src=0)

        # FSDP backward 会隐式做 reduce-scatter/平均。为教学清晰，此处显式 all_reduce：
        # rank 0 贡献 1、rank 1 贡献 2，所有 rank 最终都应看见总和 3。
        local_gradient = torch.tensor([float(self.rank + 1)], dtype=torch.float32)
        dist.all_reduce(local_gradient, op=dist.ReduceOp.SUM)
        dist.barrier()

        return {
            "rank": self.rank,
            "local_samples": local_samples,
            "weight_after_broadcast": int(initial_weight.item()),
            "gradient_after_all_reduce": float(local_gradient.item()),
        }

    def save_and_publish(self) -> dict[str, Any]:
        """模拟 FSDP 集体保存：分片 gather 到 rank 0，随后只由 rank 0 发布。"""
        local_shard = {"rank": self.rank, "value": f"parameter-shard-{self.rank}"}
        gathered_shards: list[dict[str, Any] | None] | None = (
            [None] * self.world_size if self.rank == 0 else None
        )
        # V7.3 实际调用 get_model_state_dict(full_state_dict=True)。这里用 gather_object 表达
        # 同一个约束：所有 rank 都要进集体调用，但完整状态只交给 rank 0。
        dist.gather_object(local_shard, gathered_shards, dst=0)

        saved_by_rank0 = None
        if self.rank == 0:
            saved_by_rank0 = gathered_shards

        # 对齐 FSDPTrainer.save_pretrained()：确保 rank 0 的“写盘”已结束，其他 rank 才继续。
        dist.barrier()

        self.weight_version += 1
        return {
            "rank": self.rank,
            "saved_by_rank0": saved_by_rank0,
            "sglang_reload_called": self.rank == 0,
            "weight_version": self.weight_version,
        }

    def close(self) -> int:
        """在所有示例集体调用完成后释放进程组。"""
        if dist.is_initialized():
            dist.destroy_process_group()
        return self.rank


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only Ray + torch.distributed communication demo")
    parser.add_argument("--world-size", type=int, default=2, help="Ray actor / distributed rank 数，默认 2")
    parser.add_argument("--sample-count", type=int, default=4, help="DP 切分的样本数，默认 4")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.world_size < 1:
        raise ValueError("--world-size must be at least 1")
    if args.sample_count < args.world_size:
        raise ValueError("--sample-count must be at least --world-size")
    if args.sample_count % args.world_size != 0:
        raise ValueError(
            "--sample-count must be divisible by --world-size; otherwise a rank can skip a collective "
            "backward path, which is the V7.3 deadlock hazard"
        )


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    for row in sorted(rows, key=lambda item: item["rank"]):
        print(f"  rank {row['rank']}: {row}")


def main() -> int:
    if _IMPORT_ERROR is not None:
        print(f"Missing dependency: {_IMPORT_ERROR}", file=sys.stderr)
        print("Install the demo dependencies with: uv sync --extra train", file=sys.stderr)
        return 2

    args = _parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        return 2

    workers = []
    process_group_ready = False
    try:
        ray.init(num_cpus=args.world_size, include_dashboard=False, log_to_driver=False)
        worker_cls = ray.remote(num_cpus=1)(CommunicationWorker)

        # 1. 先起 rank 0，再经 Ray RPC 取它选出的端口；这是 V7.3 的“端口广播”。
        rank0 = worker_cls.remote(rank=0, world_size=args.world_size)
        master_addr, master_port = ray.get(rank0.master_info.remote())
        workers = [rank0]
        workers.extend(
            worker_cls.remote(
                rank=rank,
                world_size=args.world_size,
                master_addr=master_addr,
                master_port=master_port,
            )
            for rank in range(1, args.world_size)
        )
        print(f"rank 0 selected rendezvous endpoint: {master_addr}:{master_port}")

        # 2. 必须先发出所有 RPC，再统一等待；逐个 ray.get 会让 rank 0 卡在 rendezvous 中。
        init_rows = ray.get([worker.initialize.remote() for worker in workers])
        process_group_ready = True
        _print_rows("1. Process group created through concurrent init.remote()", init_rows)

        # 3. 训练阶段：driver 将同一批数据 fan-out，各 rank 在 actor 内自行 DP-split。
        samples = [f"sample-{index}" for index in range(args.sample_count)]
        train_rows = ray.get([worker.train_collectives.remote(samples) for worker in workers])
        expected_gradient = sum(range(1, args.world_size + 1))
        assert all(row["weight_after_broadcast"] == 42 for row in train_rows)
        assert all(row["gradient_after_all_reduce"] == expected_gradient for row in train_rows)
        _print_rows("2. Broadcast, DP split, and all_reduce", train_rows)

        # 4. 保存必须再 fan-out 到所有 rank；只有 rank 0 收到完整状态并模拟调用 SGLang reload。
        publish_rows = ray.get([worker.save_and_publish.remote() for worker in workers])
        assert publish_rows[0]["saved_by_rank0"] is not None
        assert sum(row["sglang_reload_called"] for row in publish_rows) == 1
        assert {row["weight_version"] for row in publish_rows} == {1}
        _print_rows("3. Gather-to-rank-0 and rank-0-only publish", publish_rows)

        print("\nPASSED: the V7.3 communication order completed on CPU Gloo.")
        return 0
    finally:
        if process_group_ready:
            ray.get([worker.close.remote() for worker in workers])
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
