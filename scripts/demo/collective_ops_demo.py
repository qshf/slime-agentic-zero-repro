"""Minimal torch.distributed collective demo.

Run on CPU:
    torchrun --standalone --nproc_per_node=4 scripts/collective_ops_demo.py

Run on GPUs:
    torchrun --standalone --nproc_per_node=4 scripts/collective_ops_demo.py \
        --backend nccl --device cuda
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"This visual demo expects 4 ranks, got {world_size}")

    def local_value(value: int) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.int64, device=device)

    def print_from_rank0(message: str) -> None:
        dist.barrier()
        if rank == 0:
            print(message, flush=True)
        dist.barrier()

    # Every rank starts with its own value: rank 0..3 -> 1..4.
    x = local_value(rank + 1)
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"reduce(sum, dst=0): rank0={x.item()}", flush=True)
    dist.barrier()

    x = local_value(100 if rank == 0 else rank + 1)
    dist.broadcast(x, src=0)
    print_from_rank0(f"broadcast(src=0): all_ranks={x.item()}")

    x = local_value(rank + 1)
    gathered = [torch.empty_like(x) for _ in range(world_size)] if rank == 0 else None
    dist.gather(x, gather_list=gathered, dst=0)
    if rank == 0:
        values = [int(t.item()) for t in gathered]
        print(f"gather(dst=0): rank0={values}", flush=True)
    dist.barrier()

    x = torch.empty(1, dtype=torch.int64, device=device)
    scatter_list = [local_value(10 * (i + 1)) for i in range(world_size)] if rank == 0 else None
    dist.scatter(x, scatter_list=scatter_list, src=0)
    print_from_rank0(f"scatter(src=0): all_ranks_receive=[10, 20, 30, 40]")

    x = local_value(rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print_from_rank0(f"all_reduce(sum): all_ranks={x.item()}")

    x = local_value(rank + 1)
    parts = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(parts, x)
    values = [int(t.item()) for t in parts]
    print_from_rank0(f"all_gather: all_ranks={values}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
