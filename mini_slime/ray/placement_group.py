"""V4: Ray 装配工厂 —— 对齐源 slime/ray/placement_group.py 的三个工厂函数。

源项目 train.py:9-24 的装配三步：
  pgs = create_placement_groups(args)                                # 分配 GPU bundle + 让 ray 就绪
  rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])  # 单个 @ray.remote actor
  actor_model, _ = create_training_models(args, pgs, rollout_manager)# RayTrainGroup(本地包装)

nano V4 保留这三个工厂的**名字与角色**，但按偏离表去掉 GPU placement group（fake trainer 不占
GPU、SGLang 在独立 docker 里不归 ray 调度）。详见 docs/decisions/v4.md 偏离表。
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

import ray

from mini_slime.args import Args
from mini_slime.ray.actor_group import RayTrainGroup
from mini_slime.rollout_manager import RolloutManager

# 项目根：让 ray worker 进程能 import mini_slime / toy_rl（worker 不继承 driver 运行时的
# sys.path.insert，故通过 runtime_env 的 PYTHONPATH 显式传入——单机本地 ray，文件走文件系统）。
# V6.3：还要带上 driver 的 site-packages（venv），否则 ray worker 只有项目根、import 不到
# torch/transformers（V6.3 torch 训练在 worker actor 里跑）。用 driver 现有 sys.path 拼进去。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_SITE_PACKAGES = [p for p in (site.getsitepackages() if hasattr(site, "getsitepackages") else []) + sys.path if "site-packages" in p]
_WORKER_PYTHONPATH = ":".join([_PROJECT_ROOT, *dict.fromkeys(_SITE_PACKAGES)])


def create_placement_groups(args: Args) -> dict:
    """对齐源 create_placement_groups（placement_group.py:79）：分配资源 + 让 ray 就绪。

    偏离（见 v4.md）：源用 `PlacementGroupSchedulingStrategy` 把 actor/engine 钉到
    `{"GPU":1,"CPU":1}` bundle；nano fake/torch trainer 无需 Ray 调度 GPU，故 `ray.init(num_gpus=0)`。
    V7.3 fsdp 后端需 Ray 给训练 actor 分卡 → `ray.init(num_gpus=fsdp_world_size)`（让 Ray 看得到卡，
    每个 num_gpus=1 的 actor 独占一张；driver 侧用 CUDA_VISIBLE_DEVICES 限定到 SGLang 未占的空闲卡）。
    """
    if not ray.is_initialized():
        # fsdp/megatron 后端要 Ray 调度 GPU 给训练 actor；fake/torch 无 GPU 调度需求（=0）。
        # megatron world_size = TP × PP × DP（对齐 actor_group.py）。
        if args.train_backend == "fsdp":
            num_gpus = args.fsdp_world_size
        elif args.train_backend == "megatron":
            num_gpus = (
                args.tensor_model_parallel_size
                * args.pipeline_model_parallel_size
                * args.megatron_data_parallel_size
            )
        else:
            num_gpus = 0
        ray.init(
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            log_to_driver=False,  # 压掉 worker 日志刷屏，测试输出更干净
            runtime_env={"env_vars": {"PYTHONPATH": _WORKER_PYTHONPATH}},
        )
    # 占位：nano 无 GPU bundle。源这里是 {"actor": (pg, idx, gpu_ids), "rollout": (...)}。
    return {"actor": None, "rollout": None}


def create_rollout_manager(args: Args):
    """对齐源 create_rollout_manager（placement_group.py:181）：把 RolloutManager 建成单 actor。

    偏离（见 v4.md）：源直接 `@ray.remote class RolloutManager`；nano 用 `ray.remote(cls)` 在**构造时**
    包装，让**同一个类**既能被 V3 单进程 train.py 直接用、又能在 V4 包成 actor——一份类两种用法，
    不重写。返回 actor handle（源还返回 num_rollout_per_epoch，nano 用固定 num_rollout，故省略）。
    """
    RolloutManagerActor = ray.remote(num_cpus=1, num_gpus=0)(RolloutManager)
    return RolloutManagerActor.remote(args)


def create_training_models(args: Args) -> tuple[RayTrainGroup, None]:
    """对齐源 create_training_models（placement_group.py:132）：actor_model = RayTrainGroup(...)。

    critic 已按路线图砍掉（GRPO 式无 critic），返回 None 占位对齐源的 (actor, critic) 二元组。
    源还传 pgs / rollout_manager 做 GPU 调度与句柄互联；nano fake 无需，故签名收敛为只吃 args。
    """
    actor_model = RayTrainGroup(args)
    return actor_model, None
