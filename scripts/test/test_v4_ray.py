"""V4 行为验证：Ray 化闭环（actor 拆进程 + ray.get 同步点）。

断言（对齐 v4.md 验证清单）：
  1. 分进程：RolloutManager actor pid、TrainRayActor pid、主进程 pid **两两不同**。
  2. 闭环：跑 num_rollout 轮，metrics 字段齐全/类型对；reward_mean∈[0,1]；
     weight_version == num_rollout + 1（含训练前的初始 update_weights）。
  3. ObjectRef 流：async_train 返回 list[ObjectRef]，长度 == world_size；ray.get 可解析。
  4. train_data 契约：len 对齐 batch_size，每条 tokens/loss_mask 等长。

用法:
  python scripts/test_v4_ray.py --offline   # 本地: stub hook(不连 SGLang), 纯跑 ray 装配/闭环
  python scripts/test_v4_ray.py             # 服务器: 真 SGLang + ray 端到端
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime.args import Args
from mini_slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from mini_slime.train_ray import train


def _offline_args() -> Args:
    """离线：把 generate hook 指到不连 SGLang 的 stub（reward 仍用 calculator 规则打分）。"""
    return Args(
        num_rollout=2,
        batch_size=2,
        custom_generate_function_path="toy_rl.agent.stub_hooks.generate",
        custom_rm_path="toy_rl.agent.stub_hooks.reward_func",
    )


def test_separate_processes(args: Args) -> None:
    """不变量1: rollout_manager / trainer / 主进程 两两不同 PID。"""
    create_placement_groups(args)
    rm = create_rollout_manager(args)
    actor_model, _ = create_training_models(args)
    actor_model.async_init(args)

    main_pid = os.getpid()
    rm_pid = ray.get(rm.pid.remote())
    trainer_pids = actor_model.pids()

    print(f"  main_pid={main_pid} rollout_pid={rm_pid} trainer_pids={trainer_pids}")
    assert rm_pid != main_pid, "RolloutManager 应在独立进程"
    assert all(p != main_pid for p in trainer_pids), "TrainRayActor 应在独立进程"
    assert all(p != rm_pid for p in trainer_pids), "rollout 与 trainer 应在不同进程"
    print("  separate_processes OK")


def test_objectref_flow(args: Args) -> None:
    """不变量3: async_train 返回 list[ObjectRef]，长度==world_size，ray.get 可解析。"""
    create_placement_groups(args)
    rm = create_rollout_manager(args)
    actor_model, _ = create_training_models(args)
    actor_model.async_init(args)

    rollout_data = ray.get(rm.generate.remote(0))
    refs = actor_model.async_train(0, rollout_data)
    assert isinstance(refs, list) and len(refs) == 1, f"async_train 应返回 1 个 ObjectRef(DP=1)，得到 {len(refs)}"
    assert all(isinstance(r, ray.ObjectRef) for r in refs), "应是 ObjectRef"
    results = ray.get(refs)
    assert isinstance(results[0], dict) and "reward_mean" in results[0]

    # 不变量4: train_data 契约
    for k in ("tokens", "loss_masks", "rewards", "response_lengths"):
        assert len(rollout_data[k]) == args.batch_size, f"{k} 长度 != batch_size"
    for i in range(args.batch_size):
        assert len(rollout_data["tokens"][i]) == len(rollout_data["loss_masks"][i])
    print("  objectref_flow OK")


def test_full_loop(args: Args) -> None:
    """不变量2: 完整闭环, metrics 字段/类型/weight_version/reward 范围。"""
    metrics_log = train(args)
    assert len(metrics_log) == args.num_rollout
    for m in metrics_log:
        for k in ("gen_time", "train_time", "sync_time", "reward_mean", "tokens_per_rollout"):
            assert k in m, f"metrics 缺字段 {k}"
        assert isinstance(m["gen_time"], float) and isinstance(m["tokens_per_rollout"], int)
        assert m["tokens_per_rollout"] > 0
        assert 0.0 <= m["reward_mean"] <= 1.0, f"reward_mean 越界: {m['reward_mean']}"
    # 含训练前初始 update_weights，故 == num_rollout + 1
    expected = args.num_rollout + 1
    assert metrics_log[-1]["weight_version"] == expected, (
        f"weight_version {metrics_log[-1]['weight_version']} != {expected}(num_rollout+1)"
    )
    print("  full_loop OK")


def main() -> int:
    offline = "--offline" in sys.argv
    print(f"Running V4 tests ({'offline/stub' if offline else 'server/SGLang'})...")
    args = _offline_args() if offline else Args(num_rollout=2, batch_size=2)

    failed = 0
    # 三个测试复用同一个 ray runtime（create_placement_groups 幂等：ray.is_initialized 守卫）。
    for t in (test_separate_processes, test_objectref_flow, test_full_loop):
        try:
            t(args)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")

    ray.shutdown()
    print(f"\n{'FAILED' if failed else 'PASSED'} ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
