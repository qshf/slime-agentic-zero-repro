"""V5 行为验证：同步 vs 异步——异步 overlap 掉 train 耗时，总 wall-clock 更短。

核心断言（对齐 v5.md）：**同一份 Args、同样的模拟耗时下，异步 total < 同步 total**。
saving ≈ (num_rollout - 1) * fake_train_seconds——异步把除最后一轮外的每一轮 train(N) 都藏进了
gen(N+1) 的时间窗里。

时序公平性处理：
  - **预热 ray**：先跑一次 create_placement_groups，把 `ray.init`（~秒级）挪出计时区，否则"先跑的那条"
    会白背 init 成本，污染对照。
  - 预热后 sync/async 各自新建 actor（~ms，两条对称）+ 跑 loop，计时只包 train() 调用，差值由 overlap 主导。

用法:
  python scripts/test_v5_async.py --offline   # 本地: stub gen(fake_gen_seconds) + fake_train_seconds，确定性硬断言
  python scripts/test_v5_async.py             # 服务器: 真 SGLang gen + fake_train_seconds，观测报告
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from mini_slime.args import Args
from mini_slime.ray.placement_group import create_placement_groups
from mini_slime import train_async, train_ray


def _offline_args() -> Args:
    """离线：stub gen（不连 SGLang）+ 双旋钮制造可控耗时。

    num_rollout=3、gen=0.1s/条、train=0.2s/轮 → 理论 saving≈(3-1)*0.2=0.4s，margin ≫ ms 级噪声，
    确定性稳。gen 时间窗（batch_size*0.1=0.2s）≥ train（0.2s），train 可被完整藏住。
    """
    return Args(
        num_rollout=3,
        batch_size=2,
        fake_gen_seconds=0.1,
        fake_train_seconds=0.2,
        custom_generate_function_path="toy_rl.agent.stub_hooks.generate",
        custom_rm_path="toy_rl.agent.stub_hooks.reward_func",
    )


def _server_args() -> Args:
    """服务器：真 SGLang gen（默认 hook）+ 较大 fake_train_seconds，让 overlap 盖过 SGLang gen 方差。"""
    return Args(num_rollout=3, batch_size=2, fake_train_seconds=5.0)


def _assert_correct(tag: str, metrics_log: list[dict], args: Args) -> None:
    """两条路径共用的正确性断言：字段齐全、reward∈[0,1]、weight_version 符合 interval。"""
    assert len(metrics_log) == args.num_rollout, f"{tag}: 轮数 {len(metrics_log)} != {args.num_rollout}"
    for m in metrics_log:
        assert "reward_mean" in m and "weight_version" in m and "tokens_per_rollout" in m, f"{tag}: 缺字段"
        assert 0.0 <= m["reward_mean"] <= 1.0, f"{tag}: reward_mean 越界 {m['reward_mean']}"
        assert m["tokens_per_rollout"] > 0, f"{tag}: tokens 应 > 0"
    # 初始 update_weights(=1) + 每 interval 轮同步一次
    expected_v = 1 + args.num_rollout // args.update_weights_interval
    assert metrics_log[-1]["weight_version"] == expected_v, (
        f"{tag}: weight_version {metrics_log[-1]['weight_version']} != {expected_v}"
    )


def main() -> int:
    offline = "--offline" in sys.argv
    print(f"Running V5 tests ({'offline/stub' if offline else 'server/SGLang'})...")
    args = _offline_args() if offline else _server_args()

    # 预热 ray：把 ray.init 成本挪出下面两段计时区（否则先跑的一条白背 init）。
    create_placement_groups(args)

    failed = 0
    try:
        print("\n--- 同步基线 (train_ray, ≡ 源 train.py) ---")
        t0 = time.time()
        sync_metrics = train_ray.train(args)
        sync_total = time.time() - t0

        print("\n--- 异步 (train_async, ≡ 源 train_async.py) ---")
        t0 = time.time()
        async_metrics = train_async.train(args)
        async_total = time.time() - t0

        _assert_correct("sync", sync_metrics, args)
        _assert_correct("async", async_metrics, args)

        saving = sync_total - async_total
        print(
            f"\nsync_total={sync_total:.3f}s  async_total={async_total:.3f}s  "
            f"saving={saving:.3f}s (理论≈{(args.num_rollout - 1) * args.fake_train_seconds:.3f}s)"
        )

        # 头条断言：异步 total < 同步 total。离线确定性硬断言；服务器 gen 方差大，仅观测报告。
        if offline:
            assert async_total < sync_total, f"异步未更快: async={async_total:.3f} !< sync={sync_total:.3f}"
            print("  async_faster_than_sync OK")
        else:
            status = "OK" if async_total < sync_total else "观测未达（gen 方差内，见 v5.md）"
            print(f"  async_faster_than_sync {status}")
        print("  correctness OK")
    except AssertionError as e:
        failed += 1
        print(f"  FAIL: {e}")
    except Exception as e:
        failed += 1
        print(f"  FAIL: {type(e).__name__}: {e}")

    ray.shutdown()
    print(f"\n{'FAILED' if failed else 'PASSED'} ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
