#!/usr/bin/env python3
"""V9.2: 在线端到端对比 —— 真实 agentic RL 闭环（rollout → train → update_weights）。

2×2 矩阵：trainer × 编排，四格都跑：
  A  无infra(TorchActor 单卡) + 同步 (train_ray.py)
  B  无infra(TorchActor 单卡) + 异步 (train_async.py)
  C  有infra(Megatron TP=2) + 同步 (train_ray.py)
  D  有infra(Megatron TP=2) + 异步 (train_async.py)

横向比(A↔B, C↔D): 异步 overlap 在真 trainer 下省了多少（V5 欠账）。
纵向比(A↔C, B↔D): Megatron 并行在 train 阶段快了多少（V9 主问题）。

验收::
  python scripts/bench/v9_end_to_end.py          # 跑全部四格
  python scripts/bench/v9_end_to_end.py --cell A # 只跑一格
  python scripts/bench/v9_end_to_end.py --cell C --repeats 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
MODEL_PATH = "/home/ubuntu/models/Qwen/Qwen3-0.6B"
SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_GEN_URL = "http://localhost:30000/generate"

NUM_ROUNDS = 3   # 含 warmup 第 0 轮；steady = [1:]（2 轮中位数）
WARMUP_ROUNDS = 1

CELLS: list[dict] = [
    dict(id="A", backend="torch",    mode="sync",  label="无infra+同步"),
    dict(id="B", backend="torch",    mode="async", label="无infra+异步"),
    dict(id="C", backend="megatron", mode="sync",  label="有infra+同步"),
    dict(id="D", backend="megatron", mode="async", label="有infra+异步"),
]

# 偏离说明：计划指定 AgentFlow 路径，但 AgentFlow 的具体模块路径在 V9 范围内未独立确认。
# 用 calculator（默认）保证两条路径 rollout 完全一致，不影响 trainer 对比的有效性。
# 上线前可通过 --gen-path / --rm-path 覆盖为真实 AgentFlow 路径。
DEFAULT_GEN_PATH = "toy_rl.agent.calculator_hooks.generate"
DEFAULT_RM_PATH = "toy_rl.agent.calculator_hooks.reward_func"
DEFAULT_DATA_PATH = "toy_rl.agent.calculator_data.load_data_source"


# ------------------------------------------------------------------
# Args 构造
# ------------------------------------------------------------------
def make_args(
    backend: str,
    gen_path: str = DEFAULT_GEN_PATH,
    rm_path: str = DEFAULT_RM_PATH,
    data_path: str = DEFAULT_DATA_PATH,
) -> "Args":
    """构造一格实验的 Args。backend 是唯一要换的旋钮（对齐方案 V9.2）。"""
    from mini_slime.args import Args

    args = Args(
        train_backend=backend,
        train_model_path=MODEL_PATH,
        num_rollout=NUM_ROUNDS,
        sglang_base_url=SGLANG_BASE_URL,
        sglang_generate_url=SGLANG_GEN_URL,
        custom_generate_function_path=gen_path,
        custom_rm_path=rm_path,
        data_source_path=data_path,
        # V9: fake 旋钮全关，真时间说话（V5 欠账的关键）
        fake_train_seconds=0.0,
        fake_gen_seconds=0.0,
        update_weights_interval=1,  # 先固定 1；后续扫 {1,2,4}
        megatron_qkv_format="thd",  # V9：Megatron 统一 sequence packing，禁止 BSHD padding
        # opt-in trace → 捕获 policy_version_gap（同步路径才有，异步路径补 0）
        learner_trace=True,
        learner_contract_validate=False,
    )
    if backend == "megatron":
        args.tensor_model_parallel_size = 2  # V9.2 基准并行度

    return args


# ------------------------------------------------------------------
# 单格实验
# ------------------------------------------------------------------
def run_cell(backend: str, mode: str, **kwargs) -> list[dict]:
    """跑 2×2 矩阵的一格。编排代码复用 V5，不新写（对齐方案 V9.2）。

    backend: "torch"（无 infra，单卡）| "megatron"（有 infra，TP=2）
    mode:    "sync"（train_ray.py ≡ 源 train.py）| "async"（train_async.py ≡ 源 train_async.py）
    """
    import ray

    if ray.is_initialized():
        ray.shutdown()

    args = make_args(backend, **kwargs)

    if mode == "sync":
        from mini_slime.train_ray import train
    else:
        from mini_slime.train_async import train

    metrics_log = train(args)
    ray.shutdown()
    return metrics_log


# ------------------------------------------------------------------
# 结果聚合
# ------------------------------------------------------------------
class CellResult(NamedTuple):
    wait_gen_median: float   # 异步路径等 gen 的中位数（同步路径 = gen_time）
    train_median: float      # train 中位数
    total: float             # 全程 wall-clock（含 warmup）
    reward_mean: float       # steady 轮 reward 均值
    policy_version_gap: float  # staleness（异步>0，同步=0）
    wait_time_ratio: float   # wait_gen / (wait_gen + train)


def _extract_metrics(metrics_log: list[dict], warmup: int = WARMUP_ROUNDS) -> CellResult:
    """从一次 run 的 metrics_log 提取 steady 指标。

    train_ray 产 gen_time/train_time/sync_time；train_async 产 wait_gen_time/train_time。
    对齐方案 §2 的 wait_time_ratio = train_wait_time / (train_wait_time + train_time)。
    """
    steady = metrics_log[warmup:] if len(metrics_log) > warmup else metrics_log

    # wait_gen: 同步路径用 gen_time（包含真实生成时间），异步路径用 wait_gen_time（理想=0）
    wait_gen_vals = [m.get("wait_gen_time", m.get("gen_time", 0.0)) for m in steady]
    train_vals = [m["train_time"] for m in steady]
    gap_vals = [m.get("policy_version_gap", 0.0) for m in steady]
    reward_vals = [m["reward_mean"] for m in steady]

    wg = float(np.median(wait_gen_vals))
    tr = float(np.median(train_vals))
    ratio = wg / (wg + tr) if (wg + tr) > 0 else 0.0

    total = sum(
        m.get("wait_gen_time", m.get("gen_time", 0.0)) + m["train_time"]
        + m.get("sync_time", 0.0)
        for m in metrics_log
    )

    return CellResult(
        wait_gen_median=wg,
        train_median=tr,
        total=total,
        reward_mean=float(np.mean(reward_vals)),
        policy_version_gap=float(np.median(gap_vals)),
        wait_time_ratio=ratio,
    )


# ------------------------------------------------------------------
# 汇报 + 图
# ------------------------------------------------------------------
def print_table(results: dict[str, CellResult]) -> None:
    """打印 2×2 对比表（对齐方案 §2 的横向/纵向比）。"""
    header = f"{'格':12s} {'wait_gen(s)':>12} {'train(s)':>10} {'total(s)':>10} " \
             f"{'reward':>8} {'gap':>6} {'wait_ratio':>12}"
    print("\n" + header)
    print("-" * len(header))
    for label, r in results.items():
        print(
            f"{label:12s} {r.wait_gen_median:12.3f} {r.train_median:10.3f} "
            f"{r.total:10.2f} {r.reward_mean:8.3f} {r.policy_version_gap:6.1f} "
            f"{r.wait_time_ratio:12.3f}"
        )

    # 横向 / 纵向对比摘要
    if all(k in results for k in ("A 无infra+同步", "B 无infra+异步")):
        a, b = results["A 无infra+同步"], results["B 无infra+异步"]
        print(f"\n  横向（无infra）: async/sync total = {b.total/a.total:.3f}×  "
              f"(B {b.total:.1f}s / A {a.total:.1f}s)")
    if all(k in results for k in ("C 有infra+同步", "D 有infra+异步")):
        c, d = results["C 有infra+同步"], results["D 有infra+异步"]
        print(f"  横向（有infra）: async/sync total = {d.total/c.total:.3f}×  "
              f"(D {d.total:.1f}s / C {c.total:.1f}s)")
    if all(k in results for k in ("A 无infra+同步", "C 有infra+同步")):
        a, c = results["A 无infra+同步"], results["C 有infra+同步"]
        print(f"  纵向（同步）: megatron/torch train = {c.train_median/a.train_median:.3f}×  "
              f"(C {c.train_median:.3f}s / A {a.train_median:.3f}s)")
    if all(k in results for k in ("B 无infra+异步", "D 有infra+异步")):
        b, d = results["B 无infra+异步"], results["D 有infra+异步"]
        print(f"  纵向（异步）: megatron/torch train = {d.train_median/b.train_median:.3f}×  "
              f"(D {d.train_median:.3f}s / B {b.train_median:.3f}s)")


def save_csv(results: dict[str, CellResult], path: str) -> None:
    import csv

    fields = CellResult._fields
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + list(fields))
        for label, r in results.items():
            w.writerow([label] + list(r))
    print(f"CSV → {path}")


def plot_2x2(results: dict[str, CellResult], path: str) -> None:
    """2×2 堆叠条形图：wait_gen + train 各一色（对齐方案 §2 图示）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图。")
        return

    labels = list(results.keys())
    wait_vals = [results[l].wait_gen_median for l in labels]
    train_vals = [results[l].train_median for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, wait_vals, label="wait_gen", color="#4c8cbf")
    ax.bar(x, train_vals, bottom=wait_vals, label="train", color="#e07b39")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("median seconds (steady rounds)")
    ax.set_title("V9.2 端到端对比: 2×2 矩阵")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"图 → {path}")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="V9.2 在线端到端对比")
    parser.add_argument(
        "--cell", choices=["A", "B", "C", "D"], default=None,
        help="只跑指定格（默认跑全部四格）"
    )
    parser.add_argument("--repeats", type=int, default=3, help="每格重复次数（取中位数）")
    parser.add_argument(
        "--gen-path", default=DEFAULT_GEN_PATH,
        help="generate hook 路径（默认 calculator）"
    )
    parser.add_argument(
        "--rm-path", default=DEFAULT_RM_PATH,
        help="reward_func hook 路径（默认 calculator）"
    )
    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH,
        help="data_source hook 路径（默认 calculator）"
    )
    parser.add_argument(
        "--out-dir", default="docs/decisions",
        help="CSV + 图的输出目录"
    )
    a = parser.parse_args()

    cells_to_run = [c for c in CELLS if a.cell is None or c["id"] == a.cell]
    hook_kwargs = dict(gen_path=a.gen_path, rm_path=a.rm_path, data_path=a.data_path)

    # 先打印预期（V9 实验纪律：先测后编解释是自欺）
    print("=== V9.2 在线端到端对比 ===\n")
    print("预设预期（基于 V9 方案）：")
    print("  B < A（无infra 异步 < 同步）: gen 是大头，overlap 应藏掉几乎整个 train。")
    print("  D vs C 收益 < B vs A: Megatron train 更快，能藏的更少，overlap 边际收益递减。")
    print("  C vs A speedup << 2×: 0.6B 计算量小、TP all-reduce 通信占比高、无 NVLink。")
    print("  wait_gen ≈ 0 只在异步格，同步格 wait_gen = 真实 gen 时间。")
    print("  policy_version_gap: 异步=1，同步=0。\n")

    results: dict[str, CellResult] = {}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cell in cells_to_run:
        label = f"{cell['id']} {cell['label']}"
        print(f"\n{'='*50}")
        print(f"格 {label}  (backend={cell['backend']}, mode={cell['mode']}, repeats={a.repeats})")
        print(f"{'='*50}")

        runs: list[CellResult] = []
        for rep in range(a.repeats):
            print(f"\n  -- repeat {rep + 1}/{a.repeats} --")
            t_wall = time.time()
            metrics_log = run_cell(cell["backend"], cell["mode"], **hook_kwargs)
            elapsed = time.time() - t_wall
            cr = _extract_metrics(metrics_log)
            runs.append(cr)
            print(
                f"  repeat {rep + 1}: total={cr.total:.2f}s  "
                f"wait_gen={cr.wait_gen_median:.3f}s  train={cr.train_median:.3f}s  "
                f"reward={cr.reward_mean:.3f}  gap={cr.policy_version_gap:.0f}  "
                f"wall={elapsed:.1f}s"
            )

        # 取中位数（共享 GPU 噪声大）
        median_result = CellResult(
            wait_gen_median=float(np.median([r.wait_gen_median for r in runs])),
            train_median=float(np.median([r.train_median for r in runs])),
            total=float(np.median([r.total for r in runs])),
            reward_mean=float(np.median([r.reward_mean for r in runs])),
            policy_version_gap=float(np.median([r.policy_version_gap for r in runs])),
            wait_time_ratio=float(np.median([r.wait_time_ratio for r in runs])),
        )
        results[label] = median_result

    # 输出
    print_table(results)
    csv_path = str(out_dir / "v9_end_to_end.csv")
    png_path = str(out_dir / "v9_end_to_end.png")
    save_csv(results, csv_path)
    plot_2x2(results, png_path)


if __name__ == "__main__":
    main()
