"""Ray Debugger 快速演示。

用法：
  1. 先启动 Ray head：ray start --head
  2. 再运行：python scripts/ray_debug_demo.py
  3. 也可测试 post-mortem：python scripts/ray_debug_demo.py raise

这个脚本会在 Ray 任务里触发 breakpoint()，方便你用 VS Code 的 Ray Debugger 附加。
"""

from __future__ import annotations

import os
import sys

import ray


os.environ.setdefault("RAY_DEBUG", "1")


@ray.remote
def debug_task(x: int) -> int:
    print(f"[task] pid={os.getpid()} x={x}")
    breakpoint()  # 这里会暂停，等待 VS Code 附加
    return x * x


@ray.remote
def post_mortem_task(x: int) -> int:
    print(f"[post-mortem] pid={os.getpid()} x={x}")
    raise RuntimeError("故意抛异常，验证 Ray post-mortem debugger")


if __name__ == "__main__":
    ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "1"}})
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "raise":
            ray.get(post_mortem_task.remote(7))
        else:
            result = ray.get(debug_task.remote(9))
            print(f"result={result}")
    finally:
        ray.shutdown()
