"""V9.0: Timer —— 对齐源 slime/utils/timer.py。

偏离（docs/decisions/v9.md 偏离表）：
  - SingletonMeta 从源 misc.py 内联进来（nano 无 misc 模块）。
  - 省略 inverse_timer / with_defer（V9 不用这两个工具）。
  - 其余逻辑与源完全对齐：start/end/add/log_dict/context + timer 装饰/上下文两用。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps
from time import time

import torch.distributed

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """对齐源 slime/utils/misc.py:SingletonMeta。"""

    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Timer(metaclass=SingletonMeta):
    """进程级单例计时器。对齐源 slime/utils/timer.py:Timer。

    单例保证同一进程里任何地方 Timer() 拿到同一个实例，累积计时不会分散。
    各埋点用 start/end 包夹，内部累加到 timers dict；log_dict() 一次性读出。
    同步语义：train_batch 的调用方负责在关键点前后加 cuda.synchronize()（计划见 v9.md），
    Timer 本身不做同步——对齐源。
    """

    def __init__(self):
        self.timers: dict[str, float] = {}
        self.start_time: dict[str, float] = {}

    def start(self, name: str) -> None:
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} start")

    def end(self, name: str) -> None:
        assert name in self.start_time, f"Timer {name} not started."
        elapsed = time() - self.start_time[name]
        self.add(name, elapsed)
        del self.start_time[name]
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} end (elapsed: {elapsed:.1f}s)")

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self.timers = {}
            # A caller may return early after start() (for example an empty
            # GRPO microbatch). A full reset starts a new measurement epoch,
            # so stale open scopes must not poison the next train step.
            self.start_time = {}
        else:
            self.timers.pop(name, None)
            self.start_time.pop(name, None)

    def add(self, name: str, elapsed_time: float) -> None:
        self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def log_dict(self) -> dict[str, float]:
        return self.timers

    @contextmanager
    def context(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


def timer(name_or_func):
    """两用：既可作装饰器（@timer），也可作上下文管理器（with timer("name")）。

    对齐源 slime/utils/timer.py:timer。
    """
    if isinstance(name_or_func, str):
        return Timer().context(name_or_func)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper
