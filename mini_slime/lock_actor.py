"""Ray lock actor：防止并发 NCCL broadcast 死锁。

对齐源：slime 没有单独文件，但在 update_weight_from_distributed.py:235 用
rollout_engine_lock.acquire/release 防止多个 PP stage 同时 broadcast。

为什么需要 lock：
  - NCCL broadcast 是集体操作：所有参与 rank 必须同步进入
  - 如果多个训练进程同时向同一个 SGLang engine broadcast，会死锁
  - 用 Ray lock 串行化 broadcast 请求

简化说明（nano vs 源）：
  - 源项目每个 PP stage 建独立 group，但仍需 lock 防止跨 group 冲突
  - nano 当前只有一个 group，但仍保留 lock（架构对齐 + 未来扩展 PP）
"""

import ray


@ray.remote
class LockActor:
    """分布式锁（Ray actor），防止并发 NCCL 操作死锁。"""

    def __init__(self):
        self._locked = False

    def acquire(self) -> bool:
        """尝试获取锁。返回 True 表示获取成功。"""
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        """释放锁。"""
        self._locked = False
