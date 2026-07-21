"""V3: WeightUpdater —— "谁负责把训练后权重同步回推理引擎" 的角色。

对齐源项目 —— actor.update_weights()（slime/backends/fsdp_utils/actor.py:725）内部委托给一个
权重同步机制（UpdateWeightFromDistributed / UpdateWeightFromTensor），把训练后权重 **bucket
-by-bucket 广播** 回 SGLang 推理引擎。这是"训练器 → 推理引擎"的桥。

独立成文件（而非并进 trainer.py）的理由：让主线一的**三个角色**（RolloutManager 产数据 /
Trainer 训 / WeightUpdater 同步权重）在文件层面就清晰可见——这正是 V3 的教学目标。源项目里
它也是独立的一层（actor 只是调用方）。

偏离说明（详见 docs/decisions/v3.md 偏离表）：
  - fake：无真训练器 → 无真权重张量，无真 RolloutManager 引擎句柄 → 无处广播。
    V3 只 bump 版本号 + 记"第几次同步"，供主循环打点。
  - 真 bucket-by-bucket 广播到 SGLang 留 **V6+**（需真训练 backend + 真引擎）。
"""

from __future__ import annotations


class WeightUpdater:
    """训练后权重 → 推理引擎的同步器。对齐源 actor.update_weights 委托的权重同步机制（fake）。"""

    def __init__(self) -> None:
        self.version = 0      # 当前推理引擎上的权重版本号（每同步一次 +1）
        self.num_syncs = 0    # 累计同步次数（供主循环打点/断言）

    def update_weights(self, state_dict=None) -> int:
        """把训练后权重推回推理引擎。

        源: 遍历权重 bucket，逐块广播到 SGLang engine（update_weights_from_distributed）。
        V3 fake: 不接收/广播真张量（state_dict 恒为 None），只推进版本号并计数。
        返回新的权重版本号。
        """
        self.num_syncs += 1
        self.version += 1
        return self.version
