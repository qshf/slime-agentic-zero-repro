"""V3: WeightUpdater —— "谁负责把训练后权重同步回推理引擎" 的角色。

对齐源项目 —— actor.update_weights()（slime/backends/fsdp_utils/actor.py:725）内部委托给一个
权重同步机制（UpdateWeightFromDistributed / UpdateWeightFromTensor），把训练后权重 **bucket
-by-bucket 广播** 回 SGLang 推理引擎。这是"训练器 → 推理引擎"的桥。

独立成文件（而非并进 trainer.py）的理由：让主线一的**三个角色**（RolloutManager 产数据 /
Trainer 训 / WeightUpdater 同步权重）在文件层面就清晰可见——这正是 V3 的教学目标。源项目里
它也是独立的一层（actor 只是调用方）。

偏离说明（详见 docs/decisions/{v3,v6,v7}.md 偏离表）：
  - fake（V0-A3）：无真训练器 → 无真权重张量、无引擎句柄 → 无处广播。只 bump 版本号 + 计数。
  - V6.3 torch：走 **disk reload** 最小路径——训练后 save_pretrained 落盘 → SGLang
    /update_weights_from_disk 重载。源用 tensor/distributed bucket 广播（免落盘、更快），
    nano 取 disk reload 是因为不接管 SGLang 进程内存、单机最简可用；tensor 广播留 V7.4 加固。
  - V7.3 fsdp：save_pretrained 是**集体操作**（各 rank gather 分片到 rank0），故所有 rank 都要
    调它；但落盘 + SGLang POST + 权威 version 只 rank0 一次（对齐源 actor.py:744-746 的
    barrier + if rank==0）。传入 rank 做门控。
"""

from __future__ import annotations


class WeightUpdater:
    """训练后权重 → 推理引擎的同步器。对齐源 actor.update_weights 委托的权重同步机制。"""

    def __init__(self, save_path: str | None = None, torch_actor=None, rank: int = 0) -> None:
        self.version = 0      # 当前推理引擎上的权重版本号（每同步一次 +1）
        self.num_syncs = 0    # 累计同步次数（供主循环打点/断言）
        self.save_path = save_path      # V6.3 torch / V7.3 fsdp：落盘路径（None=fake，不落盘）
        self.torch_actor = torch_actor  # 持有可训模型（TorchActor / FSDPTrainer），用于 save_pretrained
        self.rank = rank                # V7.3 fsdp：本 rank（集体 save，但落盘/POST 只 rank0）
        self.generate_url = None        # SGLang /update_weights_from_disk 端点（可选）

    def update_weights(self, state_dict=None) -> int:
        """把训练后权重推回推理引擎。

        源: 遍历权重 bucket，逐块广播到 SGLang engine（update_weights_from_distributed）。
        fake: 不接收/广播真张量，只推进版本号并计数。
        torch（V6.3）: save_pretrained 落盘 → 若配了 generate_url 则触发 SGLang /update_weights_from_disk。
        fsdp（V7.3）: **所有 rank** 调 save_pretrained（集体 gather 分片到 rank0），但只 **rank0** 落盘
                     （save_pretrained 内部已 rank0 门控）+ **rank0** POST SGLang（避免重复 reload）。
        返回新的权重版本号（各 rank 一致递增，rank0 权威）。
        """
        if self.torch_actor is not None and self.save_path:
            # save_pretrained 对 fsdp 是集体操作：每个 rank 都必须进（内部只 rank0 真写盘）。
            self.torch_actor.save_pretrained(self.save_path)
            # SGLang 只有一个引擎、disk 上只有一份权重：只让 rank0 POST 一次触发 reload。
            if self.generate_url and self.rank == 0:
                self._reload_sglang()
        self.num_syncs += 1
        self.version += 1
        return self.version

    def _reload_sglang(self) -> None:
        """触发 SGLang 从磁盘热重载权重（对齐源"训练后引擎换新权重"语义的最小 disk 版）。"""
        import requests

        url = self.generate_url.replace("/generate", "/update_weights_from_disk")
        try:
            requests.post(url, json={"model_path": self.save_path}, timeout=120)
        except Exception as exc:  # noqa: BLE001 — 同步失败不该炸整个训练循环，记录即可
            print(f"[WeightUpdater] SGLang reload failed: {exc}")
