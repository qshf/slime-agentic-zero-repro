"""V3: RolloutManager —— "谁负责产数据" 的角色。

对齐源项目 slime/ray/rollout.py 的 RolloutManager（:441），但**去掉 Ray 与 DP 切分**（那是 V4）：
  源: @ray.remote 类, generate(rollout_id) 内 _get_rollout_data → _convert_samples_to_train_data
      → _split_train_data_by_dp
  V3: 普通类, generate(rollout_id) 内 逐条跑 generate hook → reward hook
      → _convert_samples_to_train_data（不切 DP）

命名对齐源项目（rollout.py:455 / :539）——这是本版对 v3.md 计划草稿的一处**朝源项目更靠拢的修正**：
  - self.generate_rollout : 动态加载的 generate hook（源 rollout.py:455 self.generate_rollout）
  - def generate(rollout_id): RolloutManager 的编排入口（源 rollout.py:539 def generate）
  v3.md 草稿曾把两者命名反了（hook=generate / 方法=generate_rollout）；按铁律"对齐源项目对应
  位置的写法"，这里改回与源一致的命名。

偏离说明（详见 docs/decisions/v3.md 偏离表）：
  - 无 Ray / 无 DP 切分：V3 单进程 dp_size=1，编排调用序列与源等价。
  - generate hook 是 **per-sample**（generate(args, sample)->sample），而源的 rollout_function
    是 **batch 级**（一次跑完整个 dataset）。nano 把"遍历 batch"这层放在 RolloutManager 里，
    源放在 custom rollout function 里。语义等价（都产出一批 Sample），只是分层位置不同——nano
    这样分层能让 per-sample 的 agent loop 更聚焦（V1/V2 已定的 per-sample 契约）。
  - 硬编码小题库 data_source：源 data_source_cls(args) 从数据集加载。A1 起 nano 也把它做成
    `data_source_path` hook（可切换 calculator / MemAgent 数据源）；各 loader 内仍是硬编码小集
    （见 toy_rl/agent/calculator_data.py、toy_rl/agent/memagent/data.py）。详见 docs/decisions/a1.md。
"""

from __future__ import annotations

import copy

from mini_slime.args import Args
from mini_slime.hooks import load_function
from toy_rl.sample import Sample


class RolloutManager:
    """产数据的角色。对齐源 slime/ray/rollout.py:441 RolloutManager（去 Ray/DP）。"""

    def __init__(self, args: Args) -> None:
        self.args = args
        # 对齐源 rollout.py:455-457：data_source / generate / reward 都由 load_function 按路径动态加载。
        # A1 起数据源可切换（对齐源 data_source_cls）：calculator 走 calculator_data，MemAgent 走 memagent/data。
        self.data_source = load_function(args.data_source_path)(args)  # -> list[Sample]
        self.generate_rollout = load_function(args.custom_generate_function_path)
        self.reward_func = load_function(args.custom_rm_path)

    def _next_batch(self, rollout_id: int) -> list[Sample]:
        """按 rollout_id 从 data_source 滚动取一个 batch（对齐源按 rollout_batch_size 取窗口）。

        源用全局 dataset 指针；V3 用 (rollout_id * batch_size) 起点 + 取模环绕，
        保证每轮都能取满 batch_size 条（题库小于 num_rollout*batch_size 时循环复用）。
        返回**深拷贝**：generate hook 会就地回填 Sample，拷贝避免污染 data_source 供下一轮复用。
        """
        n = self.args.batch_size
        start = (rollout_id * n) % len(self.data_source)
        return [copy.deepcopy(self.data_source[(start + i) % len(self.data_source)]) for i in range(n)]

    async def generate(self, rollout_id: int) -> dict:
        """对齐源 rollout.py:539 def generate(rollout_id)：产出一批训练数据 dict。

        源: _get_rollout_data → _convert_samples_to_train_data → _split_train_data_by_dp
        V3: 逐条跑 generate hook（真 SGLang rollout）+ reward hook → _convert（不切 DP）
        A1: 数据源直接给 Sample（可能带 metadata["context"]），hook 就地回填 tokens/loss_mask/reward。
        """
        samples: list[Sample] = []
        for s in self._next_batch(rollout_id):
            s = await self.generate_rollout(self.args, s)              # 真 SGLang rollout（复用 V1/V2 loop）
            s.reward = (await self.reward_func(self.args, s))["reward"]  # per-sample reward hook
            samples.append(s)
        return self._convert_samples_to_train_data(samples)

    def _convert_samples_to_train_data(self, samples: list[Sample]) -> dict:
        """对齐源 rollout.py:737 —— **V3 最核心的教学点**：训练器不认识 Sample，只认字段对齐的 dict。

        Sample 列表 → dict，是 RolloutManager 交给 Trainer 的唯一契约。字段与源一致（子集）：
          源: tokens / response_lengths / rewards / raw_reward / truncated / sample_indices / loss_masks
          V3: tokens / loss_masks / rewards / response_lengths（其余留后续）

        V6.1 起补齐两处 V3 登记的偏离：
          - response_lengths 用**真 response 段长度**（sum(loss_mask)）：agent 的 loss_mask 里 1 的
            个数就是 response token 数（prompt/tool 段为 0）。补上 V3 的"全长占位"偏离。
          - rollout_log_probs：orchestrator 走 /generate 后 sample.rollout_log_probs 有真值；
            旧版（chat 端点）留空 [] → 该字段对旧 agent 透明（trainer 只在有值时用）。
        """
        return {
            "tokens": [s.tokens for s in samples],
            "loss_masks": [s.loss_mask for s in samples],
            "rewards": [s.reward for s in samples],
            "response_lengths": [sum(s.loss_mask) if s.loss_mask else len(s.tokens) for s in samples],
            "rollout_log_probs": [s.rollout_log_probs for s in samples],
        }

    def pid(self) -> int:
        """当前进程 PID。V4 起把本类包成 @ray.remote actor（见 mini_slime/ray/placement_group.py），
        测试用它确认 RolloutManager 与 Trainer/主进程在**不同进程**。V3 单进程用不到。"""
        import os

        return os.getpid()
