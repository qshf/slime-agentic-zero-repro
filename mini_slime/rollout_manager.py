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
  - 硬编码小题库 data_source：源 data_source_cls(args) 从数据集加载，nano 用几条 calculator 题。
"""

from __future__ import annotations

from mini_slime.args import Args
from mini_slime.hooks import load_function
from toy_rl.sample import Sample


# --- data_source（对齐源 self.data_source，V3 用硬编码小题库）------------------------
# 源: data_source_cls = load_function(args.data_source_path); self.data_source = data_source_cls(args)
# V3: 只需几条能跑通闭环的 calculator 题；(prompt, label) 元组列表即可。
# 题目难度：4B 心算就能答对小算术（会绕过工具、tool 路径练不到），故换成大数乘除/多步
# 表达式——超出可靠心算范围，逼模型真去调 calculator。label 均由 eval 校验过。
_PROMPTS: list[tuple[str, str]] = [
    ("347 * 89 = ?", "30883"),
    ("638 * 47 = ?", "29986"),
    ("72 * 84 = ?", "6048"),
    ("(123 + 456) * 7 = ?", "4053"),
    ("1024 / 16 = ?", "64"),
    ("9876 - 5432 = ?", "4444"),
]


def _load_prompts() -> list[tuple[str, str]]:
    """对齐源 data_source 加载（V3 返回硬编码题库）。"""
    return list(_PROMPTS)


class RolloutManager:
    """产数据的角色。对齐源 slime/ray/rollout.py:441 RolloutManager（去 Ray/DP）。"""

    def __init__(self, args: Args) -> None:
        self.args = args
        self.data_source = _load_prompts()
        # 对齐源 rollout.py:455-457：generate / reward 都由 load_function 按路径动态加载。
        self.generate_rollout = load_function(args.custom_generate_function_path)
        self.reward_func = load_function(args.custom_rm_path)

    def _next_batch(self, rollout_id: int) -> list[tuple[str, str]]:
        """按 rollout_id 从 data_source 滚动取一个 batch（对齐源按 rollout_batch_size 取窗口）。

        源用全局 dataset 指针；V3 用 (rollout_id * batch_size) 起点 + 取模环绕，
        保证每轮都能取满 batch_size 条（题库小于 num_rollout*batch_size 时循环复用）。
        """
        n = self.args.batch_size
        start = (rollout_id * n) % len(self.data_source)
        return [self.data_source[(start + i) % len(self.data_source)] for i in range(n)]

    async def generate(self, rollout_id: int) -> dict:
        """对齐源 rollout.py:539 def generate(rollout_id)：产出一批训练数据 dict。

        源: _get_rollout_data → _convert_samples_to_train_data → _split_train_data_by_dp
        V3: 逐条跑 generate hook（真 SGLang rollout）+ reward hook → _convert（不切 DP）
        """
        samples: list[Sample] = []
        for prompt, label in self._next_batch(rollout_id):
            s = Sample(prompt=prompt, label=label)
            s = await self.generate_rollout(self.args, s)              # 真 SGLang rollout（复用 V1/V2 loop）
            s.reward = (await self.reward_func(self.args, s))["reward"]  # per-sample reward hook
            samples.append(s)
        return self._convert_samples_to_train_data(samples)

    def _convert_samples_to_train_data(self, samples: list[Sample]) -> dict:
        """对齐源 rollout.py:737 —— **V3 最核心的教学点**：训练器不认识 Sample，只认字段对齐的 dict。

        Sample 列表 → dict，是 RolloutManager 交给 Trainer 的唯一契约。字段与源一致（子集）：
          源: tokens / response_lengths / rewards / raw_reward / truncated / sample_indices / loss_masks
          V3: tokens / loss_masks / rewards / response_lengths（其余留后续）

        偏离（详见 v3.md 偏离表）：
          - response_lengths 用**全长 len(tokens)**：nano tokens 含 prompt 段（loss_mask=0），
            源的 response_length 只算 response 部分。结构占位等价，真 response 边界留 V6。
          - 无 rollout_log_probs：V1/V2 打 chat 端点拿不到真 log_probs，留 V6 接 /generate。
        """
        return {
            "tokens": [s.tokens for s in samples],
            "loss_masks": [s.loss_mask for s in samples],
            "rewards": [s.reward for s in samples],
            "response_lengths": [len(s.tokens) for s in samples],
        }
