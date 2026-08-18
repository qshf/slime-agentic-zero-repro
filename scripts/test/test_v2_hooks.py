"""V2 行为验证：hook 动态加载机制。

验证:
1. load_function 能按 "module.path.func" 字符串加载到函数对象(离线)。
2. load_function 对错误路径报错(离线)。
3. 加载到的 generate/reward 签名正确、能被主循环无差别调用(离线，用桩 sample)。
4. 端到端: 通过配置路径加载 generate+reward, 跑真实 rollout(需 SGLang)。

用法:
  python scripts/test_v2_hooks.py --offline   # 本地
  python scripts/test_v2_hooks.py             # 服务器(需 SGLang)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "toy_rl").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError(f"Could not locate project root from {_HERE}")

from mini_slime.hooks import load_function
from toy_rl.sample import Sample


def test_load_function() -> None:
    """不变量1: 按路径加载到真身。"""
    gen = load_function("toy_rl.agent.calculator_hooks.generate")
    rm = load_function("toy_rl.agent.calculator_hooks.reward_func")
    assert callable(gen) and gen.__name__ == "generate"
    assert callable(rm) and rm.__name__ == "reward_func"
    print("  load_function OK")


def test_load_function_bad_path() -> None:
    """不变量2: 错误路径报错。"""
    for bad in ("nonexistent", "toy_rl.agent.calculator_hooks.no_such_func"):
        raised = False
        try:
            load_function(bad)
        except (ValueError, ImportError, AttributeError):
            raised = True
        assert raised, f"错误路径 {bad!r} 应报错"
    print("  load_function_bad_path OK")


async def test_reward_via_hook() -> None:
    """不变量3: 通过 hook 加载的 reward_func 能对桩 sample 打分(离线)。"""
    from toy_rl.agent.calculator_hooks import Args
    rm = load_function("toy_rl.agent.calculator_hooks.reward_func")
    args = Args()
    s = Sample(prompt="2+3=?", label="5", response="<answer>5</answer>")
    out = await rm(args, s)
    assert isinstance(out, dict) and out["reward"] == 1.0, f"reward dict 不符: {out}"
    s2 = Sample(prompt="2+3=?", label="5", response="<answer>6</answer>")
    out2 = await rm(args, s2)
    assert out2["reward"] == 0.0
    print("  reward_via_hook OK")


async def test_end_to_end() -> None:
    """不变量4: 通过配置路径加载 generate+reward, 跑真实 rollout(需 SGLang)。"""
    from toy_rl.agent.calculator_hooks import Args
    args = Args()
    generate = load_function(args.custom_generate_function_path)
    reward_func = load_function(args.custom_rm_path)

    sample = Sample(prompt="2 + 3 = ?", label="5")
    sample = await generate(args, sample)          # 主循环不认识 calculator, 只调 hook
    reward_out = await reward_func(args, sample)
    sample.reward = reward_out["reward"]

    print("\n  --- 端到端 trajectory (hook 加载) ---")
    print(f"  response: {sample.response!r}")
    print(f"  tokens: {len(sample.tokens)}, trainable: {sample.trainable_token_count()}")
    print(f"  reward: {sample.reward}")

    assert len(sample.tokens) == len(sample.loss_mask)
    assert sample.reward in (0.0, 1.0)
    print("  end_to_end OK")


def main() -> int:
    offline = "--offline" in sys.argv
    print("Running V2 tests...")
    failed = 0

    for t in (test_load_function, test_load_function_bad_path):
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")

    try:
        asyncio.run(test_reward_via_hook())
    except AssertionError as e:
        failed += 1
        print(f"  FAIL reward_via_hook: {e}")

    if offline:
        print("\n(--offline: 跳过需 SGLang 的 end_to_end)")
    else:
        try:
            asyncio.run(test_end_to_end())
        except Exception as e:
            failed += 1
            print(f"  FAIL end_to_end: {type(e).__name__}: {e}")

    print(f"\n{'FAILED' if failed else 'PASSED'} ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
