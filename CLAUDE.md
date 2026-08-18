# slime-agentic-zero-repro — 项目上下文 primer

> 新会话起手必读。**本文件只留操作性规则 + 当前状态**；各版本的详细决策、踩坑、验证数据都在 [docs/decisions/](docs/decisions/) 和 git log 里，不在此重复。

## 1. TL;DR

- **项目**：从 0 复现一个最小 Agentic RL 训练系统，**在复现中学习** slime-agentic 的系统设计。
- **源项目**：slime-agentic —— 基于 Ray + SGLang + Megatron/FSDP 的 Agentic RL 训练框架（58K LOC）。
- **当前状态**：主线一（V0–V5 系统骨架）✅ / 主线二（A1 MemAgent、A2 AgentFlow、A3 ToolOrchestra）✅ / 主线三（V6 真训练闭环 → V7 FSDP → V7.4/7.5/7.6 → V8 Megatron TP → V8.2 多卡 TP×PP×CP）✅ 全部收官。
- **下一版 V9 吞吐实验**：前八版全在问「对不对」，V9 第一次问「多快」。计划见 [v9-plan.md](docs/decisions/v9-plan.md)，尚未开工。

## 2. 铁律（每版必须遵守）

**① 代码范式遵从原项目**：nano 的代码结构 / 接口签名 / 命名 / 数据流必须对齐 slime-agentic 在对应位置的写法。**只允许在其基础上做得更清晰（更好），不允许更乱、更 hack、更偏离（更差）**。判断法：写任一段前先问「源项目对应位置怎么做的」，对齐它。

**② 锚点优先级**：**slime-agentic 是唯一首要锚点**。其它 nano 项目（如 nano_hermes_agent 的工具系统）只是**次级借鉴**，仅在「不与 slime-agentic 冲突、且能让代码更清晰」时借其工程手法（registry/dispatch 接缝、统一返回格式）；**绝不借它与 slime-agentic 相悖的部分**（如 native tool-calling `tools=` schema —— 会破坏 RL 的 token/log_prob 保真度）。

**③ 偏离必须写明三要素**：只要没对齐源的写法（简化、次级借鉴、暂缓实现都算），必须在**代码注释 + 当版 `docs/decisions/*.md`** 里写清：①源项目对应位置怎么做的；②nano 为何偏离；③语义是否等价 / 何时补齐。**没有说明的偏离视为违反铁律**。

**④ 分支准则**：每版切一个 `vN` 分支，从上一版分支末端切出；当版的全部提交（实施计划 doc + 代码实现 + 验证结果）都落在 `vN` 上，**绝不提交到别的版本分支**。提交前先 `git branch --show-current` 确认。

**⑤ 服务器只用 git 恢复，绝不 rsync 盖 git 工作树**：服务器 checkout 是 git 管理的，版本不匹配时用 `git fetch origin && git reset --hard origin/<vN>`。rsync 会把本地其它版本的文件混进 checkout、污染分支状态。**必须走 SSH**（服务器 HTTPS 443 超时；SSH key 已配好）。

## 3. 路径与仓库

- **源项目**：`/Users/qshf/my-project/slime-agentic`（github: LMIS-ORG/slime-agentic @ main。只读，用于对照）
- **nano 项目**：`/Users/qshf/my-project/slime-agentic-zero-repro`（主分支 main）
- **infra 仓库**：`/Users/qshf/my-project/agentic-rl-infra-lab`（容器配方 + infra 契约来源，「infra as spec」）
- **当前活跃分支**：`v9`
- **git 远端**：`git@github.com:qshf/slime-agentic-zero-repro.git`（SSH，仓库名无前导横杠）
- **工作流**：**本地只开发**（写码 + 推 git）→ **SSH 5090 服务器**（`/home/ubuntu/slime-agentic-zero-repro`，git 管理）拉取 / 跑通 / 验证。只有 V0（纯 fake）本地可验，V1 起都要 GPU。

## 4. 运行环境

**环境说明读取路径**（详细配置不写在这里，按需读）：

| 内容 | 路径 |
| --- | --- |
| 服务器环境详细配置（容器/依赖/踩坑配方） | [docs/ops/server-env-bench5090.md](docs/ops/server-env-bench5090.md) |
| 容器镜像构建配方 | `/Users/qshf/my-project/agentic-rl-infra-lab/docker/Dockerfile.fa2-mcore` + `build_fa2_mcore.sh` |
| 跨会话记忆（服务器实时状态、GPU 占用、端口） | `~/.claude/projects/-Users-qshf-my-project-slime-agentic-zero-repro/memory/` |
| 版本路线图 | [docs/system-roadmap.md](docs/system-roadmap.md) |
| 各版本决策 / 偏离登记 / 验证数据 | [docs/decisions/](docs/decisions/) |

**当前环境**：

- **本地**：开发环境（uv 管理），仅 V0 可本地验证。
- **服务器 5090**（4×RTX 5090 32G，`ssh 5090`）：
  - 镜像 `agentic-rl-infra-lab:fa2-mcore`（TE 2.18 + Megatron-Core 0.18.2 + FA2 2.8.3 共存）
  - 常驻开发容器 `v9-dev`（挂 4 卡，`PYTHONPATH=/workspace`）
  - 模型 Qwen3-0.6B @ `/home/ubuntu/models/Qwen/Qwen3-0.6B`
  - SGLang 端口 30000（GPU 被别的进程占时需 `--mem-fraction-static 0.62` 才起得来）

## 5. 快速开始

```bash
# 本地
git checkout v9 && git push origin v9

# 服务器
ssh 5090
cd /home/ubuntu/slime-agentic-zero-repro
git fetch origin && git reset --hard origin/v9

# 进常驻容器跑验证（示例：V8.2 并行等价门）
docker exec -it v9-dev bash
torchrun --nproc-per-node 1 scripts/test/test_v8.2_parallel.py --layers 4 --fp32
```

各版本完整测试流程见对应的 `docs/decisions/*.md`。

## 6. 跨版本沉淀的方法学（不随版本失效的部分）

- **等价门要分层**：**fp32 才是数学精确证明，bf16 只到舍入量级**。度量用**全局 rel_L2 + cosine**（向量级、大信号主导），不用逐参数 max_abs/max_rel —— 后者会被小信号参数（k_proj 之类，bf16 下噪声≈信号）放大成虚高失败。
- **fp32 不可达时必须上 negative control**：FA2 内核拒收 fp32，CP>1 只能走 flash → 这类门物理上没有 fp32 档。改用反例（故意破坏隔离/切分）证明度量有判别力；**反例若也"通过"，则整轮验收作废**。
- **「前向对、梯度错」是漏梯度规约的指纹**：loss Δ 恰为 0 而梯度系统性偏，且最差参数指向某一类权重。已踩到三处（qk-layernorm 跨 TP / tie embedding 跨 PP / 全梯度跨 DP×CP），**只有 fp32 抓得到**。
- **等价门跑之前先关随机性**：`TransformerConfig` 默认 `dropout=0.1`，且各 TP rank 故意持不同 dropout RNG → 不关就是假阳性（实测 cosine 掉到 0.25，看着像切错其实纯随机）。
- **性能实验的纪律**（V9 要用）：必 `cuda.synchronize()` 再停表（不同步测的是 kernel launch 时间）；必丢 warmup；必记 `nvidia-smi` 快照（**共享 GPU 上的吞吐数字不可比**）；**先写下可证伪的预期再测**（先测后编解释是性能工程最常见的自欺）；**只测量不优化**（发现瓶颈记待办，当场改会让两者互相污染）。
- **新后端接线必须先过精度门（V9.1 确立）**：每当把新的并行后端（Megatron / FSDP / 未来的其它后端）接进训练闭环，**第一步**是在 TP=1 PP=1 下跑 `scripts/bench/v9_accuracy_gate.py`，验证新后端与 TorchActor 的 loss/grad 在 HF 参数空间数值等价，再往多卡并行扩展。判据分层（与 V8 一致）：fp32 档 loss |Δ| < 1e-4 / grad cosine > 0.9999 / rel_L2 < 5e-3；bf16 档 loss |Δ| < 5e-2 / cosine > 0.999。**原因**：并行等价门（V8）只证明"TP=N 与 TP=1 在同一后端内一致"，无法捕获 loss 公式本身接错（如错误的缩放因子、missing mask、ratio 计算偏离）—— 这类错误只有跨后端比才能暴露。精度门是后者的唯一检验点。

## 7. 待办 / 活跃问题

- [ ] **V9 吞吐实验**：计划已出，尚未开工。
- [ ] **GPU 共享坑**：4 张卡常被别的进程（如 `vllm-qwen36-27b` 各占 28.5G）占满 → SGLang 起不来 / 训练 OOM / 吞吐数字不可比。开跑前先 `nvidia-smi` 确认。
- [ ] **A100 补跑**（V7.6 packing 门的跨架构复验）：A100 实例已释放，仅记录不作完成条件。
