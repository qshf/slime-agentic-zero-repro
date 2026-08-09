# v8 · 5090（sm_120）专属后续方案

> 状态：建议（2026-08-09）
> 定位：主线 [`v7-onward-plan.md`](v7-onward-plan.md) 已统一为 **FA2 varlen 双机通用**，A100 与 5090 跑同一份代码。本文只记 **5090（Blackwell sm_120）独有** 的显卡相关后续——这些不进主线代码，作为「显卡差异备忘 + opt-in 探索」存在。
> 依据：infra `agentic-rl-infra-lab/docs/04-flash-attn-blackwell-recheck-plan.md`、`docs/investigations/te-sm120-cudnn-bwd/`。

---

## 0. 一句话

5090 上**训练走 FA2 varlen（与 A100 同代码，主线已定）**。以下三条是 5090 专属的边角：FA4 b25 作前沿备选、TE cuDNN backward 有 bug 必须规避、镜像按后端区分。**没有一条改变主线 attention 后端选择（FA2）**。

---

## 1. 三路后端在 5090 上的实测状态（2026-08-09）

环境：RTX 5090 sm_120，torch 2.11+cu130，bf16。详见 infra doc 04 §3。

| 路 | 后端 | 5090 状态 | 训练可用 | 备注 |
|---|---|---|---|---|
| **A · FA2** | flash-attn 2.8.3 varlen | ✅ **PASSED** | ✅ **首选** | 官方 `cu13torch2.10` 预编译 wheel 已含 sm_120 kernel，`--no-deps` 秒装；与 A100 同一 wheel |
| **B · FA4** | flash-attn 4.0.0b25 | ✅ **PASSED**（b25） | ⚠️ 备选 | b21 修了 CuTe SM120 编译期（#2671）；吞吐仍是 FA2 的 0.86–0.98×，未反超 |
| **C · TE cuDNN** | TransformerEngine 2.17 + cuDNN 9.25 | forward ✅ / **backward ❌** | ❌ **禁用** | FusedAttention backward 内核在 sm_120 损坏，梯度错误→训练发散；见 §3 |

**主线选择不变**：FA2。FA4 与 TE 都不取代它。

---

## 2. FA4 b25 —— 前沿备选（不作首选）

- **可用**：`4.0.0b25`（2026-08-05）在 sm_120 varlen forward+backward PASSED，数值与 SDPA reference 吻合到 bf16 精度。b21 的 `Fix CuTe SM120 compile-time argument handling`（#2671）+ `Implement Pack-GQA on SM120`（#2656）正对 b15 那个 `Operation creation failed` 失败。
- **为何暂不首选**：吞吐 = FA2 的 **0.86–0.98×**（beta 未极致调优），显存与 FA2 一致，**尚未反超 FA2**。后续版本可能反超，届时重估。
- **安装陷阱**：`te-spike` 基础镜像自带 b15，Dockerfile 若只写 `pip install "flash-attn-4[cu13]"`（无 `--upgrade`）会停在 b15。必须 `--upgrade` 或 pin `==4.0.0b25`。
- **与 TE 兼容**：b25 自带 `flash_attn/cute/interface.py`，`import transformer_engine` 与 `from flash_attn.cute.interface import ...` 同进程共存（区别于 **FA2 与 TE 互斥**）。但 `fa4-spike` 镜像缺 system cuDNN，TE 实跑仍需 `te-cudnn-system-spike`。
- **repo 落地方式（若探索）**：opt-in flag（如 `train_packing="fa4"`），不改主线默认 FA2。A100 上 FA4 架构门槛拒绝（需 sm_90+），故 FA4 分支天然是 5090-only，正好留在本文。

---

## 3. TE cuDNN backward bug —— 必须规避（5090 专属回归）

**结论**：5090（sm_120）上 TE `DotProductAttention(qkv_format="thd")` 走 cuDNN FusedAttention 后端时，**backward 梯度错误**（forward 正确）。与 packing、序列条数、长度是否 128 倍数、`cu_seqlens_padded`、CP 均无关——单条序列即可复现。

| 后端 | `[128]` 单条 | `[128,96]` packed |
|---|---|---|
| cuDNN FusedAttention | ❌ q_grad_cos=0.44 | ❌ q_grad_cos=0.03, norm=2e10 |
| UnfusedDotProductAttention | ✅ q_grad_cos=1.0000 | ✅ q_grad_cos=1.0000 |

- **对训练的影响**：凡在 5090 上用 TE cuDNN THD FusedAttention 训练者，backward 梯度错误 → 训练发散。**主线选 FA2 varlen 天然避开此坑**；本条是"若有人想用 TE 当训练后端"的明确禁令。
- **Workaround（仅当非用 TE 不可）**：`NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0` 强制 Unfused，梯度正确但 O(N²) HBM、长序列慢，只作数值对照回退。
- **完整调查 + 最小复现 + 上游 issue**：infra `docs/investigations/te-sm120-cudnn-bwd/README.md`、[NVIDIA/TransformerEngine#3333](https://github.com/NVIDIA/TransformerEngine/issues/3333)。
- **为何 infra `04_megatron_te_thd_spike.py` 曾 PASSED（假阴性）**：用 `square().mean()` loss（梯度压到 ~1e-3）+ 绝对 `max_abs_error<0.08` 判据，错误梯度是"量级偏大 ~3.8×"而非 NaN，绝对值仍落阈值内蒙混。cosine/量级比才抓得到——repo 侧 V7.5 的度量教训（全局 rel_L2 + cosine）与此同源。

---

## 4. 镜像区分（5090 后端隔离）

三后端装一起会抢 `flash_attn` 包名 + 底层 cuDNN 库，5090 上按后端拆镜像（infra doc 04 §3b.0）：

| 镜像 tag | 装什么 | 跑哪路 |
|---|---|---|
| `te-cudnn-system-spike` | TE + system cuDNN | TE cuDNN（`NVTE_FLASH_ATTN=0`）——仅 forward 对照 |
| `fa4-spike`（`FROM te-spike`） | FA4 b25 + SDPA | FA4 / SDPA 同进程 |
| （fa4-spike 内临时 pip 装 FA2 wheel） | FA2 2.8.3 | FA2 varlen——装完 `flash_attn` 即变 FA2，与 TE 互斥，隔离最后跑 |

repo 训练镜像基底沿用 infra 的 `te-cudnn-system-spike`（TE 只作对照层）；真训练前装 FA2 wheel。若频繁在 TE/FA2 间切，固化两个 tag（`te-cudnn-system-spike` + `fa2-system-spike`）避免反复 pip uninstall/install。

---

## 5. 与主线的关系（不分叉代码）

| 项 | 主线（双机） | 5090 专属 |
|---|---|---|
| 训练 attention 后端 | **FA2 varlen**（V7.6） | 同——无差异 |
| packing 隔离 | FA2 cu_seqlens | 同 |
| FA4 探索 | 不涉及（A100 架构拒绝） | opt-in flag，5090-only |
| TE cuDNN | 两机均降为 forward 对照 | **backward 禁用**（bug），本文记明 |
| 镜像 | 训练镜像基底 + FA2 wheel | 按后端多 tag（对照用） |

**核心**：5090 不再需要独立代码分支。所有专属项要么是 opt-in flag（FA4），要么是禁令/备忘（TE bug、镜像），主线代码与 A100 完全共享。

---

## 6. 待办

- [ ] FA4 后续版本（> b25）若吞吐反超 FA2，重估是否升为 5090 首选。
- [ ] TE#3333 若 maintainer 修复或要求补验（其他 head_dim/dtype/mask，或另租 sm_80 交叉确认），按 infra investigations README 的复现命令跑。
- [ ] 是否固化 `fa2-system-spike` 镜像 tag。
