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
| **C · TE cuDNN** | TransformerEngine 2.17 + cuDNN 9.25 | forward ✅ / **backward ❌** | ❌ **禁用** | FusedAttention backward 内核在 sm_120 损坏，梯度错误→训练发散；见 §3。**注意作用域**：坏的是 **fused 后端**——同一 TE `DotProductAttention` 换成 flash 后端（`NVTE_FLASH_ATTN=1`，即 megatron `--attention-backend flash`）梯度正确（§3.1），这是 CP 唯一可走的路 |

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
- **Workaround（当非用 TE 不可，例如 CP —— CP 只有 TE 后端支持）**：
  **首选 `NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0`**（= megatron `--attention-backend flash`，源项目所有训练脚本的默认），
  梯度正确且保留 varlen 性能，**支持 CP**（§3.1 实测）。
  次选 `NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0` 强制 Unfused，梯度正确但 O(N²) HBM、长序列慢、**且不支持 CP**，只作数值对照回退。
- **完整调查 + 最小复现 + 上游 issue**：infra `docs/investigations/te-sm120-cudnn-bwd/README.md`、[NVIDIA/TransformerEngine#3333](https://github.com/NVIDIA/TransformerEngine/issues/3333)（**本项目自己报的**，2026-08-09 提交，截至 2026-08-10 仍 **open、无 maintainer 回复**）。
- **与 [TE#2186](https://github.com/NVIDIA/TransformerEngine/issues/2186) 不是同一条 bug**（#3333 正文已作此区分，本轮复核确认）：#2186 是 **THD + CP 的 tail-padding** corner case（Hopper sm90，短序列导致某 CP rank 的后半块全是 padding → `dk/dv` 爆成 NaN，已 closed/assigned），**依赖 CP**；本条 **单卡、无 CP、单条序列即复现**，不依赖 packing 也不依赖 padding。两者只是都落在 THD 上。
- **上游的版本门在我们这版本已"放行"，但实际仍坏（本轮新查，是 #3333 的有力补强）**：
  `utils.py:992-999` 对 sm_120 的 THD 有一条 gate —— `cudnn_version < (9,18,1)` 时禁用 FusedAttention。
  容器实测 `cudnn_version=(9,25,0)`，**高于该门槛故不触发**；TE 自报
  `Available backends = {... FusedAttention=True (sub-backend 1) ...}` → `Selected backend = FusedAttention (sub-backend 1)`，
  **即上游认为这条路在本版本已修好**，而实测 backward 仍错。
  换言之这不是"用了上游明令不支持的组合"，而是**上游放行的配置本身是坏的**。
- **为何 infra `04_megatron_te_thd_spike.py` 曾 PASSED（假阴性）**：用 `square().mean()` loss（梯度压到 ~1e-3）+ 绝对 `max_abs_error<0.08` 判据，错误梯度是"量级偏大 ~3.8×"而非 NaN，绝对值仍落阈值内蒙混。cosine/量级比才抓得到——repo 侧 V7.5 的度量教训（全局 rel_L2 + cosine）与此同源。

### 3.1 补充实测（2026-08-10，V8.2 计划期）：**这条 bug 是 fused 内核专属的，不构成对 CP 的限制**

本轮为 V8.2 验 CP 时独立复现了本条 bug，并测清了它与 CP 的相互作用（本节补的是原文没覆盖的 **CP 维度**）。
复现脚本 [`scripts/probe_v8.2_cp_packing.py`](../../scripts/probe_v8.2_cp_packing.py)（C1–C4 四个检查，跑法见其 docstring）。

**注意本节被改写过一次**：第一轮只测了 fused 与 unfused，得出「CP 只能走 fused + padding，与 packing 互斥」；
第二轮补测 **flash 后端**（`NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0`，即 megatron `--attention-backend flash`）后
**该结论作废**。保留改写经过见 [v8.2-plan.md](v8.2-plan.md) §1.1。

| 实测 | 结果 |
|---|---|
| 单段 THD vs BSHD **forward**（C1，三个后端） | `max_abs_diff=0.000e+00`（同一算法，逐值相等） |
| 单段 THD vs BSHD **backward**（C2，**fused**） | **差 24.7×**：`dq` THD `5.325e+01` vs BSHD `2.156e+00`，cosine 仅 `0.356` ← 本条 bug 的独立复现 |
| 同上（C2，**Unfused**） | `dq/dk/dv` cosine ≥ `0.99999994`、ratio `1.00×` ← **反证锅在 fused 内核** |
| 同上（C2，**flash / FA2 2.8.3**） | ✅ `dq` cosine `1.00000000`、ratio `1.00×`（dk/dv `0.99999988`）← **flash 路径没有这条 bug** |
| **fused 的 BSHD backward**（C3，原文未覆盖） | ✅ **正确**：fused vs unfused cosine `dq` 0.99996924 / `dk` 0.99999744 / `dv` 1.00003910，差异均 bf16 舍入量级 |
| **flash vs unfused**（C3） | ✅ `dq` 0.99996847 / `dk` 0.99999839 / `dv` 1.00003910，**bshd 与 thd 两组逐位相同** |
| **Unfused + CP=2** | ❌ `ValueError: No dot product attention backend is available` ← **Unfused 不支持 CP** |
| **fused + CP=2 + BSHD**（C4） | ✅ 通（`\|dq\|=106.6`、S_local=128=256/2，日志见 `unbatched P2P op` = ring KV 交换真通信） |
| **fused + CP=2 + THD**（C4） | ❌ `thd_second_half_lse_correction` assert ← 上游对 sm_120 的 carve-out |
| **flash + CP=2 + BSHD**（C4） | ✅ 通（`\|dq\|=106.6186` finite） |
| **flash + CP=2 + THD**（C4） | ✅ **通**（`\|dq\|=102.6641` finite）← **CP × packing 在 flash 后端下可组合** |

**结论（两条）**：

1. **坏的是 cuDNN fused 内核的 THD backward，不是 THD 布局本身**。flash 后端走同一 THD 输入、
   给出与 BSHD 逐值一致的梯度（cosine 1.000）。§3 开头那条「TE cuDNN backward 禁用」依然成立，
   但它的作用域是 **fused 后端**，不该外推到「TE 不能用」或「CP 不能配 packing」。
2. **CP 的可用路径 = TE spec + flash 后端**（BSHD/THD 都可），这正是源项目的默认配方 ——
   源所有 megatron 训练脚本都带 `--attention-backend flash`，`docs/zh/developer_guide/debug.md:15`
   写明理由是「避免 CP 下 fused attention 的数值不稳定」。**源遇到过同一类问题，解法是换后端。**

**上表的后端是探针自证的，不是靠 env 推断的**：`NVTE_*` 是**请求**，TE 可能否掉它
（unfused 遇 CP 就直接不可用）。探针跑过真前向后从 TE 的 `_attention_backends` 缓存读回实选后端并断言，
实测三行分别自报 `FlashAttention(2.8.3)`（版本号直接证明是 FA2 而非 FA4）、
`FusedAttention(NVTE_F16_arbitrary_seqlen)`、`UnfusedDotProductAttention`；
CP=2 的 BSHD/THD 两行也各自内联报告，均为 `FlashAttention(2.8.3)`。
故意错标（`TAG=flash` + env 请求 fused）会被 C0 拒绝 —— 该断言有判别力。

> **不要去 patch 那个 sm_120 carve-out**：实测 patch 掉后 fused CP=2 THD 会"跑通"，但梯度是错的——
> 那是把上游有意设的护栏拆掉换来一个静默错误。正确做法是换 flash 后端。

---

## 4. 镜像：收敛到单一 `fa2-mcore`（2026-08-10 改写）

**本节原来的多 tag 方案已作废**。原方案基于一条实测推翻了的前提——infra `docs/ops/docker-env-a100-vs-5090.md` §4
写的「**FA2 + TE 互斥**：FA2 wheel 覆盖 `flash_attn` namespace → TE import 崩」。

**那条只在「留下 `flash_attn_4-*.dist-info`」时成立**。TE `backends.py` 里有**两个独立的 try 块**，
各自靠**包元数据**判断装没装，却 import **同一个 `flash_attn/` 目录**：

| 块 | 判据 | import |
|---|---|---|
| FA2（`:96-109`） | `get_pkg_version("flash-attn")` | `from flash_attn.flash_attn_interface import flash_attn_varlen_func` |
| FA4（`:163-169`） | `get_pkg_version("flash-attn-4")` | `from flash_attn.cute.interface import ...` |

`pip install --no-deps` FA2 只把 `flash_attn/` 的内容换成 FA2 的（`flash_attn.cute` 随之消失），
**不动 `flash_attn_4-*.dist-info`** → `get_pkg_version("flash-attn-4")` 仍返回 `4.0.0b15`、
不抛 `PackageNotFoundError` → TE 走进 `:167` 的 `else` 去 import 已不存在的 `flash_attn.cute`
→ `ModuleNotFoundError`（**该 try 只捕获 `PackageNotFoundError`，这个异常没人接**）→ `import transformer_engine` 崩。
**元数据说 FA4 在、文件系统说不在，TE 信元数据。连 dist-info 一起删则两者共存无碍**
（容器实测：`flash-attn 2.8.3` / `flash-attn-4 PackageNotFoundError`）。

**订正结论**：TE 与 FA2 **从不互斥**——TE 有一整块专为 FA2 写的路径，还为 Blackwell 单设版本下限
（`backends.py:101` `version_required_blackwell`，正对 sm_120）。真正互斥的是 **FA2 ↔ FA4**
（同抢 `flash_attn` 顶层包名）。之前那次崩是**卸载不干净**，被误诊成了软件冲突。

于是 5090 上只留一个镜像：

| 镜像 tag | 装什么 | 跑什么 |
|---|---|---|
| **`agentic-rl-infra-lab:fa2-mcore`** | TE 2.17 + megatron-core 0.18.2 + **FA2 2.8.3** + accelerate | 全部——V7.6 packing、V8 Megatron TP、V8.2 CP，及后端对照（`NVTE_*` 可覆盖） |

配方入库在 infra 仓库：`docker/Dockerfile.fa2-mcore` + `docker/build_fa2_mcore.sh`
（`FROM te-cudnn-system-spike` → 删 FA4 目录**与 dist-info** → 装 FA2 → 装 accelerate →
`ENV NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0`）。

**为何必须是「TE 与 FA2 同镜像」而不是二选一**：CP 只有 TE 后端支持
（mcore `dot_product_attention.py:57-59` 对 local spec 直接 assert `cp==1`），
而 sm_120 上 TE 的 cuDNN fused THD backward 是坏的（§3，TE#3333）→ CP 必须走 flash 后端 →
**同时需要 TE 与 FA2**。infra `scripts/06_attn_backend_throughput.py` 那种
`from flash_attn import flash_attn_varlen_func` 的直调路径在这里用不上（它绕开 TE，也就绕开了 CP）；
nano 走的是 TE 内部那行 `backends.py:109 from flash_attn.flash_attn_interface import flash_attn_varlen_func`。

**验收（2026-08-10，镜像由入库脚本重建后复验，数字与合并前逐位相同）**：

| 检查 | 结果 |
|---|---|
| 三者共存 | `FA2 2.8.3` / `TE 2.17.0` / `megatron-core ok` |
| **FA2 真被调用**（给 `flash_attn_varlen_func` 打计数器，同 V7.6 验 varlen 那招） | `[before] varlen=0` → `[after] varlen=1`；TE 自报 `use_flash_attention=1  flash_attention_backend=Version('2.8.3')` |
| C0 后端断言 | TE 自选 `FlashAttention(2.8.3)` |
| C1 forward / C2 backward | `max_abs_diff=0.000e+00` / `dq` cosine `1.00000000` |
| C4 CP=2 BSHD / THD | PASS `\|dq\|=106.6186` / PASS `\|dq\|=102.6641` |
| negative control（`TAG=flash` + env 请求 fused） | `❌ 后端不符`、`EXIT=1` ← 断言仍有判别力，且证明 `ENV` 是默认值非锁 |

**版本号字符串不算证据**：`FlashAttention(2.8.3)` 只证明 TE 看见了这个包；计数器才证明 kernel 被调进。

---

## 5. 与主线的关系（不分叉代码）

| 项 | 主线（双机） | 5090 专属 |
|---|---|---|
| 训练 attention 后端 | **FA2 varlen**（V7.6） | 同——无差异 |
| packing 隔离 | FA2 cu_seqlens | 同 |
| FA4 探索 | 不涉及（A100 架构拒绝） | opt-in flag，5090-only |
| TE cuDNN | 两机均降为 forward 对照 | **backward 禁用**（bug），本文记明 |
| 镜像 | 训练镜像基底 + FA2 wheel | **单一 `fa2-mcore`**（TE+megatron+FA2 共存，§4） |

**核心**：5090 不再需要独立代码分支。所有专属项要么是 opt-in flag（FA4），要么是禁令/备忘（TE bug、镜像），主线代码与 A100 完全共享。

---

## 6. 待办

- [ ] FA4 后续版本（> b25）若吞吐反超 FA2，重估是否升为 5090 首选。**注意**：`fa2-mcore` 里 FA2 与 FA4 仍互斥（同占 `flash_attn` 包名），若要试 FA4 需另建 tag。
- [ ] TE#3333 若 maintainer 修复或要求补验（其他 head_dim/dtype/mask，或另租 sm_80 交叉确认），按 infra investigations README 的复现命令跑。
- [x] ~~是否固化 `fa2-system-spike` 镜像 tag~~ → 已固化为 **`fa2-mcore`**（§4），并把配方入库到 infra `docker/`。5090 上其余 lab 镜像（`fa2-spike`/`te-spike`/`te-spike-deps`）已删。
- [ ] infra `docs/ops/docker-env-a100-vs-5090.md` §4 那行「FA2 + TE ❌ 互斥」需按 §4 的 dist-info 机制订正（该仓库正在改结构，未擅动）。
