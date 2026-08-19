# 08 · GPU 内幕：kernel、融合、FlashAttention、混合精度

> 对应原文：Diving into the GPUs – Fusing, Threading, and Mixing（A primer on GPUs / Improving performance with kernels / Fused kernels / FlashAttention / Mixed precision training）

---

## 1. GPU 的两条层级

**计算侧**：GPU = 一组 **SM**（streaming multiprocessor），每个 SM 含一组 core。**H100：132 个 SM × 128 core = 16,896 core**，每个 core 还能同时处理多个线程。

**内存侧**（从快到慢、从小到大）：

| 层级 | 作用域 |
|---|---|
| **Register** | 线程私有，最小最快 |
| **Shared memory / L1** | **同一个 SM 内的线程共享** |
| **L2** | 所有 SM 共享 |
| **Global memory（HBM）** | 最大（H100 的 80GB）**也最慢** |

**调度单位**：
- 线程按 **32 个一组**成 **warp**，warp 内所有线程**锁步执行同一条指令**（作用在不同数据上）；
- warp 再组成 **block**（512 或 1024 线程），**每个 block 分配给一个 SM**；一个 SM 可并行跑多个 block，但资源不够时部分 block 会排队。

在 GPU 上跑的那段代码叫 **kernel**（CUDA/Triton 写，编译成 PTX）；还需要 CPU 侧的 **host code** 负责分配显存、搬数据、发射 kernel。

---

## 2. 写 kernel 的工具阶梯

| 工具 | 难度 | 速度 | 灵活性 |
|---|---|---|---|
| **PyTorch** | 易 | 慢 | — |
| **`@torch.compile`** | 易 | 快 | **不灵活** |
| **Triton** | 较难 | 更快 | 较灵活 |
| **CUDA** | 最难 | 最快 | **最灵活**（如果你写对了） |

**实用路径**：先写朴素 PyTorch → 加 `@torch.compile`（仅一个装饰器就能有显著提升）→ 不够就 `export TORCH_LOGS="output_code"` **把 compile 生成的 Triton kernel 打印出来当起点改**，不要从零手写。Triton 的极限在于它管不到 shared memory 和 SM 内调度的细节，要那个层级就得下到 CUDA。

---

## 3. CUDA 四个经典优化

### ① Memory coalescing（内存合并访问）

global memory 用 DRAM 实现，**每次访问会以 burst 方式并行读出一串连续地址**。若一个 warp 里 thread 0 访问 M、thread 1 访问 M+1、……硬件会**把这些请求合并成一次大访问**。

矩阵乘的朴素写法（每线程算一个输出元素）里，同 warp 的 (0,0) 和 (1,0) 会读 A 的**不同行同一列** —— 而矩阵是**行主序**存的，所以这两个地址离得很远，**每次迭代都无法合并**。

改法只是换了一下 x/y 的算法（2D block 改 1D）：

```cuda
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

让同 warp 的线程**共享同一个 x、不同 y** → 读 A 的同一行、B 的不同列 → 行主序下可合并。

**效果：内存吞吐 ×10，kernel 执行时间 ÷10。**

### ② Tiling（分块 + shared memory）

朴素实现里，block 内多个线程会**重复**从 global memory 读同样的数据。Tiling 把 A 的一块和 B 的一块**协作载入 shared memory 一次**，让 block 内所有线程复用；载完 `__syncthreads()`，再在 tile 上做乘加、累加到中间结果，逐 tile 推进。

**效果：内存吞吐到 410 GB/s，kernel 时间 −43%，约 6.6 TFLOPS。**

### ③ Thread coarsening（线程粗化）

tiling 之后看 warp 状态，发现大量 `mio_throttle` 停顿——**warp 在等 shared memory 访问返回**。解法是**把多个线程合并成一个"粗线程"**，每个粗线程负责多个输出元素，从而**大幅减少 shared memory 访问次数**（用更少但更宽的 load）。

### ④ 减少 control divergence（控制流分歧）

SM 按 **SIMD** 执行 warp：一条指令，warp 内所有线程同时执行。如果 `if` 让同 warp 的线程走了不同分支，**warp 必须串行执行这些分支**，其余线程空等。

对策：重构代码减少分支、用能让线程走相同路径的数据结构、或用 predication。

---

## 4. Fused kernel（融合）

CPU 侧可以非阻塞地往 GPU 发任务。除了用来重叠通信/计算，更一般的原则是：**不惜代价避免在 host 与 GPU kernel 之间来回**。

一串 kernel 各自「从 global memory 读 → 算 → 写回 global memory」会反复往返；**融合成一个 kernel 后，中间值一直留在 SM 本地**，一次算完。

**最适合融合的是逐点（point-wise）操作序列** —— 每个 token 上独立执行、彼此无关。Transformer 里典型例子就是 **LayerNorm 那一串逐点运算**。

---

## 5. FlashAttention

朴素 attention 要在 **HBM** 里物化两个巨大矩阵：`S = QKᵀ`（注意力分数）和 `P = softmax(S)`，然后再读回 SRAM 做下一步 —— **HBM 带宽远低于 SRAM，这就是瓶颈**。

FlashAttention 的做法：
1. **把 S 分成能塞进 SM shared memory 的小块**来算；
2. 更进一步，**根本不物化 S** —— 只保留计算 softmax 归一化因子所需的**统计量**（online softmax），在 SRAM 里直接算出一部分 O。

**双重收益**：
- 不物化 S → **省掉模型里（长上下文下）最大的那个激活矩阵**；
- 也**消掉了朴素实现里 O(S²) 开销的很大一部分**。

> 影响之大，以至于 **Transformer 出现后不久涌现的各种线性注意力/次二次近似方法，基本都被这个"精确且快"的实现取代了。**

**FA1 → FA2 → FA3 的改进不在注意力机制本身，而在贴合 GPU**：①尽量减少非 matmul 操作；②在 warp 和 thread block 之间更好地划分工作（FA2）；③针对 Hopper 的 FP8 和 Tensor Core 优化（FA3）。

FlashAttention 对**能加速哪些注意力模式有限制** → 更灵活的变体见 FlexAttention。

> 对本项目：V7.6 用的 FA2 varlen packing 正是这一族；nano 实测的「FA2 内核只收 fp16/bf16」也是这种深度贴硬件的实现的必然副作用——**它不是通用算子，是为特定精度的 Tensor Core 路径写的**。

---

## 6. 混合精度训练

### 格式表

| 格式 | 总位数 | 符号 | 指数 | 尾数 |
|---|---|---|---|---|
| float32 | 32 | 1 | 8 | 23 |
| float16 | 16 | 1 | **5** | 10 |
| **bfloat16** | 16 | 1 | **8** | **7** |
| float8 (e4m3) | 8 | 1 | 4 | 3 |
| float8 (e5m2) | 8 | 1 | 5 | 2 |

（bfloat16 的 "b" 来自 Google **B**rain。）

**取舍就是：位数少了，你只能选择牺牲尾数还是指数。**

- **float32 跨 80 个数量级**；
- **float16 牺牲了大量动态范围**；
- **bfloat16 保住了 fp32 的完整范围**，代价是**精度更差**；
- **fp8**：e5m2 能维持 fp16 的范围，e4m3 范围更小。

**分辨率**（epsilon = 1.00 之后第一个可表示的数）：fp32 约 1.19e-7、fp16 约 1e-3、bf16 再高 10 倍。在 [1,2] 区间里 **e4m3 只能表示 7 个数，e5m2 只有 3 个**。

### FP16/BF16 训练的三个必需技巧

把所有张量和运算直接切到 float16 **会发散**。原始混合精度论文给了三招：

1. **FP32 权重副本（master weights）**：fp16 权重可能被舍入到 0，或者更新量太小、加法时下溢；**一旦变成 0 就永远是 0**（没有梯度信号了）。
2. **Loss scaling**：梯度普遍远小于 1，容易下溢。反向前把 loss 放大、反向后把梯度缩回来。**因为在梯度裁剪和 optimizer step 之前就还原了，所以不影响训练。**
3. **FP32 累加**：求和/求平均这类操作在 16 位下会上溢或下溢 → **中间结果用 FP32 累加，最后再转回 16 位**。

### FP8 预训练

**动机**：即使通信完美重叠，最终还是会撞上硬件的理论 FLOPS 上限。**H100 上 FP8 的 GEMM 理论算力是 BF16 的两倍。**

**挑战：稳定性。** 低精度下数值不稳定常导致 loss 发散，且**模型大小固定时学习率越高越不稳**。

**首个公开报告的超大规模 FP8 混合精度训练是 DeepSeek-V3**：他们逐个分析了前向（Fprop）、激活反向（Dgrad）、权重反向（Wgrad），像 BF16 混合精度一样**把部分聚合和 master weights 保持在高精度**，运算本身走 FP8。关键手法是**按 tile 归一化量化范围**：**输入/激活用 1×128，权重和 scale 用 128×128** —— 这样**离群值对归一化的影响被局部化**。

| 方案 | GEMM 精度 | master weights | 累积梯度 | 权重 | 梯度 | 优化器状态 | 合计 |
|---|---|---|---|---|---|---|---|
| BF16 + FP32 混合精度（基线） | BF16 | FP32 | FP32 | BF16 | BF16 | FP32+FP32 | **20 字节** |
| 同上但不做 FP32 梯度累积 | BF16 | FP32 | — | BF16 | BF16 | FP32+FP32 | 16（−20%） |
| Transformer Engine | FP8 | — | — | FP32 | FP32 | FP32+FP32 | 16（−20%） |
| FP8-LM O3 | FP8 | FP16 | FP16 | FP8 | FP8 | FP8+FP16 | **9（−55%）** |
| DeepSeek-V3 | FP8 | FP32 | FP32 | FP8 | BF16 | BF16+BF16 | 15（−25%） |
| Nanotron FP8 | FP8 | BF16 | FP32 | FP8 | FP8 | FP8+FP8 | 10（−50%） |

**FP8 在 2025 年初仍属实验性**，但收益明显，很可能取代 BF16 成为标准。再往后 **Blackwell 宣布支持 FP4 训练** —— 更快，也毫无疑问带来新的稳定性挑战。

---

## 7. 与本项目的关系

- **bf16 保范围、牺牲精度**这条正是 nano 从 V7.5 起那套「**fp32 才是精确证明、bf16 只到舍入**」分层验收门的物理依据：bf16 尾数只有 7 位，28 层前向累积下来 rel_L2 到 10% 级别是**正常舍入**而非 bug。这一章给了这个判断一个可引用的出处。
- **FA2/FA3 是深度贴硬件的实现**这条，解释了 V7.6 偏离 #8 和 V8.2 偏离 N4 的同一个根因：**FA2 拒收 fp32 不是疏忽，是它本来就只为 Tensor Core 的半精度路径而写。**
- 「融合最适合逐点算子序列」对应 V8 用的 `fused_vocab_parallel_cross_entropy`；V8 撞的那个「fused kernel 原地改输入、bf16 下被 `.float()` 隐式拷贝盖住、只有 fp32 才暴露」的坑，也属于这一节的实践面。
