# 04 · 上下文并行 CP：Ring Attention 与 Zig-Zag

> 对应原文：Context Parallelism / Ring Attention / Zig-Zag Ring Attention

---

## 1. CP 与 SP 的确切区别

TP+SP 之后每卡显存已经压得很低，但**序列上到 128k+ 时仍然爆**——因为**进了 TP 区就必须处理完整序列**。而且即使开 full 重算（约 +30% 算力），层边界那些必须留下的激活**仍然随序列长度线性增长**。

CP 的想法和 SP 一样都是**沿序列切**，区别在**作用范围**：

- **SP**：只在 TP 之外的区域（LayerNorm/dropout）沿 s 切，进 TP 区要 all-gather 回完整 s；
- **CP**：**在整个模型上**沿 s 切，**包括原本由 TP 处理的那些模块** —— 于是这些模块被**沿两个维度同时切开**。

对大多数模块（MLP、LayerNorm）而言，沿序列切**毫无影响**——每个 token 独立处理。也**不需要 TP 那种昂贵通信**，因为**只切输入、不切权重矩阵**。就像 DP 一样，算完梯度后在 CP 组内做一次 all-reduce 同步即可。

---

## 2. 唯一的例外：attention

注意力里每个 token 都要访问**所有**其它 token 的 K/V（因果注意力下至少是所有更早的 token）。而 CP 把序列切开放在不同 GPU 上 → **attention 必须在 GPU 之间完整交换 K/V**。

朴素做法非常贵。解法是 **Ring Attention**。

> 📝 CP 和 FlashAttention 有共同的技术底座：**都依赖 online softmax** 来避免物化完整的注意力矩阵。区别在于 FlashAttention 优化的是**单卡内**的 attention 计算，CP 是**把序列摊到多卡**上省显存。

---

## 3. Ring Attention

每张卡在**等别人数据的同时算自己手上那份**。每个时间步，每张卡做三件事：

1. **非阻塞**地把当前 K/V 发给环上的下一台机器（最后一步除外）；
2. 在**当前手上**的 K/V 上算注意力分数 `Softmax(QKᵀ/√d)·V`；
3. 等着收上一台机器发来的 K/V，然后回到第 1 步（现在的"当前 K/V"就是刚收到的那份）。

4 卡就重复 4 次，注意力算完。理想情况下**下一份 K/V 在本轮计算结束前就已到达**，计算不停顿——这就是 "Ring" 的由来。

---

## 4. 朴素 Ring 的致命问题：因果 mask 导致负载严重不均

softmax 是**按行**算的，一张卡收齐了某一行需要的全部 token 就能算这一行。

假设 4 卡、序列按顺序切成 [1-4] [5-8] [9-12] [13-16]：

- **GPU 1** 拿到 token 1-4，因果 mask 下它**只需要自己这份**，一步都不用等 → 立刻算完，**而且工作量最小**；
- **GPU 2** 要等第二轮拿到 1-4 才能算全 1-8；
- **GPU 4** 要等到最后、且要算的方块最多。

三角形的注意力矩阵 → **负载严重倾斜**，最闲的卡和最忙的卡差好几倍。

---

## 5. Zig-Zag：把早期和晚期 token 混着分

不再按纯顺序把 token 分给 GPU，而是**打乱顺序，让每张卡上既有早期 token 又有晚期 token**。数一数着色的方块就会发现：**计算量在各卡间平衡了**。

> 与 **Striped Attention** 略有不同，细节见原文引的 GitHub discussion。

代价：为了算完所有行，**每张卡都需要来自所有其它卡的信息**（朴素切分下 GPU1 不需要任何人的）。

### 两种通信实现

| | all-gather 实现 | all-to-all（ring）实现 |
|---|---|---|
| 做法 | 所有卡同时把完整 K/V 收齐（ZeRO-3 风格） | 环状逐块交换 |
| 临时显存 | **高**（每卡要同时存下全部 K/V） | **低**（每卡只多存一块） |
| 通信形态 | 一步到位，但显存开销大 | 分散、与计算重叠，但多步带来额外基础延迟 |

**all-to-all 显存效率更好、通信模式更复杂；all-gather 简单但临时显存大。**

---

## 6. 与本项目的关系（含一条超出 playbook 的实测）

- **V8.2 的 `cp_utils.py`「2-chunk 对称切分」就是 Zig-Zag 这一族**：把序列切成 `2·cp` 块，每个 rank 拿一头一尾两块，正是本章说的「早期与晚期 token 混合」。
- **playbook 说 Zig-Zag 的动机是「负载均衡」（性能问题）；nano 的 G3c 反例实测发现它同时是「正确性必需」**——把 2-chunk 对称切分换成朴素连续切分后，cosine 从 0.99921 掉到 **0.588**。原因是 mcore/TE 的 CP 内核**假定**了这个 chunk 布局并据此重建因果结构，换布局就算错。也就是说：在具体实现里，"怎么切" 是内核契约的一部分，不只是调度优化。这条在 playbook 里没有。
- **CP 的 all-reduce 同步梯度那句**（"just like data parallelism, after computing the gradients, an all-reduce is initiated to synchronize gradients across the CP group"）**正是 V8.2 那条「计划里没有的新发现」**——nano 用裸 torch AdamW 没有 mcore DDP wrapper，**没人做这个 all-reduce**，于是 CP>1 时出现「loss Δ 恰为 0 而梯度系统性偏」的指纹。CP=1 时它是 no-op，所以 V8 从没暴露。**playbook 这一句就是那个 bug 的正解，只是当时没读到。**
- CP 与 FlashAttention 共用 online softmax 这条，也解释了 V8.2 的一条硬限制：**CP 只有 TE/flash 后端支持 → FA2 拒收 fp32 → CP 门不可能有 fp32 档**。
