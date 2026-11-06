# 方法架构文档: CrossAttnScorer —— 问题条件化的视觉 token 打分器

> 配套实验报告: EXPERIMENT_distill_v{1,2,3,4,5,6,7,8}.md / EXPERIMENT_heads_lastlayer.md
> 代码: harness/train_scorer_v8.py (模块定义 CrossAttnScorer) + harness/eval_local.py (method=distilled)
> 定稿配置 (v8, 2026-11-05 选定): 1×CrossAttnBlock(dim 768) + 全分布 KL(教师=h24_h2)
> + 熵匹配辅助 β·(H_s−H_t)², β=0.05。与旧 2-block d512 架构三基座统计打平(EXPERIMENT_distill_v8.md),
> 取结构更简者定稿。

## 1. 定位

一遍推理的视觉 token 重要性打分器，插在**视觉编码器+projector 之后、LLM 之前**
（与 VisionSelector 的 LIS 同一插槽）。与 LIS 的本质区别：**问题条件化**——
LIS 只吃视觉特征（问题无关），本模块让每个图像 token 先读问题再被打分。

## 2. 输入

| 输入 | 来源（全冻结） | 形状 | 说明 |
|---|---|---|---|
| V | 视觉塔 32 blocks + PatchMerger 输出 | N×3584 | 即 masked_scatter 进 inputs_embeds 前的视觉特征 |
| Q | LLM embed_tokens 查表 | N1×3584 | 图像 span 之后的文本 token（问题+模板尾） |

N 随图像动态变化（~260-1400），N1 随问题变化（~15-50）——模块对两者长度都自适应。

## 3. 架构（10,430,721 参数 = 10.43M）

```
V (N×3584) ──P_v──→ (N×768) ──┐
                              ├─ CrossAttnBlock ×1 (8头×96维)
Q (N1×3584) ──P_q──→ (N1×768) ┘        ↓
                              ln_out → (N×768) → MLP(768→256→1) → logits (N,)
```

### 3.1 输入投影（2 × 2,753,280 = 5,506,560）

```
P_v: Linear(3584→768) + bias
P_q: Linear(3584→768) + bias
```

### 3.2 CrossAttnBlock ×1（4,725,504）

组成（x=图像序列, kv=问题序列, 均为 768 维, 8 头 × 96 维）：

```
ln_q → W_q:  x  (N,768)  → (8, N, 96)
ln_kv → W_k:  kv (N1,768) → (8, N1, 96)
       W_v:  kv (N1,768) → (8, N1, 96)
scores = Q·Kᵀ/√96 → (8, N, N1)；问题 padding 位 -inf → softmax(N1 维)
out    = scores @ V   → (8, N, 96) → 合并 (N, 768)
x = x + W_o(out)                     （残差）
x = x + FFN(ln_f(x))                 （FFN: 768→1536→768, 残差）
```

| 子件 | 参数 |
|---|---|
| ln_q / ln_kv / ln_f | 3×1,536 = 4,608 |
| W_q / W_k / W_v / W_o（均无偏置） | 4×589,824 = 2,359,296 |
| FFN（768→1536→768，含偏置） | 2,361,600 |
| **合计** | **4,725,504** |

**机制**：每个图像 token 作 query 查询问题 token 的 K/V，把"和本问题相关"的
语义读进自己的表示，再进入打分头。cross-attn 的 K/V 长度任意 → 问题长度可变
不需要任何特殊处理；padding 位以 -inf 掩码（exp(-inf)=0，权重恒 0）。

### 3.3 输出头（198,657）

```
ln_out: LayerNorm(768)
scorer: Linear(768→256) → GELU → Linear(256→1)
```

### 3.4 参数总账（已和训练日志打印值对账: 10,430,721）

```
P_v + P_q        = 5,506,560
1 × CrossAttnBlock = 4,725,504
ln_out + scorer  =   198,657
────────────────────────────
合计             = 10,430,721  (10.43M, 比 VisionSelector 的 LIS 12.85M 小 19%)
```

移植到其他模型（同一模块，仅输入维度 D 变化）：LLaVA-1.5-7B / Qwen3-VL-8B
（D=4096）→ 11,217,153 ≈ 11.22M。

## 4. 训练（蒸馏）

- **教师**：LLM 最后一层（L27）头 {24, 2} 在最后一个 prompt 行对图像 token 的
  注意力分数，原始值平均后归一化：t = (s24+s2)/2, t̂ = t/Σt
- **损失**：
  ```
  L = KL(t̂ ‖ softmax(logits))  +  β · (H(softmax(logits)) − H(t̂))²
      前项: 全分布匹配（v1 以来未动）    后项: 熵匹配, β=0.05（v7 甜点区 [0.05,0.2] 取下端）
  ```
- **数据**：chartqa 28,138 + ocr_vqa 80,000 + coco%10 36,410 + textvqa-train 22,492
  （167,040；首轮问题；eval 同款 prompt 与分辨率；答案不进损失）
- **协议**：骨干全冻结，DDP 7 卡（GPU 0 被外部进程占用，全局 batch 56 = 7×8），
  1 epoch，AdamW lr 1e-4 cosine，2,048 留出验证，按 val top-k IoU 选 checkpoint

## 5. 推理（一遍，开销≈零）

```
图像 → 冻结视觉塔 → V → CrossAttnScorer → logits (N,)
问题 → 冻结 embed  → Q ↗
top-k = max(1, int(N×budget)) → 保留索引送入 LLM generate
任意 budget 推理（连续分数，无需重训）
```

## 6. 最终成绩（v8 @0.2 全量, vs selector 配对 McNemar）

| 条件 | ChartQA | TextVQA |
|---|---|---|
| **v8(定稿, β=0.05)** | **68.52**（-1.16 ns 打平 selector） | **79.00**（-0.88 ns 打平 selector） |
| v7d(旧 2-block d512, β=0.05) | 68.92（v8 比之 -0.40 ns） | 79.12（v8 比之 -0.12 ns） |

selector 基线：69.68 / 79.88；full cache：83.84 / 82.93。
⚠️ textvqa 为 in-domain（selector 零样本），论文须声明。
跨基座（v8 同配置）：LLaVA-1.5-7B 16.72/45.85、Qwen3-VL-8B 67.08/76.56，
均与各自旧 2-block 版本统计打平（详见 EXPERIMENT_distill_v8.md）。
