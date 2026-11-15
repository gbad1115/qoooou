# AnchorMerge v2 评测结果汇总

## 1. 总览

所有结果均为百分比（%）。MT-GQA 报告 per-turn accuracy；MT-VQA-v2 报告官方十答案 VQA soft accuracy。ConvBench 成绩按要求暂时留空。**Full cache 指 `mode=none`，即保留全部 N 个视觉 token；解码仍禁用 KV cache并逐 token 全量重算。**

| Backbone | MT-GQA Full cache | MT-GQA fused@10% | MT-GQA fused@20% | MT-VQA-v2 Full cache | MT-VQA-v2 fused@10% | ConvBench |
|---|---:|---:|---:|---:|---:|---|
| LLaVA-1.5-7B | 57.64‡ | 56.5953 | 57.9742 | 0.4151 | 33.4930 |  |
| LLaVA-NeXT-7B | 58.9428 | 60.4531 | 62.1686 | 0.2746 | 45.8812 |  |
| LLaVA-NeXT-13B | 59.7390 | 59.3039 | 61.4545 | — | — |  |
| Qwen2.5-VL-7B | 51.0876 | 55.6021 | 57.4161 | 0.0000† | 1.1111† |  |

† Qwen 的 MT-VQA-v2 子集 Full cache 与 fused 结果均使用 FP32；其 MT-GQA 结果使用 BF16。两个 LLaVA 系列均使用 BF16。  
‡ LLaVA-1.5 的 MT-GQA Full cache 57.64 来自旧 harness，未使用当前 harness 重测。

---

## 2. MT-GQA：全量结果

数据：testdev-balanced，**4,061 个三轮对话 / 12,183 turn**。每图压缩一次，三轮复用。贪婪解码，最多 16 token，BF16、全量重算、禁用 KV cache。

| Backbone | 预算 | Fused | Random | Anchor | Δ Fused−Random | 显著性 p | Full cache (none) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | `n1=58`（约10%） | **56.5953** | 52.5404 | 51.6293 | +4.0549 | 4.5e-23 | 57.64‡ |
|  | `n1=115`（约20%） | **57.9742** | 54.2888 | — | +3.6854 | 1.1e-24 |  |
| LLaVA-NeXT-7B | `n1=244`（约10%） | **60.4531** | 55.4872 | 55.4872 | +4.9659 | 4.2e-32 | 58.9428 |
|  | `n1=488`（约20%） | **62.1686** | 56.6938 | — | +5.4748 | 1.6e-41 |  |
| LLaVA-NeXT-13B | `n1=244`（约10%） | **59.3039** | 54.9044 | 54.6089 | +4.3995 | 3.0e-27 | 59.7390 |
|  | `n1=488`（约20%） | **61.4545** | 56.1684 | — | +5.2861 | 3.0e-40 |  |
| Qwen2.5-VL-7B | `n1=41`（约10%） | **55.6021** | 41.7877 | 42.1407 | +13.8144 | 1.0e-180 | 51.0876 |
|  | `n1=83`（约20%） | **57.4161** | 44.9397 | — | +12.4764 | 9.4e-167 |  |

‡ LLaVA-1.5 的未压缩 57.64 来自旧 harness，未使用当前 harness 重测。

说明：

- `anchor` 是同一批 medoid 锚点、关闭融合的对照；当前汇总中 20% 档未列入 anchor 数字，因此留作 `—`。
- 显著性为 turn 级配对 McNemar，Yates 校正，p 值使用 `chi2.sf`。
- 受 16-token 截断影响，不能直接声称“压缩后优于未压缩”。可靠结论是 fused 显著优于同预算 random/anchor。

---

## 3. MT-VQA-v2：当前有图子集结果

数据口径：**522 个三轮对话 / 1,566 turn**，每个对话一张图。原始数据集为 25,000 对话 / 75,000 turn，但当前仅有 522 张所需图像可用，因此这里是 **image-available subset**，不是全量结果，也不能默认视为随机子集。

判分：VQA-v2 官方十答案 soft accuracy。生成上限 16 token，全量重算、禁用 KV cache。表中同时给出 Full cache（`none`，全部视觉 token）与 fused 10% 预算结果。

| Backbone | 模式 | 精度 | Turn 1 | Turn 2 | Turn 3 | Overall | 截断率 |
|---|---|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | Full cache | BF16 | 0.4406 | 0.3640 | 0.4406 | **0.4151** | 28.35% |
|  | fused@10% (`n1=58`) | BF16 | 31.9157 | 33.1992 | 35.3640 | **33.4930** | 13.41% |
| LLaVA-NeXT-7B | Full cache | BF16 | 0.0575 | 0.5747 | 0.1916 | **0.2746** | 53.26% |
|  | fused@10% (`n1=244`) | BF16 | 42.3946 | 47.5479 | 47.7011 | **45.8812** | 8.75% |
| Qwen2.5-VL-7B | Full cache | FP32† | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 68.39% |
|  | fused@10% (`n1=41`) | FP32† | 2.2797 | 0.6897 | 0.3640 | **1.1111** | 42.98% |

注意：

- VQA soft accuracy 要求规范化后的答案精确匹配，不能与 MT-GQA 的词边界包含准确率直接比较。
- Full cache 的回答明显更冗长，截断率也更高。很多语义上正确的完整句子因不符合精确匹配格式而得 0 分，因此这些接近 0 的数字主要反映**当前 16-token + 原始 prompt + 精确匹配协议**，不能直接解释为未压缩模型缺乏视觉理解能力。
- Qwen 的 Full cache 与 fused 均使用 FP32；LLaVA 两者均使用 BF16。
- 所有 VQA 分数均已独立重算，误差为 0.0；majority-answer sanity check 为 96.2580%。

---

## 4. ConvBench：成绩留空

原始推理已经完成全量 **577 个三轮对话 / 1,731 turn**，但官方评测需要 GPT judge，目前不填写准确率。

| Backbone | ConvBench accuracy |
|---|---|
| LLaVA-1.5-7B |  |
| LLaVA-NeXT-7B |  |
| Qwen2.5-VL-7B |  |

原始预测、参考答案、类别和 third-turn demands 均已保存，未调用外部评审服务。三张截断图像使用了明确标记的修复副本，后续评审时需要保留该说明。

---

## 5. 结果来源

- MT-GQA 方法、结果与注意事项：`multi/METHOD_v2.md`
- MT-VQA-v2 / ConvBench 汇总：`multi/outputs/conv_vqa_v2/SUMMARY.md`
- 独立审计：`multi/outputs/conv_vqa_v2/AUDIT.md`、`audit.json`、`fullcache_audit.json`
- 原始推理：`multi/outputs/mt_gqa/`、`multi/outputs/conv_vqa_v2/`
