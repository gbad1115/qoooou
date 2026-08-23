# Top-5 全排列顺序敏感性诊断：实验结论（500 题，obsexp.md）

> 设置：`results/obs.md`；详细表/图：`results/perm_diag/analysis_summary.md` + `distance_bucket_summary.csv` + `reverse_tail_analysis.csv` + `kendall_vs_reward_drop.pdf` + `kendall_vs_hcf.pdf`。
>
> **Backbone：Llama-3.1-8B-Instruct (base，无 RL)** · greedy（temp=0）· **500 题**（eval_1000 池 4000, seed=42 取前 500，含先前 200 为子集）× 5! = 120 全排列 = **60,000 次推理** · 8 路数据并行（A 方案，8× vLLM TP=1）。reward/EM/F1 全复用项目逻辑未改。Kendall 距离 = 相对检索序逆序数（0–10，reverse=10）。本结论为 500 题最终版，如实呈现未筛选。

## 一句话结论
**「Ranking 位移越大 → 行为退化越大」在群体平均层面成立且 Reverse 为退化峰值；但逐题层面不可靠（Spearman ρ≈0、CI 跨 0）。Reverse 是平均最害的结构化 counterfactual（聚合三项峰值、HCF 偏高），却非逐题 worst-case proxy（harmful percentile ≈53 中位、Tail 低于随机基线）。**

## 关键数字（500 题，retrieval-correct n=136，27.2%）

**Analysis 1 — 按 Kendall 距离聚合（retrieval-correct）**

| Kendall d | 0 | 2 | 4 | 6 | 8 | 10(reverse) |
|---|---|---|---|---|---|---|
| reward_drop | 0 | 0.089 | 0.108 | 0.122 | 0.130 | **0.141** |
| F1_drop | 0 | 0.105 | 0.126 | 0.143 | 0.154 | **0.161** |
| HCF % | 0 | 12.0 | 14.9 | 17.6 | 18.9 | **19.1** |

- retrieval-correct 子集：reward_drop / F1_drop / HCF **随位移单调升至 reverse(kd10) 取三项峰值**（500 比 200 更单调）。
- 全部 500 题：reward_drop **持平为负**（−0.006~−0.013）——非检索序排列平均 reward 略高，总体无退化趋势。

**Analysis 2 — per-question Spearman(Kendall, RewardDrop)**

| 子集 | mean ρ | median ρ | 95% CI | 正向比例 | n |
|---|---|---|---|---|---|
| 全部题 | −0.011 | −0.013 | [−0.030, +0.006] | 46.2% | 500 |
| retrieval-correct | +0.029 | +0.010 | [−0.009, +0.065] | 50.7% | 136 |

→ 整体≈0；retr-correct **弱正相关但 CI 明确跨 0、正向率≈51%**——**不稳健**。

**Analysis 3 — Reverse harmful percentile（retrieval-correct 136 题）**

| 指标 | 值 | 随机基线 |
|---|---|---|
| mean / median reverse percentile | **53.0 / 50.0** | 50 |
| Tail@5% | 2.2% | 5% |
| Tail@10% | 8.1% | 10% |
| Tail@20% | 17.7% | 20% |
| HCF：Reverse / Average / Random | **19.1%** / 15.9% / 16.9% | — |

→ reverse percentile **≈53（中位）**，Tail 率**均低于随机基线**；但 reverse 的单点 HCF 高于 avg/random，且聚合三项峰值。

## RQ1：位移越大是否越退化？
**群体成立、逐题不可靠。**
- ✅ retrieval-correct 聚合：reward_drop 0→0.141、F1_drop 0→0.161、HCF 0→19.1% 随位移**单调升至 reverse**，饱和消失（500 更稳）。
- ❌ 逐题 Spearman ρ≈+0.029、95% CI [−0.009, +0.065] **跨 0**、正向率 51%——**不稳健**；全部题 ψ≈−0.011 无趋势。
- ❌ 全题 reward_drop 为负（检索序非全局最优）。
- 结论：**聚合平均意义上"位移越大退化越大"成立且有 reverse 峰值，但非逐题可靠规律**（大量"位移大却退化小/并列"的题）。

## RQ2：Reverse 是否稳定落入 harmful tail、可作训练用单一 counterfactual？
**Reverse 是平均最害的结构化排列，但非逐题 worst-case proxy。**
- ✅ 聚合三项（reward_drop/F1_drop/HCF）reverse 均为峰值；HCF 19.1% > avg/random(~16%)。**"最大化 Kendall 位移"在 500 题上对应最高平均行为退化**——比 200 题结论更强。
- ❌ 逐题 harmful percentile ≈53（中位），Tail@10% 8.1% < 10% 基线 → reverse **不稳定落入单题 harmful tail**。
- 二者不矛盾：reverse "平均最害"（被系统性偏高 + 部分题拉高），但 reward_drop 大量并列 + 单题内常存在同等/更害排列，使其**单题排名常居中**。
- 结论：reverse **可作合理单一结构化 counterfactual**（平均最害、构造确定、覆盖全位移、HCF 偏高），**但不应作逐题 worst-case proxy**——用 reverse 做 hard negative 会系统性低估小位移有害排列。

## 200 → 500 稳健性变化
| 量 | 200 | 500 | 解读 |
|---|---|---|---|
| retr-correct n | 47 | 136 | ~3× 更稳 |
| reverse 为聚合峰? | 否(kd8峰) | **是** | 500 下 reverse 转峰值 |
| retr-correct Spearman ρ | +0.056 | +0.029 | 更大样本**削弱**逐题相关 |
| reverse percentile | 48.4 | 53.0 | 仍≈中位 |
| Tail@10% | 6.4% | 8.1% | 仍<基线 |

→ 更多数据**强化聚合单调趋势 + reverse 峰值**、**削弱逐题相关**、reverse-tail 判断不变。500 题结论更可靠。

## 对本研究的含义
1. **Reverse 作训练 counterfactual 合理且得到更强支持**：500 题下它在聚合意义上确为最害（reward/F1/HCF 三峰），覆盖全位移、构造确定——作为"单一结构化 hard 样本"比 200 题结论更站得住。
2. **但仍非逐题 worst-case proxy**：单题内常有位移更小却同等/更害的排列 + reward_drop 并列，使 reverse 单题排名中位。训练若只用 reverse，会**低估小位移有害排列**——支持 §4.2 four_orders/多排列采样方向。
3. **顺序鲁棒性优化主战场仍是"本就答对"的题**：位移-退化趋势仅显现在 retrieval-correct 子集（27%），与本工作 HCF 口径一致，**支持续用 HCF 作优化目标**；但聚合趋势 ≠ 逐题可靠，提示 HCF 的 per-question 信号本身有上限噪声。
4. **检索序非全局最优**：全题 reward_drop 为负 → reranker 在 base 上 +1–1.4 EM（§3.2）得到行为佐证。

## 局限
- 单 backbone（base Llama）、单种子采样（seed=42；推理 greedy 确定性）。500 题（retr-correct 136）已显著优于 200，但 RQ1 逐题 ρ CI 仍跨 0——若需定论"逐题是否相关"可进一步扩样本或换模型。
- reward_drop 含 cite 项，但 500 下 reward_drop / F1_drop / HCF 三维度峰值一致落在 reverse，结论对度量选择稳健。
