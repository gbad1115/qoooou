## BAR-RAG 系列实验总结

本文汇总目前在 `BAR-RAGbase` 中围绕 BAR-RAG / full_2x4 训练流程做过的主要实验，包括 Qwen3 系列、Qwen2.5 系列，以及早期 Qwen3-4B mini 复现实验和 Qwen3-32B 大模型流水线尝试。

除特别说明外，主表指标均来自 `DynamicRAG 512`，评测数据集为 `2wiki`、`hotpot`、`nq`、`triviaqa` 四个数据集，表中 `Avg EM/F1` 为四个数据集的简单平均。

---

## 1. 实验策略定义

### 1.1 `base`

- 不做 BAR-RAG 训练，直接用原始模型或未训练 baseline 做评测。
- 作用是衡量同一推理脚本、同一 DynamicRAG 数据下的基础能力。

### 1.2 `ordered rollout8`

- 保持 selector 输出文档顺序。
- generator 每个样本使用 `rollout n=8`。
- 训练目标主要是 answer correctness，即答对给奖励，没有显式顺序鲁棒约束。

### 1.3 `unordered rollout8`

- 使用 forward/reverse 文档顺序增强。
- 每个顺序仍倾向使用较多 rollout，总体更像“顺序数据增强”。
- 目标仍主要是 answer correctness。

### 1.4 `unordered_n4 / 2×4`

- 同一问题构造 forward + reverse 两种顺序。
- 每种顺序 `rollout n=4`，合计接近 `rollout8` 的总采样量。
- 目标是公平比较：在总 rollout 预算接近的前提下，引入顺序扰动。
- 关键局限：它只是把 forward/reverse 都放进训练，不一定显式惩罚“同题不同顺序答案不一致”。

### 1.5 `order_reward`

- 在 `unordered_n4` 基础上加入简单顺序相关 reward 加权。
- 目标是鼓励顺序鲁棒性，但实际效果整体偏弱。
- 主要问题是 reward 设计较粗，容易只改变平均 reward 标尺，而没有真正绑定同一问题的 forward/reverse 行为。

### 1.6 `flip_penalty`

- 对 harmful flip 做惩罚，即同一题某个顺序答对、另一个顺序答错时，对答错排列施加额外惩罚。
- 对部分模型有小幅改善，但不稳定。
- 局限是只惩罚失败排列，仍没有系统性鼓励多排列一致输出。

### 1.7 `order_balance`

- 后续在 Qwen2.5-3B 上新增的更细粒度 group reward。
- 核心思想：同一 `order_group_uid` 下比较不同顺序表现。
- 策略包括：
  - 弱顺序答对时给额外优势。
  - harmful flip 更直接惩罚。
  - 强顺序单边答对轻微降权，避免继续强化本来容易的顺序。
- 目前在 Qwen2.5-3B 上是最成功的 reward 改动。

---

## 2. Qwen3-1.7B 实验结果

模型：`Qwen3-1.7B`

### 2.1 DynamicRAG 512 主结果

| 实验 | 2wiki EM/F1 | hotpot EM/F1 | nq EM/F1 | triviaqa EM/F1 | Avg EM | Avg F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `base_1p7b` | 11.72 / 19.18 | 16.60 / 26.67 | 17.77 / 29.76 | 26.56 / 36.92 | 18.16 | 28.13 | 原始能力较弱 |
| `ordered_1p7b` | 16.99 / 23.52 | 23.83 / 33.53 | 27.34 / 38.38 | 37.11 / 46.51 | 26.32 | 35.49 | 大幅超过 base |
| `unordered_1p7b` | 17.38 / 23.91 | 23.63 / 33.20 | 27.93 / 38.63 | 37.89 / 48.21 | 26.71 | 35.99 | 略优于 ordered |
| `unordered_n4_1p7b` | 17.38 / 24.25 | 23.44 / 33.19 | 27.34 / 38.59 | 37.89 / 47.38 | 26.51 | 35.85 | 与 unordered rollout8 接近但略低 |
| `unordered_order_reward_1p7b` | 17.38 / 23.89 | 23.63 / 33.62 | 27.93 / 38.77 | 38.09 / 47.87 | 26.76 | 36.04 | 小幅最好，但提升很小 |
| `unordered_flip_penalty_1p7b` | 17.19 / 23.69 | 23.44 / 33.58 | 28.32 / 38.93 | 38.09 / 48.44 | 26.76 | 36.16 | F1 略好，EM 持平 order_reward |

### 2.2 辅助评测

`unordered_n4_1p7b` vs `unordered_order_reward_1p7b` 的补充结果：

| 实验 | random10 EM/F1 | random10 flip rate | order_probe forward EM | order_probe reverse EM | reverse drop | harmful flip all |
|---|---:|---:|---:|---:|---:|---:|
| `unordered_n4_1p7b` | 28.40 / 36.76 | 22.0 | 50.0 | 42.0 | 8.0 | 13.0 |
| `unordered_order_reward_1p7b` | 28.60 / 37.00 | 22.0 | 52.0 | 43.0 | 9.0 | 14.0 |

### 2.3 结论

- **训练本身有效**：`ordered` 相比 `base` 从 `18.16/28.13` 提升到 `26.32/35.49`，说明 BAR-RAG generator 训练对弱模型帮助明显。
- **unordered 有小收益**：`unordered rollout8` 平均 `26.71/35.99`，略高于 `ordered`。
- **2×4 没有明显赢 rollout8**：`unordered_n4` 平均 `26.51/35.85`，说明单纯把 rollout 拆成 forward/reverse 并不一定更优。
- **简单 reward 改动收益很小**：`order_reward`、`flip_penalty` 只带来约 `+0.05~0.17 F1` 级别提升，不构成强证据。

### 2.4 原因分析

- **1.7B 基础能力较弱**，主收益来自“学会更好答题”，而不是顺序鲁棒性。
- **2×4 降低了单一顺序下的 rollout 数**，从 `8` 变成 `4`，GRPO 优势估计噪声可能更大。
- **reward 没有真正 pairwise 绑定 forward/reverse**，所以模型仍可能只学到“当前排列下答对”，而不是“不同排列都稳定答对”。
- `order_reward` 让 forward 能力略升，但 reverse drop 和 harmful flip 没变好，说明它没有真正解决顺序一致性问题。

---

## 3. Qwen3-4B 实验结果

模型：`Qwen3-4B`

### 3.1 mini 闭环复现

早期 `Qwen3-4B` 主要用于跑通 mini 流水线：数据准备、selector GRPO、selector→generator、generator GRPO、merge、推理、评测。

| 实验 | Eval 规模 | EM | F1 | 结论 |
|---|---:|---:|---:|---|
| Base Qwen3-4B | 64 | 48.44 | 58.84 | baseline |
| Selector + Generator GRPO 30 step | 64 | 46.88 | 57.73 | 未提升，略退化 |

样本级 diff：64 题中，30 题两边都对，33 题两边都错，1 题回退，0 题改善，仅 3 题输出文本发生变化。

### 3.2 random10 顺序稳定性

| 实验 | mean EM | mean F1 | flip count | flip rate |
|---|---:|---:|---:|---:|
| `base_4b` | 33.80 | 43.33 | 26 | 26.0 |
| `ordered_4b` | 35.30 | 44.83 | 24 | 24.0 |
| `unordered_4b` | 35.00 | 44.35 | 22 | 22.0 |

### 3.3 结论

- **mini 流水线成功跑通**，解决了 chat template、LoRA merge、verl checkpoint、vLLM 加载等工程问题。
- **30 step GRPO 太短**，几乎没有改变模型行为，因此 mini 主指标没有提升。
- `random10` 上 `ordered/unordered` 相比 `base` 有小幅提升，`unordered` 的 flip rate 最低，但该结果更适合作为顺序稳定性参考，不适合直接等同于主实验结论。

### 3.4 失败/收益有限原因

- **训练步数不足**：30 step 对 4B + LoRA 来说明显不够。
- **评测集太小**：64 题噪声很大，单题变化即可导致较大波动。
- **早期流程主要在修工程链路**：很多时间花在 FSDP+LoRA merge、chat template、data_source reward 映射等问题上，不是最终训练配置。
- **推理分布曾经不一致**：早期 `qa_inference.py` 未正确使用 chat template，修复后才对齐训练分布。

---

## 4. Qwen3-8B 实验结果

模型：`Qwen3-8B`

### 4.1 DynamicRAG 512 主结果

| 实验 | 2wiki EM/F1 | hotpot EM/F1 | nq EM/F1 | triviaqa EM/F1 | Avg EM | Avg F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `base_8b` | 20.12 / 26.87 | 29.30 / 40.26 | 35.74 / 47.44 | 41.80 / 51.86 | 31.74 | 41.61 | baseline 已较强 |
| `ordered_8b` | 24.80 / 31.19 | 31.25 / 42.97 | 41.60 / 53.89 | 49.22 / 60.28 | 36.72 | 47.08 | 大幅提升 |
| `unordered_8b` | 25.20 / 31.68 | 31.84 / 43.20 | 42.19 / 53.96 | 50.20 / 60.71 | 37.36 | 47.39 | 最好，略优于 ordered |

### 4.2 结论

- **Qwen3-8B 是 Qwen3 系列里最清晰的成功结果之一**。
- `ordered` 相比 `base` 提升约 `+4.98 EM / +5.47 F1`。
- `unordered` 相比 `ordered` 继续提升约 `+0.64 EM / +0.31 F1`。
- 说明在模型能力足够强、训练规模足够时，BAR-RAG 训练和文档顺序增强都能带来稳定收益。

### 4.3 成功原因

- **基础模型能力足够**：8B 能利用 selector/generator 数据中的高质量信号。
- **训练不是 mini 级别**：相比 4B mini，full_2x4 训练规模更接近有效 RL 微调。
- **unordered 增强没有破坏答题能力**：Qwen3-8B 对顺序扰动有更强吸收能力，因此 unordered 能带来小幅额外收益。

---

## 5. Qwen3-32B 实验状态

模型：`Qwen3-32B`

### 5.1 当前状态

`unordered_32b` pipeline 已跑完整个训练和 merge 流程：

- generator vLLM for selector reward 启动成功。
- selector GRPO 完成。
- selector checkpoint merge 成功。
- selector→generator 数据生成成功。
- generator GRPO 完成。
- generator checkpoint merge 成功。
- pipeline 日志显示 `[DONE] unordered 32B pipeline finished.`

但当前 `results` 目录下没有找到 `full_2x4_10k_unordered_32b/dynamicrag_eval_512` 的四数据集主评测结果，因此 **32B 暂时不能和 1.7B/8B/Qwen2.5 系列做主指标横向比较**。

### 5.2 风险和注意事项

- `04_generator_training` 日志中出现过多条 `[Generator API Error] HTTPConnectionPool(... port=8194 ...)`，说明训练中可能存在 reward API 服务不可达片段。
- 虽然 pipeline 最终完成并产出 checkpoint，但如果 reward API 中途不可用，训练信号可能有污染。
- 在纳入论文/汇报结论前，需要补跑：
  - `DynamicRAG 512` 四数据集评测。
  - random10/order_probe 顺序稳定性评测。
  - 检查训练期 reward API error 对有效 step 的影响。

---

## 6. Qwen2.5-3B 实验结果

模型：`Qwen2.5-3B-Instruct`

### 6.1 DynamicRAG 512 主结果

| 实验 | 2wiki EM/F1 | hotpot EM/F1 | nq EM/F1 | triviaqa EM/F1 | Avg EM | Avg F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `base_qwen25_3b` | 19.14 / 25.33 | 25.98 / 35.82 | 33.20 / 43.97 | 42.77 / 52.07 | 30.27 | 39.30 | baseline |
| `ordered_qwen25_3b_rollout8` | 20.51 / 25.74 | 25.20 / 34.50 | 33.20 / 43.83 | 42.97 / 51.83 | 30.47 | 38.98 | EM 略升，F1 略降 |
| `unordered_n4_qwen25_3b_2x4` | 20.70 / 27.07 | 23.83 / 32.80 | 32.62 / 43.41 | 41.41 / 50.33 | 29.64 | 38.40 | 退化 |
| `unordered_order_balance_qwen25_3b` | 21.88 / 27.88 | 25.20 / 34.42 | 33.79 / 44.03 | 42.77 / 52.52 | 30.91 | 39.71 | 最好 |

### 6.2 结论

- **普通 2×4 对 Qwen2.5-3B 是失败的**：`unordered_n4` 从 `30.27/39.30` 降到 `29.64/38.40`。
- **ordered rollout8 也不明显**：EM 只提升 `+0.20`，F1 还下降 `-0.32`。
- **order_balance 是成功的**：相比 base 提升 `+0.64 EM / +0.41 F1`，相比普通 2×4 提升 `+1.27 EM / +1.31 F1`。

### 6.3 成功/失败原因

失败原因：

- **3B 模型容量有限**，单纯加入 forward/reverse 会增加训练分布复杂度，容易损害原本答题能力。
- **2×4 降低单排列 rollout 数**，优势估计更噪。
- **没有显式顺序目标时，unordered 只是数据增强**，不保证模型学会 order-invariant。

成功原因：

- `order_balance` 不只是简单加权，而是利用同组 forward/reverse 的相对表现。
- 对弱顺序答对给额外奖励，能把训练信号集中到真正需要改善的排列上。
- 对 harmful flip 的惩罚更直接，减少“只在容易顺序答对”的倾向。
- 复用 `unordered_n4` 的 generator 数据，只改变 reward，因而实验更能隔离 reward 改动的效果。

---

## 7. Qwen2.5-7B 实验结果

模型：`Qwen2.5-7B-Instruct`

### 7.1 DynamicRAG 512 主结果

| 实验 | 2wiki EM/F1 | hotpot EM/F1 | nq EM/F1 | triviaqa EM/F1 | Avg EM | Avg F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `base_qwen25_7b` | 19.34 / 23.78 | 26.76 / 35.98 | 35.35 / 45.62 | 41.60 / 51.74 | 30.76 | 39.28 | baseline |
| `unordered_qwen25_7b_rollout8` | 19.92 / 24.48 | 26.95 / 36.96 | 37.11 / 47.91 | 44.53 / 54.75 | 32.13 | 41.03 | 明显提升 |
| `unordered_n4_qwen25_7b_2x4` | 21.88 / 26.67 | 27.73 / 36.98 | 37.11 / 47.71 | 44.92 / 54.27 | 32.91 | 41.41 | 主实验最好 |
| `unordered_order_reward_qwen25_7b` | 19.14 / 23.83 | 27.34 / 36.87 | 34.57 / 45.10 | 44.53 / 53.88 | 31.40 | 39.92 | 不如 2×4 |
| `unordered_flip_penalty_qwen25_7b` | 20.51 / 25.22 | 28.32 / 37.31 | 35.74 / 46.90 | 44.34 / 53.56 | 32.23 | 40.75 | 有效但不如 2×4 |

### 7.2 random10 和 order_probe

| 实验 | random10 EM/F1 | random10 flip rate | order_probe forward EM | reverse EM | reverse drop | harmful flip all |
|---|---:|---:|---:|---:|---:|---:|
| `base_qwen25_7b` | 29.90 / 39.04 | 29.0 | 56.0 | 50.0 | 6.0 | 14.0 |
| `unordered_qwen25_7b_rollout8` | 31.50 / 40.96 | 30.0 | 61.0 | 50.0 | 11.0 | 16.0 |
| `unordered_n4_qwen25_7b_2x4` | 30.90 / 40.66 | 29.0 | 60.0 | 52.0 | 8.0 | 14.0 |

### 7.3 结论

- **7B 主实验支持 2×4**：`unordered_n4` 平均 `32.91/41.41`，优于 base 和 rollout8。
- **但顺序鲁棒性提升不是质变**：random10 上 rollout8 略高，flip rate 没改善；order_probe 上 2×4 的 reverse drop 比 rollout8 小，但仍有明显掉点。
- **简单 reward 改动不如 2×4 本身**：`order_reward` 和 `flip_penalty` 均没有超过 `unordered_n4`。

### 7.4 原因分析

- 7B 有足够能力吸收 2×4 的顺序增强，所以主指标有收益。
- 但 2×4 仍然是“顺序增强”而非“顺序一致性优化”，因此 random10/order_probe 的顺序鲁棒性指标没有显著改善。
- 简单 reward 在 7B 上可能改变了训练目标分布，但没有解决核心 pairwise consistency 问题，反而可能损害部分数据集表现。

---

## 8. 跨模型规律总结

### 8.1 主趋势

| 模型 | 最好主实验策略 | 最好 Avg EM/F1 | 对 base 的变化 | 结论 |
|---|---|---:|---:|---|
| Qwen3-1.7B | `flip_penalty/order_reward` 近似并列 | 26.76 / 36.16 | +8.60 / +8.03 | 训练显著有效，reward 收益很小 |
| Qwen3-4B | mini 未形成有效主结论 | 46.88 / 57.73 on 64 eval | -1.56 / -1.11 | mini 步数太少，主要验证工程链路 |
| Qwen3-8B | `unordered_8b` | 37.36 / 47.39 | +5.62 / +5.78 | 成功，unordered 略优于 ordered |
| Qwen3-32B | 无 DynamicRAG 主评测 | - | - | 训练完成但缺评测，暂不可比较 |
| Qwen2.5-3B | `order_balance` | 30.91 / 39.71 | +0.64 / +0.41 | 普通 2×4 失败，精细 reward 成功 |
| Qwen2.5-7B | `unordered_n4_2x4` | 32.91 / 41.41 | +2.15 / +2.13 | 2×4 主指标最好，但鲁棒性提升有限 |

### 8.2 成功策略

#### Qwen3-8B 的 `unordered`

- 成功原因：模型容量足够、训练规模足够、顺序增强没有压垮 QA 能力。
- 表现：相比 base 提升约 `+5.62 EM / +5.78 F1`，相比 ordered 也有小幅提升。

#### Qwen2.5-7B 的 `unordered_n4 / 2×4`

- 成功原因：在 rollout 预算接近的情况下，forward/reverse 扰动带来额外泛化收益。
- 但它对顺序鲁棒性的改善有限，说明其主要收益可能仍是数据增强，而不是严格 order invariance。

#### Qwen2.5-3B 的 `order_balance`

- 成功原因：对弱顺序和 harmful flip 的 reward 更有针对性。
- 这说明小模型不能只靠 unordered 数据增强，需要更明确的训练目标。

### 8.3 失败或收益有限的策略

#### 4B mini 30-step GRPO

- 失败原因：训练步数太短、eval 太小、主要处于工程打通阶段。
- 结论：不能据此否定 BAR-RAG，只能说明 mini 设置不够训练出收益。

#### 普通 `unordered_n4` 在 Qwen2.5-3B 上退化

- 失败原因：小模型容量有限，2×4 引入额外顺序分布复杂度，同时单顺序 rollout 数减少。
- 说明：低容量模型上 unordered augmentation 可能需要 reward 辅助，否则会损害 QA 能力。

#### 简单 `order_reward`

- 在 1.7B 上仅小幅提升。
- 在 Qwen2.5-7B 上不如 `unordered_n4`。
- 原因：简单加权无法真正表达“同一题不同顺序应该一致答对”。

#### `flip_penalty`

- 在 1.7B/Qwen2.5-7B 上有小收益，但不稳定。
- 原因：它只惩罚错误排列，没有充分奖励跨排列一致性，也没有处理多排列下的整体方差。

---

## 9. 对“顺序鲁棒性没有明显提升”的核心解释

当前多数 unordered / 2×4 实验存在同一个核心问题：

> 它们更多是 order augmentation，而不是 order-invariance optimization。

也就是说：

- 模型看到 forward 样本，也看到 reverse 样本。
- 但 reward 多数情况下仍是每条样本单独算 answer correctness。
- 如果 forward 答对、reverse 答错，训练会惩罚 reverse 样本，但不会明确告诉模型：“这两个样本是同一个问题，你应该在两种顺序下输出一致答案”。

因此会出现：

- DynamicRAG 主指标有提升，但 random10 flip rate 没变好。
- forward EM 提升，但 reverse drop 仍然存在。
- 2×4 比 rollout8 主指标略好，但顺序鲁棒性不是质变。

---

## 10. 后续建议

### 10.1 优先做 pair/group-level consistency reward

建议把 reward 从 sample-level 升级到 pair/group-level：

```text
final_reward = qa_reward + lambda * consistency_reward
```

其中 `consistency_reward` 可以是：

- forward/reverse 答案一致奖励。
- 多排列答案 F1 平均一致性。
- `min(reward_forward, reward_reverse)`。
- `average_reward - alpha * reward_variance`。

这样目标会变成：不仅要答对，还要不同文档顺序下稳定答对。

### 10.2 从 `2×4` 扩展到 `4×2`

当前只有 forward/reverse 两种顺序，扰动空间太小。可以尝试：

- original order。
- reverse order。
- random shuffle。
- answer-bearing docs pushed late。
- hard negative pushed early。

保持总 rollout 预算不变时，可从 `2 orders × 4 rollouts` 改成 `4 orders × 2 rollouts`。

### 10.3 只在 order-sensitive subset 上训练

筛选以下样本做二阶段 fine-tune：

- forward 答对、reverse 答错。
- reverse 答对、forward 答错。
- 多随机排列答案方差大。
- relevant doc 不在 top-1。
- hard negative 排在前面时容易错。

这样能减少大量 order-insensitive 样本对训练的稀释。

### 10.4 做 curriculum

推荐两阶段：

```text
Stage 1: ordered rollout8 学会基础 QA
Stage 2: unordered 2×4/4×2 + group consistency reward 学顺序鲁棒
```

这尤其适合 3B/1.7B 这类小模型，因为它们直接 unordered 训练容易损害基础答题能力。

### 10.5 补齐 32B 主评测

`Qwen3-32B` 已有训练和 merge 产物，但缺 DynamicRAG 主结果。建议优先补：

- `DynamicRAG 512` 四数据集。
- random10。
- order_probe。
- 训练日志 reward API error 审计。

只有补齐这些，才能判断 32B 是否存在“强模型天然鲁棒、训练收益更小”或“大模型能更好吸收 unordered”的趋势。

---

## 11. 文件与结果来源

主要结果来源：

- `results/full_2x4_10k_base_1p7b/dynamicrag_eval_512`
- `results/full_2x4_10k_ordered_1p7b/dynamicrag_eval_512`
- `results/full_2x4_10k_unordered_1p7b/dynamicrag_eval_512`
- `results/full_2x4_10k_unordered_n4_1p7b/dynamicrag_eval_512`
- `results/full_2x4_10k_unordered_order_reward_1p7b/dynamicrag_eval_512`
- `results/full_2x4_10k_unordered_flip_penalty_1p7b/dynamicrag_eval_512`
- `results/full_2x4_10k_base_8b/dynamicrag_eval_512`
- `results/full_2x4_10k_ordered_8b/dynamicrag_eval_512`
- `results/full_2x4_10k_unordered_8b/dynamicrag_eval_512`
- `results/qwen25_3b_threeclass_dynamicrag512_compare.json`
- `results/qwen25_3b_order_balance_dynamicrag512_compare.json`
- `results/full_2x4_10k_qwen25_7b_threeclass_compare/threeclass_summary.json`
- `results/full_2x4_10k_unordered_order_reward_qwen25_7b/dynamicrag_eval_512`
- `results/full_2x4_10k_unordered_flip_penalty_qwen25_7b/dynamicrag_eval_512`
- `scripts/full_2*4/MINI_RUN_SUMMARY.md`
- `scripts/full_2*4/random10-4b/results`
- `logs/full_2x4_10k_unordered_32b`

