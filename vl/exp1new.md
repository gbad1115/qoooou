# LongBench-Pro 官方指标评测结果 — Qwen3-8B + SnapKV（保留 10% KV）

日期：2026-10-30（SemSim 于 10-30 改用 bge-m3 重算）
方法：Qwen3-8B thinking mode（YaRN factor=4.0，128k 窗口）+ SnapKV（kvpress `compression_ratio=0.9`，即每层保留 10% KV，prefill 后一次压缩、decode 不压缩）
采样：temperature=0.6, top_p=0.95, top_k=20；生成长度上限 8192 tokens；全部 1500 条（含 256k 组，context 截断到 128k）
SemSim：bge-m3（sentence-transformers，/home/admin/bge-m3），cosine of normalized embeddings，每条取 3 个参考摘要中的最大值

## Overall

**28.27**（n=1500，论文指标口径，×100 报告）

## 按长度档

| 长度 | 分数 | n |
|---|---|---|
| 8k | 26.91 | 250 |
| 16k | 29.36 | 250 |
| 32k | 30.82 | 250 |
| 64k | 25.42 | 250 |
| 128k | 29.36 | 250 |
| 256k（截断至 128k） | 27.76 | 250 |

## 按一级任务

| 任务 | 分数 | n |
|---|---|---|
| T1 Retrieval & Ranking | 54.03 | 120 |
| T2 Sequencing & Structure Reconstruction | 55.26 | 120 |
| T3 Evidence-Grounded QA | 35.00 | 120 |
| T4 Summarization & Synthesis | 48.51 | 120 |
| T5 Attribution & Citation Alignment | 5.72 | 120 |
| T6 Aggregation & Clustering | 27.09 | 180 |
| T7 Consistency & Compliance Checking | 12.99 | 180 |
| T8 Structured & Numeric Reasoning | 5.60 | 180 |
| T9 Version & Code Diff Analysis | 19.28 | 120 |
| T10 Rule Induction & In-Context Learning | 46.22 | 120 |
| T11 Dialogue Memory & Long-Horizon Tracking | 20.83 | 120 |

## 按二级任务（25 子集 × 官方指标）

| 子任务 | 指标 | 分数 | n |
|---|---|---|---|
| T1.1 Global Cohesive Retrieval | NDCG@k | 67.92 | 60 |
| T1.2 Key-Snippet Retrieval | NDCG@k | 40.15 | 60 |
| T2.1 Global Timeline Reconstruction | Pairwise | 69.53 | 60 |
| T2.2 Local Causal Chain Sorting | Pairwise | 40.99 | 60 |
| T3.1 Multi-Doc Integration QA | Accuracy | 33.33 | 60 |
| T3.2 Single-Hop Fact QA | Accuracy | 36.67 | 60 |
| T4.1 Global-Coverage Constrained Summary | 0.5·SemSim+0.5·ROUGE-L | 49.54 | 60 |
| T4.2 Query-Focused Summary | 0.5·SemSim+0.5·ROUGE-L | 47.49 | 60 |
| T5.1 Full-Sentence Citation Alignment | F1 | 8.81 | 60 |
| T5.2 Key-Statement Citation Alignment | F1 | 2.64 | 60 |
| T6.1 Large-Scale Document Clustering | SubEM | 12.81 | 60 |
| T6.2 Targeted Subset Cluster Identification | F1 | 22.17 | 60 |
| T6.3 Global Frequency Analysis | Pairwise | 46.29 | 60 |
| T7.1 Global Conflict & Inconsistency Localization | F1 | 8.41 | 60 |
| T7.2 Targeted Rule or Condition Violation Detection | F1 | 28.92 | 60 |
| T7.3 Comprehensive Error & Anomaly Sweep | F1 | 1.64 | 60 |
| T8.1 Structured Multi-Source Consistency Verification | SubEM | 13.47 | 60 |
| T8.2 Single-Source Targeted Aggregation | SubEM | 1.67 | 60 |
| T8.3 Long-Context Procedural State Tracking | SubEM（论文 SubEM/F1 不一致） | 1.67 | 60 |
| T9.1 Dependency-Aware Multi-Version Impact Analysis | F1 | 19.03 | 60 |
| T9.2 Localized Interface Change Detection | F1 | 19.53 | 60 |
| T10.1 Large-Scale In-Context Rule Induction | SubEM | 42.86 | 60 |
| T10.2 Targeted Example-Based Rule Induction | SubEM | 49.58 | 60 |
| T11.1 Long-Range Entity & Commitment Tracking | Accuracy | 16.67 | 60 |
| T11.2 Short-Range Reference Resolution & State Query | Accuracy | 25.00 | 60 |

T4 详情：SemSim（bge-m3）mean 0.761 / std 0.108；ROUGE-L mean 0.209。

## 口径说明（与论文协议的偏差）

1. **SemSim**：使用 bge-m3 计算（离线环境下可用的最佳 embedding 模型；论文未指明其官方实现）。比早前用 Qwen3-8B 隐层代理（mean 0.957）显著更合理。换成 bge-m3 后 T4 从 58.3 → 48.5，Overall 从 29.06 → 28.27——早前代理对 SemSim 高估约 0.20 分。
2. **256k 组截断方式**：论文为中间截断，本实验为头部截断（保留前 128k，kvpress pipeline 行为），256k 组分数与论文口径不完全可比。
3. **NDCG@k**：k = 参考答案条数，二元相关性。
4. **SubEM**：成分级归一化包含匹配取均值（论文未给完整公式）。
5. **T8.3**：正文 SubEM 与附录 F1 不一致，本实验用 SubEM（单值答案下两者等价）。
6. **单轮运行**：论文跑 3 次取均值，本实验为 1 次（temperature 采样）。
7. 此前占位 scorer（component recall）的 Overall 为 20.8，与本口径不可直接比较。

## 文件

- 逐样本明细：`results/longbench_pro_official_per_sample.csv`（含每条 score/semsim/rouge_l/metric，SemSim 为 bge-m3 版本）
- 指标汇总：`results/longbench_pro_official_metrics.json`
- 评分代码：`benchmarks/longbench_pro/official_metrics.py`（SemSimScorer 默认加载 /home/admin/bge-m3）
- 原始预测：`results/longbench_pro_merged_predictions.csv`
