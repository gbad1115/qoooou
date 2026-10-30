# Qwen3-8B（thinking mode）Full Cache vs RPC 评测结果

日期：2026-10-30
状态：AIME25 两组全完成；MATH500 部分完成（full ~176/500，rpc ~192/500，均在续跑）

## 实验设置

- **模型**：Qwen3-8B，`/home/admin/Qwen3-8B`，thinking mode（`enable_thinking=True`）
- **采样参数**（全方法统一）：temperature=0.6, top_p=0.95, top_k=20（Qwen3 thinking 推荐值），do_sample
- **生成上限**：32768 tokens，生成到 EOS 或上限
- **RPC 配置**：P=1024, R=32, c=4，selectors=recent，aggregation=all（对层和头取平均），kernel=7 avgpool（官方实现，移植到 Qwen3 + transformers 4.56）
- **数据**：`eval/data/aime25.jsonl`（30 题）、`math500.jsonl`（由 `MATH-500/test.jsonl` 转换，官方同款 prompt）
- **评分**：官方 `eval/eval/eval.py`（math_opensource grader，pass@1）
- **运行**：8 × H20-3e 分片（shard 8），batch_size=8

## 结果汇总

| 数据集 | 方法 | n | pass@1 | 平均生成 token | 打满 32k 比例 | 单卡显存 | 运行时间 |
|---|---|---|---|---|---|---|---|
| AIME25 | Full cache | 120（30×4） | **70.0%** | 31,347 | 82.5% | 115 GB | ~2.7h |
| AIME25 | RPC 1024-32-4 | 120（30×4） | **46.7%** | 32,639 | 94.2% | 18 GB | **~1.0h（2.7× 加速）** |
| MATH500 | Full cache | 176（部分） | **95.45%** | 10,392 | 0% | 115 GB | — |
| MATH500 | RPC 1024-32-4 | 192（部分） | **95.31%** | 15,042 | 8.3% | 18 GB | — |

## 同题对比

### AIME25（30 题，每题 4 采样）

- 逐题：RPC 更差 16 题 / 更好 1 题 / 持平 13 题
- 结论：长轨迹（~32k）下压缩损失显著（-23.3 点）

### MATH500（同 152 题重叠）

| | Full | RPC |
|---|---|---|
| 准确率 | 95.4% | 94.1% |
| 逐题 | — | 更好 3 / 更差 5 / 持平 144 |

结论：中等长度轨迹（~10-15k）下 RPC 几乎无损（-1.3 点，噪声范围内）。

## 解读

1. **加速收益普适**：2.7× wall-clock、显存 6.4×↓（115→18 GB），与论文量级一致
2. **精度代价长度依赖**：
   - AIME25 平均 32k 全打满上限，经历 ~31 次压缩 → 掉 23 点
   - MATH500 平均 10-15k，~10-14 次压缩 → 基本持平
3. **生成长度反直觉**：RPC 组平均生成长度更长（AIME 32.6k vs 31.3k；MATH500 15.0k vs 10.4k），打满上限比例更高——压缩可能打断 thinking 连贯性，模型在绕路；简单题绕得回来，难题绕不回来
4. **与论文差异**：论文报告 AIME24 上 ~1.2% 掉幅（DeepSeek-R1-Distill-Qwen-7B）；本实验中 Qwen3-8B 在 32k 超长轨迹上掉幅大得多，说明结论与模型/轨迹长度分布强相关

## 数据文件

- 预测：`eval/outputs/qwen3-8b/`（`*__merged.jsonl` 或 shard 文件）
- 评分明细：`eval/eval_res/qwen3-8b/aime25__full.jsonl`、`aime25__rpc.jsonl`、`math500__full_partial.jsonl`、`/tmp/math500_rpc_partial_scored.jsonl`
