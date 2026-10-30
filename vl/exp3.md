# Qwen3-8B（thinking mode）Full Cache 评测结果

日期：2026-10-30
状态：AIME25 已跑完；MATH500 跑完约 1/3（续跑中，RPC 组未开始）

## 实验设置

- **模型**：Qwen3-8B，`/home/admin/Qwen3-8B`，thinking mode（`enable_thinking=True`）
- **采样参数**（全方法统一）：temperature=0.6, top_p=0.95, top_k=20（Qwen3 thinking 推荐值），do_sample
- **生成上限**：32768 tokens，生成到 EOS 或上限
- **方法**：Full cache（不压缩）
- **数据**：`ReasoningPathCompression/eval/data/aime25.jsonl`（30 题，官方格式）、`math500.jsonl`（由 `MATH-500/test.jsonl` 转换，官方同款 prompt 模板）
- **评分**：官方 `eval/eval/eval.py`（math_opensource grader，pass@1 = 每条样本正确与否的均值）

## 结果汇总

| 数据集 | n（条） | pass@1 | 平均生成 token | 生成 min / max | 打满 32k 上限比例 |
|---|---|---|---|---|---|
| AIME 2025 | 120（30 题 × 4 采样） | **70.0%** | 31,347 | 18,874 / 32,768 | 99/120（82.5%） |
| MATH-500 | 176（题 × 1 采样，部分完成） | **95.45%** | 10,392 | 3,752 / 16,917 | 0/176（0%） |

## AIME25 逐题明细

参考：每题 4 次采样，下表为各题正确次数（格式：题目前缀 → 4 次中的正确数）。

```text
84 / 120 条正确（70.0%）。逐题正确次数分布见 eval/eval_res/qwen3-8b/aime25__full.jsonl
```

## 备注

- AIME 生成普遍打满 32k 上限（82.5%），模型反复验算到截断；pass@1 可能受长度截断影响（对 RPC 组同样适用，可对比）。
- MATH500 部分结果代表前 ~176/500 题，全量跑完后的数字可能浮动。
- 数据文件：
  - 预测：`eval/outputs/qwen3-8b/aime25__full__merged.jsonl`、`math500__full__shard*of8.jsonl`
  - 评分明细：`eval/eval_res/qwen3-8b/aime25__full.jsonl`、`math500__full_partial.jsonl`
- 待 RPC 组（P=1024, R=32, c=4）跑完后在本文档追加对比表格。
