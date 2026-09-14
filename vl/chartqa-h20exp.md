# ChartQA 复现结果对比记录（本地复现 vs 原代码仓库记录）

- 本地复现时间：2026-09-14 00:59 – 01:51（三次运行顺序执行）
- 对比基准：本仓库 README「重新复现结果」表（commit `3af2071`，2026-08-03，`README.md:44` / `README_zh.md:30`）
- 本文档只记录事实数据，不包含推测

## 1. 本地复现实验设置

来源：各次运行的 `hawk-vlmeval-manifest.json` 与 `validation/vlmeval_chartqa_prune*.json`。三次运行的设置完全相同，仅剪枝比例不同：

| 项目 | 值 |
|---|---|
| 评测框架 | VLMEvalKit（`.runtime/VLMEvalKit`） |
| 模型 | Qwen2.5-VL-7B-Instruct（`/home/admin/Qwen2.5-VL-7B-Instruct`），模型名 `Qwen2.5-VL-7B-HAWK-p60/p80/p90` |
| 数据集 | ChartQA_TEST（本地 TSV：`data/vlmeval/ChartQA_TEST_local.tsv`，2501 行，SHA256 `b1f99ffb…c7dd`） |
| 分辨率模式 | 原生动态分辨率（native） |
| min_pixels / max_pixels | 1,003,520 / 12,845,056 |
| attn_implementation | sdpa |
| 分数归一化 | l2 |
| 注意力头权重 | 28 维向量，和为 1.0000000000000002，三次运行相同（SHA256 `7e77fa51879a361e6d9ff517a315cdc57302f3c1bd11faa1c677e8003f5153d8`） |
| 剪枝率 | 0.6 / 0.8 / 0.9（keep_ratio 0.4 / 0.2 / 0.1） |
| GPU | 8 卡（0–7），8 进程 |

运行产物均包含逐样本剪枝统计 `traces/pruning.rank0–7.jsonl`。采样显示实际 keep ratio 与请求值一致（如请求 0.4，实际 0.4006 / 0.4000）。

## 2. 结果对比

### Overall 准确率（%）

| 剪枝率 | 论文值 | 仓库 README 记录（2026-08-03） | 本次本地复现（2026-09-14） | 本地 − 仓库记录 | 本地 − 论文 |
|---:|---:|---:|---:|---:|---:|
| 60% | 83.600 | 85.160 | **85.32** | +0.160 | +1.720 |
| 80% | 76.800 | 79.040 | **79.16** | +0.120 | +2.360 |
| 90% | 65.200 | 67.440 | **67.48** | +0.04 | +2.280 |

### 分项准确率（%）

| 剪枝率 | human_test | augmented_test | Overall |
|---:|---:|---:|---:|
| 60% | 76.24 | 94.40 | 85.32 |
| 80% | 66.16 | 92.16 | 79.16 |
| 90% | 51.12 | 83.84 | 67.48 |

注：仓库 README 的记录表只给出 Overall，未记录 human/augmented 分项，因此无法做分项对比。

## 3. 实验设置对照

本地复现设置与仓库 README 声明的复现设置逐项对照：

| 设置项 | 仓库 README 声明 | 本次本地复现 | 是否一致 |
|---|---|---|---|
| 分辨率 | native，`min_pixels=1,003,520`、`max_pixels=12,845,056`（`README_zh.md:127`） | 相同 | 一致 |
| 分数归一化 | L2（`README.md:60`） | l2 | 一致 |
| 注意力头权重 | 原始 HAWK 向量，公开权重 L1 归一化和为 1；统一正数缩放不改变 top-k（`README_zh.md:140`） | 同一向量，和为 1，SHA256 三次相同 | 一致 |
| generation state | 修正后的 generation state（`README.md:60`） | 未在 manifest 中单独记录 | 无法从产物直接核对 |
| 模型 / 数据来源 | Hugging Face 官方客户端及标准缓存 | 模型本地路径 + 本地 TSV（同一数据集 ChartQA_TEST） | 来源相同，路径形式不同 |
| GPU 配置 | 示例命令为 `--gpus 0,1,2,3`（4 卡，`README_zh.md:109`） | 8 卡 8 进程 | 示例不同（非结果对比项） |

本地 manifest 通过 `hawk-vlmeval-manifest.json` 记录了完整运行参数；仓库 2026-08-03 那批运行的 manifest 未入库（`vlmeval_results/`、`validation/vlmeval_*.json` 均在 `.gitignore:16-18` 中），仅 README 表格可查。

## 4. 文件位置

| 内容 | 路径 |
|---|---|
| 本次 60% 分数 | `vlmeval_results/chartqa_prune060_native/Qwen2.5-VL-7B-HAWK-p60/Qwen2.5-VL-7B-HAWK-p60_ChartQA_TEST_acc.csv` |
| 本次 80% 分数 | `vlmeval_results/chartqa_prune080_native/Qwen2.5-VL-7B-HAWK-p80/Qwen2.5-VL-7B-HAWK-p80_ChartQA_TEST_acc.csv` |
| 本次 90% 分数 | `vlmeval_results/chartqa_prune090_native/Qwen2.5-VL-7B-HAWK-p90/Qwen2.5-VL-7B-HAWK-p90_ChartQA_TEST_acc.csv` |
| 运行配置/清单 | 各 `vlmeval_results/chartqa_prune*/hawk-vlmeval-manifest.json` |
| VLMEvalKit 模型/数据配置 | `validation/vlmeval_chartqa_prune060/080/090_native.json` |
| 逐样本剪枝统计 | 各 `vlmeval_results/chartqa_prune*/traces/pruning.rank0–7.jsonl` |

## 5. 差异汇总

1. 数值差异：三个剪枝率的 Overall 均高于仓库 README 记录，差值 +0.04 ～ +0.16（60%: +0.160，80%: +0.120，90%: +0.04）。
2. 记录差异：本次结果带 human/augmented 分项与逐样本 traces；仓库 README 记录只有 Overall。
3. 文档状态：README「重新复现结果」表仍为 2026-08-03 的数字（85.160 / 79.040 / 67.440），未更新为本次结果。
4. 入库状态：本次全部产物（`vlmeval_results/`、`validation/vlmeval_*.json`）被 `.gitignore` 排除，仅存在于本地。
