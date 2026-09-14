# TextVQA / RealWorldQA 复现结果对比记录(本地复现 vs 论文 vs README)

- 本地复现时间:2026-09-14 21:44 – 23:53(六次运行顺序执行,共约 2h09m)
- 对比基准:论文 Table 3 + 本仓库 README「重新复现结果」表(commit `3af2071`)
- 本文档只记录事实数据,不包含推测

## 1. 运行设置

来源:各次运行的 `hawk-vlmeval-manifest.json`(六个 run 设置一致,仅 keep_ratio 不同)。

| 项目 | 值 |
|---|---|
| 评测框架 | VLMEvalKit(`.runtime/VLMEvalKit`,源自 `ms-vlmeval 0.0.18`) |
| 模型 | Qwen2.5-VL-7B-Instruct(`/home/admin/Qwen2.5-VL-7B-Instruct`) |
| 模型名 | `Qwen2.5-VL-7B-HAWK-p60/p80/p90` |
| 分辨率模式 | 原生动态分辨率(native) |
| min_pixels / max_pixels | 1,003,520 / 12,845,056 |
| attn_implementation | sdpa |
| 分数归一化 | l2 |
| 注意力头权重 | 28 维向量,和为 1.0000000000000002,SHA256 `7e77fa51879a…`(与 09-14 ChartQA 复现同向量) |
| GPU | 8 卡(0–7),8 进程 |
| keep_ratio / pruning_ratio | 0.4/0.6、0.2/0.8、0.1/0.9 |

文本侧数据来源:`data/vlmeval/TextVQA_VAL_local.tsv`(5000 题 / 3166 图)与
`data/vlmeval/RealWorldQA_local.tsv`(765 题 / 765 图),由
`scripts/prepare_textvqa_local.py`、`scripts/prepare_realworldqa_local.py`
从 `data/textvqa_validation.tar.gz`、`data/realworldqa_test.tar.gz` 离线生成
(列与 HF 路径逐字一致,HAWK 评测链路一字未改)。RealWorldQA 的 A/B/C/D/答案
取自随包附带的 `InternVL-Chat-V1-5_RealWorldQA.xlsx`(即原 HF `jefehern/vlmevalkit_inference`
那份答案键)。

## 2. 结果对比

### Overall 准确率

| 数据集 | 剪枝 | 本次复现 | 论文 | README 记录复现 | 本次−论文 | 本次−README |
|---|---:|---:|---:|---:|---:|---:|
| RealWorldQA | 60% | **66.93** | 67.600 | 67.712 | −0.672 | −0.782 |
| RealWorldQA | 80% | **64.84** | 65.000 | 62.353 | −0.163 | +2.484 |
| RealWorldQA | 90% | **59.74** | 60.400 | 60.000 | −0.661 | −0.261 |
| TextVQA | 60% | **85.088** | 85.000 | 85.000 | +0.088 | +0.088 |
| TextVQA | 80% | **83.308** | 83.000 | 83.156 | +0.308 | +0.152 |
| TextVQA | 90% | **79.444** | 79.800 | 79.152 | −0.356 | +0.292 |

(RWQA 比例为 0–1 小数 ×100,见各 `*_acc.csv`,误差四舍五入到 0.01。

### 各次耗时

| run | 起止 | 耗时 | exit |
|---|---|---|---|
| realworldqa_prune060_native | 21:44:55 – 22:05:44 | 20m49s | 0 |
| realworldqa_prune080_native | 22:05:44 – 22:23:27 | 17m43s | 0 |
| realworldqa_prune090_native | 22:23:27 – 22:41:42 | 18m15s | 0 |
| textvqa_prune060_native | 22:41:42 – 23:06:32 | 24m50s | 0 |
| textvqa_prune080_native | 23:06:32 – 23:30:22 | 23m50s | 0 |
| textvqa_prune090_native | 23:30:22 – 23:53:50 | 23m28s | 0 |

## 3. 文件位置

| 内容 | 路径 |
|---|---|
| 本次 RWQA 60/80/90 分数 | `vlmeval_results/realworldqa_prune0{60,80,90}_native/Qwen2.5-VL-7B-HAWK-p*/…_RealWorldQA_acc.csv` |
| 本次 TextVQA 60/80/90 分数 | `vlmeval_results/textvqa_prune0{60,80,90}_native/Qwen2.5-VL-7B-HAWK-p*/…_TextVQA_VAL_acc.csv` |
| 分数快照(已汇出) | `validation/{realworldqa,textvqa}_prune0{60,80,90}_native_acc.csv` |
| 运行配置 | `validation/vlmeval_{realworldqa,textvqa}_prune0{60,80,90}_native.json` |
| 运行清单 | 各 `vlmeval_results/<run>/hawk-vlmeval-manifest.json` |
| 逐样本剪枝统计 | 各 `vlmeval_results/<run>/traces/pruning.rank0–7.jsonl` |
| 推理日志 | `validation/logs/{realworldqa,textvqa}_p{0.6,0.8,0.9}.log` |
| 离线数据转换器 | `scripts/prepare_textvqa_local.py`、`scripts/prepare_realworldqa_local.py` |
| 原始数据包 | `data/textvqa_validation.tar.gz`、`data/realworldqa_test.tar.gz` |

## 4. 与 ChartQA 复现口径一致性

六个 run 的 `head_weights_sha256` 为 `7e77fa51879a361e6d9ff517a315cdc57302f3c1bd11faa1c677e8003f5153d8`,
与 09-14 ChartQA 三次复现(`validation/chartqa_reproduction_comparison_2026-09-14.md`)记录的向量
完全一致;分辨率、归一化、sdpa、8 卡设置也相同。因此本批结果与 ChartQA 复现同口径,可直接并表。

## 5. 差异汇总

1. TextVQA:三档与论文差 ±0.36 以内(60% +0.088、80% +0.308、90% −0.356),与 README 记录复现
   差 ±0.29 以内——基本完全复现。
2. RealWorldQA:三档与论文差 ±0.67 以内(60% −0.67、80% −0.16、90% −0.66);80% 档本次 64.84
   高于 README 记录复现 62.353 约 +2.48(更靠近论文 65.000),疑与运行间 sdpa 非确定性 + 图像
   重编码差异相关,方向与 README 记录值位于论文下方的整体趋势一致。
3. RealWorldQA 答案键 xlsx 中存在一行 `answer="Uphill"`(非 A–D),与原 HF 路径一致未做修正,
   影响为 1/765 ≈ ±0.13%。
4. 文档状态:README「重新复现结果」表当前仍为 2026-08-03 的数字,未含本次 TextVQA/RealWorldQA
   新复现;本记录与 `validation/*_acc.csv` 同在本地(仍在 `.gitignore` 排除范围)。
