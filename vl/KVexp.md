# lookattn 复现与扩展实验记录（2026-09-16）

本文档记录在本机对 README 各节任务的完成情况、实测数字、以及 README 之外的
8 卡数据并行（DDP）改造。结论先行：

- **README 主路径全部跑通**：环境 → 数据 → 单图推理 → 冒烟训练 → OCR_VQA 正式训练 → 混合语料训练。
- **OCR_VQA 复现优于参考值**：val KL = **0.2765**（README 第 2 节参考 0.2832）。
- **新增 8 卡 DDP**（README 没有的能力）：全局 batch 语义与单卡严格等价，混合 11,000 样本训练实测 **16.4 分钟**（单卡 5,499 样本约 80 分钟）。

---

## 1. 环境（对应 README 第 3 节，有改动）

| 项 | 实际使用 | 说明 |
|---|---|---|
| Python 环境 | `/ossfs/workspace/HAWK/.venv`（Python 3.10.13） | **未按 README 建 conda**；venv 基础依赖与 requirements.txt 完全对齐（torch 2.5.1 / torchvision 0.20.1 / accelerate 1.4.0 / Pillow 11.0.0 / numpy 1.26.4 / qwen-vl-utils 0.0.11） |
| 补丁运行时 | `.runtime/python`（`HAWK 0.1.0 Transformers 4.52.0` 校验通过） | 所有入口仍必须走 `bash scripts/run.sh` |
| 模型 | `/home/admin/Qwen2.5-VL-7B-Instruct`（16 GB 完整），软链 `models/Qwen2.5-VL-7B-Instruct` | 经 aistudio modelhub 内网同步（16.6 GB 仅 9 秒） |
| 评测依赖 | venv 已含 ms-vlmeval 0.0.18 / pandas / pyarrow / openpyxl / xlsxwriter | `requirements-eval.txt` 无需再装 |
| GPU | 8 × NVIDIA H20-3e（143 GB/卡），192 核 CPU，1.5 TiB 内存 | |
| 存储 | `/ossfs` 为阿里云 NAS（NFSv3）；模型在本地盘 `/home/admin` | NAS 小文件 I/O 慢：冷 import transformers ~5 min（8 路并发 ~10 min）；tar 选择性解压 ~0.3 s/文件 |

## 2. 快速验证：单图推理（README 第 4 节）✅

```bash
bash scripts/run.sh scripts/infer_lookhawk.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --scorer checkpoints/scorer_best.pt \
  --image assets/smoke_test.png --keep-ratio 0.198
```

通过判据全部满足：`pruning.applied=true`；视觉 token **486 → 97**（实际保留率 0.1996 ≈ 0.198）；生成文本非空且正确描述画面；峰值显存 15.6 GiB。

## 3. 训练数据准备（README 第 5 节）✅

- **下载（5.2）**：`jsons/Cambrian737k.jsonl`（765.5 MB）+ `ocr_vqa.tar.gz`（3.2 GB）+ `chartqa.tar.gz`（708 MB）+ `coco.tar.gz`（18.1 GB）全部到位（IDE 上传，~1.3 MB/s）。
- **filter（5.3 阶段 1）**：80,000 / 28,138 / 364,100 行，与 README 记载逐一相符。
- **split（5.3 阶段 2）**：
  - 复现用：`data/cambrian-ocr_vqa/` = 5,000/500（`overlap=0`）
  - 混合用：`data/cambrian-{ocr_vqa,chartqa,coco}-mix/` = 6,000/600 + 2,500/250 + 1,500/150（单位均为**图片**，去重无放回）
- **选择性解压（5.4）**：`tar -T members.txt` 共解出 5,500 + 5,903 + 2,750 + 1,650 张图；混合语料训练前已校验 **0 缺失**。
- **合并（5.5）**：`data/cambrian-mix/{train,val}.jsonl` = **10,000 / 1,000** 行。
- **76K 全量（5.6）**：未尝试——trainer 全量驻留的结构限制（显存 734 GB + 内存 1.9 TB > 本机 1.5 TiB）。

## 4. 训练结果（README 第 6.1 节）

### 4.1 冒烟（data/smoke，20 步）✅

管线端到端通过：`train_kl` 2.84 → 1.02，`teacher_kl_uniform=0.63`（教师非平，损失可信）。

### 4.2 OCR_VQA 复现（5,000/500，单卡，README 第 2 节同配置）✅

`--num-pseudo 32 --score-layer 1 --lr 2e-4 --steps 2000 --batch-size 32 --save-best`，耗时约 80 分钟。

| 指标 | 本次实测 | README 参考 |
|---|---|---|
| **val KL（28 头求和）** | **0.2765** | 0.2832 |
| recall@1 | 0.9340 | — |
| recall@2 | 0.9640 | — |
| recall@16 | 0.9370 | — |
| recall@64 | 0.9615 | — |
| teacher_kl_uniform | 0.8404 | — |

产物：`outputs/lookhawk_layer1/{scorer_best.pt, scorer.pt, config.json}`。

### 4.3 混合语料（10,000/1,000，8 卡 DDP）✅

同一配方，数据为 `data/cambrian-mix/`（ocr_vqa 6,000 + chartqa 2,500 + coco 1,500）。

| 指标 | 实测 |
|---|---|
| **val KL** | **0.4464**（随机初始化基线 ≈2.8 的 1/6.3） |
| recall@1 | 0.9550 |
| recall@2 | 0.9380 |
| recall@16 | 0.9377 |
| recall@64 | 0.9564 |
| teacher_kl_uniform | 0.8492 |
| 轨迹 | 3.96 → 0.62（400 步）→ 0.46（1000 步）→ 0.4464（2000 步） |

产物：`outputs/lookhawk_layer1_mix/{scorer_best.pt, scorer.pt, config.json}`。

## 5. 8 卡数据并行改造（README 之外的新增能力）✅

### 改动

- 新增 `src/lookhawk/dist.py`：NCCL 初始化（**延迟到 fork 编码池结束后**，否则 fork-after-CUDA 有死锁风险）、梯度 all-reduce(SUM)、指标汇总、rank0 门控。
- `src/lookhawk/layer1.py`：`train_step` 增加 `loss_divisor` / `on_grads_ready` 两个可选参数（单卡默认行为不变）。
- `src/lookhawk/train.py`：`encode_jsonl` 增加 `shard=(rank, world)`，每 rank 只编码/驻留 1/8 样本。
- `scripts/train_layer1.py`：分布式化；`--batch-size` 保持**全局**语义。
- `README.md`：第 6.1 节已补多卡启动说明。

### 正确性设计

每 rank 按 `1/全局 batch` 缩放各样本损失后做梯度 SUM —— 与单卡的全局均值**严格相等**，lr / steps / cosine 配方原样适用，无需重调。验证：8 卡冒烟的 `diag kl=2.822687` 与单卡回归**逐位一致**（manual_seed + rank0 broadcast 保证初始化相同）。

### 启动

```bash
bash scripts/run.sh -m torch.distributed.run --nproc_per_node=8 \
  scripts/train_layer1.py --batch-size 32 --steps 2000 --save-best \
  --train-jsonl <train.jsonl> --val-jsonl <val.jsonl> --out-dir <dir>
```

### 实测加速（2026-09-16）

| | 单卡 | 8 卡 DDP |
|---|---|---|
| 数据 | 5,499 样本 | 11,000 样本（2×） |
| 计算耗时（模型加载后） | ~80 min | **16.4 min** |

## 6. 发现的 README / 脚本问题

1. **`scripts/eval_recall_lookhawk.py` 只认 layer-0 checkpoint**（期望 `checkpoint["config"]`），对 layer-1 的 `pseudo_token` checkpoint 报 `KeyError: 'config'`——包括自带的 `checkpoints/scorer_best.pt`。README 第 6.2 节的离线评测命令对 layer-1 产物跑不通。layer-1 的 val 指标目前以训练日志 `[eval step N]` 为准。
2. **VLMEvalKit 端到端未接 lookahead scorer**（README 已自述）：`scripts/evaluate_vlmeval.sh` 只生成 `hawk_keep_ratio` / `hawk_head_weights` 的 layer-0 配置。
3. NAS 导致的两大慢点已在 README 无记载：import ~5 min；tar 选择性解压 ~0.3 s/文件。

## 7. 评测补齐（2026-09-17，全部完成）

### 7.1 新 scorer 推理冒烟 ✅

两个 scorer 分别跑 `infer_lookhawk.py`：均正常安装、剪枝 486→97（0.1996）、生成连贯。

### 7.2 layer-1 离线评测（新脚本 `scripts/eval_recall_layer1.py`）✅

OCR_VQA scorer（500 条 val，对标 README 第 2 节）：

| 指标 | student | HAWK 基线 | README 参考 |
|---|---|---|---|
| **KL** | **0.2765** | 1.8258 | 0.2832 vs 1.8750（比值 6.60× vs 6.62×，几乎逐位吻合） |
| **recall@deploy（k≈261，README 缺失的数字）** | **0.9776** | 0.9460 | —（新增） |
| recall@64 / @128 | 0.9620 / 0.9707 | 0.9024 / 0.9265 | — |
| 逐样本获胜 | **499/500** | | README 为 500/500 |

混合 scorer（1,000 条 val）：KL **0.4463** vs 1.7566；**recall@deploy 0.9718** vs 0.9440；逐样本获胜 953/1000。

报告落盘：`outputs/lookhawk_layer1/eval.json`、`outputs/lookhawk_layer1_mix/eval.json`。

### 7.3 VLMEvalKit 端到端接线 ✅

- `evaluate_vlmeval.sh` 新增 `SCORER=<checkpoint>` 环境变量；`hawk/vlmeval_adapter.py` 在 `configure_model` 后装载 lookahead scorer（`SCORER` 要求 `KEEP_RATIO<1.0`，模型名变为 `Qwen2.5-VL-7B-Lookahead-p<百分比>`）。
- `setup_vlmevalkit.sh` 已执行（`.runtime/VLMEvalKit` 复制+补丁）；TSV 复用 HAWK 的 3 份到 `data/vlmeval/`。
- 冒烟验证（RealWorldQA × 8 样本）全链路通过：scorer 装载、剪枝生成、评测出分。README 第 6.2 节已同步更新。

### 7.4 76K 内存墙修复 ✅

- `data.py`：`pixel_values` 编码侧改 **bf16 存储**（25→12 MB/行；视觉塔入口本来就转 bf16，舍入点位不变——冒烟 `diag kl=2.822687` 与修复前**逐位一致**）。76K+val 合计 83.6K 行的编码期内存 1.9 TB → **~1.0 TB**（< 本机 1.5 TiB）。
- `layer1.py` / `train_layer1.py`：`_prepare` 消费完即弃 `pixel_values`，训练期主机内存随样本数平坦。
- 叠加 DDP 分片：显存 ~91 GB/rank、内存 ~130 GB/rank，**76K 在本机可行**。

### 7.5 剩余待办

| 项 | 状态 |
|---|---|
| ScienceQA / MME 基准 | TSV 需下载，待用户指示 |
| 全量 VLMEvalKit 基准跑分（3 个本地 TSV） | 接线已验证，随时可跑 |

## 8. 76K 全量训练（2026-09-17，README 5.6 限制正式解除）

**配方**：`ocr_vqa 50,000/5,000 + chartqa 16,000/1,600 + coco 10,000/1,000`（README 5.1 原始 76K 配方，7,600 验证，共 83,600 样本），图片解压到 `/dev/shm`（tmpfs，分钟级，避开 NAS 写小文件），8 卡 DDP、全局 batch 32、2000 步、lr 2e-4 cosine、--save-best。

**资源实测（峰值）**：VRAM ~124 GB/rank（< 140 GB）；主机内存 ~1.0 TB（< 1.5 TiB，bf16 像素 + 用后丢弃生效）；/dev/shm ~12 GB。**此前"8×A800 640 GB 放不下"的 76K，本机一次跑通。**

**耗时**：import ~11 min → 编码（8 rank × 8 workers 读 tmpfs）~5 min → `_prepare` ~60 min → 2000 步训练 ~5 min → 40 次评测 ~25 min；计算段合计 **92 分钟**。

**结果**：

| 指标 | best（step 1600，`scorer_best.pt`） | final（step 2000） |
|---|---|---|
| **val KL** | **0.6528** | 0.6730 |
| recall@1 | 0.9547 | 0.9537 |
| recall@2 | 0.9370 | 0.9366 |
| recall@16 | 0.9307 | 0.9243 |
| recall@64 | 0.9503 | 0.9463 |
| teacher_kl_uniform | 0.8441 | 0.8441 |

**三方对照**（注意 val 集构成不同，且 76K 的 2000 步仅 0.84 epoch，而 11K 混合为 6.4 epoch）：

| 语料 | val KL | 备注 |
|---|---|---|
| OCR_VQA 5K（单卡） | 0.2765 | README 复现 |
| 混合 11K（8 卡） | 0.4463 | 6.4 epoch |
| **混合 76K（8 卡）** | **0.6528** | 0.84 epoch，欠训；step ~1570 出现一次 loss 尖峰（grad_norm 3.9），best 停在其后 |

**判读**：工程目标（76K 可跑）达成；质量上 76K 受步数限制尚未吃饱，按 README 配方同口径放大步数（如 15,000 步 ≈ 6 epoch）是下一步的自然实验，8 卡下约 40 分钟训练段。

产物：`outputs/lookhawk_layer1_76k/{scorer_best.pt, scorer.pt, config.json}`。

**离线评测**（`eval_recall_layer1.py`，全量 7,599 条 val，`outputs/lookhawk_layer1_76k/eval.json`）：student KL **0.6482** vs HAWK 基线 1.7589（2.71×）；**recall@deploy 0.9668** vs 0.9446；recall@64 0.9490 / recall@128 0.9573；逐样本获胜 6,676/7,599（87.9%）。

## 9. 端到端公平对比：HAWK vs Lookahead-10K vs Lookahead-76K（2026-09-17）

VLMEvalKit 端到端精度。两侧同数据集 TSV、同 native 分辨率（min_pixels 1,003,520 / max 12,845,056）、同 keep ratio（p80=0.2 / p90=0.1）、同 sdpa / l2 归一化（逐项核对 HAWK 侧 `hawk-vlmeval-manifest.json`）。HAWK 数字复用 `HAWK/vlmeval_results/` 现成结果；lookahead 两侧各 6 路评测（3 数据集 × 2 档，GPU 0-5 一路一卡，脚本 `scripts/run_lookahead_3x2.sh`）。

| 数据集 | 档位 | HAWK | 10K mix | 76K | 最优 |
|---|---|---|---|---|---|
| RealWorldQA | p80 | 64.84 | **64.97** | 64.31 | 10K（≈HAWK） |
| RealWorldQA | p90 | **59.74** | 57.65 | 58.43 | HAWK |
| ChartQA_TEST | p80 | **79.16** | 75.56 | 76.28 | HAWK |
| ChartQA_TEST | p90 | **67.48** | 58.00 | 58.84 | HAWK |
| TextVQA_VAL | p80 | 83.31 | 83.12 | **83.45** | 76K ✅ |
| TextVQA_VAL | p90 | 79.44 | 79.30 | **79.43** | 76K（≈HAWK） |

产物：`vlmeval_results/look_*/`（76K）、`vlmeval_results/mix10k_*/`（10K）。

**结论**：

1. **TextVQA 稳定打平/微超 HAWK**（两个 scorer 都是）——训练分布中 ocr_vqa 占 2/3，与 OCR 任务匹配，方法路子成立。
2. **ChartQA 是系统性短板**，与数据规模无关（10K/76K 都差 3–9 分）：训练语料的 chartqa 是 Cambrian 机器生成图表，评测集一半为人工标注真实图表（human_test 掉分最多：76K p90 human 46.5 vs augmented 71.2）。
3. **10K→76K 几乎无提升**（6 项中 76K 赢 4 项但仅 +0.1~0.8，rwqa p80 反而 10K 好）——见下方 loss 曲线分析：76K 训练在 step ~1570 发散，有效训练量与收敛的 10K 相当。

### 9.1 三次训练的 loss 曲线

![training curves](outputs/training_curves.png)

（生成脚本 `scripts/plot_training_curves.py`，数据源为三次训练的完整日志，每 10 步一条 train_kl，各 201 点。）

| Run | 起点 | 终点 | 最低 | 形态 |
|---|---|---|---|---|
| OCR-VQA 5K（单卡） | 2.70 | 0.28 | 0.21 | 平稳收敛 |
| Mix 10K（8 卡） | 2.81 | 0.58 | 0.28 | step ~200 尖峰（1.57）后 0.45–0.55 震荡 |
| Mix 76K（8 卡） | 3.02 | 1.44 | 0.20 | **step ~1570 从 0.25 跳至 1.4–1.75，持续发散到结束** |

**关键发现**：76K 的 step-1570 事件不是瞬时尖峰而是**低 lr 阶段的持续发散**；发散前（step ~1500）train KL 已降至 ~0.25，与 5K run 相当。评测所用 `scorer_best.pt` 为发散前救回（val KL 0.6528），76K 数据潜力远未榨干。下一步候选：更严 grad clip / spike 自动回滚 / step ~1500 早停，或按 README 同口径放大步数重训。

### 9.2 离线 recall 与端到端精度的反差

76K scorer 离线指标全面优于 HAWK（val 2,000 样本：recall@p80 0.976 vs 0.947、recall@p90 0.967 vs 0.928、逐样本获胜 1,994/2,000，见 `outputs/lookhawk_layer1_76k/eval_levels.json`），但端到端精度 6 项中 5 项仍低于 HAWK——**留住更多答案相关 token 不等于生成更好**，recall 提升可能集中在非关键 token，或 token 组合/位置对生成质量有影响。离线指标与端到端指标的关系需要进一步研究。

## 10. 数据与产物索引

```text
datasets/jsons/Cambrian737k.jsonl     标注（765.5 MB）
datasets/{ocr_vqa,chartqa,coco}.tar.gz 图片归档（22.8 GB）
datasets/*_cambrian.jsonl             filter 全量分流
data/cambrian-ocr_vqa/                复现 split（5,000/500）
data/cambrian-{ocr_vqa,chartqa,coco}-mix/ 混合 split
data/cambrian-mix/                    合并语料（10,000/1,000）
outputs/lookhawk_layer1/              OCR_VQA scorer（KL 0.2765）
outputs/lookhawk_layer1_mix/          混合 10K scorer（KL 0.4464）
outputs/lookhawk_layer1_76k/          混合 76K scorer（best KL 0.6528）+ eval.json + eval_levels.json
outputs/training_curves.png           三次训练 loss 曲线（第 9.1 节）
outputs/smoke*/                       冒烟与回归/DDP 验证产物
vlmeval_results/look_*/               76K scorer 端到端评测（3 数据集 × p80/p90）
vlmeval_results/mix10k_*/             10K scorer 端到端评测（同上）
scripts/run_lookahead_3x2.sh          6 路评测启动器（SCORER/PREFIX 可配）
scripts/plot_training_curves.py       loss 曲线生成脚本
```

---

## 11. 86K 一轮 epoch 训练（2026-09-19，coco +10K，batch 64，seed 42 shuffle）

**动机**：76K run 在 step ~1570 发散（§9.1），且事后发现 `Layer1Trainer.next_batch_ids`
只在跨 epoch 边界才 reshuffle——**第一个 epoch 是顺序喂的**。76K 的前 ~59% 步数实际按
ocr_vqa→chartqa→coco 固定课程训练，这很可能参与了发散。本轮把三个变量一起改：

- **数据**：coco 10K → 20K（`prepare_cambrian_subset.py` 新增 `--exclude-members`，
  新 10K 与旧 11K（含 val）构造性不相交），总量 50,000+16,000+20,000 = **86,000**，
  **val 保持 76K 原样（7,600）**，与上一轮逐样本可比。
- **shuffle**：两处落实。① `data/cambrian-mix-86k/train.jsonl` 物理 shuffle（seed 42，
  前 1000 行构成 571/247/182 ≈ 全局比例 ✓）；② 修 trainer——`__init__` 里建 `order`
  时立即 shuffle（`seed=42`），第一个 epoch 也是乱序。
- **配方**：全局 batch 64（8/rank）、**1,343 steps ≈ 1 epoch**、lr 2e-4 cosine 不变。

**训练器显存改造（本轮新坑）**：`Prepared` 从「全量常驻 GPU」改为「host 常驻 +
`stage()` 用时再搬」，GPU 从 ~124 GB/rank 降到 **~5 GB/rank**（模型 + 单样本临时量）。
注意第一版用 pinned memory 会**死锁**：8 rank × ~112 GB 的 `cudaHostAlloc` 在无 swap、
tmpfs 不可回收的 1.5 TiB 机器上触发 direct-reclaim 停摆（线程挂在
`wait_on_page_bit_common`，进度 ~0.1 MB/s）。改成**普通 pageable** 后正常——
单样本 ~10 MB 的 H2D 拷贝 ~1 ms，相对前向开销不可见；且释放的 pixel 显存arena被
后续的 embeds 复用，host 峰值 ~1.09 TB 与 76K 的编码峰值持平。改动后回归冒烟
step-1 `train_kl=2.424329` 与改前逐位一致（staging 对值无损；step 10 起的 1e-3 漂移是
GPU backward 原子序不确定性，非本改动引入）。

**训练实测**（8 × H20）：import ~11 min → encode ~6 min → prepare ~78 min →
1,343 步训练 ~6 min（0.28 s/step）→ 27 次评测 ~17 min；合计 ~2 h。
**全程无发散**：grad_norm 稳定在 0.3~4.8（76K 同期曾跳 3.9 后持续发散）。

| 指标 | 86K 本轮（best, step ~1150） | 76K（best） | 10K mix |
|---|---|---|---|
| **val KL** | **0.4548** | 0.6528 | 0.4463 |
| recall@1 | 0.9500 | 0.9547 | 0.9550 |
| recall@64 | 0.9560 | 0.9503 | 0.9564 |

（val KL 从 step ~800 起就在 0.4548~0.4605 平台化；1 epoch 欠训于 10K 的 6.4 epoch 属预期。）

**端到端**（VLMEvalKit，与 §9 完全同口径；`vlmeval_results/mix86k_*`）：

| 数据集 | 档位 | HAWK | 10K | 76K | **86K** |
|---|---|---|---|---|---|
| RealWorldQA | p80 | 64.84 | 64.97 | 64.31 | **64.71** |
| RealWorldQA | p90 | **59.74** | 57.65 | 58.43 | 57.65 |
| ChartQA_TEST | p80 | **79.16** | 75.56 | 76.28 | 75.52 |
| ChartQA_TEST | p90 | **67.48** | 58.00 | 58.84 | 57.76 |
| TextVQA_VAL | p80 | 83.31 | 83.12 | **83.45** | 83.28 |
| TextVQA_VAL | p90 | 79.44 | 79.30 | **79.43** | 79.12 |

**离线评测**（`eval_recall_layer1.py`，全量 7,599 条 val，`outputs/lookhawk_layer1_86k/eval.json`）：
KL **0.4548** vs HAWK 1.7589（3.87×）；**recall@deploy 0.9718** vs 0.9446；
逐样本获胜 **7,245/7,599（95.3%）**（76K 为 87.9%）。脚本已适配 host-resident
`Prepared`（循环内 `trainer.stage(...)`）。

**结论**：

1. 训练侧问题全部解决：shuffle 修复 + 整 epoch 后 **不再发散**，val KL 0.4548
   显著优于 76K 的 0.6528（同 val 集，teacher_kl_uniform=0.8441 逐位一致佐证）。
2. 但端到端结论与 §9 **不变**：TextVQA 打平、RWQA p80 打平、ChartQA 仍差 3.6/9.7 分
   （human_test 46.16 是主因）——**数据加量 +10K coco 对端到端无感**，
   进一步坐实 ChartQA 差距是分布问题而非规模问题。
3. 离线 recall/KL 与端到端精度的脱节依旧（§9.2），是下一步的核心问题。

**产物**：`outputs/lookhawk_layer1_86k/{scorer_best.pt, scorer.pt, config.json, eval.json}`、
`data/cambrian-coco-86k-extra/`（新 10K split）、`data/cambrian-mix-86k/{train,val}.jsonl`、
`vlmeval_results/mix86k_*/`、`logs/train_86k.log`。

**其他改动**：`eval_recall_layer1.py` 适配 host-resident `Prepared`（循环内
`trainer.stage(...)`，否则 device 报错）；`prepare_cambrian_subset.py` 新增
`--exclude-members`（增量切分与旧 val 保持不相交）。

## 12. 推理侧 L2 合并消融（2026-09-22，训练不动、checkpoint 复用）

**动机**：355 条 ChartQA p90「HAWK 对 / 86K 错」样本的对照分析
（`validation/chartqa_p90_vis/`，每样本 result.json + 两方法保留 token 标图）显示，
86K softmax 合并的分数高度尖峰化（max/mean≈18.9，top-10% token 占 41% 质量），
而 HAWK 逐头 L2 归一化 + 头权重的分数接近均匀（max/mean≈1.04）。softmax 的指数放大
让少数自信头主导跨头均值。假设：把 86K 推理时的合并换成 HAWK 式 L2 归一化可缩小差距。

**改动**（仅推理，训练路径与 checkpoint 不变）：

- `layer1.py::pre_rope_read` 新增 `apply_softmax=False`（返回 pre-softmax logits）；
- `layer1.py::visual_scores_l2`：行均值在 logit 空间 → 逐头 L2 归一化 → 头等权 mean
  （可选 `head_weights` 参数，本次未用）；L2 打在 logit 上，打在 softmax 概率上无效
  （形状不变）；
- `inference.py`：scorer 新增 `score_merge` 属性（`softmax` 默认 / `l2`），
  `load_lookahead_for_model` 支持 `LOOKHAWK_SCORE_MERGE` 环境变量，VLMEvalKit 零改动切换；
  trace 的 `score_normalization` 记录为 `l2_logit/mean`。

**逐样本对照（355 条 HAWK对/86K错，p90，冒烟 3 条验证）**：L2 后 max/mean 从
14.8~25.5 压到 1.31~1.41（与 HAWK 同量级），保留集合换血 15~17%。

**端到端**（VLMEvalKit，与 §9/§11 完全同口径，`vlmeval_results/mix86k_l2_*`）：

| 数据集 | 档位 | HAWK | 86K softmax | **86K L2** | L2−softmax |
|---|---|---|---|---|---|
| RealWorldQA | p80 | 64.84 | 64.71 | 63.53 | −1.18 |
| RealWorldQA | p90 | **59.74** | 57.65 | 58.30 | **+0.65** |
| ChartQA_TEST | p80 | 79.16 | 75.52 | 73.88 | −1.64 |
| ChartQA_TEST | p90 | **67.48** | 57.76 | **58.56** | **+0.80** |
| TextVQA_VAL | p80 | 83.31 | 83.28 | 82.95 | −0.33 |
| TextVQA_VAL | p90 | 79.44 | 79.12 | 78.41 | −0.71 |

ChartQA p90 细分：human_test 46.16→47.44（+1.28），augmented 69.36→69.68（+0.32）。

**结论**：

1. 合并几何只值 ±1 分量级：L2 只在 p90 高压下小幅帮忙（ChartQA +0.80 / RWQA +0.65），
   p80 与 TextVQA 全退。与 HAWK 的 9.7 分差距只缩到 8.92 分——**尖峰化不是差距主体**。
2. TextVQA 全退反证了文字先验在 OCR 任务上是优点：与 §11 的逐样本数据一致
   （HAWK 保留的文字 token 比例 73.2% 反而高于 86K 的 59.8%）——问题不在"关注文字"，
   而在 86K 的查询是问题无关的答案先验，HAWK 是问题条件化的内容匹配。
3. 后续候选：L2 + 头权重（`score_head_weights` 开关已留）；伪 token 与问题 token
   混合查询（向条件化靠，是差距主体的候选修复）。

**产物**：`vlmeval_results/mix86k_l2_{rwqa,chartqa,textvqa}_p{80,90}/`（manifest 含
`lookhawk_score_merge: l2`）、`validation/chartqa_p90_vis/`（355 样本对照标图）、
`validation/chartqa_p90_reinfer.jsonl`（355 样本两方法全量 token 分数与保留索引）、
`validation/{find_hawk_right_mine_wrong,reinfer_p90_capture,visualize_kept_tokens,analyze_kept_distribution,analyze_text_bias}.py`。

## 13. 层 0 迁移与组合信号实验（2026-09-22）

**动机链**：§12 的 L2 合并只值 ±1 分 → 怀疑目标信号本身有问题 → 两个决定性诊断
（`validation/diag_teacher_overlap.py`、`diag_hawkl0_vs_teacher.py`，各 24 条 val）：

- **层 1 上问题注意力 ≈ 答案注意力**：top-20% IoU 0.886、Spearman 0.992——层 1 读数
  被通用显著性（文字区域）主导，查询内容几乎无影响。这解释了 86K 的"问题无关答案先验"
  形态，也判了层 1 混合教师（问题+答案）死刑：实测 pilot（w=0.5, 10K）ChartQA p90
  = 57.60 ≈ 基线 58.00（混合≈没混）。
- **部署版 HAWK 信号（层 0）与层 1 答案教师只有 IoU 0.667**——层间差异是真信息来源。

**实验**：伪 token 训练从层 1 移到**层 0**（`--score-layer 0`，10K mix，8 卡，2000 步，
纯答案教师；推理侧新增 `LOOKHAWK_SCORE_MERGE`（softmax/l2 合并）与
`LOOKHAWK_COMBO_BETA`（与 HAWK 信号的混合权重）两个环境变量开关，训练零改动）。

**ChartQA p90 结果**：

| 方法 | Overall | human | aug |
|---|---|---|---|
| HAWK（参照） | **67.48** | 51.12 | 83.84 |
| **层0伪token** | **65.76** | 50.56 | 80.96 |
| 层0伪token β=0.5 组合 HAWK | 65.80 | 50.56 | 81.04 |
| 86K 层1伪token（§11） | 57.76 | 46.16 | 69.36 |

**结论**：

1. **换层是迄今最大单项改进**：+8.0 分（57.76→65.76），与 HAWK 差距 9.72→1.72。
   且 KL 拟合更差（2.52 vs 0.45）端到端反而更好——"离线拟合度≠下游质量"第三次成立。
2. **β=0.5 组合无效**：`diag_combo_overlap.py` 测得层 0 上伪 token 与 HAWK 的
   top-10% IoU = 0.824（top-20% = 0.868）——两信号高度同源，混合≈恒等，扫 β 无意义。
3. 层 0 学生 KL 2.52 远高于层 1 可达的 0.45——但事后核查 checkpoint 元信息，
   **best 停在 step 400（val KL 1.81），终值 2.52，即 10K 上 ~1 epoch 后持续过拟合**。
   修正此前"欠拟合"的说法：层 0 不是欠拟合而是早过拟合，86K 重训应配早停/
   按 epoch 对齐步数，而非沿用 2000 步。

**产物**：`outputs/lookhawk_layer0_mix10k/`、`vlmeval_results/l0mix10k_{pure,combo50}_chartqa_p90/`、
`validation/diag_{teacher_overlap,hawkl0_vs_teacher,combo_overlap}.py`、
`logs/train_layer0_mix10k.log`；推理开关文档见 `src/lookhawk/inference.py`。

## 14. 无 HAWK 组合与"问题特异性"的定量结论（2026-09-22）

**背景纠错**：§13 的组合 β=0.5（65.80）中，问题半边调用了 HAWK 原版配方
（层 0 问题查询 + HAWK 28 维头权重 + 逐头 L2）——混入了他人方法的部件。
纯层 0 伪 token（65.76）则完全无 HAWK 成分。本节把组合重做为本方法自洽的形态。

**改动**（`src/lookhawk/inference.py`）：删除 `_hawk_question_scores`，新增
`_question_visual_scores`——问题行取问题文本区间（`vision_end+1 : 用户轮 im_end`，
无模版 token，与训练侧同一规则），读数规则与伪 token 完全同源（pre-RoPE QK、
softmax 只在视觉列、行均值、**头等权均值**），组合
`score = (1−β)·L1norm(question) + β·L1norm(pseudo)`。本方法路径中不再有任何
HAWK 部件（无头权重、无其 L2 选择）。冒烟 3 样本验证 trace 标记
`combo_beta0.5:pseudo(softmax_visual/mean)+question`。

**ChartQA p90 结果**（`vlmeval_results/l0mix10k_combo50q_chartqa_p90/`）：

| 方法 | Overall | human | aug | 含 HAWK 部件 |
|---|---|---|---|---|
| HAWK（基线） | 67.48 | 51.12 | 83.84 | — |
| **纯层0伪token** | **65.76** | 50.56 | 80.96 | 无 |
| 组合 β=0.5（无 HAWK 版） | 65.36 | 49.20 | 81.52 | 无 |
| 组合 β=0.5（§13，含 HAWK） | 65.80 | 50.56 | 81.04 | 有 |

**结论**：

1. **问题注意力混合路线正式关闭**：无 HAWK 版组合 65.36 < 纯伪 token 65.76，
   问题半边不贡献正增益反而稀释。加上 §13 的 82% 重合度与 β 无效应答，三种形态
   （HAWK 版、无 HAWK 版、不同 β）一致证明：问题信号的信息已包含在伪 token
   学到的先验里。β=0.5 无效的另一半机制：HAWK 分数近均匀（L2），伪 token 分数
   是 softmax 尖峰，L1 归一化混合后 HAWK 侧只贡献近常数底噪，改变不了排序。
2. **问题特异性在 token 筛选环节的真实价值 = 1.72 分**（HAWK 67.48 − 纯伪 token
   65.76），且集中在 human_test（51.12 vs 50.56）。机理解释：伪 token 是 32 个
   固定内容探针，样本特异性由 key 侧（图像内容嵌入）承担；问题的特异性由
   生成端承担（问题文本始终在 LLM 输入里，剪枝不删文本）——筛选只需召回证据，
   不需定位答案。
3. **当前最优形态 = 纯层 0 伪 token（65.76）**。剩余 1.72 分的追赶空间在训练侧：
   86K 重训层 0（对症 §13 结论 3 的过拟合：8.6 倍数据 + 早停/按 epoch 对齐步数）、
   `num_pseudo` 32→64/128 扩容。

## 15. 86K 层 0 重训 + 6 路评测（2026-09-23，纯伪 token 形态定版）

**配方**：`--score-layer 0 --seed 42 --batch-size 32 --steps 2688`（86,000/32 = 恰好 1 epoch）、
`--save-best`、8 卡 DDP、纯答案教师（无问题混合、无 HAWK 部件）。流水线脚本
`scripts/run_l0_86k_pipeline.sh`（restore_images → 训练 → 6 路评测 → 汇总，全后台，
状态落 `logs/l0_86k_pipeline.status`）。训练 22:16→02:30（~4.2 h），评测 02:30→04:04。

**训练轨迹**：val KL 1.89(step 500) → 1.94(1000) → 1.92(2000) → 2.23(2500) → 2.25(2688)。
10K 时的早过拟合（1.81→2.52）在 86K 上明显缓解（1.89→2.25），`scorer_best.pt` 取自
KL 最优点（step ~500 附近），评测全部用 best。

**端到端结果**（`vlmeval_results/l086k_*/`，与 §9/§11/§13 完全同口径）：

| 数据集 | 档位 | HAWK | 86K 层1（§11） | 10K 层0（§13） | **86K 层0** | Δ vs HAWK |
|---|---|---|---|---|---|---|
| RealWorldQA | p80 | 64.84 | 64.71 | — | 64.05 | −0.79 |
| RealWorldQA | p90 | 59.74 | 57.65 | — | **60.39** | **+0.65 ✅** |
| ChartQA | p80 | 79.16 | 75.52 | — | 78.48 | −0.68 |
| ChartQA | p90 | 67.48 | 57.76 | 65.76 | **66.28** | **−1.20** |
| TextVQA | p80 | 83.31 | 83.28 | — | 83.18 | −0.13 |
| TextVQA | p90 | 79.44 | 79.12 | — | 79.42 | −0.02 |

ChartQA p90 细分：human_test 50.32 / augmented 82.24（HAWK 51.12 / 83.84）。

**结论**：

1. **86K 层 0 与 HAWK 基本打平，且 RWQA p90 首次反超（+0.65）**：6 项中
   1 项反超、2 项打平（±0.13 内）、3 项小幅落后（最大 −1.20）。
   ChartQA p90 差距从最初（§11）的 9.72 → 1.20。
2. **10K→86K 在层 0 有效**（ChartQA p90 +0.52）——与层 1 时代"加数据无感"相反，
   印证层 1 的瓶颈在层不在数据，层 0 的瓶颈在容量/数据。
3. 残余差距集中在 ChartQA human_test（−0.8）：真实图表的细粒度定位仍是
   固定先验的已知上限（§14 结论 2）。后续候选：`num_pseudo` 扩容、
   更长训练（86K 2-3 epoch + save-best）。

## 18. RoPE 打分消融（2026-09-30，单一变量：pre-RoPE → mRoPE，负结果定案）

**动机**：§17 p8 的 ChartQA p90 分歧分析（HAWK对/p8错 106 例 + HAWK错/p8对 72 例，
逐样本含两侧全量 scores 与 kept_rel；`validation/p8_diff_analysis_report{,2}.md`、
`p8_diff_sample_metrics.csv`）给出三个事实：

1. 两打分器全量分数 Spearman 中位 **0.992**、top-10% 保留集 Jaccard 0.83——层 0 的
   排序几乎完全由 key（图像内容）决定，查询只影响残余 ~1%。全部分歧由 top-k 边界
   ~22 个 token 决定，且这些 token 在对方排名里也紧贴保留线（百分位 0.88 vs 0.90）。
2. **p8 的边际预算 ~40% 浪费在网格最外圈**：对称差集中 p8 独有 token 40–45% 在边框
   （顶/底/左/右 ≈ 32/32/21/16%），HAWK 独有仅 8–10%（Wilcoxon p<1e-4）；该偏置
   越重 p8 越容易输（组间 p=0.028），且与问题是否含数字无关、human/aug 均成立——
   是探针对图像 key 的固有偏好（疑似 blank-patch key 范数异常 + softmax 尖峰放大，
   未直接验证）。
3. 问题含显式数字 → HAWK 胜面 68%（字面锚点直接内容匹配）；无数字 → 50:50。
   错误预测中 16–29% 只是 relaxed-accuracy 阈值边界运气，非选择差异。

边框税的候选修法之一是"打分引入位置信息"（外圈 patch 位置可辨、可学可罚）。本节
检验最朴素的一种：**训练与推理的打分 Q/K 都应用 mRoPE，其余逐项不变**。

**改动**：`pre_rope_read(..., apply_rope=)`（cos/sin 来自 `text_decoder.rotary_emb`
与同批 position_ids，`apply_multimodal_rotary_pos_emb`，rope 先于 repeat_kv，与
真实注意力逐行一致）；教师/学生同侧切换；checkpoint 写入 `rope_scoring`，推理按
checkpoint 自动切换（stats 显示 `rope/softmax_visual/mean`）。实现位置：
`src/lookhawk/layer1.py`、`src/lookhawk/inference.py`、`scripts/train_layer1.py
--rope-scoring`。冒烟（train 20 步 + 单图推理）验证通过后上量。

**配方**：与 §17 p8 逐项一致——层 0、8 探针、86K、seed 42、batch 32 全局、
2688 步（1 epoch）、lr 2e-4 cosine、save-best、纯答案教师，8 卡 DDP。
流水线 `scripts/run_l0_86k_p8_rope_pipeline.sh`（restore_images → 训练 → 6 路评测）。

**训练轨迹**：预计算 13532s（与 §15 的 13268s 一致）。eval KL 4.13@50 → 3.34@150 →
**3.239@1400（best）** → 4.56@2688，与层 0 各轮相同的早停形态。RoPE 教师显著更尖：
teacher_kl_uniform **1.13 vs pre-RoPE 0.77**（位置信息让答案注意力更集中）。

**端到端**（与 §9/§15 完全同口径；`vlmeval_results/l086kp8rope_*`）：

| 数据集 | 档位 | HAWK | p8 pre-RoPE（§17） | **p8 RoPE** | Δ vs pre-RoPE |
|---|---|---|---|---|---|
| RealWorldQA | p80 | 64.84 | — | **65.36** | —（vs HAWK +0.52） |
| RealWorldQA | p90 | 59.74 | 60.00 | **59.74** | −0.26 |
| ChartQA | p80 | 79.16 | — | **74.72** | —（vs HAWK −4.44） |
| ChartQA | p90 | 67.48 | 66.12 | **53.44** | **−12.68** |
| TextVQA | p80 | 83.31 | — | **81.49** | —（vs HAWK −1.82） |
| TextVQA | p90 | 79.44 | 79.64 | **75.50** | −4.14 |

ChartQA p90 细分：human 41.36 vs 50.08（−8.72）、augmented 65.52 vs 82.16（**−16.64**）。

**结论**：

1. **pre-RoPE 是承重设计，本轮反向坐实**。唯一改动（打分 Q/K 加 mRoPE）让 ChartQA
   p90 掉 12.7 分。探针坐在 prompt 末尾，RoPE 使每个视觉 token 的分数混入
   "离探针的距离 + (t,h,w) 地址"，内容纯度被位置污染；保留率越低（p90）污染越致命，
   空间精度需求越高（图表 ≫ 自然图像）污染越致命——两个梯度都与机制自洽。
2. **边框税的解法不在"引入位置信息"这条路上**。回到动机分析给出的另两条候选：
   训练教师侧 mask 最外圈，或推理侧对外圈 logit 加固定负偏置。
3. RWQA p80 的 +0.52 是 6 项中唯一亮点，单点不构成路线证据（p90 持平、其余 4 项
   皆负）。
4. 运维记录：① 评测首轮 6 路并发在 host cgroup OOM 被杀 2 路、CUDA OOM 崩 2 路
   （注意 `run_lookahead_3x2.sh` 的 exit code 取自脚本尾命令，会误报 0）；两两串行
   补跑通过（`scripts/run_p8rope_eval_recovery.sh`）。② 本机系统 python3
   （torch 2.5.1+cu121）在 H20 上 bf16 GEMM 随机 SIGFPE（libcublasLt trap divide
   error）；一切入口必须走 `PYTHON_BIN=/ossfs/workspace/HAWK/.venv/bin/python` 经
   `scripts/run.sh`（它会从 LD_LIBRARY_PATH 剔除系统 CUDA 路径，否则 venv torch
   自身 import 都失败）。③ 模型权重：运行用 /home/admin（本地盘），持久备份在
   `.model/Qwen2.5-VL-7B-Instruct`（NAS；机器重建会清 /home/admin）。

**产物**：`outputs/lookhawk_layer0_86k_p8_rope/`（scorer_best.pt @ step 1400，
val KL 3.239，`rope_scoring: true`；原 §17 p8 checkpoint 未动）、
`vlmeval_results/l086kp8rope_*`、`scripts/run_l0_86k_p8_rope_pipeline.sh`、
`scripts/run_p8rope_eval_recovery.sh`、`validation/analyze_p8_diff{,2}.py`、
`validation/check_rope_position_bias.py`（位置偏置直接验证脚本，未运行）。

