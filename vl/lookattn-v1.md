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

## 7. 待办（评测补齐，按代价排序）

| 项 | 缺什么 | 代价 |
|---|---|---|
| 新 scorer 推理冒烟 | 不缺，直接跑 `infer_lookhawk.py --scorer outputs/.../scorer_best.pt` | 几分钟 |
| layer-1 离线评测脚本 | 全 k 点 recall（含部署点位 recall@~261）、HAWK 无训练基线对照、rank_loss | ~30 min 代码，无下载 |
| VLMEvalKit 端到端（RealWorldQA / ChartQA_TEST / TextVQA_VAL） | ① `setup_vlmevalkit.sh`（本地复制+补丁）② TSV 复用 `/ossfs/workspace/HAWK/data/vlmeval/` ③ 把 scorer 接进 `evaluate_vlmeval.sh` + `vlmeval_adapter.py` | ~1 h 代码；ScienceQA / MME 的 TSV 需另行下载 |
| 76K 全量训练 | DDP 分片后显存已够（~91 GB/rank）；还差"编码完即弃 pixel_values"（内存墙 1.9 TB） | 需改 `layer1.py`/`data.py` |

## 8. 数据与产物索引

 Lookahead-76K vs HAWK 端到端精度对比

| 数据集         | 档位  |  HAWK | Lookahead (76K) |       Δ |
| ----------- | --- | ----: | --------------: | ------: |
| RealWorldQA | p80 | 64.84 |           64.31 |   -0.53 |
| RealWorldQA | p90 | 59.74 |           58.43 |   -1.31 |
| ChartQA     | p80 | 79.16 |           76.28 |   -2.88 |
| ChartQA     | p90 | 67.48 |           58.84 |   -8.64 |
| TextVQA     | p80 | 83.31 |           83.45 | +0.15 ✅ |
| TextVQA     | p90 | 79.44 |           79.43 | -0.01 ≈ |

  解读

  - TextVQA：达标。p80 反超 HAWK，p90 持平 —— 说明方法在文本识别类任务上可行。
  - RealWorldQA：略低（-0.5~-1.3），基本接近。
  - ChartQA：差距明显，p90 掉了 8.6 个点。ChartQA 是图表理解，视觉 token 的信息分布和 OCR 文本差别大，而我们的 scorer 是在 Cambrian
  混合语料上训的，teacher 是真实答案 token——对图表类问题，32 个 lookahead 行的表达能力在高剪枝率下不够用。

  另一个值得注意的点：离线评测里 recall@p90 我们是 0.967 > HAWK 0.928（留得更多答案相关 token），但端到端精度反而更低——说明光留住对的 token 
  不够，留住的 token 组合/位置对生成质量也有影响，或者 recall 提升集中在非关键 token 上。

  可能的原因方向（供你决策下一步）：
  1. 76K 训练只跑了 2000 step（约 0.85 epoch），且 step~1570 有过 loss 尖峰，best 停得早——训练不充分
  2. 训练语料里 ChartQA 类数据占比低，分布不匹配
  3. 高剪枝率（p90）下 32 行 lookahead 容量不足


```text
datasets/jsons/Cambrian737k.jsonl     标注（765.5 MB）
datasets/{ocr_vqa,chartqa,coco}.tar.gz 图片归档（22.8 GB）
datasets/*_cambrian.jsonl             filter 全量分流
data/cambrian-ocr_vqa/                复现 split（5,000/500）
data/cambrian-{ocr_vqa,chartqa,coco}-mix/ 混合 split
data/cambrian-mix/                    合并语料（10,000/1,000）
outputs/lookhawk_layer1/              OCR_VQA scorer（KL 0.2765）
outputs/lookhawk_layer1_mix/          混合 scorer（KL 0.4464）
outputs/smoke*/                       冒烟与回归/DDP 验证产物
```
