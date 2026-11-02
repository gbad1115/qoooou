# VisionSelector 研究会话记忆（跨会话交接文档）

> 用法：新会话开始时把本文件路径发给 Claude，让它先读完再工作。
> 最后更新：2026-10-22（熵匹配 β 曲线完成后）

## 0. 项目目标

在 VisionSelector（ACM MM 2026，视觉 token 压缩）之上做自己的方法/论文。核心假设演化路线：
**深层注意力才是好的选择信号 → 信号集中在少数检索头 → 蒸馏回一遍推理**。
模型是 **Qwen2.5-VL-7B**（不是仓库表面的 LLaVA-OV-1.5 路线）。

## 1. 环境（离线机器，无外网）

| 项 | 位置 |
|---|---|
| 基座模型 | `/ossfs/workspace/lookattn/.model/Qwen2.5-VL-7B-Instruct` |
| selector 权重（自训） | `/dev/shm/vs_out/orig-repro/VisionSelector-Qwen2.5-VL-7B`（⚠️ tmpfs，重启即失；artifacts 里只有 config 无权重，需备份） |
| 评测 venv | `/dev/shm/vs_env/.venv`（torch 2.6.0 + transformers **4.50.0 钉死**，勿升级；重建流程 `harness/build_env.sh`） |
| GPU | 8× H20-3e 143GB |
| harness（全部实验代码） | `/ossfs/workspace/visionselector_artifacts/harness/` |
| 评测数据 | `/ossfs/workspace/lookattn/data/vlmeval/{ChartQA_TEST,TextVQA_VAL}_local.tsv`（2500/5000 题） |
| 训练分数 | `/ossfs/workspace/visionselector_artifacts/oracle_scores_train/*.pt`（144,548+22,492 样本，4 通道 fp32：0=h24_lt, 1=h2_lt, 2=h24_qm, 3=h2_qm） |
| textvqa-train | `/ossfs/workspace/textvqa/`（parquet，图像内嵌 bytes；已转 images+jsonl，22,492 题） |

## 2. 实验纪律（必须遵守）

1. **公平性铁律**：改 token 预算的改法一律否决；固定 k = max(1, int(N_img×0.2))，只改选择策略。
2. **单变量改动**：每个变体只动一个因素，配 null 实验。
3. **配对 McNemar**（Yates 校正）：`harness/paired_matrix.py` 的 score_rows/mcnemar 约定；TextVQA 按 0.5 二值化配对。chi2>3.84 为显著。
4. **评分必须用仓库原码**：chartqa relaxed_correctness / textvqa VQA soft acc（`score_local.py`）。
5. 实验报告写 `EXPERIMENT_*.md` 存 `/ossfs/workspace/visionselector_artifacts/`；总账在 `SUMMARY_oracle_heads_distill.md`。
6. **seed 复刻教训**：L42/L7 的 +1.88pp 曾被 seed 间波动推翻——任何 <2pp 的增益都要 seed 复刻后才算数。

## 3. 已完成的实验与关键数字（@0.2 全量，chartqa/textvqa）

### 基线
- orig（不压缩）：83.84 / 82.93
- **selector（要超越的对象）**：69.68 / 79.88

### Oracle 分析（两遍探针，EXPERIMENT_oracle_lastlayer.md）
- 最后一层（L27）注意力选 top-20% 重跑：lasttok 68.04 / **81.29**（textvqa 显著超 selector +1.41）
- 任务结构依赖：chartqa 深层注意力太尖覆盖不足，textvqa 精准定位强
- lasttok（最后一行 query）≥ qmean（问题行均值）

### 头部分析（EXPERIMENT_heads_lastlayer.md）
- L27 的 28 头 = 检索头~17 / 局部头(8,7) / 透传死头(10-13, 100%注意力在自身) / 中间头(25,26,0 看系统提示)
- **{24,2} 两个教师头**：h24 广覆盖型（读题全程看图 0.22），h2 答案位专看型（平时不看图 0.069、答案位 0.40，比值 5.8×）
- 头组合全量：**h24_h2 = 73.20 / 80.57（双数据集显著超 selector，chartqa 项目纪录）**
- 死头惰性 IoU 0.99；组内"看得多≠看得准"（Spearman -0.20/+0.22）

### 蒸馏线（EXPERIMENT_distill_v{1,2,3,4,5,6,7}.md）
学生 = CrossAttnScorer 8.01M（P_v/P_q 3584→512 + 2 层 cross-attn 图query/问KV + MLP）：
- v1 全分布 KL：67.76 / 73.48，val IoU 0.492
- v2a 边界hinge / v3a 阶跃加权 / v3b 掩码 KL：全部 ±1 内 → **损失形状变量关闭**
- v4 深Q：67.76 / 76.25，IoU 不变 → 问题深度排除
- v5 单头 h2 目标：65.72 / 73.88 → 自洽目标证伪
- v6 归因：textvqa +5.47 增益 100% 来自 textvqa-train 数据；正确性过滤（双✓硬过滤）证伪
- **v7 熵匹配辅助损失 β·(H_s−H_t)²**：β∈[0.05,0.2] 整块甜点区
- 诊断终论：学生 Spearman 0.69 但 top-20% 边界≈随机；教师互证 IoU 0.48；**缺口在图像侧深层上下文，pre-LLM 不可得**

## 4. 当前定稿方法（最佳一遍配置）

```
CrossAttnScorer 8.01M + 全分布KL(教师=h24_h2 oracle) + 熵匹配辅助 β·(H_s−H_t)²
训练混合 = chartqa 28,138 + ocr_vqa 80,000 + coco%10 36,410 + textvqa-train 22,492
（= 167,040，只用首轮问题，eval 同款 prompt/分辨率）
```

| β | ChartQA | TextVQA | vs selector |
|---|---|---|---|
| 0.05（v7d） | 68.92 | **79.12** | 双 ns 打平 |
| 0.1（v7c） | 68.76 | 78.99 | 双 ns 打平 |
| **0.2（v7b，chartqa 峰）** | **69.08** | 78.98 | -0.60 ns / -0.90 边缘 |

**双基准统计打平 selector 的蒸馏学生诞生。** ⚠️ textvqa 为 in-domain（selector 零样本），论文须声明。

## 5. 待办（按优先级）

- [ ] **seed 复刻**（β=0.05 或 0.1 换种子 1-2 发，~50min/发）——增益站住的必要条件
- [ ] **定稿配置预算扫描** k=0.1/0.3（~20min）——"任意预算推理"表格；之前学生低预算掉得快（k=0.1 差 5-9.5pp）
- [ ] LLaVA-1.5-7B 移植（D=4096 → 8.53M；Qwen3-VL 需 vendor 代码+升 transformers，搁置）
- [ ] 方向储备：early-exit scorer（读第 L 层特征）、级联选择（两遍付费）
- [ ] 论文骨架见 SUMMARY 第 4 节

## 6. 工具与坑

- 跑评测：`cd harness && $VS_ENV/bin/python eval_local.py --method {selector,orig,oracle_lasttok,distilled} ...`，8卡分片见 run_*.sh
- `analysis_distill.py` 需 `--stu_glob`（文件带预算后缀时）；tag≠文件名前缀曾有坑
- glob `shard*.jsonl` 会同时匹配 `.scores.jsonl`——靠字典序侥幸正确，写新 glob 用 `shard[0-9].jsonl`
- 长任务一律 nohup + disown 挂后台，日志落 /ossfs（/dev/shm 会被清）
- Bash 工具会偶发分类器暂不可用，重试即可；后台重定向目录需先 mkdir

## 7. 关键结论一句话

深层注意力里存在比 selector 更好的选择信号（oracle 73.20/80.57），集中在 {24,2} 两个检索头；该信号浅层蒸馏有信息天花板（六个损失/输入变体全部止步），但配合 in-domain 数据（textvqa-train）+ 熵匹配辅助损失（β 0.05-0.2），一遍蒸馏学生在两个基准上都与 selector 统计打平。
