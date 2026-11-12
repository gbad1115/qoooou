# AnchorMerge v2：Training / Inference 实现细节（逐行核对版）

> 用途：论文 method+implementation 部分的写作依据。**每一项都对照当前代码抄录**，
> 文件行号一并给出，便于复核。方法主体（压缩五步）见 `METHOD_v2.md` §3 与
> `cvpr_files/anchor_method.tex`；本文只覆盖**损失、训练循环、推理与判分**。
>
> 核对基准：`v2_model.py`、`train_v2.py`(LLaVA-1.5)、`train_llavanext_v2.py`、
> `train_qwen_v2.py`、`eval_v2.py`、`eval_llavanext_v2.py`、`eval_qwen_v2.py`、
> `train.py`(LLaVATap/Tokeniser)、`train_qwen.py`(QwenTap/build_positions)、
> `eval_mt_gqa.py`(build_prompt/is_correct)。
>
> ⚠️ 本文描述的是**当前实现**，其中残差池**包含 anchor 自身**（其残差恒为 0）。
> `anchor_method.tex` 写的是排除 anchor（$\mathcal{R}_k=S_k\setminus\{i_k^\star\}$）。
> 二者关系为 $\mu_{\text{code}}=(1-w_{a})\,\mu_{\text{tex}}$（方向完全相同），
> 详见 `METHOD_v2.md` §2.1 与相关讨论。**所有已测结果均由"包含 anchor"的实现产出。**

---

## 0. 文件与骨干对应表

| 文件 | 用途 | 骨干 |
|---|---|---|
| `v2_model.py` | `AnchorMergeV2` + `VisualReconstructor` + `reconstruction_loss` | 全部（d 可变） |
| `train_v2.py` | 训练（bsr=4，无梯度累积） | LLaVA-1.5-7B |
| `train_llavanext_v2.py` | 训练（bsr=1，累积 4） | LLaVA-NeXT 7B/13B |
| `train_qwen_v2.py` | 训练（bsr=1，累积 4）+ 坏 grid 过滤 | Qwen2.5-VL-7B |
| `eval_v2.py` / `eval_llavanext_v2.py` / `eval_qwen_v2.py` | MT-GQA 推理 | 各自 |
| `analyze_frac.py` / `analyze_llavanext.py` / `analyze_qwen_v2.py` | 汇总 + McNemar | 各自 |

---

## 1. 冻结/可训练边界与数值精度

**冻结（`requires_grad_(False)`）**：vision tower、projector/patch-merger、整个 LLM。
**唯一可训练模块**：`AnchorMergeV2`，其参数为 **fp32**（构造时不指定 dtype，`.to(device)` 只搬设备）。

**精度流**（三骨干一致）：

```
pixel_values ──(bf16/fp16)──> 冻结视觉编码 ──> 视觉 token（骨干 dtype）
                                              │
                                     .float() │  ← 进压缩器前升 fp32
                                              ▼
                              AnchorMergeV2（fp32 参数与计算）
                                              │
                         splice 时 .to(embeds.dtype)  ← 回骨干 dtype
                                              ▼
                                  冻结 LLM 前向 → CE
```

- 代码位置：LLaVA-1.5 `train_v2.py:113-116`（`x→bf16`，`y=tap.projector(x).float()`，
  `y` 在 `torch.no_grad()` 下算）；NeXT `train_llavanext_v2.py:126-131`；
  Qwen `train_qwen_v2.py:169`（`y = f.unsqueeze(0).float()`）。
- **骨干 dtype**：LLaVA-1.5 / NeXT 用 `torch.bfloat16`（`train.py:152` 的默认值，
  两处 `LLaVATap(...)` 调用均未覆盖）。**Qwen 训练用 `float16`**，评测用 `bfloat16`，原因见 §11.1。
- 梯度路径：CE 的梯度**穿过冻结的 LLM 回传**到压缩器（LLM 权重不更新但参与反向）；
  splice 处的 bf16 cast 是可微的。重构损失的梯度只在压缩器内部（目标 `y` 已 detach）。

**参数量**（`d` = LLM 隐层维度；`hidden=256`, `gate_hidden=64`, `recon_hidden=128`）：

| 骨干 | d | 训练期（含重构 decoder） | 推理（删 decoder） |
|---|---|---|---|
| LLaVA-1.5-7B | 4096 | 6,499,074 | 4,723,586 |
| LLaVA-NeXT-7B | 4096 | 6,499,074 | 4,723,586 |
| LLaVA-NeXT-13B | 5120 | 8,073,986 | 5,904,258 |
| Qwen2.5-VL-7B | 3584 | 5,711,618 | 4,133,250 |

推理时 `need_recon=False`，decoder **完全不参与前向**（`v2_model.py:196`）。

---

## 2. 数据、采样与批处理

### 2.1 数据

- 混合：chartqa 28,138 + ocr_vqa 80,000 + coco 36,410 + textvqa 22,492 = **167,040**；
  每条只取**第一个 human→gpt 轮**（单轮）。
- 划分：`train_val_split(items, n_val=2000, seed=17)` → **train 165,040 / val 2,000**。
  划分**先于**抽样，因此所有数据量实验共用同一 val。
- 抽样：`stratified_subsample(train_items, frac, seed=17)` 按源等比抽样（seed 固定 → 可复现）。
  30% → **49,511**（chartqa 8,342 / coco 10,796 / ocr_vqa 23,711 / textvqa 6,662）。
- 启动断言：`EXPECTED_MIXTURE` 计数不符则直接报错，防止 `load_mixture` 因某源首图缺失
  而**静默跳过整源**（此坑已真实踩过，见 `METHOD_v2.md` §6）。

### 2.2 批处理配置（三种骨干都保持 global batch = 24）

| 骨干 | batch_per_rank | 梯度累积 | GPU 数 | global batch | total_steps |
|---|---|---|---|---|---|
| LLaVA-1.5 | 4 | 1 | 6 | 24 | 2062（30% 数据一个 epoch） |
| NeXT 7B/13B | **1** | **4** | 6 | 24 | 2062 |
| Qwen2.5-VL | **1** | **4** | 6 | 24 | 2062 |

NeXT/Qwen 用 `bsr=1`：NeXT 的 AnyRes 使 `pixel_values` 各图 patch 数不同（ragged，无法 stack）；
Qwen 则是为压低 GEMM 的 `M=batch×maxlen`（见 §11.1）。

`total_steps = min(steps_per_epoch × epochs, max_steps)`；启用累积时
`steps_per_epoch = len(dl) // grad_accum`。cosine 日程以 `total_steps` 为周期。

### 2.3 视觉特征提取（三者不同）

| 骨干 | 取法 | 输出 |
|---|---|---|
| LLaVA-1.5 | `vision_tower(pv, output_hidden_states=True).hidden_states[-2][:,1:]` → `.detach()` → `multi_modal_projector` | `(B, 576, 4096)`，`[:,1:]` 去掉 CLS |
| NeXT | `model.get_image_features(pv, image_sizes, vision_feature_layer=-2, strategy="default")` → **`model.pack_image_features(..., image_newline=model.image_newline)`** → 按 `feature_lens` split | 每图 `(N_i, 4096)`，N=1368–2928 |
| Qwen2.5-VL | `model.visual(pv, grid_thw)`（patch merger 之后，**无 projector**）→ 按 `grid_thw.prod(dim=1)//4` split | 每图 `(N_i, 3584)` |

NeXT 用 HF 自带的 `pack_image_features` 而非自行实现 unpadding；已实测 n₁=N 时
我的拼接前向与 HF 原生 `pixel_values` 前向的 CE **差值恰为 0.00e+00**（7B 与 13B 均验证）。

### 2.4 文本编码与标签（两套不同约定，务必区分）

**LLaVA-1.5 / NeXT**（`train.py:115-133` 的 `Tokeniser`）：

```
PROMPT_FMT = "USER: <image>\n{q} ASSISTANT:"      # train.py:39
prompt_ids = tok(PROMPT_FMT, add_special_tokens=True)      # 带 BOS
at         = prompt_ids.index(image_token_id)               # image_token_id = 32000
ids        = prompt_ids[:at] + [32000]*n1 + prompt_ids[at+1:]
ans        = tok(" " + answer, add_special_tokens=False) + [eos]     # ← 答案前有前导空格
labels     = [-100]*len(ids) + ans                            # 只监督答案 + eos
```

**Qwen**（`train_qwen.py:98-120` 的 `QwenTokeniser`）：

```
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
ids  = tok(prompt_text, add_special_tokens=False)
at   = ids.index(image_pad_id)                    # image_token_id = 151655
ids  = ids[:at] + [151655]*n1 + ids[at+1:]
ans  = tok(answer, add_special_tokens=False) + [im_end_id]   # ← 无前导空格，结束符 151645
labels = [-100]*len(ids) + ans
```

差异要点：LLaVA 用 vicuna 裸模板 + `" "+answer` + `</s>`(2)；Qwen 用 chat template +
`answer` + `<|im_end|>`(151645)。**标签范围一致**：prompt 全 −100，仅答案 token 与结束符参与 CE。

### 2.5 拼接（splice）与 padding

```python
embeds = tok_emb(input_ids)                       # (B, L, d)
pos    = (input_ids[b] == image_token_id).nonzero().flatten()   # 恰好 n1 个
embeds[b, pos] = compressed[b].to(embeds.dtype)   # 按下标顺序 = raster 顺序
```
（`train.py:203-208` `_splice`；Qwen 版在 `train_qwen.py:197-199`；NeXT 版 `llava_next_tap.py:splice`）

- **前提**：每个样本的图像占位符数必须恰好等于 `n1`（由 `Tokeniser.encode` 保证），
  否则广播失败。因此 NeXT/Qwen 训练时 `n1` 按 batch 统一取 `min(--n1, min_i N_i)`。
- padding：LLaVA 用 `pad = eos if pad_token_id is None`，`labels` 填 −100，`attn` 填 0；
  Qwen 用 `input_ids` 填 0、`pos` 填 0、`labels` 填 −100、`attn` 填 0（`train_qwen.py:186-189`）。

### 2.6 Qwen 的 mRoPE 位置构造（仅 Qwen 需要）

```python
def grid_for(n1):                       # 最大因子分解，g1 ≤ √n1
    g1 = isqrt(n1); while n1 % g1: g1 -= 1
    return g1, n1 // g1

def build_positions(prefix_len, n1, suffix_len, device):   # -> (3, L)
    g1, g2 = grid_for(n1); base = prefix_len
    pos[:, :prefix]                        = arange(prefix)
    hj = arange(g1).repeat_interleave(g2);  wj = arange(g2).repeat(g1)
    pos[0, prefix:prefix+n1] = base         # 时间通道恒定
    pos[1, prefix:prefix+n1] = base + hj    # 高
    pos[2, prefix:prefix+n1] = base + wj    # 宽
    pos[:, prefix+n1:]       = base + max(g1,g2) + arange(suffix_len)
```
LLaVA-1.5 / NeXT **不传 position_ids**，用 HF 默认的 1-D 位置。

---

## 3. 损失一：答案交叉熵 `L_CE`

### 3.1 定义

```
L_CE = − (1/|A|) Σ_{t∈A} log softmax(z_t)[y_t]
```

其中 `A` = 该 batch 内 `labels ≠ −100` 的位置集合（即答案 token + 结束符），`z_t` 为学生 LLM
在**压缩输入** `[prompt(with n1 visual tokens) ; answer]` 上的 logits。

### 3.2 实现

代码里**没有手写 CE**，直接用 HF 返回值：

```python
# LLaVA-1.5: train_v2.py:101-108
embeds = tap._splice(c.to(torch.bfloat16), ids)
out = tap.model(inputs_embeds=embeds, attention_mask=attn, labels=labels,
                output_hidden_states=False)
ce = out.loss
```
HF 内部做标准右移（`shift_logits = logits[..., :-1, :]`、`shift_labels = labels[..., 1:]`）、
`ignore_index=-100`、并用 `attention_mask` 过滤 padding，最后 CrossEntropyLoss 默认 `reduction='mean'`。

```python
# NeXT: train_llavanext_v2.py:104-108（逐样本前向后取均值）
out = tap.model(inputs_embeds=tap.splice(c, ids_t), attention_mask=..., labels=lab_t)
ces.append(out.loss)
...
res = {"ce": torch.stack(ces).mean(), ...}
```
```python
# Qwen: train_qwen_v2.py:191，复用 train_qwen.py 的 answer_loss（内部拼 position_ids）
ce = tap.answer_loss(compressed, encoded, n1, device)
```

### 3.3 ⚠️ 一个必须知道的口径差异

- **LLaVA-1.5**：`bsr=4` 一次前向 → HF 的 `mean` 是**整个 batch 内所有有效 token 的平均**
  ⇒ **长答案的样本权重更大**（token 加权）。
- **NeXT / Qwen**：`bsr=1` 逐样本前向，再对 4 个微批取均值 ⇒ **每个样本等权**
  （样本级 token-mean 再平均）。

两者在答案长度分布不均时数值不同。论文若要严谨比较骨干，应说明这一点；
或把 LLaVA-1.5 也改为 `bsr=1 + 累积`。**本文所有结果是在这个不一致下得到的**（如实记录）。

### 3.4 没有教师

v1 的 KD（答案分布蒸馏）与 hidden-state 蒸馏**已完全删除**：代码中不存在教师前向，
学生前向也显式 `output_hidden_states=False`。`L_CE` 是唯一的任务信号。

---

## 4. 损失二：重构损失 `L_REC`

### 4.1 重构 decoder（`v2_model.py:35-62`，仅训练期存在）

```python
in_proj  : Linear(d → 128)               # C 作为 K/V 的投影
pos_proj : Linear(d → 128)               # 正弦位置编码的投影
ln1, ln2 : LayerNorm(128)
attn     : nn.MultiheadAttention(128, num_heads=4, batch_first=True)
ffn      : Linear(128→512) → GELU → Linear(512→128)
out      : Linear(128 → d)

forward(c, n_src):                       # c: (B, n1, d) → R: (B, n_src, d)
    kv = in_proj(c)                                     # (B, n1, 128)
    PE = sinusoidal(n_src, d)                           # (n_src, d)  ← 注意在 d 维上生成
    q  = pos_proj(PE).expand(B, -1, -1)                  # (B, n_src, 128)
    h  = q + attn(ln1(q), kv, kv)                        # pre-norm 残差，cross-attention
    h  = h + ffn(ln2(h))                                 # pre-norm 残差
    R  = out(h)                                          # (B, n_src, d)

sinusoidal(n, d):  div = exp(arange(0,d,2) * (−ln(10000)/d))
                   PE[:, 0::2] = sin(pos*div);  PE[:, 1::2] = cos(pos*div)
```

写作时容易写错的三点：

1. **位置编码先在 `d` 维生成、再被 `pos_proj` 线性降到 128**（不是直接在 128 维生成正弦）。
2. **只有 query 过了 `ln1`；K/V 用的是未归一化的 `in_proj(c)`**。
3. `n_src` 是**目标 token 数**：LLaVA-1.5 恒为 576；NeXT/Qwen 按图传 `n_src=N`
   （`v2_model.py:196` 的 `self.reconstructor(c, n_src=n)`）。**decoder 的可学习权重与 `n_src` 无关**，
   所以同一份权重适配任意 N。

### 4.2 损失公式（`v2_model.py:200-207`）

设重构输出 `R ∈ ℝ^{B×n_src×d}`，目标 `X ∈ ℝ^{B×n_src×d}`（= 压缩器输入的那份视觉 token，
**已 detach**），两项都在 **fp32** 下计算：

```python
r, t = R.float(), X.float()
mse = (r - t).pow(2).mean()                       # 对 B、n_src、d 全部元素求均值
rel = mse / t.pow(2).mean().clamp_min(1e-8)       # 尺度无关：除以目标平均能量
cos = 1.0 - F.cosine_similarity(r, t, dim=-1).mean()   # 逐 token 余弦后对 (B, n_src) 求均值
L_REC = rel + 0.1 * cos
```

写成数学式：

$$
\mathcal{L}_{\text{rel}}=\frac{\frac{1}{|\Omega|}\sum_{\Omega}(R-X)^2}{\max\!\big(\frac{1}{|\Omega|}\sum_{\Omega}X^2,\;10^{-8}\big)},
\qquad
\mathcal{L}_{\text{cos}}=1-\frac{1}{B\cdot n_{src}}\sum_{b,i}\frac{R_{b,i}^{\top}X_{b,i}}{\lVert R_{b,i}\rVert\,\lVert X_{b,i}\rVert}
$$

$$
\mathcal{L}_{\text{REC}}=\mathcal{L}_{\text{rel}}+0.1\,\mathcal{L}_{\text{cos}}
$$

其中 $\Omega$ 是 $(B, n_{src}, d)$ 的全部元素索引。

**注意分母是全局标量**（整个 batch 的 $X$ 平均能量），不是逐样本/逐 token 归一化。
所以日志里 `recon_rel=0.70` 的含义是"重构平方误差约为目标平均能量的 70%"，
**不是"保留了 30% 的信息"**——写作时不要误读。

### 4.3 目标张量 X 的来源（关键，容易写错）

`X` **就是压缩器的输入**，同一份张量、在 `torch.no_grad()` 下算出：

```python
# LLaVA-1.5  train_v2.py:113-116
x = tap.vision_features(pixel_values.to(device, bfloat16))    # 内部已 detach
with torch.no_grad():
    y = tap.projector(x.to(bfloat16)).float()                 # ← 这就是 X
c, aux = merger(y, n1=args.n1, use_fusion=True, need_recon=True)
...
rec, rec_diag = reconstruction_loss(aux["recon"], y)          # 目标 = 同一个 y
```

即"**从 m 个压缩 token 重构出全部 M 个原始视觉 token**"，目标不经过任何再加工，
也不使用问题/答案文本。多图的 NeXT/Qwen 是**每图单独算 `reconstruction_loss` 再取均值**
（`train_qwen_v2.py:194-196`：`torch.stack([...]).mean()`），因为各图 N 不同。

### 4.4 总损失与有效权重

```python
# train_v2.py:256-257
loss = args.ce_weight * losses["ce"] + args.recon_weight * losses["recon"]
#       ce_weight   = 1.0                  recon_weight = 0.1
```

$$
\boxed{\;\mathcal{L}=1.0\cdot\mathcal{L}_{\text{CE}}+0.1\cdot\mathcal{L}_{\text{REC}}
      =1.0\cdot\mathcal{L}_{\text{CE}}+0.1\cdot\mathcal{L}_{\text{rel}}+0.01\cdot\mathcal{L}_{\text{cos}}\;}
$$

`recon_weight=0.1` 与内部的 `0.1`（cos 项）都是**沿用 v1、未做任何调参**的取值。

梯度累积时损失先除以 `grad_accum` 再 `backward()`（`train_llavanext_v2.py` / `train_qwen_v2.py`
的 `loss = (...) / args.grad_accum`），一个累积窗口内累加后才 `opt.step()`。

---

## 5. 优化器、日程与梯度

| 项 | 值 | 代码 |
|---|---|---|
| 优化器 | AdamW，只含 merger 参数 | `torch.optim.AdamW(merger.parameters(), ...)` |
| lr | 2e-4 | `--lr` |
| betas | (0.9, 0.95) | 硬编码 |
| weight_decay | 0.01 | 硬编码 |
| 梯度裁剪 | **全局范数 1.0**，对 `merger.parameters()` | `clip_grad_norm_` |
| warmup | 前 100 步**线性**：`lr·(step+1)/100` | |
| 衰减 | 其后 cosine 到 `0.1×lr` | `--min-lr-ratio 0.1` |
| `zero_grad` | `set_to_none=True`；在每个累积窗口开始时、以及 `step()` 之后各调一次 | |
| DDP | `find_unused_parameters=True` | 三骨干一致 |

```python
def lr_at(step):
    if step < warmup:  return lr * (step + 1) / warmup
    t = (step - warmup) / max(total_steps - warmup, 1)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * t)))
```

裁剪时机：**每个累积窗口结束时裁一次**（不是每个微批），与 `step()` 同步。

---

## 6. 训练期验证、可行性判据与选点

### 6.1 配对验证

触发步：`{100, 200, 400, 600} ∪ {k·save_every} ∪ snap_steps`；我们的运行里 `save_every=500`，
`snap_steps={1031, 2062}`（NeXT/Qwen）或 `{1031,2062}`（LLaVA-1.5 为 `1031,2062`）。

固定取 **val 的前 64 条**（`--diag-items 64`），对同一批图像分别跑：

- `fused`：`use_fusion=True`（正常压缩）
- `base`：`use_fusion=False` ⇒ 代码里 `g = zeros` ⇒ `c = a`（**同一批 medoid 锚点，不写入**）

两者锚点、顺序、预算完全相同，唯一差别是有没有融合 ⇒ 这是"增益是否来自融合"的训练内对照。

产出指标：
```
task_gain  = base_ce    − fused_ce          > 0 表示任务侧变好
rec_gain   = base_recon − fused_recon       > 0 表示重构变好
```
外加监控量（不参与判定）：`gate_abs`、`anchor_cos`、`assign_ent`、`write_ratio`、
`residual_norm`、`anchor_norm`、`out_norm`、`count_mean`。

### 6.2 否决条件（`train_v2.py:feasibility_decision`，NeXT/Qwen 直接 import 复用同一函数）

依次判定，命中即停：

1. `fused_ce` 或 `fused_recon` 非有限 → 停（`non-finite loss`）
2. `step ≥ 200` 且 `task_gain < 0` 且 `rec_gain < 0` → 停
3. `step ≥ 400` 且连续两次 `task_gain < 0` → 停
4. `step ≥ 600` 时：
   - `gate_abs < 1e-5` 且 `task_gain ≤ 0` → 停（门没开且没收益）
   - `task_gain ≤ 0` → 停
   - **`rec_gain < −0.01`** → 停（任务变好但重构**实质性**退化）
   - `task_gain < 0.5 × 上次 task_gain` → 停（早期收益快速衰减）
   - 全部通过 → `feasible=True`
5. 若训练跑满预算仍没触发 `step ≥ 600` 分支，则在结束时写
   `feasible = (最后一次 task_gain > 0)`，理由 `full budget reached`

**相对 v1 的两处重校准**（都有实测依据，写在函数 docstring 里）：

- **移除** v1 的 `anchor_cos < 0.75 ∧ gate_abs > 0.5` 否决项，二者降级为纯监控量。
  理由：该联立条件是 v1 门恒为 ~0.011 的产物；v2 共享门按设计开到 0.7–1.0，
  使它在一个 `task_gain=+3.28`、`rec_gain=+0.03`、实测精度创纪录的运行上误触发。
- `rec_gain` 阈值由 `≤ 0` 改为 `< −0.01`。
  理由：一次运行因 `rec_gain = −0.0027`（相对 0.36%，噪声级）被杀，而 v1 真实重构退化量级约 0.03，
  相差一个数量级。改后重放验证：噪声级通过、`−0.05` 仍被拦。

### 6.3 选点与产物

- **`best.pt` = 验证 `fused_ce` 最低的那个 checkpoint**（CE-only，不含 KD  anymore）。
- 同时保存：`last.pt`（每 `save_every` 步与 `snap_steps` 处）、`snap_step{N}.pt`、
  `final.pt`（训练结束时的最终权重，仅 NeXT/Qwen 训练脚本写）。
- `diagnostics.jsonl` 每次验证一行；`train_log.jsonl` 每 `log_every` 步一行；
  `feasibility.json` 首次给出裁决时写入；`meta.json` 记录全部超参。
- **`compressor_config` 不含 `n1`**（这是预算无关性的前提），内容为
  `{d, hidden, gate_hidden, recon_hidden, tau, gate_scale, write_cap}`
  （Qwen 额外含 `backbone` 与 `backbone_dtype`；NeXT 含 `backbone`）。
- ⚠️ 已知记录缺陷：`runs/nextv2_f30_13b/ckpt/best.pt` 的 `backbone` 字段被硬编码成
  `llava-v1.6-vicuna-7b`（实际是 13b，可由 `d=5120`、参数数 8,073,986 确认）。代码已改为从
  `--model-path` 推导，**已产出的文件未回填**。

---

## 7. 推理：三骨干共用的部分

### 7.1 总流程

```
读图像 → 骨干视觉编码 → (可选)压缩 → 构造多轮 prompt → 拼接视觉 embedding
       → 贪婪解码 ≤16 token（遇结束符即停，且不把结束符计入输出）
       → is_correct 判分
```

- **每张图只压缩一次**，三个 turn 复用同一份视觉 embedding。
- 每轮**重新 prefill 整段对话**（把之前几轮的模型预测插回对话历史），
  不使用上一轮的 KV cache。
- 贪婪：`argmax(logits[0,-1])`，无采样、无 temperature、无 beam、无重复惩罚。
- **上限 16 个新 token**；命中结束符（LLaVA `eos=2`；Qwen `<|im_end|>=151645`）立即停止，
  **结束符本身不 append 到输出**。
- 解码实现：**全量重算，不用 KV cache**（`use_cache=False`；每生成一个 token 把整段序列
  重新前向）。原因见 §11.2 —— 本机 KV-cache 增量解码必 SIGFPE。
- 不传 `attention_mask`（等价全 1）、不传 `position_ids`（LLaVA/NeXT 用默认 1-D）。
- 输出解码：`tok.decode(made, skip_special_tokens=True).strip()`。

### 7.2 多轮 prompt 构造（`eval_mt_gqa.build_prompt`，LLaVA 与 NeXT 共用）

```python
for i, (q, a) in enumerate(zip(questions, answers)):
    head = "USER: <image>\n" if i == 0 else "USER: "     # 仅第 1 轮含图像占位符
    text = head + q + " ASSISTANT:"
    if a is not None: text += " " + a + "</s>"            # 历史轮插入模型自己的预测
    parts.append(text)
prompt = "".join(parts)
```
调用方式：第 t 轮传 `questions[:t+1]` 与 `preds + [None]`（最后一轮 `a=None` ⇒ 以 ` ASSISTANT:` 结尾）。

Qwen 走 `processor.apply_chat_template(messages, add_generation_prompt=True)`，
消息结构与上式对应（第 0 轮 content 里含 `{"type":"image"}`，历史轮含
`{"role":"assistant"}`）。

### 7.3 四种模式

| mode | 视觉输入 | 是否需要 ckpt |
|---|---|---|
| `fused` | `merger(y, n1, use_fusion=True)` | ✅ |
| `anchor` | `merger(y, n1, use_fusion=False)` ⇒ `g=0` ⇒ **c = a 精确** | ✅ |
| `random` | 从 `y` 中按 `sha1(image_id)` 种子取 n1 行，**按 raster 序还原** | ❌ |
| `none` | 全部 N 个 token（NeXT/Qwen 才实现） | ❌ |

`random` 的确定性实现（三骨干一致）：
```python
seed = int(hashlib.sha1(str(image_id).encode()).hexdigest()[:8], 16) % 2**31
gen  = torch.Generator().manual_seed(seed)
idx  = torch.randperm(N, generator=gen)[:n1].sort().values
out  = torch.gather(y, dim=1, idx.unsqueeze(-1).expand(-1, -1, d))   # y = projector 之后的真实 token
```
要点：**采样自与 `C` 同一个空间**（projector/merger 之后的真实视觉 token），
排序回 raster 序，逐图确定性可复现。`anchor` 因不含任何可学习参数，
在不同 checkpoint / 骨干运行间**逐位相同**（已多次实测）。

### 7.4 预算与钳位

| 骨干 | 典型 N | 10% | 20% |
|---|---|---|---|
| LLaVA-1.5 | 576（固定） | `n1=58`（10.07%） | `n1=115`（19.97%） |
| LLaVA-NeXT 7B/13B | 1368–2928，MT-GQA 均值 2242 | `n1=244`（≈10.9%） | `n1=488`（≈21.8%） |
| Qwen2.5-VL | 126–529，MT-GQA 均值 351.6 | `n1=41`（≈11.7%） | `n1=83`（≈23.6%） |

NeXT/Qwen 的 N 逐图不同，故评测按**每张图**钳位 `n1 ← min(n1, N)`；训练按 **batch** 钳位。
实测钳位触发：NeXT 0 次，Qwen 44/8248 微批（0.53%）。

### 7.5 数据与分片

- MT-GQA testdev-balanced：**4,061 对话 × 3 轮 = 12,183 turn**。
- 分片：`dialogues = dialogues[shard::n_shards]`，评测用 `n_shards=8`（每卡一分片）。
- 图像路径：`DATASET_ROOT / d["image_path"]`，
  `DATASET_ROOT = /ossfs/workspace/Rethinking Token Reduction for Large Vision-Language Models_dataset`。
- 每 shard 落盘一个 JSON，含 `args`、`ckpt_step`、`token_stats`（`N_min/N_max/N_mean/clamped_images`）、
  `rows[{dialogue_id, image_id, questions, references, predictions, correct[3]}]`。

---

## 8. 判分与显著性检验

### 8.1 `is_correct`（`eval_mt_gqa.py:38-62`，GQA 官方口径）

```python
def _norm(text):
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))   # 去掉所有标点
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())      # 去冠词 + 合并空格

def is_correct(response, reference):
    r, f = _norm(response), _norm(reference)
    if not f: return False
    if f in ("yes", "no"):                      # 参考答案是 yes/no
        first = r.split()[0] if r.split() else ""
        if first in ("yes", "no"): return first == f
        return (f == "no") == (NEG_RE.search(r) is not None)
    return re.search(rf"\b{re.escape(f)}\b", r) is not None    # 词边界包含
```
`NEG_RE = \b(no|not|never|none|neither|nor|without|cannot|isn't|aren't|doesn't|don't|didn't|can't|won't)\b`

**写作时须知的一个细节**：`_norm` 会先删掉所有标点，因此 `NEG_RE` 中带撇号的
`isn't / aren't / doesn't / don't / didn't / can't / won't` **永远匹配不上**
（文本里已变成 `isnt / arent / ...`）。这是继承自既有 harness 的行为，
本文所有结果都用的是这个版本，不要"顺手修正"，否则与历史数字不可比。

准确率：
$$
\text{acc}=\frac{1}{3|D|}\sum_{d\in D}\sum_{t=1}^{3}\mathbb{1}\big[\text{is\_correct}(\hat y_{d,t}, y_{d,t})\big]
$$

### 8.2 McNemar（turn 级配对，Yates 连续性校正）

```python
b = #{turns: fused 对 ∧ null 错}      c = #{turns: fused 错 ∧ null 对}
chi2 = (|b − c| − 1)² / (b + c)                       # b+c=0 时返回 p=1
p    = stats.chi2.sf(chi2, 1)                          # 生存函数，不是 1−cdf
```
**必须用 `sf`**：χ² 大于约 150 时 `1 − cdf` 会下溢成**恰好 0**，把 `p=3e-297` 误印成 `p=0`。
这个 bug 在早期版本里出现过，已修正（`analyze_frac.py` / `analyze_llavanext.py` /
`analyze_qwen_v2.py` 均用 `sf`）。

`b`（改善 turn 数）/ `c`（退化 turn 数）也一并报告，因为 Δ 相同但 b/c 分布不同时含义不同。

---

## 9. 监控量定义（论文若引用需一致）

全部在 `v2_model.py:185-195` 的 `no_grad` 块里算，对**所有段与 batch 求均值**：

| 名称 | 定义 |
|---|---|
| `gate_abs` | `mean(|g_k|)`，`g_k = gate_scale·tanh(gate_net(v_k))` |
| `anchor_cos` | `mean(cos(c_k, a_k))`，在**范数锚定之后**计算 |
| `assign_ent` | `mean(−Σ_i w_i ln w_i)`（nats），段内聚合权重的熵 |
| `anchor_norm` | `mean(‖a_k‖)` |
| `out_norm` | `mean(‖c_k‖)`，按构造 `= anchor_norm` |
| `residual_norm` | `mean(‖δ_k‖)` |
| `write_ratio` | `mean(‖g_k·δ_k‖ / ‖a_k‖)` ← **相对量**，逐 token 除以自身范数 |
| `count_mean` | `mean(L_k)` |
| `recon_rel` / `recon_cos` | §4.2 的两项 |

`write_ratio` 之所以重要：视觉 token 范数**远非一致**（实测图内 CV：LLaVA-1.5 **1.14**、
Qwen2.5-VL **0.40**、Qwen3-VL **0.71**；极差 12–75 倍），所以绝对 `‖δ‖` 无法跨 token 比较，
必须除以各自 `‖a_k‖`。这也是范数锚定做的是"锚定到该 token 自身范数"而不是"归一化到 1"的原因。

---

## 10. 一次完整运行的命令

```bash
# ── LLaVA-1.5：30% 数据 / 2062 步 / n1=58 ──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.run --nproc_per_node=6 \
  --master_port=36207 train_v2.py --out runs/v2f30_1ep \
  --model-path /home/admin/llava-1.5-7b-hf \
  --epochs 4.0 --max-steps 2062 --n1 58 --batch-per-rank 4 \
  --workers 6 --lr 2e-4 --warmup 100 --min-lr-ratio 0.1 --seed 17 --n-val 2000 \
  --ce-weight 1.0 --recon-weight 0.1 --data-frac 0.30 --snap-steps 1031,2062 \
  --log-every 20 --save-every 500 --diag-items 64

# ── NeXT（7B/13B 只换 --model-path）：bsr=1 + 累积 4 ──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.run --nproc_per_node=6 \
  --master_port=36251 train_llavanext_v2.py --out runs/nextv2_f30 \
  --model-path /home/admin/llava-v1.6-vicuna-7b-hf \
  --epochs 4.0 --max-steps 2062 --n1 244 --batch-per-rank 1 --grad-accum 4 \
  --data-frac 0.30 --snap-steps 1031,2062 --diag-items 64

# ── Qwen（注意 fp16 与坏 grid 过滤，见 §11.1）──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.run --nproc_per_node=6 \
  --master_port=36271 train_qwen_v2.py --out runs/qwenv2bf16 \
  --backbone-dtype bfloat16 --epochs 4.0 --max-steps 2062 --n1 41 \
  --batch-per-rank 1 --grad-accum 4 --data-frac 0.30 --snap-steps 1031,2062 \
  --bad-grid-census runs/qwen_grid_census.json --bad-grids runs/qwen_bf16_bad_grids.txt

# ── 推理：每个 (mode, n1) 起 8 个分片 ──
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 eval_v2.py --mode fused --n1 58 \
    --ckpt runs/v2f30_1ep/ckpt/best.pt --model-path /home/admin/llava-1.5-7b-hf \
    --shard $i --n-shards 8 \
    --out outputs/mt_gqa/v2f30_1ep_n58_fused_full_shard$i.json &
done; wait
python3 analyze_frac.py            # 或 analyze_llavanext.py / analyze_qwen_v2.py
```

必设环境变量：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false`；
另需把 `/usr/local/cuda/lib64` 从 `LD_LIBRARY_PATH` 里剔除（本机残留 cuda-10.0/11.1）。

---

## 11. 本机特有的强制约束（不遵守会崩或静默出错）

### 11.1 Qwen：训练 fp16 + 坏 grid 过滤，评测 bf16

- **bf16 + torchrun 下 Qwen 视觉塔确定性 SIGFPE**，触发条件是**特定的 (h,w) patch grid**，
  且**非单调**：(16,16)N=64 与 (18,18)N=81 崩，而 (14,14)N=49 与 (20,20)N=100 通过；
  每个 grid 重复 3 次结果一致（确定性，非抖动）。单进程 + 无 dist 时 bf16 反而正常。
- 黑名单 **23 个 grid**（`runs/qwen_bf16_bad_grids.txt`），在本训练子集中只涉及
  **189 / 49,511 = 0.382%** 个样本 → 由 `--bad-grid-census/--bad-grids` **在建 DataLoader 之前剔除**。
  **不能在训练循环里跳过**：某 rank 少调一次 `backward()` 会让 DDP 的 all-reduce **死锁**。
- 因此 Qwen 存在 **fp16 训练 / bf16 评测** 的不对称。代价已实测：同一评测协议下，
  bf16 训练的 checkpoint `fused@41 = 55.6021`，而 fp16 训练的只有 `49.5280`（差 6.1 分）。

### 11.2 全骨干：禁用 KV cache 解码

增量解码（M=1 + 增长的 past_key_values）在本机 **bf16 与 fp16 都 SIGFPE**
（Qwen 崩在 `modeling_qwen2_5_vl.py:715` 的 `q_proj`，NeXT 用 `generate` 也崩），
而 prefill 式前向（M ≈ 70–500）正常。故**三个骨干统一用全量重算解码**。
代价：长序列很慢（NeXT-13B 的 `none` 档 21.7 s/对话，因序列约 2500 token × 17 次重算）。

### 11.3 全骨干：必须 bf16 评测，fp16 会静默掉分

同 25 个对话、同解码路径：未压缩 **bf16 = 62.67% vs fp16 = 30.67%**（差 32 分且不报错）。
短序列（压缩后 L≈90）两者接近，**只测压缩档会误判"fp16 没问题"**。

### 11.4 评测协议的固有偏置：16-token 上限

`is_correct` 用词边界匹配 + 生成上限 16 token，而**未压缩模型答案冗长**
（NeXT none 平均 9.67–10.18 token、Qwen none 12.92 token），常被截断在答案出现之前
（Qwen none 截断率 **34.0%**、NeXT none 5.6–6.2%）。压缩后答案简短（2.3–4.3 token，
截断率 0.2–0.5%）。写作时**不能声称"压缩超过未压缩"**；剔除截断后该结论不成立
（NeXT-13B 在 10% 预算上甚至低于未压缩 1.42 分）。详见 `METHOD_v2.md` §7.2。

---

## 12. 写作时的自检清单

- [ ] `L_REC` 的分母写成"全局平均能量"而非逐样本归一化（`recon_rel=0.70` ≠ 保留 30% 信息）
- [ ] `L_REC = rel + 0.1·cos`，外层再乘 0.1 ⇒ 实际 cos 权重是 **0.01**
- [ ] 位置编码先在 **d 维**生成正弦、再 `Linear(d→128)` 投影
- [ ] cross-attention 的 **K/V 未归一化**，只有 Q 过 `ln1`
- [ ] `n_src`（重构目标长度）= 输入视觉 token 数 N，**与预算 m 无关**；decoder 参数与 N 无关
- [ ] 残差池按**当前实现包含 anchor**（tex 目前写的是排除）——两者 μ 仅差 `(1−w_a)` 倍、方向恒同
- [ ] LLaVA 答案有**前导空格**、结束符是 `</s>`(2)；Qwen 无前导空格、结束符是 `<|im_end|>`(151645)
- [ ] CE 加权方式：LLaVA-1.5 是 batch 内 **token 加权**，NeXT/Qwen 是 **样本等权**
- [ ] `compressor_config` **不含** `n1`；预算只在推理时给 `--n1`
- [ ] `anchor` 对照 = `use_fusion=False`，其输出**精确等于** medoid 锚点本身
- [ ] `random` null 在 **projector/merger 之后的空间**采样，并按 raster 序还原、逐图 sha1 播种
- [ ] 解码：贪婪、≤16 token、遇结束符即停且不计入输出、**全量重算无 KV cache**
- [ ] McNemar 用 **Yates 校正**且 p 值用 **`chi2.sf`**（避免下溢误报 0）
- [ ] `is_correct` 里 `_NEG` 的缩写形式因先去标点而永远不匹配——沿用即可，勿修改
- [ ] Qwen：训练 fp16 + 坏 grid 过滤 189 条；评测 bf16
- [ ] 不要写"压缩优于未压缩"（§11.4）
