# AnchorMerge v2：重构 Decoder 与正弦位置嵌入详解

> 整理日期：2026-11-13  
> 网络结构依据：`multi/v2_model.py:25-62`  
> 适用范围：AnchorMerge v2 的训练期视觉特征重构模块  
> 损失表述：统一使用单一的 \(\mathcal{L}_{\mathrm{REC}}\)，即目标能量归一化的均方误差  
> 注意：重构 decoder 仅用于训练，推理时完全移除。

---

## 1. 模块的作用

AnchorMerge 将原始视觉特征

\[
X\in\mathbb{R}^{N\times d}
\]

压缩为

\[
C\in\mathbb{R}^{n_1\times d},\qquad n_1\leq N.
\]

训练时，一个轻量级重构 decoder 尝试从压缩表示 \(C\) 中恢复原始视觉特征：

\[
R=\mathcal{D}_{\psi}(C,N),
\]

其中：

- \(N\)：原始视觉 token 数；
- \(n_1\)：压缩后的视觉 token 数；
- \(d\)：视觉特征维度，与 LLM 隐层维度一致；
- \(R\in\mathbb{R}^{N\times d}\)：重构出的视觉特征；
- \(\mathcal{D}_{\psi}\)：参数为 \(\psi\) 的重构 decoder。

重构目标是压缩器的输入 \(X\)，也就是同一份经过 projector/patch-merger、但尚未压缩的视觉 token。\(X\) 在 `torch.no_grad()` 下计算并作为固定目标，不通过重构损失更新视觉骨干。

该模块的作用是提供辅助监督：除了要求 \(C\) 有助于回答当前问题，还要求它尽可能保留恢复全部原始视觉 token 所需的信息。

---

## 2. Decoder 不是普通自回归 Decoder

这里的 decoder 是一个**单层、非自回归的 Transformer cross-attention 重构器**：

- 不逐 token 自回归生成；
- 不使用 masked self-attention；
- 不使用卷积或反卷积上采样；
- 不为每个 \(N\) 保存一张可学习的位置参数表；
- 一次并行生成全部 \(N\) 个重构 token。

其核心计算方式是：

\[
\boxed{
N\text{ 个目标位置查询}
\xrightarrow{\text{从 }n_1\text{ 个压缩 token 中读取信息}}
N\text{ 个重构视觉 token}
}
\]

每个目标位置都有一个独立 query；该 query 通过 cross-attention 读取全部压缩 token，再由输出投影恢复为 \(d\) 维视觉特征。

---

## 3. 具体网络结构

实现类为 `VisualReconstructor`。设 batch 中的压缩特征为

\[
C\in\mathbb{R}^{B\times n_1\times d}.
\]

decoder 的主要组件如下：

| 组件 | 当前实现 |
|---|---|
| 压缩特征输入投影 | `Linear(d, 128)` |
| 位置查询投影 | `Linear(d, 128)` |
| Cross-attention | `nn.MultiheadAttention(128, num_heads=4, batch_first=True)` |
| Attention 前归一化 | `LayerNorm(128)` |
| FFN 前归一化 | `LayerNorm(128)` |
| FFN | `Linear(128, 512) → GELU → Linear(512, 128)` |
| 输出投影 | `Linear(128, d)` |
| 残差结构 | Attention 与 FFN 均采用 pre-norm 残差连接 |
| Decoder 层数 | 1 层 |

---

## 4. 逐步前向计算

### 4.1 将压缩 token 映射到 decoder 隐层

首先对 \(C\) 做线性投影：

\[
U=W_{\mathrm{in}}C+b_{\mathrm{in}},
\qquad
U\in\mathbb{R}^{B\times n_1\times128}.
\]

\(U\) 被同时传给 cross-attention 的 key 和 value 输入路径；`MultiheadAttention` 内部再完成相应的 key/value 投影。

这些压缩 token 是 decoder 重构时可以读取的全部视觉信息来源。

### 4.2 为每个原始 token 位置构造 Query

先生成 \(N\) 个固定正弦位置嵌入：

\[
P=\operatorname{PE}(N)\in\mathbb{R}^{N\times d}.
\]

然后使用可学习线性层将它们投影到 decoder 的 128 维隐空间：

\[
Q=PW_{\mathrm{pos}}+b_{\mathrm{pos}},
\qquad
Q\in\mathbb{R}^{B\times N\times128}.
\]

这里需要注意：

1. 正弦位置嵌入是先在 **\(d\) 维**生成的；
2. 随后才通过 `pos_proj` 投影到 **128 维**；
3. 正弦嵌入本身固定、不可学习；
4. 可学习的是位置嵌入之后的线性变换。

### 4.3 Cross-attention 与残差连接

\[
H=Q+\operatorname{MHA}\big(\operatorname{LN}_1(Q),U,U\big),
\]

其中：

\[
H\in\mathbb{R}^{B\times N\times128}.
\]

第 \(i\) 个位置 query 可以读取全部 \(n_1\) 个压缩 token，从而为原始第 \(i\) 个位置收集相关信息。

严格按照当前代码，只有 query 经过 `ln1`；key/value 使用未经该 LayerNorm 的 \(U\)。

### 4.4 FFN 与输出投影

\[
Z=H+\operatorname{FFN}\big(\operatorname{LN}_2(H)\big),
\]

其中：

\[
\operatorname{FFN}(h)
=W_2\operatorname{GELU}(W_1h+b_1)+b_2,
\]

隐层维度变化为 \(128\rightarrow512\rightarrow128\)。

最后映射回原始视觉特征维度：

\[
R=ZW_{\mathrm{out}}+b_{\mathrm{out}},
\qquad
R\in\mathbb{R}^{B\times N\times d}.
\]

单图情况下可省略 batch 维，得到 \(R\in\mathbb{R}^{N\times d}\)。

---

## 5. 正弦位置嵌入是什么？

对于原始序列中的第 \(i\) 个 token，标准正弦位置嵌入定义为：

\[
P_{i,2j}
=\sin\left(\frac{i}{10000^{2j/d}}\right),
\]

\[
P_{i,2j+1}
=\cos\left(\frac{i}{10000^{2j/d}}\right).
\]

不同位置 \(i\) 对应不同的正弦/余弦组合，因此每个目标位置都有可区分的编码。

这里的 position 指的是**原始视觉 token 序列的一维索引**：

\[
i=0,1,\ldots,N-1.
\]

对于以 raster 顺序展开的视觉 token，它保留的是展开后的序列次序。它不是显式的二维坐标编码，也不是 Qwen 的 mRoPE，更不是 LLM 推理时使用的位置编码。

---

## 6. 为什么重构需要正弦位置查询？

位置查询解决的是：

> **当前应该为原始的哪一个 token 位置生成重构特征？**

如果为所有输出位置使用完全相同的 query，decoder 就缺少区分目标位置的依据，无法明确：

- 第 1 个输出应对应 \(x_1\)；
- 第 2 个输出应对应 \(x_2\)；
- …
- 第 \(N\) 个输出应对应 \(x_N\)。

加入位置嵌入后，第 \(i\) 个 query 为：

\[
q_i=W_{\mathrm{pos}}\operatorname{PE}(i)+b_{\mathrm{pos}}.
\]

它携带目标位置 \(i\) 的信息，并从压缩表示 \(C\) 中检索相关内容，最终生成 \(r_i\)。

因此，位置查询同时决定：

1. **输出 token 数**：构造 \(N\) 个 query，就生成 \(N\) 个输出；
2. **输出与原始 token 的对应关系**：第 \(i\) 个输出对应第 \(i\) 个原始 token；
3. **重构序列的顺序**：输出保持原始视觉 token 的序列顺序。

简言之，\(C\) 提供“用于重构的视觉内容”，位置 query 指定“要重构哪个位置”。

---

## 7. 为什么位置嵌入可以支持变长 \(N\)？

正弦位置嵌入根据传入的 \(N\) 临时生成，因此不需要学习一个固定大小的位置嵌入矩阵。

- LLaVA-1.5：通常令 `n_src=N=576`；
- LLaVA-NeXT / Qwen：按每张图传入实际的 `n_src=N`；
- decoder 的所有可学习权重都与 \(N\) 无关。

因此，同一个 decoder 可以处理不同骨干和不同图像产生的变长视觉序列。

这种可变性只属于训练期辅助重构过程，不意味着推理时需要运行 decoder。

---

## 8. 重构损失与总训练目标

重构损失直接定义为目标能量归一化的均方误差，并统一记为 \(\mathcal{L}_{\mathrm{REC}}\)：

\[
\boxed{
\mathcal{L}_{\mathrm{REC}}
=
\frac{\operatorname{MSE}(R,X)}
{\max\left(\operatorname{Mean}(X^2),\epsilon\right)}
}
\]

其中，\(\epsilon=10^{-8}\) 用于保证数值稳定。展开为元素级求和形式：

\[
\mathcal{L}_{\mathrm{REC}}
=
\frac{
\frac{1}{|\Omega|}\sum_{(b,i,j)\in\Omega}
\left(R_{b,i,j}-X_{b,i,j}\right)^2
}{
\max\left(
\frac{1}{|\Omega|}\sum_{(b,i,j)\in\Omega}X_{b,i,j}^2,
\;10^{-8}
\right)
}.
\]

这里，\(\Omega\) 是 \((B,N,d)\) 张量中全部元素的索引集合。分母是整个 batch 中目标特征的全局平均能量，而不是逐样本或逐 token 归一化。该归一化降低了不同特征尺度的影响，同时约束重构特征的方向与幅值。

将答案交叉熵与重构损失组合，得到总训练目标：

\[
\boxed{
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+\lambda_{\mathrm{REC}}\mathcal{L}_{\mathrm{REC}}
}
\]

当前设定为 \(\lambda_{\mathrm{REC}}=0.1\)，因此：

\[
\boxed{
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+0.1\mathcal{L}_{\mathrm{REC}}
}
\]

\(\mathcal{L}_{\mathrm{CE}}\) 仅在答案 token 和结束符位置计算；\(\mathcal{L}_{\mathrm{REC}}\) 在重构特征 \(R\) 与固定目标 \(X\) 之间计算。两项损失共同更新 AnchorMerge，而重构 decoder 仅由重构分支训练。

---

## 9. 推理时是否保留 Decoder？

不保留。

推理时调用压缩器使用 `need_recon=False`，因此：

- 不生成位置查询；
- 不执行 cross-attention；
- 不生成 \(R\)；
- 不计算重构损失；
- 不需要原始长度 \(N\) 的 decoder 输出。

实际部署只保留 AnchorMerge 压缩器。重构 decoder 不参与 LLM 前向，也不增加推理延迟。

---

## 10. 论文写作时容易写错的细节

- 应写成 **learned linear transformations of fixed sinusoidal positional embeddings**，不要让读者误以为正弦嵌入本身可学习。
- 位置嵌入先在 \(d\) 维生成，再投影到 128 维。
- 这是单层、非自回归 cross-attention decoder，不是自回归 Transformer decoder。
- 位置查询对应原始视觉序列的一维 token 索引，不要直接写成二维空间坐标。
- Cross-attention 中每个目标位置 query 都可以读取全部 \(n_1\) 个压缩 token。
- 当前实现中只有 query 经过第一个 LayerNorm，key/value 不经过该归一化。
- 重构目标 \(X\) 是压缩器的输入特征，不是 ViT 投影前的特征，也不是文本或答案特征。
- 这些位置查询只属于训练期 decoder，不会送入 LLM。
- decoder 在推理时被完全删除，不能计入部署开销。

---

## 11. 可直接用于论文的英文表述

### 完整版本

> The reconstruction decoder is a single-layer, non-autoregressive Transformer cross-attention block with a hidden dimension of 128 and four attention heads. The compressed features \(C\) are first linearly projected to form the inputs to the key and value pathways. To reconstruct the feature at each original token position, we generate a fixed \(d\)-dimensional sinusoidal positional embedding and transform it into a 128-dimensional query using a learned linear projection. These positional queries distinguish the target token locations and determine the output order. Each query retrieves information from all compressed tokens through cross-attention, followed by a position-wise feed-forward network with a \(128\!\rightarrow\!512\!\rightarrow\!128\) structure and GELU activation. A final linear projection maps the decoded representations back to \(d\) dimensions, producing \(R\in\mathbb{R}^{N\times d}\). Both attention and feed-forward sublayers use pre-normalization and residual connections. The decoder is used only as an auxiliary training module and is discarded at inference.

### 与原文衔接的简短版本

> Specifically, the decoder first projects \(C\) into a 128-dimensional space to provide the key/value inputs. For each of the \(N\) original token positions, a fixed sinusoidal embedding is mapped to a 128-dimensional query through a learned linear projection. A four-head cross-attention layer allows each positional query to retrieve information from all compressed tokens. The result is processed by a residual feed-forward network and projected back to \(d\) dimensions to obtain \(R\). The positional queries specify which original token each output should reconstruct and preserve the original token order. Since sinusoidal embeddings are generated on the fly, the same decoder supports variable \(N\) without introducing position-dependent parameters.

### Reconstruction objective and overall training loss

> We define the reconstruction loss as the mean squared error normalized by the average energy of the target features:
> \[
> \mathcal{L}_{\mathrm{REC}}
> =\frac{\operatorname{MSE}(R,X)}
> {\max\left(\operatorname{Mean}(X^2),\epsilon\right)},
> \]
> where \(\epsilon=10^{-8}\) is used for numerical stability. The normalization is computed over all elements of the target feature tensor. We jointly optimize AnchorMerge and the reconstruction decoder using
> \[
> \mathcal{L}=\mathcal{L}_{\mathrm{CE}}
> +\lambda_{\mathrm{REC}}\mathcal{L}_{\mathrm{REC}},
> \]
> where \(\mathcal{L}_{\mathrm{CE}}\) is computed only over answer tokens and the end-of-sequence token, and \(\lambda_{\mathrm{REC}}=0.1\).

### 可选澄清句

> The sinusoidal embeddings encode one-dimensional indices in the original visual token sequence; they are used only as reconstruction queries and are independent of the positional encoding used by the LLM.

---

## 12. 对应代码骨架

```python
class VisualReconstructor(nn.Module):
    def __init__(self, d, hidden=128, n_src=576):
        self.in_proj = nn.Linear(d, hidden)
        self.pos_proj = nn.Linear(d, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.out = nn.Linear(hidden, d)

    def forward(self, c, n_src):
        kv = self.in_proj(c)
        pe = sinusoidal(n_src, d, c.device, c.dtype)
        q = self.pos_proj(pe).expand(batch_size, -1, -1)
        h = q + self.attn(self.ln1(q), kv, kv)[0]
        h = h + self.ffn(self.ln2(h))
        return self.out(h)
```

以上骨架用于解释计算流程；实际执行以 `multi/v2_model.py` 为准。
