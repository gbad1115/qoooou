# Case Study：顺序翻转 harmful-flip 实例 (nq_294)
> 来源：OBS Top-5 全排列顺序敏感性诊断（base Llama-3.1-8B-Instruct, greedy, 1000 题×120 排列）。`results/perm_diag/permutation_level_results.csv`。
> 这是 reverse(逆序) harmful flip 的代表性样例：正序答对、逆序答错，且错答与 gold **同类可混淆**（都是该曲的 rap 嘉宾）。

## 元信息
- **question_id**: nq_294  | **dataset**: NQ
- **question**: who sings the rap in baby by justin bieber
- **gold answers**: ['Ludacris']
- **正序(forward, 原始检索序 Doc[1..5]) 答案**: `Ludacris`  ✓ EM=1
- **逆序(reverse, Doc[5..1]) 答案**: `Chipmunk`  ✗ EM=0  (harmful flip)
- **Kendall 距离**: reverse = 10（相对检索序最大位移，归一化 1.0）

---

## 原始 Top-5 文档（正序 Doc[1..5]，模型实际看到的全文）

### Doc [1]  — Title: Baby (Justin Bieber song)

```
episode, "OMG" (parody of TMZ), there is a clip of Justin Beberry (portrayal of Bieber) singing a parody version called, "Gravy". Baby (Justin Bieber song) "Baby" is a song by Canadian recording artist Justin Bieber. It was released as the lead single from the latter half of Bieber's debut album, "My World 2.0". The track was written by Bieber with Christopher "Tricky" Stewart and Terius "The-Dream" Nash, both of whom worked with Bieber on "One Time", and also by R&B singer Christina Milian and labelmate, rapper Ludacris. It was available for digital download on January 18, 2010. The song received
```

### Doc [2]  — Title: Baby (Justin Bieber song)

```
Baby (Justin Bieber song) "Baby" is a song by Canadian recording artist Justin Bieber. It was released as the lead single from the latter half of Bieber's debut album, "My World 2.0". The track was written by Bieber with Christopher "Tricky" Stewart and Terius "The-Dream" Nash, both of whom worked with Bieber on "One Time", and also by R&B singer Christina Milian and labelmate, rapper Ludacris. It was available for digital download on January 18, 2010. The song received airplay directly after release, officially impacting mainstream and rhythmic radio on January 26, 2010. The song is uptempo R&B, blending together
```

### Doc [3]  — Title: Baby (Justin Bieber song)

```
dance-pop and hip-hop elements, while using influences of doo-wop music. The song has received positive reviews from critics who complimented the song's effective lyrics and chorus, and commended Ludacris' part and the song's ability to have an urban twist. The song was a commercial success, peaking at number one in France and Scotland, and charting in the top ten of the United Kingdom, Canada, the United States, Australia, Norway, Japan, Ireland, Hungary, Belgium (Flanders), Slovakia and New Zealand. The accompanying music video takes place in a mall/bowling-alley setting. The video features several cameos such as Drake, Lil Twist and Tinashe,
```

### Doc [4]  — Title: Baby (Justin Bieber song)

```
also performed an acoustic rendition of the remix version with Drake at the 2010 Juno Awards. The official remix in the United Kingdom features British hip-hop artist Chipmunk. "Baby" was written by Bieber with Christopher "Tricky" Stewart and Terius "The-Dream" Nash, both of whom worked with Bieber on "One Time", R&B singer and The-Dream's then wife Christina Milian and labelmate, Ludacris. When asked how the collaboration initially came about, Bieber said, "[Ludacris and I] both live in Atlanta. I met him a year prior to this and we figured it was a perfect collaboration for him, so we invited him
```

### Doc [5]  — Title: Baby (Justin Bieber song)

```
an acoustic rendition of the song along with a freestyle by good friend Drake at the 2010 Juno Awards. In April 2010, Bieber noted that due to his voice changing, he could no longer hit all the notes in "Baby", and for live performances, the key is lowered. Bieber performed the song live in Sydney, Australia on April 26, 2010, in the "Sunrise" studios after his public performance was cancelled. Bieber also performed the song on May 11, 2010, on The Oprah Winfrey Show. Bieber performed the song on June 4, 2010, on the Today Show along with "Never Say
```

---

## 正序(forward, Doc[1..5]) 完整生成

```
<evidence>Doc [3] states that the song has a hip-hop element, and Doc [4] mentions Ludacris as a rapper and labelmate of Justin Bieber.</evidence>
<answer> Ludacris </answer>
```

**抽取答案**：`Ludacris` ✓

---

## 逆序(reverse, Doc[5→1]，即原 Doc[5],[4],[3],[2],[1]) 完整生成

```
<evidence>Doc [2] states that the official remix features British hip-hop artist Chipmunk, and Doc [3] mentions that the song has a hip-hop element.</evidence>
<answer> Chipmunk </answer>
```

**抽取答案**：`Chipmunk` ✗

---

## 翻转机制（顺序核对过）

- **gold `Ludacris` 出现在原 Doc[1][2][3][4]**——尤其 Doc[3]「commended **Ludacris' part**」、Doc[1]/[2]/[4] 的写作名单里也有「rapper Ludacris」。
- **错答 `Chipmunk` 只出现在原 Doc[4]**：「The official remix in the United Kingdom features British hip-hop artist **Chipmunk**」
- 正序下 gold 信息遍布前 4 篇、Ludacris 出现得早而密 → 模型引用 Doc[3]/[4] → 答 `Ludacris` ✓。
- 逆序 `[5,4,3,2,1]` 重编号后：原 Doc[4]（含 Chipmunk 那句、也含 Ludacris）变成 **逆序 Doc[2]**，排在原 Doc[3]（Ludacris part）= 逆序 Doc[3] **之前**。模型在逆序里先碰到 Chipmunk 句子、就地锚定 → 引用逆序-Doc[2] → 答 `Chipmunk` ✗。

## 代表性说明

- 错答 `Chipmunk` 与 gold `Ludacris` **同类**：都是 Justin Bieber《Baby》这首歌的 rap 嘉宾（Ludacris=原版 feature，Chipmunk=英国混音版 feature），构成可混淆的近邻实体。
- 翻转可由文档顺序解释（错答所在原 Doc[4] 在正序里靠后、逆序后前移到 gold 主证据之前），非随机噪声。
- 这正是 HCF（P(reverse错|forward对)）所指现象：在 275 个正序答对的题中，逆序使 63 个（22.9%）翻成答错；本例是其中机制清晰、错答同类的代表。

## 复现

- 数据：`data/dynamicrag/normalized/eval_1000/nq.jsonl` (id=nq_294)
- 模型：`/ainative/.../Llama-3.1-8B-Instruct` (base, greedy temp=0, chat template)
- prompt 模板：项目 prompt3（与 `qa_inference.py`/`order_stability_eval.py` 一致）
- greedy 确定性，复跑结果与本文件生成一致。
