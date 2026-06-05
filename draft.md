## BAR-RAG Experiment Summary and Method Draft

### 1. Current motivation

The current project studies whether retrieval-augmented generation can be improved by explicitly modeling the interaction between evidence selection and answer generation. The core observation is that a generator can be sensitive not only to which documents are selected, but also to the order in which the selected documents are presented.

A key empirical motivation is the selector-order probe: after fixing a set of selected evidence documents, we compare the model behavior under forward, reverse, and random document orders. Across existing 1.7B, 4B, and 8B experiments, reversing the selector-ranked evidence order usually causes a larger harmful flip rate than random permutations. This suggests that reverse order is a strong and structured counterfactual perturbation rather than a purely noisy shuffle.

### 2. High-level method

Our method can be described as a two-stage reinforcement learning framework for retrieval-augmented QA.

#### Stage 1: selector optimization

Given a question and a candidate document pool, the selector is trained to choose a compact set of useful evidence documents. The selector is optimized with GRPO-style reinforcement learning. Its reward is computed through downstream answer generation: selected documents are passed to a generator, and the generated answer is evaluated against the reference answer using accuracy-oriented signals such as exact match, F1, citation validity, and format constraints.

The goal of this stage is to train a selector that retrieves evidence not only by lexical relevance, but by its utility for downstream generation.

#### Stage 2: generator optimization

After selector training, the trained selector is used to construct generator training data. For each question, selected documents are converted into generator prompts.

We compare two generator data construction modes:

- **Ordered setting**: keep the selector-produced evidence order and train the generator on this single order.
- **Unordered / order-robust setting**: construct paired document-order variants, mainly forward and reverse orders, and use group-aware GRPO training to encourage the generator to be robust to structured order perturbations.

The unordered setting is motivated by the order-probe finding that reverse order often represents a harder counterfactual than random shuffling.

### 3. Existing experiment groups

The current experiments are based on the `full_2x4_10k` data pipeline. Although the directory name contains `10k`, the filtered training subset currently used for the main experiments contains about **1078** examples.

#### Main training variants

| Variant | Description | Current status |
|---|---|---|
| `base` | Generator training baseline without the ordered/unordered comparison focus | Completed for several smaller scales |
| `ordered` | Two-stage training with selector order preserved for generator training | Completed for 1.7B / 4B / 8B style comparisons |
| `unordered` | Two-stage training with forward/reverse document-order augmentation and group UID | Completed for smaller scales; 32B currently being tested |

#### Model scales considered

| Scale | Purpose |
|---:|---|
| `1.7B` | Fast validation of the pipeline and order sensitivity |
| `4B` | Mid-scale sanity check; some path naming is historically irregular due to `full_2*4` directory |
| `8B` | Cleaner mid-scale comparison; order-probe results are relatively consistent |
| `32B` | Current main validation target for stronger-model behavior |

### 4. Selector-order probe results

The order-probe evaluates whether changing only the evidence order can flip model answers. Each model group was evaluated on 100 questions with 5 views per question: forward, reverse, and three random orders.

| Scale | Model | Forward EM | Reverse EM | Reverse harmful flip | Random harmful flip avg | Observation |
|---|---|---:|---:|---:|---:|---|
| 1.7B | base | 31 | 23 | 38.71% | 31.18% | reverse is strongest |
| 1.7B | ordered | 51 | 39 | 27.45% | 22.88% | reverse is strongest |
| 1.7B | unordered | 52 | 43 | 23.08% | 19.23% | reverse is strongest |
| 4B | base | 66 | 56 | 31.82% | 27.78% | reverse is strong; one random view is slightly worse |
| 4B | ordered | 68 | 59 | 27.94% | 24.02% | reverse is strongest |
| 4B | unordered | 67 | 61 | 26.87% | 24.38% | reverse is comparable to strongest random |
| 8B | base | 54 | 55 | 24.07% | 15.43% | reverse is clearly stronger than random average |
| 8B | ordered | 70 | 60 | 21.43% | 13.81% | reverse is clearly stronger than random average |
| 8B | unordered | 69 | 60 | 21.74% | 13.52% | reverse is clearly stronger than random average |
| 32B | unordered | 70 | 66 | 15.71% | 13.81% | reverse drop is smaller than smaller unordered models; random01 remains a hard perturbation |

Preliminary conclusion: reverse order is not always strictly worse than every random permutation, but it consistently has higher harmful flip rate than the average random perturbation. This supports using forward/reverse pairs as structured counterfactuals in generator-stage training.

### 5. Current 32B experiment

A Qwen3-32B unordered two-stage pipeline has been prepared under:

```text
scripts/full_2x4_32b/
```

Main entry:

```bash
bash scripts/full_2x4_32b/run_unordered_32b_all.sh
```

The 32B setup reuses the existing filtered data and does not rerun the filtering step. The intended hardware layout is:

- **Selector training stage**: GPUs `0-3` for selector training, GPUs `4-7` for reward-generator vLLM.
- **Selector-to-generator data construction**: GPUs `0-7` for selector inference.
- **Generator training stage**: GPUs `0-7` for 32B generator GRPO.

At the time of this draft, the 32B unordered two-stage run and selector-order probe evaluation have completed. Key training and evaluation facts:

- **Training pipeline**: completed end-to-end through generator checkpoint merge.
- **Final generator checkpoint**: `checkpoints/full_2x4_10k_unordered_32b/generator/global_step_269/actor/huggingface`.
- **Order-probe evaluation**: 100 questions × 5 views = 500 examples.
- **Evaluation output**: `scripts/full_2x4_1p7b/order_probe/results/unordered_32b/order_probe_metrics.txt`.
- **Main metrics**: forward EM/F1 = 70.00/76.42; reverse EM/F1 = 66.00/74.82; reverse harmful flip = 11/70 = 15.71%; reverse drop = 4.00 EM.
- **Random perturbations**: random01 EM/F1 = 60.00/70.83, random02 = 67.00/76.20, random03 = 63.00/71.68; average random harmful flip = 13.81%.

Compared with 1.7B / 4B / 8B unordered models, the 32B unordered model has the highest forward EM among unordered runs and the smallest reverse EM drop. The reverse harmful flip rate is also lower than the smaller unordered models, although one random view (`random01`) remains substantially worse in EM.

### 6. Draft paper-style method description

We propose a two-stage reinforcement learning framework for order-robust retrieval-augmented generation. In the first stage, a selector policy receives a question and candidate evidence documents, then selects a subset of documents. The selector is optimized with downstream generation rewards, so that evidence selection is aligned with answer correctness rather than isolated retrieval relevance.

In the second stage, we train the generator using documents produced by the optimized selector. Instead of treating the selector order as a fixed context layout, we construct structured counterfactual evidence orders. In particular, we use forward and reverse document orders to form grouped training instances. The generator is then optimized with group-aware GRPO, encouraging answer correctness and citation consistency under evidence-order perturbations.

This design is motivated by empirical evidence that reversing the selector-ranked document order induces more harmful answer flips than random shuffling on average. Therefore, reverse order provides a strong but controlled perturbation for improving generator robustness.

### 7. Current comparison story

The current narrative can be framed as follows:

1. **Order sensitivity exists**: changing only document order can change answer correctness.
2. **Reverse order is a strong structured perturbation**: reverse generally has higher harmful flip rate than random average.
3. **Training with structured order counterfactuals may improve robustness**: unordered forward/reverse training is designed to reduce harmful reliance on presentation order.
4. **Scaling test suggests improved robustness**: the 32B unordered result keeps high forward accuracy while reducing reverse harmful flips and reverse EM drop relative to smaller unordered models.

### 8. Missing pieces before a paper-ready version

- Add final 32B results and runtime.
- Add standard QA metrics for base / ordered / unordered, not only order-probe metrics.
- Clarify whether the main claim is robustness, accuracy, citation quality, or all three.
- Add formal reward definitions for selector and generator stages.
- Add ablations:
  - forward only vs forward/reverse;
  - reverse vs random augmentation;
  - with vs without group UID;
  - different model scales.
- Clean up historical path naming, especially the `full_2*4` 4B directory.
