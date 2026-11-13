# V10 results: new teacher (qwen25 L20 {21,8}; llava L26 {21,23}; qwen3 L34 {7,5}), loss = KL + 0.2 x truncated suffix ListMLE (j>=i, top-10%), 1 epoch, final-step last.pt, no val; nested subsets 30%/50%/100% of the 167,040-row mixture

## qwen25 @20% retention
| dataset | full cache | p30 | p50 | full | v8 (old teacher) |
|---|---:|---:|---:|---:|---:|
| chartqa | 83.84 | 51.32 | 51.76 | 51.76 | 68.52 |
| textvqa | 82.93 | 78.89 | 79.42 | 79.51 | 79.0 |
| docvqa | 94.60 | 84.65 | 85.67 | 86.05 | 71.87 |
| ocrbench | 840.00 | 605.00 | 619.00 | 608.00 | 425 |
| ai2d | 82.45 | 75.55 | 75.23 | 73.93 | 74.03 |
| pope | 87.72 | 78.73 | 81.97 | 85.43 | 83.07 |
| mme | 2318.53 | 2216.94 | 2244.06 | 2241.46 | 2119.01 |
| scienceqa | 87.98 | 85.31 | 83.60 | 84.12 | 81.35 |
| mmmu | 51.89 | 48.22 | 48.78 | 48.89 | 48.89 |
| realworldqa | 69.28 | 61.96 | 63.66 | 65.10 | 61.7 |

## llava @20% retention
| dataset | p30 | p50 | full | v8 (old teacher) | full-cache fp16 |
|---|---:|---:|---:|---:|---:|
| chartqa | 17.08 | 17.24 | 16.76 | 16.72 | 18.04 |
| textvqa | 45.91 | 45.84 | 46.39 | 45.85 | 47.36 |
| docvqa | 20.13 | 19.94 | 20.31 | 19.1 | 23.46 |
| ocrbench | 192.00 | 185.00 | 188.00 | 186 | 196 |
| ai2d | 51.33 | 51.26 | 51.36 | 51.68 | 52.82 |
| pope | 83.18 | 83.13 | 83.24 | 82.12 | 85.52 |
| mme | 1694.77 | 1678.73 | 1699.31 | 1719.88 | 1778.43 |
| scienceqa | 64.23 | 64.19 | 64.76 | 64.43 | 64.04 |
| mmmu | 34.15 | 34.03 | 34.15 | 33.68 | 34.03 |
| realworldqa | 51.90 | 52.55 | 53.20 | 50.85 | 55.56 |

## qwen3 (Qwen3-VL-8B, teacher L34 {7,5}) @20% retention
| dataset | full cache | p30 | p50 | full | v8 (old teacher, ckpt_v8/best.pt) |
|---|---:|---:|---:|---:|---:|
| chartqa | 84.48 | 63.92 | 63.36 | 63.80 | 67.08 |
| textvqa | 81.87 | 75.52 | 76.52 | 76.66 | 76.56 |
| docvqa | 94.88 | 67.85 | 69.11 | 69.35 | 69.46 |
| ocrbench | 861.00 | 430.00 | 413.00 | 433.00 | 483.00 |
| ai2d | 83.87 | 71.28 | 70.69 | 71.50 | 72.60 |
| pope | 88.96 | 84.63 | 85.33 | 86.33 | 79.29 |
| mme | 2416.57 | 1996.96 | 2050.07 | 2100.92 | 1928.79 |
| scienceqa | 94.04 | 84.02 | 84.22 | 84.98 | 85.46 |
| mmmu | 51.22 | 50.00 | 51.22 | 51.44 | 51.11 |
| realworldqa | 70.72 | 63.14 | 63.79 | 64.18 | 62.22 |
| **mean (8 pct datasets)** | **81.26** | **70.05** | **70.53** | **71.03** | **70.47** |

Notes (qwen3, run 2026-11-13 14:20-21:11):
* mme/ocrbench use raw units, excluded from the 8-dataset mean.
* Data volume scales monotonically for qwen3 too: p30 70.05 < p50 70.53 < full 71.03
  (same direction as qwen25 70.58/71.26/71.85 and llava 45.99/46.02/46.27).
* v10 full beats the old v8 scorer on the mean by +0.56, driven by pope (+7.04) and
  mme (+172); v8 stays ahead on ocrbench (-50), chartqa (-3.28), ai2d (-1.10).
* Protocol caveat: v8 column is ckpt_v8/**best.pt** (val-selected) while v10 is the
  user-specified final-step **last.pt** with no val -- not an apples-to-apples ckpt choice.
* full-cache column read from bench10/qwen3/orig/; v8 column from bench10/qwen3/distilled/.
* Extraction of the 167,040-row teacher mixture took 4h39m at the default 192 torch
  threads per shard, then 0.11-0.30 s/row after capping to 4 threads mid-run
  (coco+textvqa). Scores are bit-identical either way (verified max|diff| = 0.0 on 6 rows),
  so all four datasets' targets are the same measurement.
