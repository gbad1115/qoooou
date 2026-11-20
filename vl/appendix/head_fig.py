#!/usr/bin/env python
"""fig_head.py -- retrieval-head figures (qrad_method.tex sec:heads / Appendix app:2)
for three backbones: Qwen2.5-VL-7B (layer 20), LLaVA-1.5-7B (layer 26),
Qwen3-VL-8B (layer 34). Self-contained: all numbers are hard-coded below, nothing is
read from disk.

Two figures per model, one per metric, each a dual-axis combo like obs_heads_plot.py:
  bars (left axis)  = alpha_h   or   rho_h      for all H heads of the chosen layer
  line (right axis) = per-head token-selection accuracy (top-20% budget)

Head selection rule (obs_heads.py:387-390), reproduced exactly at run time:
  band       = the H//4 heads with the largest alpha_h   (top 25%)
  h_alpha    = argmax alpha_h (unconstrained)
  h_rho_band = argmax rho_h within band   (the head the method actually uses)
  h_rho      = argmax rho_h unconstrained; when it falls OUTSIDE the band it is drawn
               with its own hatch to show it lost to the banded pick.
               (qwen25 head 0, qwen3 head 6: outside. llava head 23: already inside,
               so h_rho == h_rho_band there and no rejected marker is drawn.)
Heads outside the band are drawn hatched in the rho figure so the reader can see at a
glance which heads were ineligible for the rho-band pick.

Provenance of the hard-coded numbers (verbatim copies, ACC rounded to 6 decimals):
  qwen25: visionselector_artifacts/obs_qwen25/heads/heads.json
  llava : visionselector_artifacts/obs_llava/heads/heads.json
  qwen3 : visionselector_artifacts/obs_qwen3/heads/heads.json
each with n_samples=600 (300 OCRVQA + 300 TextVQA), strict_q_fallbacks=0. Accuracy =
per-head top-20% selection over those 600 rows, OCRVQA normalized EM plus TextVQA
VQA-acc binarized at 0.5 (obs_heads.py correct()) -- identical metric on all three
backbones, so the accuracy lines are comparable across models.

Usage:  fig_head.py [model|all] [out_dir]      (model in qwen25|llava|qwen3;
        default all, default out_dir = current directory)
Writes fig_heads_<m>_banded.{pdf,png} and fig_heads_rho_<m>_banded.{pdf,png}
"""
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'font.size': 13, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

# === DATA BEGIN === (generated from heads.json; edit via build, not by hand)
MODELS = {
    "qwen25": {
        "label": "Qwen2.5-VL-7B", "layer": 20, "n_samples": 600,
        "ALPHA": [
            0.12844045856967568, 0.026761309377810297, 0.06164279628389825,
            0.046149841338434874, 0.13497540945963313, 0.25501614600856554,
            0.16958391134316722, 0.14736441529045502, 0.45123245571856385,
            0.17912504756939598, 0.36942333715346953, 0.07470364501466975,
            0.08358916682346414, 0.03460468829104987, 0.2933942689994971,
            0.20467571506897608, 0.1819301365308153, 0.40577905611474613,
            0.04677144865163086, 0.27075579862420757, 0.05887709475743274,
            0.5576063798740506, 0.1533898997005114, 0.2571567538442711,
            0.31508638101940356, 0.17105241197627039, 0.49444003274664283,
            0.45504602395308513
        ],
        "RHO": [
            7.177777905528347, 2.9556863337596795, 1.5620512119217607,
            3.0130075480534275, 3.180721106537332, 4.673331363094874, 2.3325727881587337,
            5.771337080279677, 6.840788100394626, 6.121764694736191, 3.578508902620478,
            5.498973475389165, 3.3441321631944096, 1.754427461283451, 3.1085596571062375,
            1.8705260258563088, 2.3965129415807103, 1.7754016762015417,
            1.3001055532396897, 1.852642826750319, 1.1611826588893421,
            2.4017750951524386, 4.83485521644276, 1.9716330024471647, 2.581274044913791,
            2.8625861866404714, 3.5242295896543423, 3.6397698922067403
        ],
        "ACC": [
            69.5, 73.5, 81.666667, 80.166667, 72.166667, 81.666667, 82.666667, 85.5,
            80.666667, 80.333333, 83.333333, 77.166667, 81.5, 82.833333, 81.666667,
            82.666667, 84.0, 82.333333, 82.333333, 83.833333, 80.5, 84.5, 82.166667,
            83.333333, 81.5, 67.666667, 80.0, 82.833333
        ],
        "PAIR_ACC": {'pair': 84.33333333333334, 'pair:21,8': 84.5},
    },
    "llava": {
        "label": "LLaVA-1.5-7B", "layer": 26, "n_samples": 600,
        "ALPHA": [
            0.053859634020676214, 0.06037777737248689, 0.049622391311762236,
            0.21140798746918638, 0.08405565314227716, 0.03265350313081095,
            0.036380747040578476, 0.05018881404772401, 0.060364394029602406,
            0.04609719636694839, 0.01912966757779941, 0.0474622088096415,
            0.01174704710933535, 0.016713986843824385, 0.26568590100854633,
            0.32892904974520204, 0.06283474258612841, 0.030317876847305646,
            0.03316241238731891, 0.3984765101100008, 0.013003335162842026,
            0.4923628777700166, 0.014599299265537411, 0.27742761068046096,
            0.2862713957997039, 0.2395756526229282, 0.29786409783487516,
            0.47586602697769803, 0.25391242862679064, 0.023504979806796958,
            0.08696691864092523, 0.03463603003571431
        ],
        "RHO": [
            0.8084374751931329, 1.5019899365330596, 1.2242615469437312,
            3.0623183445224917, 1.772989635585465, 1.6112577338858167,
            1.7674735365259648, 1.0835189911681788, 1.8704378796173513,
            1.5284709693072716, 2.274982676738877, 1.4667804716577724,
            0.5280339123632041, 1.1577713998843122, 1.6183333015454466, 1.92714056031738,
            2.3560271739807774, 0.8633794134543875, 1.4679695015862462,
            1.6342788260967234, 1.1473688934068615, 2.1804490145805353,
            0.9298900810538452, 3.131181261785406, 1.8666508994661126,
            1.7187236922030393, 2.3505034151720388, 2.3225500083823176,
            2.214168478676091, 0.89992920235858, 1.9520745835561042, 1.0344510504900082
        ],
        "ACC": [
            42.0, 48.333333, 50.666667, 49.5, 40.833333, 43.0, 40.333333, 39.666667,
            46.666667, 43.666667, 44.5, 43.333333, 39.0, 40.333333, 49.0, 48.666667,
            48.833333, 46.333333, 45.166667, 41.833333, 41.0, 49.666667, 39.666667,
            49.833333, 50.333333, 49.833333, 50.166667, 49.833333, 47.5, 46.333333,
            47.333333, 41.333333
        ],
        "PAIR_ACC": {'pair': 50.0, 'pair:21,23': 50.0},
    },
    "qwen3": {
        "label": "Qwen3-VL-8B", "layer": 34, "n_samples": 600,
        "ALPHA": [
            0.2665892144075284, 0.23853090423857792, 0.2321712764845385,
            0.20924554800847545, 0.07516620535015439, 0.3220901506983985,
            0.08098575786687434, 0.33473933723134297, 0.028041159649146722,
            0.013658971620413164, 0.0038495267793526486, 0.0186677213315852,
            0.025618475522690764, 0.0053219682153818814, 0.0559157852421049,
            0.0790097929428642, 0.08781876460338632, 0.09599374097927163,
            0.0027078837064133648, 0.21043384121730924, 0.027993018488244464,
            0.22616453228518366, 0.005364254633071444, 0.0030805237570408887,
            0.006155784431769765, 0.0030223816107657815, 0.003107555344225451,
            0.018212900077730108, 0.27024471273645756, 0.15445775766236086,
            0.16489464145153762, 0.09587443678678634
        ],
        "RHO": [
            1.9859437793258152, 2.500059558528322, 2.0910378358900474,
            2.6523975541204856, 2.856985750831577, 2.727648760239531, 2.9972678064676708,
            2.343887474234754, 1.2085924685270983, 1.4396150587142094,
            0.5118161961730539, 1.4657534729058117, 0.853825333466912,
            0.15551976305104528, 1.3605700263657599, 0.7434663057478421,
            1.2835570318403158, 1.4358002360678814, 0.3463562005194768,
            1.4159683518408108, 1.9219372703346935, 1.1613737398878126,
            0.3272754417532832, 0.2906454060519123, 0.6278852060695805,
            0.17047719980598564, 0.3228896729058478, 2.4155065900124004,
            1.3408451187063897, 1.6379946875488771, 1.3110053079579185,
            1.3467496257973133
        ],
        "ACC": [
            69.166667, 70.666667, 69.333333, 70.333333, 67.666667, 71.166667, 65.5, 71.5,
            57.333333, 57.166667, 56.333333, 67.166667, 52.833333, 51.666667, 60.833333,
            59.833333, 70.333333, 65.666667, 61.666667, 68.666667, 51.166667, 69.5,
            50.166667, 55.666667, 70.666667, 72.833333, 60.5, 68.166667, 68.833333,
            71.666667, 68.166667, 69.833333
        ],
        "PAIR_ACC": {'pair': 72.83333333333334, 'pair:7,5': 71.83333333333334},
    },
}
# === DATA END ===

# --- palette (shared with obs_heads_plot.py) ----------------------------------
BAR, BAR_HI, LINE = "#9eb9d9", "#375b89", "#d1495b"   # blues follow figure03
INEL_FILL, INEL_EDGE = "#e9eef6", "#8d97a3"
HATCH = "///"
ARG_FILL, ARG_EDGE, ARG_HATCH = "#f7e4e6", "#b23a48", "xxx"
GRID, SPINE, INK, MUTED = "#e4e4e2", "#9a9a96", "#111111", "#5c5c58"
SYM = {"alpha": (r"$\alpha_h$", "Visual attention mass"),
       "rho": (r"$\rho_h$", "Generation-to-question ratio")}


def spearman(x, y):
    def ranks(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v))
        return r
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def select(alpha, rho):
    """The obs_heads.py:387-390 rule, verbatim."""
    H = len(alpha)
    band = sorted(range(H), key=lambda h: -alpha[h])[:H // 4]      # alpha top-25%
    in_band = np.zeros(H, bool)
    in_band[band] = True
    h_alpha = int(np.argmax(alpha))
    h_rho = int(np.argmax(rho))
    h_rho_band = max(band, key=lambda h: rho[h])
    return band, in_band, h_alpha, h_rho, h_rho_band


def draw(S, metric, out_stem):
    """argmax-of-rho marker is drawn only when the unconstrained argmax lost to the
    banded pick (h_rho != h_rho_band)."""
    alpha, rho, acc = S["alpha"], S["rho"], S["acc"]
    H, in_band = S["H"], S["in_band"]
    h_alpha, h_rho, h_rho_band = S["h_alpha"], S["h_rho"], S["h_rho_band"]
    argmax_marker = h_rho if (metric == "rho" and h_rho != h_rho_band) else None

    vals = alpha if metric == "alpha" else rho
    sym, name = SYM[metric]
    sel = h_alpha if metric == "alpha" else h_rho_band

    fig, ax = plt.subplots(figsize=(7.0, 3.1), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(1, H + 1)                           # heads numbered 1..H

    for h in range(H):
        if argmax_marker is not None and h == argmax_marker:
            bar = ax.bar(h + 1, vals[h], width=0.72, color=ARG_FILL, edgecolor=ARG_EDGE,
                         hatch=ARG_HATCH, linewidth=0.7, zorder=2)
        elif metric == "rho" and not in_band[h]:
            bar = ax.bar(h + 1, vals[h], width=0.72, color=INEL_FILL, edgecolor=INEL_EDGE,
                         hatch=HATCH, linewidth=0.6, zorder=2)
        else:
            bar = ax.bar(h + 1, vals[h], width=0.72, color=BAR, edgecolor="none",
                         linewidth=0.0, zorder=2)
        if h == sel:                                  # the head we actually use
            rect = bar[0]
            rect.set_color(BAR_HI)
            rect.set_edgecolor("#0d366b")
            rect.set_hatch(None)
            rect.set_linewidth(0.9)

    headroom = 1.30 if metric == "alpha" else 1.45
    ax.set_ylim(0, float(vals.max()) * headroom)
    ax.set_ylabel(f"{name} {sym}", fontsize=9, color=INK)
    ax.set_xlabel(f"Head index (all {H} heads)", fontsize=9, color=INK)
    ax.set_xticks(x[::2])
    ax.tick_params(labelsize=8, colors=INK, length=2.4, width=0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(x, acc, color=LINE, linewidth=1.1, marker="o", markersize=3.0,
             markerfacecolor=LINE, markeredgecolor="white", markeredgewidth=0.4,
             zorder=4)
    alo, ahi = acc.min(), acc.max()
    pad = (ahi - alo) * 0.30
    ax2.set_ylim(alo - pad, ahi + pad)
    ax2.set_ylabel("Accuracy (%)", fontsize=9, color=INK)
    ax2.tick_params(labelsize=8, colors=INK, length=2.4, width=0.7)
    ax2.grid(False)

    for axx in (ax, ax2):
        for side, sp in axx.spines.items():
            sp.set_color(SPINE)
            sp.set_linewidth(0.7)
            sp.set_visible(side in ("left", "right", "bottom"))

    line_handle = plt.Line2D([], [], color=LINE, linewidth=1.1, marker="o", markersize=3.0,
                             markerfacecolor=LINE, markeredgecolor="white", label="Accuracy")
    if metric == "alpha":
        handles = [Patch(facecolor=BAR, edgecolor="none",
                         label=r"Visual attention mass $\alpha_h$"),
                   line_handle]
    else:
        handles = [Patch(facecolor=BAR, edgecolor="none",
                         label=r"$\rho_h$ of top 25% $\alpha_h$ heads"),
                   Patch(facecolor=INEL_FILL, edgecolor=INEL_EDGE, hatch=HATCH,
                         label=r"$\rho_h$ of the remaining heads")]
        if argmax_marker is not None:
            handles.append(Patch(facecolor=ARG_FILL, edgecolor=ARG_EDGE, hatch=ARG_HATCH,
                                 label=r"Head with max $\rho_h$, rejected"))
        handles.append(line_handle)
    ax.legend(handles=handles, fontsize=7.0, frameon=False, loc="upper left",
              handlelength=2.0, borderpad=0.25, labelspacing=0.28)

    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    print(f"wrote {out_stem}.pdf / .png   (metric={metric}, selected head {sel})")


def run(name, out_dir):
    M = MODELS[name]
    alpha = np.array(M["ALPHA"], float)
    rho = np.array(M["RHO"], float)
    acc = np.array(M["ACC"], float)
    band, in_band, h_alpha, h_rho, h_rho_band = select(alpha, rho)
    S = {"alpha": alpha, "rho": rho, "acc": acc, "H": len(alpha), "in_band": in_band,
         "h_alpha": h_alpha, "h_rho": h_rho, "h_rho_band": h_rho_band}

    print(f"== {M['label']}  layer {M['layer']}  n_samples {M['n_samples']}  "
          f"H {len(alpha)} ==")
    draw(S, "alpha", f"{out_dir}/fig_heads_{name}_banded")
    draw(S, "rho", f"{out_dir}/fig_heads_rho_{name}_banded")
    print(f"alpha top-25% band (eligible)      : {band}")
    print(f"h_alpha  = argmax alpha_h          : {h_alpha}  "
          f"(alpha {alpha[h_alpha]:.4f}, acc {acc[h_alpha]:.2f})")
    if h_rho == h_rho_band:
        print(f"h_rho    = argmax rho_h            : {h_rho}  "
              f"(rho {rho[h_rho]:.3f}, acc {acc[h_rho]:.2f}, inside band -> selected)")
    else:
        print(f"h_rho    = argmax rho_h            : {h_rho}  "
              f"(rho {rho[h_rho]:.3f}, acc {acc[h_rho]:.2f}, NOT eligible)")
    print(f"h_rho_band = argmax rho within band: {h_rho_band}  "
          f"(rho {rho[h_rho_band]:.3f}, alpha {alpha[h_rho_band]:.4f}, "
          f"acc {acc[h_rho_band]:.2f})")
    print(f"Spearman(alpha, acc) = {spearman(alpha, acc):+.4f}   "
          f"Spearman(rho, acc) = {spearman(rho, acc):+.4f}")
    print(f"teacher pair accuracies: {M['PAIR_ACC']}")
    print()


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("all",) and sys.argv[1] not in MODELS:
        sys.exit(f"unknown model {sys.argv[1]!r}; expected one of {sorted(MODELS)} or 'all'")
    names = list(MODELS) if len(sys.argv) < 2 or sys.argv[1] == "all" else [sys.argv[1]]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        run(name, out_dir)


if __name__ == "__main__":
    main()
