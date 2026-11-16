#!/usr/bin/env python3
"""Offline renderer for both visualization tracks. NO torch / transformers / GPU:
needs only numpy, matplotlib, Pillow plus the files in this directory.

    python3 render_offline.py mtgqa  [outdir]   4-column paper figure + 3 single panels
    python3 render_offline.py textvqa <fastv|divprune|ours> [outdir]
                                               VisionSelector-style panel (faded bg +
                                               patch squares coloured by score)

Data provenance (baked into this bundle, computed once on the GPU box):
  data_n100991.json   MT-GQA n100991 / dialogue mt_gqa_000012, LLaVA-1.5-7B, 24x24 grid:
                        fastv_idx      58 raster indices, official FastV (layer-2
                                       last-prompt-token attention, Q1-guided)
                        divprune_idx   58 raster indices, official DivPrune greedy
                                       max-min cosine-distance diversity
                        anchormerge    58 medoid anchor indices + token_weight[576]:
                                       per-token residual softmax weight w (raster
                                       order) used by the heat-map panel
  textvqa_idx722.json    merged VisionSelector viz json (Qwen2.5-VL, 37x37 grid,
                                       k=273): methods.{ours,fastv,divprune}.selected
                                       with scores (student / attention / rank)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, GRAY, BLACK = "#1a7f37", "#c62828", "#6b6b6b", "#111111"

# VisionSelector colour ramp: blue (low) -> teal -> green -> gold (high),
# 66 points sampled from the project's own colourbar (viz/..._selection.png)
CBAR66 = (
    (0.000,0.267,0.902), (0.000,0.278,0.890), (0.000,0.290,0.875), (0.000,0.302,0.859), (0.000,0.314,0.843), (0.000,0.329,0.827),
    (0.000,0.341,0.812), (0.000,0.353,0.796), (0.000,0.365,0.780), (0.000,0.376,0.769), (0.000,0.388,0.753), (0.000,0.400,0.741),
    (0.000,0.412,0.725), (0.000,0.424,0.710), (0.000,0.439,0.694), (0.000,0.451,0.678), (0.000,0.463,0.663), (0.000,0.475,0.647),
    (0.000,0.486,0.635), (0.000,0.498,0.620), (0.000,0.510,0.604), (0.000,0.522,0.588), (0.000,0.533,0.573), (0.000,0.549,0.557),
    (0.000,0.561,0.541), (0.000,0.573,0.525), (0.000,0.584,0.510), (0.000,0.596,0.498), (0.000,0.608,0.482), (0.000,0.620,0.467),
    (0.000,0.631,0.451), (0.000,0.643,0.435), (0.000,0.659,0.420), (0.027,0.663,0.408), (0.059,0.667,0.396), (0.090,0.675,0.380),
    (0.114,0.678,0.373), (0.145,0.682,0.357), (0.176,0.686,0.345), (0.208,0.694,0.333), (0.239,0.698,0.318), (0.271,0.702,0.306),
    (0.302,0.706,0.294), (0.333,0.714,0.278), (0.365,0.718,0.267), (0.388,0.722,0.255), (0.420,0.725,0.243), (0.451,0.729,0.231),
    (0.482,0.737,0.216), (0.514,0.741,0.204), (0.545,0.745,0.192), (0.576,0.753,0.176), (0.608,0.757,0.165), (0.631,0.761,0.153),
    (0.663,0.765,0.141), (0.694,0.769,0.129), (0.725,0.776,0.114), (0.757,0.780,0.102), (0.788,0.784,0.090), (0.820,0.792,0.075),
    (0.851,0.796,0.063), (0.882,0.800,0.051), (0.906,0.804,0.039), (0.937,0.808,0.027), (0.969,0.816,0.012), (0.890,0.729,0.000),
)
SQUARE_ALPHA = 0.52     # measured from the project's bare png (white bg .475 / coloured .552)


def load_img(name, px=None):
    im = Image.open(os.path.join(HERE, name)).convert("RGB")
    return im if px is None else im.resize((px, px))


def stars(ax, idx, grid, px, s=95, face="#00e5ff", edge=RED, lw=0.9, hollow=False):
    idx = np.asarray(idx)
    r, c = idx // grid, idx % grid
    kw = dict(marker="*", s=s, linewidth=lw, zorder=5)
    if hollow:
        kw.update(facecolor="none", edgecolor="white")
    else:
        kw.update(facecolor=face, edgecolor=edge)
    ax.scatter((c + 0.5) * px / grid, (r + 0.5) * px / grid, **kw)


def heat_overlay(disp, w2d, grid, px, mix=0.55):
    wn = np.asarray(w2d, float)
    wn = (wn - wn.min()) / (wn.max() - wn.min() + 1e-9)
    rgb = plt.get_cmap("jet")(wn)[..., :3]
    up = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((px, px),
                                                                      Image.BILINEAR))
    return np.clip(up * mix + disp * mix, 0, 255).astype(np.uint8)


def put_lines(ax, lines, width=50, dy=0.078):
    import textwrap
    y = 0.99
    for text, color, size, weight in lines:
        for ln in (textwrap.wrap(text, width) or [""]):
            ax.text(0.02, y, ln, transform=ax.transAxes, va="top", ha="left",
                    fontsize=size, color=color, fontweight=weight, linespacing=1.25)
            y -= dy * (0.82 if size < 8 else 1.0)
        y -= 0.012


# ------------------------------------------------------------------ MT-GQA track
def mtgqa(outdir):
    d = json.load(open(os.path.join(HERE, "data_n100991.json")))
    g, px = d["grid"], d["view_px"]
    disp = np.array(load_img(d["image_file"], px))
    W = np.array(d["anchormerge"]["token_weight"]).reshape(g, g)
    am = d["anchormerge"]

    os.makedirs(outdir, exist_ok=True)
    # single panels
    for name, draw in (("panel_fastv", lambda ax: stars(ax, d["fastv_idx"], g, px)),
                       ("panel_divprune", lambda ax: stars(ax, d["divprune_idx"], g, px)),
                       ("panel_anchor", None)):
        if name == "panel_anchor":
            continue
        fig = plt.figure(figsize=(4.2, 4.2))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(disp); draw(ax)
        fig.savefig(f"{outdir}/{name}.png", dpi=150, facecolor="white"); plt.close(fig)
    fig = plt.figure(figsize=(4.9, 4.2))
    ax = fig.add_axes([0, 0, 0.855, 1]); ax.axis("off")
    ax.imshow(heat_overlay(disp, W, g, px))
    stars(ax, am["anchor_idx"], g, px)      # cyan fill + red edge, like the others
    cax = fig.add_axes([0.885, 0.06, 0.035, 0.88])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="jet",
                          norm=Normalize(vmin=W.min(), vmax=W.max())), cax=cax)
    cb.set_label("residual merge weight  w", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(f"{outdir}/panel_anchor.png", dpi=150, facecolor="white"); plt.close(fig)

    # 4-column paper figure
    FW = 17.2
    w, gap, x0 = 0.2435, 0.0045, 0.008
    col_in = w * FW
    text_in, pad_in = 1.55, 0.30
    FH = col_in + text_in + pad_in
    h_img = col_in / FH
    y_img = 1.0 - pad_in / 2 / FH - h_img
    h_txt, y_txt = text_in / FH, pad_in / 2 / FH
    fig = plt.figure(figsize=(FW, FH))
    panels = [disp, disp, disp, heat_overlay(disp, W, g, px)]
    marks = [None, d["fastv_idx"], d["divprune_idx"], am["anchor_idx"]]
    for i in range(4):
        x = x0 + i * (w + gap)
        ax = fig.add_axes([x, y_img, w, h_img]); ax.axis("off")
        ax.imshow(panels[i])
        if marks[i] is not None:
            stars(ax, marks[i], g, px)
        tx = fig.add_axes([x, y_txt, w, h_txt]); tx.axis("off")
        if i == 0:
            lines = []
            for t, (q, ref) in enumerate(zip(d["questions"], d["references"]), 1):
                lines += [(f"Q{t}: {q}", BLACK, 8.2, "normal"),
                          (f"      ref: {ref}", GRAY, 7.0, "normal")]
            put_lines(tx, lines)
        else:
            heads = {1: "FastV", 2: "DivPrune", 3: "AnchorMerge (ours)"}
            preds, ok = {1: (d.get("fastv_pred"), None), 2: (d.get("divprune_pred"), None),
                         3: (d["anchormerge_pred"], d["anchormerge_correct"])}[i]
            head = f"{heads[i]}  @58 (10%)"
            if preds and ok:
                head += f"   {sum(ok)}/3 correct"
            lines = [(head, BLACK, 9.2, "bold")]
            for t in range(3):
                if preds and ok:
                    good = ok[t]
                    lines.append((f"Q{t+1}: {preds[t]}{' ✓' if good else ' ✗'}",
                                  GREEN if good else RED, 8.2, "normal"))
                else:
                    lines.append((f"Q{t+1}: —", GRAY, 8.2, "normal"))
            put_lines(tx, lines)
    fig.savefig(f"{outdir}/paper_fig.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", outdir, ": panel_fastv/divprune/anchor.png + paper_fig.png")


# ---------------------------------------------------------------- TextVQA track
def textvqa(method, outdir):
    # single merged json: shared metadata + per-method selected/scores
    d = json.load(open(os.path.join(HERE, "textvqa_idx722.json")))
    gh, gw = d["grid"]
    pxs = d["patch_px"]
    rw, rh = d["resized_wh"]
    m = d["methods"][method]
    sel = np.array(m["selected"], float)
    idx, scores = sel[:, 0].astype(int), sel[:, 1]
    # background stays the ORIGINAL image, untouched (matches the project's bare pdf);
    # only the kept-token squares are blended on top
    img = np.array(Image.open(os.path.join(HERE, d["image_file"])).convert("RGB")
                   .resize((rw, rh)))
    cmap = LinearSegmentedColormap.from_list("vs", CBAR66)
    # the project colours by RANK over kept tokens (its colourbar label says so);
    # raw scores are far too skewed for FastV attention and would collapse to blue
    ranks = np.argsort(np.argsort(scores))
    tvals = ranks / max(len(scores) - 1, 1)
    fig, ax = plt.subplots(figsize=(rw / 100, rh / 100), dpi=100)
    ax.imshow(img)
    for t, tv in zip(idx, tvals):
        r, c = divmod(int(t), gw)
        ax.add_patch(Rectangle((c * pxs, r * pxs), pxs, pxs, facecolor=cmap(tv),
                               edgecolor="white", linewidth=0.5, alpha=SQUARE_ALPHA,
                               zorder=3))
    ax.set_xlim(0, rw); ax.set_ylim(rh, 0); ax.axis("off")
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/vsstyle_{method}_722.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mtgqa"
    if mode == "mtgqa":
        mtgqa(sys.argv[2] if len(sys.argv) > 2 else ".")
    else:
        textvqa(sys.argv[2] if len(sys.argv) > 2 else "ours",
                sys.argv[3] if len(sys.argv) > 3 else ".")
