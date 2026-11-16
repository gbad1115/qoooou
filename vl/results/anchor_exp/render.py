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
  textvqa_idx722_*.json  VisionSelector project viz jsons (Qwen2.5-VL, 37x37 grid,
                                       k=273, scores = student / rank / attention)
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
# sampled from the project's own colourbar
VS_RAMP = [(0.0, 0.267, 0.902), (0.0, 0.463, 0.663), (0.0, 0.659, 0.420),
           (0.514, 0.741, 0.204), (0.890, 0.729, 0.0)]


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
    d = json.load(open(os.path.join(HERE, f"textvqa_idx722_{method}.json")))
    gh, gw = d["grid"]
    pxs = d["patch_px"]
    rw, rh = d["resized_wh"]
    sel = np.array(d["selected"], float)
    idx, scores = sel[:, 0].astype(int), sel[:, 1]
    img = np.array(Image.open(os.path.join(HERE, "img_textvqa722.jpg")).convert("RGB")
                   .resize((rw, rh)))
    faded = np.clip(img * 0.18 + 255.0 * 0.82, 0, 255).astype(np.uint8)
    cmap = LinearSegmentedColormap.from_list("vs", VS_RAMP)
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    fig, ax = plt.subplots(figsize=(rw / 100, rh / 100), dpi=100)
    ax.imshow(faded)
    for t, s in zip(idx, scores):
        r, c = divmod(int(t), gw)
        ax.add_patch(Rectangle((c * pxs, r * pxs), pxs, pxs, facecolor=cmap(norm(s)),
                               edgecolor="white", linewidth=0.5, alpha=0.80, zorder=3))
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
