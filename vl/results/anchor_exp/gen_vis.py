#!/usr/bin/env python3
"""Per-method panel generators for the 4-column paper figure (image n100991 by default).

    python3 viz_method_panels.py fastv     [img] -> /tmp/panel_fastv.png
    python3 viz_method_panels.py divprune  [img] -> /tmp/panel_divprune.png
    python3 viz_method_panels.py anchor    [img] -> /tmp/panel_anchor.png   (heat-map style)
    python3 viz_method_panels.py all       [img] -> all three

Each panel is a bare 336x336 model-view square with NO text, so columns can be
assembled in LaTeX or via render_paper_fig.py.

Selection protocols (identical to render_prune_compare.py):
  fastv    : official FastV - layer idx=2 attention of the LAST prompt token onto the
             576 image tokens (mean over heads), top-58; prompt carries Q1 of the
             dialogue, i.e. question-guided.
  divprune : official DivPrune - greedy max-min diversity on the pairwise cosine
             distance of post-projector tokens, k=58 (question-agnostic).
  anchor   : AnchorMerge - medoid anchor per raster segment + per-segment residual
             softmax weights w rendered as a jet heat overlay, anchors as white stars.

Results are cached in /tmp (anchviz_*.npz, prunecmp_*.npz); a cache miss triggers the
GPU computation (LLaVA-1.5-7B bf16, same feature path as eval_v2).
"""
from __future__ import annotations
import os, sys, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch
torch.set_num_threads(4)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

DEVICE = "cuda"
MODEL = "/home/admin/llava-1.5-7b-hf"
DR = "/ossfs/workspace/Rethinking Token Reduction for Large Vision-Language Models_dataset"
N1, GRID, PROC = 58, 24, 336
DIALOGUE = "mt_gqa_000012"          # supplies Q1 for FastV's question-guided selection

METHOD = sys.argv[1] if len(sys.argv) > 1 else "all"
IMG = sys.argv[2] if len(sys.argv) > 2 else "n100991"
assert METHOD in ("fastv", "divprune", "anchor", "all")


# ---------------------------------------------------------------- data / caches
def img_path_for(iid):
    for line in open(f"{DR}/processed/mt_gqa_testdev_balanced_seed42.jsonl"):
        d = json.loads(line)
        if d["image_id"] == iid:
            return f"{DR}/{d['image_path']}"
    raise KeyError(iid)


def q1_of_dialogue():
    for f in sorted(glob.glob("/ossfs/workspace/multi/outputs/mt_gqa/"
                              "v2f30_1ep_best_n58_fused_short_answer_full_shard*.json")):
        for r in json.load(open(f))["rows"]:
            if r["dialogue_id"] == DIALOGUE:
                return r["questions"][0]
    raise KeyError(DIALOGUE)


def pairwise_cosine_distance(v):
    z = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return 1.0 - z @ z.t()


def divprune_select(v, k):
    """Verbatim port of VisionSelector-main/.../divprune.py (greedy max-min diversity)."""
    d = pairwise_cosine_distance(v)
    s = torch.empty(k, dtype=torch.long, device=v.device)
    for i in range(k):
        m2 = d if i == 0 else torch.index_select(d, 0, s[:i])
        scores = (torch.topk(m2, 2, dim=0, largest=False).values[1, :] if i == 0
                  else torch.min(m2, dim=0).values)
        s[i] = torch.argmax(scores)
    return s.sort().values.cpu().numpy()


def features_and_all(img_path):
    """One model load fills every missing cache entry."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from v2_model import AnchorMergeV2
    image = Image.open(img_path).convert("RGB")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    with torch.no_grad():
        pv = processor.image_processor(images=[image], return_tensors="pt")["pixel_values"]
        layer = int(getattr(model.config, "vision_feature_layer", -2))
        out = model.vision_tower(pv.to(DEVICE, torch.bfloat16), output_hidden_states=True)
        y = model.multi_modal_projector(out.hidden_states[layer][:, 1:].to(torch.bfloat16)).float()

        # FastV
        prompt = "USER: <image>\n" + q1_of_dialogue() + " ASSISTANT:"
        ids = processor.tokenizer(text=prompt, return_tensors="pt").input_ids.to(DEVICE)
        itid = int(model.config.image_token_index)
        at = int((ids[0] == itid).nonzero().flatten()[0])
        ids = torch.cat([ids[:, :at], ids.new_full((1, y.shape[1]), itid), ids[:, at + 1:]], 1)
        emb = model.get_input_embeddings()(ids)
        emb[0, at:at + y.shape[1]] = y[0].to(emb.dtype)
        avg = model(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                    output_attentions=True).attentions[2].mean(1)[0]
        fastv = avg[-1, at:at + y.shape[1]].topk(N1).indices.sort().values.cpu().numpy()

        # DivPrune
        div = divprune_select(y[0], N1)

        # AnchorMerge anchors + per-segment residual weights
        ck = torch.load("/ossfs/workspace/multi/runs/v2f30_1ep/ckpt/best.pt",
                        map_location="cpu", weights_only=False)
        cfg = ck.get("compressor_config") or {}
        merger = AnchorMergeV2(d=int(cfg.get("d", 4096)), tau=float(cfg.get("tau", 1.0)),
                               gate_scale=float(cfg.get("gate_scale", 1.0)),
                               write_cap=float(cfg.get("write_cap", 0.0)))
        merger.load_state_dict(ck["compressor"]); merger = merger.to(DEVICE).eval()
        edges = AnchorMergeV2.edges(y.shape[1], N1, y.device)
        anchor = np.zeros(N1, dtype=int)
        W = np.zeros((GRID, GRID))
        for k in range(N1):
            lo, hi = int(edges[k]), int(edges[k + 1])
            seg = y[:, lo:hi]
            ai = AnchorMergeV2._medoid(seg)
            a = torch.gather(seg, 1, ai.view(1, 1, -1).expand(-1, -1, y.shape[-1]))[:, 0]
            r = seg - a.unsqueeze(1)
            sim = torch.nn.functional.cosine_similarity(seg, a.unsqueeze(1), dim=-1)
            w = torch.softmax((merger.res_score(r).squeeze(-1) + sim) / merger.tau, dim=1)[0]
            anchor[k] = lo + int(ai.item())
            for j in range(hi - lo):
                W[(lo + j) // GRID, (lo + j) % GRID] = w[j].item()
    del model, merger
    torch.cuda.empty_cache()
    np.savez(f"/tmp/prunecmp_{IMG}.npz", fastv=fastv, div=div, anchor=anchor,
             img_path=img_path, q1=q1_of_dialogue())
    np.savez(f"/tmp/anchviz_{IMG}.npz", anchor_idx=anchor, W=W,
             seg_of=np.zeros(1), img_path=img_path)
    return fastv, div, anchor, W


def load_all():
    """(fastv, div, anchor, W, img_path): read both caches when present, else recompute."""
    pa, pc = f"/tmp/anchviz_{IMG}.npz", f"/tmp/prunecmp_{IMG}.npz"
    if os.path.exists(pa) and os.path.exists(pc):
        za, zc = np.load(pa), np.load(pc)
        return zc["fastv"], zc["div"], za["anchor_idx"], za["W"], str(za["img_path"])
    fastv, div, anchor, W = features_and_all(img_path_for(IMG))
    return fastv, div, anchor, W, img_path_for(IMG)


# ---------------------------------------------------------------- panel renderers
def new_panel():
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    return fig, ax


def stars(ax, idx, s=95):
    r, c = np.asarray(idx) // GRID, np.asarray(idx) % GRID
    ax.scatter((c + 0.5) * PROC / GRID, (r + 0.5) * PROC / GRID, s=s, marker="*",
               facecolor="#00e5ff", edgecolor="#c62828", linewidth=0.9, zorder=5)


def panel_fastv(disp, idx, out):
    fig, ax = new_panel()
    ax.imshow(disp); stars(ax, idx)
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


def panel_divprune(disp, idx, out):
    fig, ax = new_panel()
    ax.imshow(disp); stars(ax, idx)
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


def panel_anchor(disp, anchor, W, out):
    """Heat-map style: jet overlay of per-segment residual weights + white anchor stars."""
    Wn = (W - W.min()) / (W.max() - W.min() + 1e-9)
    Wimg = plt.get_cmap("jet")(Wn)[..., :3]
    Wup = np.array(Image.fromarray((Wimg * 255).astype(np.uint8))
                   .resize((PROC, PROC), Image.BILINEAR))
    fig, ax = new_panel()
    ax.imshow(np.clip(Wup * 0.55 + disp * 0.55, 0, 255).astype(np.uint8) / 255.0)
    r, c = anchor // GRID, anchor % GRID
    ax.scatter((c + 0.5) * PROC / GRID, (r + 0.5) * PROC / GRID, s=44, marker="*",
               facecolor="none", edgecolor="white", linewidth=0.8, zorder=5)
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


fastv, div, anchor, W, img_path = load_all()
disp = np.array(Image.open(img_path).convert("RGB").resize((PROC, PROC)))
todo = {"fastv": ["fastv"], "divprune": ["divprune"], "anchor": ["anchor"],
        "all": ["fastv", "divprune", "anchor"]}[METHOD]
for m in todo:
    out = f"/tmp/panel_{m}.png"
    if m == "fastv":
        panel_fastv(disp, fastv, out)
    elif m == "divprune":
        panel_divprune(disp, div, out)
    else:
        panel_anchor(disp, anchor, W, out)
    print("saved", out)
