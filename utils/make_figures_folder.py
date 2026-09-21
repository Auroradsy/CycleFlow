#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Every unpaired-folder figure, regenerated from whatever runs have finished.

Re-run it as arms complete; it picks up any `h2z_*` checkpoint directory that
already has a `model.pth` and skips the rest, so a half-finished ablation still
produces a coherent folder.

    python -m utils.make_figures_folder --dataset h2z
    python -m utils.make_figures_folder --dataset mnist
    python -m utils.make_figures_folder --dataset mnist --tags mnist_latcyc

Figures, in the order they should be read:

  00_host_reference.png     the plain CycleGAN we split, both directions.  Every
                            other figure is a comparison against this.
  01_fid_calibration.png    FID has no natural scale, so the bar chart carries
                            its own floor and ceiling: real-vs-real is the best
                            any model can do, and real-source-vs-real-target is
                            what "change nothing" scores.  A run that lands on
                            the ceiling did not translate, however large or
                            small its FID happens to look in isolation.
  02_fid_path.png           FID of each intermediate flow state against real
                            A u B.  This is the unpaired replacement for the
                            ADNI hole metric: it asks whether a frame that is
                            neither horse nor zebra is a plausible image at all,
                            with no ground-truth mid-frames required.
  10_cross_vs_host_<tag>.png  input / host / this arm's cross path / this arm's
                            self path.  The self row is the control: if it is
                            clean and the cross row is not, the autoencoders are
                            fine and the flow is the problem.
  40_morph_{a2b,b2a}_<tag>.png  the native morph (via utils.plot_morph_folder).

The calibration constants are cached to fid_reference.json -- they depend only
on the dataset, and recomputing Inception statistics for 2400 images on every
invocation would dominate the runtime.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.unpaired_dataset import list_images                            # noqa: E402
from model import MMCLASTcg                                              # noqa: E402
from model.backbone import ResnetGenerator                               # noqa: E402
from utils import fid as F                                               # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
CKPT = os.path.join(EXPS, "checkpoints")
# Set by main() from --dataset / the individual overrides.  Module-level
# because every figure function writes into OUT and draws at SIZE, and
# threading them through nine signatures would obscure the figures themselves.
OUT = os.path.join(EXPS, "snapshot_results", "horse2zebra")
SIZE = 256
DOM = ("horse", "zebra")          # display names for domain A and B
PREFIX = "h2z_"

# Per-dataset presets.  `dom` is cosmetic (titles and row labels); everything
# else changes what is actually read and written.
PRESETS = {
    "h2z": dict(root="/ix/lzhan/siyuan/datasets/processed_datas/horse2zebra", prefix="h2z_",
                host="h2z_host", out="horse2zebra", size=256,
                dom=("horse", "zebra")),
    "mnist": dict(root="/ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired", prefix="mnist_",
                  host="mnist_host", out="mnist_petct", size=64,
                  dom=("CT", "PET")),
}


def load_img(p, size=None, ch=3):
    size = SIZE if size is None else size
    im = Image.open(p).convert("RGB" if ch == 3 else "L").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = a[:, :, None]
    return torch.from_numpy(a).permute(2, 0, 1)


def batch(paths, idx, size=None):
    return torch.stack([load_img(paths[i], size) for i in idx]).to(DEV) * 2 - 1


def show(ax, t):
    ax.imshow(((t + 1) * 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    ax.set_xticks([]); ax.set_yticks([])


def _host_n_blocks(ck, default):
    """n_blocks lives in ck["args"], not at the top level.  ck.get("n_blocks", D)
    therefore always returned D silently -- correct only when D happened to
    match the run.  Read the real value and fall back only if it is absent."""
    a = ck.get("args") or {}
    return int(a.get("n_blocks", ck.get("n_blocks", default)))


def load_host(warm):
    ck = torch.load(warm, map_location=DEV)
    g = {}
    for key, name in [("G_T1toFA", "a2b"), ("G_FAtoT1", "b2a")]:
        n = ResnetGenerator(3, 3, 64, _host_n_blocks(ck, 9)).to(DEV).eval()
        n.load_state_dict(ck[key], strict=True)
        g[name] = n
    return g


def load_arm(tag, ckpt_name="model.pth"):
    ck = torch.load(os.path.join(CKPT, tag, ckpt_name), map_location=DEV)
    a = ck["args"]
    m = MMCLASTcg(a["ngf"], a["n_blocks"], a["n_flow"], a["flow_hidden"],
                  bool(a["pre_relu"]), img_ch=a.get("img_ch", 1)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, a


# ---------------------------------------------------------------------------
def fid_reference(root, host, cache):
    """Floor / ceiling / host, all under the protocol the runs themselves use."""
    if os.path.exists(cache):
        return json.load(open(cache))
    trA, trB = list_images(f"{root}/trainA"), list_images(f"{root}/trainB")
    teA, teB = list_images(f"{root}/testA"), list_images(f"{root}/testB")
    cdir = os.path.join(OUT, "fid_cache")
    # crop=SIZE on every call: the default is 256, and the caches these write
    # are the same files fid_against reads back below.  Leaving it out computed
    # the floor and ceiling at 256 on a 64px dataset and fed 256-crop reference
    # stats to 64-crop fakes.
    kw = dict(crop=SIZE, img_ch=3, batch=8)
    sk = dict(crop=SIZE)
    muA, sA = F.folder_stats(trA, DEV, os.path.join(cdir, "trainA.npz"), **sk)
    muB, sB = F.folder_stats(trB, DEV, os.path.join(cdir, "trainB.npz"), **sk)
    mtA, stA = F.folder_stats(teA, DEV, os.path.join(cdir, "testA.npz"), **sk)
    mtB, stB = F.folder_stats(teB, DEV, os.path.join(cdir, "testB.npz"), **sk)
    with torch.no_grad():
        d = {
            "floor_a2b": F.frechet(mtB, stB, muB, sB),   # real B test vs B train
            "floor_b2a": F.frechet(mtA, stA, muA, sA),
            "ceil_a2b": F.frechet(mtA, stA, muB, sB),    # untouched A vs real B
            "ceil_b2a": F.frechet(mtB, stB, muA, sA),
            "host_a2b": F.fid_against(teA, host["a2b"], trB, DEV,
                                      os.path.join(cdir, "trainB.npz"), **kw),
            "host_b2a": F.fid_against(teB, host["b2a"], trA, DEV,
                                      os.path.join(cdir, "trainA.npz"), **kw),
        }
    os.makedirs(OUT, exist_ok=True)
    json.dump(d, open(cache, "w"), indent=2)
    return d


def fig_host(root, host):
    idx = list(range(6))
    A, B = list_images(f"{root}/testA"), list_images(f"{root}/testB")
    xa, xb = batch(A, idx), batch(B, idx)
    with torch.no_grad():
        rows = [(f"{DOM[0]} (input)", xa),
                (f"host: {DOM[0]} -> {DOM[1]}", host["a2b"](xa)),
                (f"{DOM[1]} (input)", xb),
                (f"host: {DOM[1]} -> {DOM[0]}", host["b2a"](xb))]
    fig, ax = plt.subplots(4, 6, figsize=(13, 9))
    for r, (lab, t) in enumerate(rows):
        for c in range(6):
            show(ax[r, c], t[c])
        ax[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("plain CycleGAN host — the model MMCLAST-cg is split from", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "00_host_reference.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("saved", p)


def fig_calibration(ref, arms):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, tt in [(axes[0], "a2b", f"{DOM[0]} -> {DOM[1]}"),
                      (axes[1], "b2a", f"{DOM[1]} -> {DOM[0]}")]:
        names = ["real\n(floor)", "host"] + [t.replace(PREFIX, "") for t, _ in arms]
        vals = [ref[f"floor_{d}"], ref[f"host_{d}"]] + \
               [v.get(f"fid_{'A2B' if d == 'a2b' else 'B2A'}", float("nan")) for _, v in arms]
        cols = ["#4c9f70", "#3b6ea5"] + \
               ["#c1584a" if (v is not None and not np.isnan(v)
                              and v > 0.9 * ref[f"ceil_{d}"]) else "#8a6bbf"
                for v in vals[2:]]
        b = ax.bar(names, vals, color=cols)
        ax.axhline(ref[f"ceil_{d}"], ls="--", c="#8c8c8c")
        ax.text(len(names) - 0.4, ref[f"ceil_{d}"], "  do nothing", va="bottom",
                ha="right", fontsize=9, color="#5c5c5c")
        for r, v in zip(b, vals):
            if not np.isnan(v):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}", ha="center",
                        va="bottom", fontsize=9)
        ax.set_title(tt); ax.set_ylabel("FID (lower is better)")
        ax.set_ylim(0, max(ref[f"ceil_{d}"], max([v for v in vals if not np.isnan(v)])) * 1.18)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("FID needs its own scale: a bar at the dashed line did not translate",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "01_fid_calibration.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("saved", p)


def fig_path_fid(arms):
    have = [(t, v["fid_path"]) for t, v in arms if v.get("fid_path")]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t, ys in have:
        ax.plot(range(len(ys)), ys, marker="o", label=t.replace(PREFIX, ""))
    ax.set_xlabel("flow block state  $f^0 \\ldots f^L$")
    ax.set_ylabel("FID of the decoded frame vs real A $\\cup$ B")
    ax.set_title("Are the intermediate frames plausible images?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "02_fid_path.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("saved", p)


def fig_cross(root, host, tag):
    m, _ = load_arm(tag)
    idx = list(range(6))
    xa = batch(list_images(f"{root}/testA"), idx)
    with torch.no_grad():
        rows = [(f"input {DOM[0]}", xa), ("host G_A2B", host["a2b"](xa)),
                (f"{tag}\ncross A->B", m.cross_A2B(xa)),
                (f"{tag}\nself A->A", m.self_A(xa))]
    fig, ax = plt.subplots(4, 6, figsize=(13, 9))
    for r, (lab, t) in enumerate(rows):
        for c in range(6):
            show(ax[r, c], t[c])
        ax[r, 0].set_ylabel(lab, fontsize=8)
    fig.suptitle(f"{tag} — cross path against the host, with the self path as control",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, f"10_cross_vs_host_{tag}.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("saved", p)
    del m; torch.cuda.empty_cache()


def read_eval(tag):
    p = os.path.join(CKPT, tag, "final_eval.txt")
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if "=" not in line:
            continue
        k, v = line.strip().split("=", 1)
        if k == "fid_path_vs_AuB":
            d["fid_path"] = [float(x) for x in v.split()]
        else:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def main():
    global OUT, SIZE, DOM, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="h2z")
    ap.add_argument("--root", default=None)
    ap.add_argument("--warm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 3, 7, 11])
    a = ap.parse_args()

    P = PRESETS[a.dataset]
    PREFIX, SIZE, DOM = P["prefix"], P["size"], P["dom"]
    a.root = a.root or P["root"]
    a.warm = a.warm or os.path.join(CKPT, P["host"], "last.pth")
    OUT = a.out or os.path.join(EXPS, "snapshot_results", P["out"])

    os.makedirs(OUT, exist_ok=True)
    tags = a.tags if a.tags is not None else sorted(
        t for t in os.listdir(CKPT)
        if t.startswith(PREFIX) and t != P["host"]
        and os.path.exists(os.path.join(CKPT, t, "model.pth")))
    print(f"finished arms: {tags or '(none yet)'}")

    host = load_host(a.warm)
    ref = fid_reference(a.root, host, os.path.join(OUT, "fid_reference.json"))
    print("reference:", json.dumps({k: round(v, 2) for k, v in ref.items()}))

    fig_host(a.root, host)
    arms = [(t, read_eval(t) or {}) for t in tags]
    fig_calibration(ref, arms)
    fig_path_fid(arms)
    for t in tags:
        fig_cross(a.root, host, t)

    # the native morph, one figure per direction per arm
    from utils.plot_morph_folder import load as _load, fig_morph
    from data.unpaired_dataset import UnpairedFolderDataset
    from utils.image import to_pm1
    for t in tags:
        m, targs = _load(t)
        ds = UnpairedFolderDataset(a.root, "test", train=False,
                                   crop_size=targs["crop_size"],
                                   img_ch=targs.get("img_ch", 3))
        for d in ["a2b", "b2a"]:
            side = 0 if d == "a2b" else 1
            imgs = [to_pm1(ds[i][side].unsqueeze(0).to(DEV)) for i in a.idx]
            print(f"===== {t} {d}")
            fig_morph(m, imgs, t, d, OUT, DOM)
        del m; torch.cuda.empty_cache()
    print("\nOut:", OUT)


if __name__ == "__main__":
    main()
