#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMCLAST-cg — path morph along the shared flow, plus self-vs-cross.

Two figures:

  30_mmclast_cg_morph_<tag>.png    every block state of f decoded by BOTH
                                   decoders (T1 view dissolving / FA view
                                   emerging).  Rows alternate D_A / D_B.
  31_mmclast_cg_selfcross_<tag>.png  self-recon vs cross-recon side by side,
                                   which separates "decoder can't" from
                                   "the flow's output is off-manifold".

The morph is NATIVE: the frames are the flow's own block outputs, not an
interpolation between two codes.  That distinction is the whole point — a
linear interpolant would be available to any autoencoder.

  python -m utils.plot_morph --tag morph
"""
import os
import sys
import argparse
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                     # repo root, one level up
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("ADNI_Z_LO", "40")
os.environ.setdefault("ADNI_Z_HI", "49")

from data.paired_dataset import (                                        # noqa: E402
    build_cache, PairedADNISliceDataset, subject_level_split)
from model import MMCLASTcg                                              # noqa: E402
from utils.image import to_pm1, first_frame as _n, ssim as S                      # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(_ROOT, "checkpoints")
FIGS = os.path.join(_ROOT, "snapshot_results")


def load(tag):
    p = os.path.join(CKPT, tag, "model.pth")
    ck = torch.load(p, map_location=DEV)
    a = ck["args"]
    m = MMCLASTcg(a["ngf"], a["n_blocks"], a["n_flow"], a["flow_hidden"],
                  bool(a["pre_relu"])).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, a


@torch.no_grad()
def fig_morph(m, ds, slices, tag):
    rows = []
    for si in slices:
        t1, fa, _ = ds[si]
        T = to_pm1(t1.unsqueeze(0).to(DEV)); F = fa.unsqueeze(0)
        states = m.walk(m.enc_A(T))                       # z, b1..bL (bL = u)
        dA = [_n(m.dec_A(s)) for s in states]
        dB = [_n(m.dec_B(s)) for s in states]
        rows.append((f"slice {si}\nD_A(path)\n(T1 view)", t1[0].numpy(), dA, fa[0].numpy()))
        rows.append((f"slice {si}\nD_B(path)\n(FA view)", t1[0].numpy(), dB, fa[0].numpy()))
        d = [float(np.abs(dB[i] - dB[0]).mean()) for i in range(len(dB))]
        print(f"  slice {si}: D_B(path) mean|Δ| vs frame0 = " +
              " ".join(f"{x:.3f}" for x in d), flush=True)

    L = len(rows[0][2]) - 1
    labels = ["z\n(A rep)"] + [f"b{i+1}" for i in range(L - 1)] + ["u = f(z)\n(B rep)"]
    cols = ["T1 (src)"] + labels + ["FA (tgt)"]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(1.5 * len(cols), 1.65 * len(rows)))
    for r, (lab, src, seq, tgt) in enumerate(rows):
        for c, im in enumerate([src] + seq + [tgt]):
            A = ax[r, c]; A.imshow(im, cmap="gray", vmin=0, vmax=1)
            A.set_xticks([]); A.set_yticks([])
            if c == len(seq):                              # the endpoint u
                for s in A.spines.values():
                    s.set_color("#d62728"); s.set_linewidth(1.8)
        ax[r, 0].set_ylabel(lab, fontsize=7.5, rotation=0, ha="right", va="center")
    for c, t in enumerate(cols):
        ax[0, c].set_title(t, fontsize=8.5)
    fig.suptitle(f"MMCLAST-cg ({tag}) — native path morph along the single shared flow\n"
                 "z --b1..bL--> u  (f applied ONCE: A-space → B-space); red = u, B's "
                 "representation.  Frames are the flow's own block outputs, not an interpolation.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGS, f"30_mmclast_cg_morph_{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


@torch.no_grad()
def fig_self_cross(m, ds, slices, tag):
    cols = ["T1 (GT)", "self T1\nD_A(z)", "cross FA→T1\nD_A(f⁻¹u)",
            "FA (GT)", "self FA\nD_B(u_B)", "cross T1→FA\nD_B(f z)"]
    rows = []
    print(f"{'slice':>5s} | {'selfT1':>7s} {'crossT1':>7s} | {'selfFA':>7s} {'crossFA':>7s} | {'gap':>6s}")
    for si in slices:
        t1, fa, _ = ds[si]
        T = to_pm1(t1.unsqueeze(0).to(DEV)); F = to_pm1(fa.unsqueeze(0).to(DEV))
        zA = m.enc_A(T); uB = m.enc_B(F)
        u = m.a_to_b(zA); z = m.b_to_a(uB)
        sT, sF = _n(m.dec_A(zA)), _n(m.dec_B(uB))
        cT, cF = _n(m.dec_A(z)), _n(m.dec_B(u))
        g1, g2 = t1[0].numpy(), fa[0].numpy()
        gap = float(((u - uB).flatten(1).norm(dim=1) / (uB.flatten(1).norm(dim=1) + 1e-8)))
        print(f"{si:5d} | {S(sT,g1):7.3f} {S(cT,g1):7.3f} | {S(sF,g2):7.3f} {S(cF,g2):7.3f} | {gap:6.3f}")
        rows.append((f"slice {si}\ngap {gap:.2f}", [g1, sT, cT, g2, sF, cF],
                     [None, S(sT, g1), S(cT, g1), None, S(sF, g2), S(cF, g2)]))

    fig, ax = plt.subplots(len(rows), 6, figsize=(1.75 * 6, 1.9 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (lab, imgs, sc) in enumerate(rows):
        for c, im in enumerate(imgs):
            A = ax[r, c]; A.imshow(im, cmap="gray", vmin=0, vmax=1)
            A.set_xticks([]); A.set_yticks([])
            if sc[c] is not None:
                A.set_xlabel(f"{sc[c]:.3f}", fontsize=8)
            if c in (2, 5):
                for s in A.spines.values():
                    s.set_color("#d62728"); s.set_linewidth(1.6)
        ax[r, 0].set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
    for c, t in enumerate(cols):
        ax[0, c].set_title(t, fontsize=8.5)
    fig.suptitle(f"MMCLAST-cg ({tag}) — self vs cross reconstruction\n"
                 "red = the decoder is fed a FLOW-PRODUCED code; "
                 "gap = ‖f(z)−E_B(FA)‖ / ‖E_B(FA)‖", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGS, f"31_mmclast_cg_selfcross_{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="morph")
    ap.add_argument("--slices", type=int, nargs="+", default=[50, 150, 250])
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    m, cfg = load(a.tag)
    print(f"loaded checkpoints/{a.tag} (variant={cfg['variant']}, "
          f"w_latcyc={cfg['w_latcyc']}, w_path_gan={cfg['w_path_gan']})")
    build_cache()
    _, te = subject_level_split(42, 0.20, "label_4")
    ds = PairedADNISliceDataset(te, "label_4")
    fig_morph(m, ds, a.slices, a.tag)
    fig_self_cross(m, ds, a.slices, a.tag)


if __name__ == "__main__":
    main()
