#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What makes the path decodable? — morph ablation across MMCLAST-cg variants.

One row per (variant, slice), showing D_B applied to every block state of the
shared flow, plus the per-frame mean|Δ| curve that quantifies how much actually
changes along the path.

The question this answers: eight earlier attempts produced either a path with
no visible change (constrain the intermediates -> flow_work collapses) or one
that decoded to noise (leave them free -> off-manifold).  The variants here
differ only in whether the flow's output is required to be a fixed point of the
B autoencoder (L_latcyc) and whether intermediate frames must satisfy a
realism critic (L_path).  Neither term constrains the flow's DISTRIBUTION,
which is what -bNLL did at the cost of collapsing flow_work to 0.34.

  python -m utils.plot_ablation --tags base latcyc morph
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
from utils.image import to_pm1, first_frame as _n                      # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPS = os.environ.get("MMCLAST_EXPS", os.path.join(_ROOT, "exps"))
CKPT = os.path.join(EXPS, "checkpoints")
FIGS = os.path.join(EXPS, "snapshot_results")

NICE = {"base":            "base\nno L_latcyc\nno L_path",
        "latcyc":          "+ L_latcyc\n(fixed point)",
        "morph":           "+ L_latcyc\n+ L_path\n(adversarial)",
        "morph_s3ctl":     "L_path\nFORWARD only\n(control)",
        "morph_bi_s3ctl":  "L_path\nBOTH directions",
        "morph_bi":        "L_path\nBOTH directions\n(full run)"}


def load(tag):
    p = os.path.join(CKPT, tag, "model.pth")
    if not os.path.exists(p):
        return None, None
    ck = torch.load(p, map_location=DEV); a = ck["args"]
    m = MMCLASTcg(a["ngf"], a["n_blocks"], a["n_flow"], a["flow_hidden"],
                  bool(a["pre_relu"])).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    stats = {}
    fe = os.path.join(CKPT, tag, "final_eval.txt")
    if os.path.exists(fe):
        for line in open(fe):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                stats[k] = v
    return m, stats


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base", "latcyc", "morph"])
    ap.add_argument("--slices", type=int, nargs="+", default=[50, 250])
    ap.add_argument("--direction", default="a2b", choices=["a2b", "b2a"],
                    help="a2b = T1->FA (walk f forward);  b2a = FA->T1 (walk f back)")
    ap.add_argument("--out", default="32_mmclast_cg_morph_ablation_a2b.png")
    a = ap.parse_args()
    fwd = a.direction == "a2b"
    os.makedirs(FIGS, exist_ok=True)
    build_cache()
    _, te = subject_level_split(42, 0.20, "label_4")
    ds = PairedADNISliceDataset(te, "label_4")

    rows, curves = [], {}
    for tag in a.tags:
        m, st = load(tag)
        if m is None:
            print(f"skip {tag}: no model.pth"); continue
        cur = []
        for si in a.slices:
            t1, fa, _ = ds[si]
            if fwd:
                code = m.enc_A(to_pm1(t1.unsqueeze(0).to(DEV)))
                src_im, tgt_im = t1[0].numpy(), fa[0].numpy()
            else:
                code = m.enc_B(to_pm1(fa.unsqueeze(0).to(DEV)))
                src_im, tgt_im = fa[0].numpy(), t1[0].numpy()
            states = m.walk(code, inverse=not fwd)
            lab = NICE.get(tag, tag)
            if st:
                lab += f"\nfw {float(st.get('flow_work', 'nan')):.2f}"
            # BOTH decoders.  D_B alone is misleading: an unconstrained flow can
            # still look fine there (its endpoint is trained) while the same
            # states tear apart under D_A.  The path is only usable if every
            # state is decodable by the decoder that owns each end.
            for dec_name, dec in (("D_A (T1 view)", m.dec_A), ("D_B (FA view)", m.dec_B)):
                seq = [_n(dec(s)) for s in states]
                d = [float(np.abs(seq[i] - seq[0]).mean()) for i in range(len(seq))]
                rows.append((f"{lab}\nslice {si}\n{dec_name}", src_im, seq, tgt_im))
                print(f"{tag:8s} slice {si} {dec_name:14s}: mean|Δ| = "
                      + " ".join(f"{x:.3f}" for x in d))
                if dec is (m.dec_B if fwd else m.dec_A):
                    cur.append(d)
        curves[tag] = np.mean(cur, 0)

    L = len(rows[0][2]) - 1
    ends = (("T1 (src)", "z\n(A rep)", "u = f(z)\n(B rep)", "FA (tgt)") if fwd else
            ("FA (src)", "u\n(B rep)", "z = f\u207b\u00b9(u)\n(A rep)", "T1 (tgt)"))
    cols = ([ends[0], ends[1]] + [f"b{i+1}" for i in range(L - 1)] + [ends[2], ends[3]])
    nc = len(cols) + 1                                   # + the mean|Δ| panel
    fig, ax = plt.subplots(len(rows), nc, figsize=(1.5 * nc, 1.7 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (lab, src, seq, tgt) in enumerate(rows):
        for c, im in enumerate([src] + seq + [tgt]):
            A = ax[r, c]; A.imshow(im, cmap="gray", vmin=0, vmax=1)
            A.set_xticks([]); A.set_yticks([])
            if c == len(seq):
                for s in A.spines.values():
                    s.set_color("#d62728"); s.set_linewidth(1.8)
        ax[r, 0].set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
        A = ax[r, nc - 1]
        d = [float(np.abs(seq[i] - seq[0]).mean()) for i in range(len(seq))]
        A.plot(range(len(d)), d, "o-", color="#d62728", ms=3.5, lw=1.4)
        A.set_ylim(-0.01, 0.20); A.grid(alpha=0.3, lw=0.5)
        A.tick_params(labelsize=6)
        A.set_xticks(range(len(d)))
        A.set_xticklabels(["z"] + [f"b{i+1}" for i in range(len(d) - 1)], fontsize=5.5)
    for c, t in enumerate(cols):
        ax[0, c].set_title(t, fontsize=8)
    ax[0, nc - 1].set_title("mean|Δ| vs z\n(path progress)", fontsize=7.5)
    fig.suptitle(f"MMCLAST-cg — what makes the flow's own path decodable "
                 f"({'T1 → FA' if fwd else 'FA → T1'})\n"
                 "every frame is a block state of f decoded by D_A (T1 view) or D_B (FA view).  A usable morph "
                 "needs the change to appear in the FA view\nwhile the T1 view stays a valid brain — not the "
                 "other way round.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGS, a.out)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
