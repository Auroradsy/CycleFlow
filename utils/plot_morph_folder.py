#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path morph for the unpaired image-folder runs (horse2zebra and friends).

The ADNI version (`utils.plot_morph`) reads slices out of the paired cache and
scores every frame against a ground-truth slice.  Neither exists here, so this
is a separate script rather than a flag: the frames are shown, and the only
number attached to them is mean|d| against frame 0 — how far the walk has
actually moved, which is the quantity that separates a working arm from one
whose path collapsed to a constant.

Each block state of f is decoded by BOTH decoders, so one row shows the source
domain dissolving and the next shows the target emerging.  The frames are the
flow's own block outputs, not an interpolation between two codes.

  python -m utils.plot_morph_folder --tag h2z_morph_ra
  python -m utils.plot_morph_folder --tag h2z_latcyc --directions a2b b2a
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
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.unpaired_dataset import UnpairedFolderDataset                  # noqa: E402
from model import MMCLASTcg                                              # noqa: E402
from utils.image import to_pm1, to_01                                    # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPS = os.environ.get("MMCLAST_EXPS", os.path.join(_ROOT, "exps"))
CKPT = os.path.join(EXPS, "checkpoints")
FIGS = os.path.join(EXPS, "snapshot_results")


def load(tag, ckpt_name="model.pth"):
    """`model.pth` is written only after S3; pass `stage2.pth` to look at the
    checkpoint an ablation forks from while its own S3 is still running."""
    ck = torch.load(os.path.join(CKPT, tag, ckpt_name), map_location=DEV)
    a = ck["args"]
    m = MMCLASTcg(a["ngf"], a["n_blocks"], a["n_flow"], a["flow_hidden"],
                  bool(a["pre_relu"]), img_ch=a.get("img_ch", 1)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, a


def _img(t):
    """(1,C,H,W) in [-1,1] -> HxW or HxWx3 array for imshow."""
    x = to_01(t)[0].detach().cpu().numpy().transpose(1, 2, 0)
    return x[:, :, 0] if x.shape[2] == 1 else x


@torch.no_grad()
def fig_morph(m, imgs, tag, direction="a2b", outdir=None, dom=("A", "B")):
    """Rows alternate D_A / D_B; columns are the flow's block states."""
    a2b = direction == "a2b"
    enc, states_of = (m.enc_A, lambda z: m.walk(z)) if a2b else \
                     (m.enc_B, lambda u: m.walk(u, inverse=True))
    nA, nB = f"{dom[0]}/A", f"{dom[1]}/B"
    src, tgt = (nA, nB) if a2b else (nB, nA)

    n_col = m.n_blocks + 1
    fig, ax = plt.subplots(2 * len(imgs), n_col,
                           figsize=(2.0 * n_col, 2.0 * 2 * len(imgs)))
    ax = np.atleast_2d(ax)
    report = []
    for r, x in enumerate(imgs):
        states = states_of(enc(x))
        # D_A first, then D_B: reading down a column is "source view, target
        # view" of the SAME latent state.
        for k, dec in enumerate([m.dec_A, m.dec_B]):
            seq = [dec(s) for s in states]
            d = [float((f - seq[0]).abs().mean()) for f in seq]
            view = f"A({dom[0]})" if k == 0 else f"B({dom[1]})"
            report.append(f"  img{r} D_{view:9s} mean|d| = " +
                          " ".join(f"{v:.3f}" for v in d))
            for c, f in enumerate(seq):
                p = ax[2 * r + k, c]
                p.imshow(_img(f), cmap=None if f.shape[1] == 3 else "gray",
                         vmin=0, vmax=1)
                p.set_xticks([]); p.set_yticks([])
                if r == 0 and k == 0:
                    p.set_title(f"$f^{{{c}}}$" if a2b else f"$f^{{-{c}}}$", fontsize=10)
                if c == 0:
                    p.set_ylabel(f"D_{view}", fontsize=8)
    fig.suptitle(f"{tag} — walk {src} -> {tgt}, every block state decoded by both",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    d = outdir or FIGS
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"40_morph_{direction}_{tag}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("\n".join(report)); print("saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_root", default="")
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 3, 7, 11])
    ap.add_argument("--directions", nargs="+", default=["a2b", "b2a"])
    ap.add_argument("--dom", nargs=2, default=["A", "B"],
                    metavar=("A_NAME", "B_NAME"),
                    help="display names for the two domains, e.g. --dom CT PET")
    ap.add_argument("--outdir", default=None,
                    help="where to write the figures (default exps/snapshot_results)")
    a = ap.parse_args()

    m, targs = load(a.tag)
    root = a.data_root or targs["data_root"]
    ds = UnpairedFolderDataset(root, "test", train=False,
                               crop_size=targs["crop_size"],
                               img_ch=targs.get("img_ch", 3))
    for d in a.directions:
        side = 0 if d == "a2b" else 1
        imgs = [to_pm1(ds[i][side].unsqueeze(0).to(DEV)) for i in a.idx]
        print(f"===== {a.tag} {d}")
        fig_morph(m, imgs, a.tag, d, a.outdir, tuple(a.dom))


if __name__ == "__main__":
    main()
