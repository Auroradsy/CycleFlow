#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The SSIM numbers that ARE meaningful without pairing, scored post-hoc.

Unpaired data admits no SSIM against ground truth, which is why the runs report
FID.  But two SSIMs still exist, and exactly one of them is a fair comparison
against the host:

  cycle SSIM   SSIM(A -> B -> A, A).  Both a CycleGAN and MMCLAST-cg have this
               round trip, so it compares directly.  MMCLAST-cg should win by
               construction: f is an EXACT bijection, so f^-1(f(z)) cancels
               analytically and the cycle costs nothing, whereas CycleGAN has to
               push two independent generators together with lambda_cyc.
               It is NOT a translation-quality metric — a model that outputs its
               input scores 1.0.

  SSIM(out,in) how much the translation changed the image.  Reported only to
               show that it is the WRONG metric here: the host translates most
               and therefore scores WORST, inverting the FID ranking.  Anyone
               reaching for "SSIM" on this dataset gets this number and draws
               the opposite of the correct conclusion.

Scored after the fact from model.pth rather than inside train.py, so arms that
finished before this existed are covered too.

    python -m utils.eval_h2z_ssim                 # every finished h2z_* arm
    python -m utils.eval_h2z_ssim --tags h2z_morph_ra
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.unpaired_dataset import UnpairedFolderDataset                  # noqa: E402
from model import MMCLASTcg                                              # noqa: E402
from model.backbone import ResnetGenerator                               # noqa: E402
from utils.image import to_pm1, to_01                                    # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
CKPT = os.path.join(EXPS, "checkpoints")
OUT = os.path.join(EXPS, "snapshot_results", "horse2zebra")


def ssim_batch(pred, gt):
    from skimage.metrics import structural_similarity as ssim
    p = to_01(pred).cpu().numpy(); g = to_01(gt).cpu().numpy()
    return [ssim(g[i].transpose(1, 2, 0), p[i].transpose(1, 2, 0),
                 data_range=1.0, channel_axis=2) for i in range(p.shape[0])]


@torch.no_grad()
def score(fwd, bwd, loader):
    """fwd: A->B, bwd: B->A.  Works for the host pair and for an MMCLAST arm."""
    acc = {k: [] for k in ["cyc_A", "cyc_B", "delta_A2B", "delta_B2A"]}
    for xa, xb, _ in loader:
        A = to_pm1(xa.to(DEV)); B = to_pm1(xb.to(DEV))
        fb, fa = fwd(A), bwd(B)
        acc["cyc_A"] += ssim_batch(bwd(fb), A)
        acc["cyc_B"] += ssim_batch(fwd(fa), B)
        acc["delta_A2B"] += ssim_batch(fb, A)
        acc["delta_B2A"] += ssim_batch(fa, B)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _host_n_blocks(ck, default):
    """n_blocks lives in ck["args"], not at the top level.  ck.get("n_blocks", D)
    therefore always returned D silently -- correct only when D happened to
    match the run.  Read the real value and fall back only if it is absent."""
    a = ck.get("args") or {}
    return int(a.get("n_blocks", ck.get("n_blocks", default)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ix/lzhan/siyuan/datasets/processed_datas/horse2zebra")
    ap.add_argument("--warm", default=os.path.join(CKPT, "h2z_host", "last.pth"))
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    ds = UnpairedFolderDataset(a.root, "test", train=False, crop_size=256, img_ch=3)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4)

    ck = torch.load(a.warm, map_location=DEV)
    host = {}
    for key, name in [("G_T1toFA", "a2b"), ("G_FAtoT1", "b2a")]:
        g = ResnetGenerator(3, 3, 64, _host_n_blocks(ck, 9)).to(DEV).eval()
        g.load_state_dict(ck[key], strict=True)
        host[name] = g

    tags = a.tags if a.tags is not None else sorted(
        t for t in os.listdir(CKPT)
        if t.startswith("h2z_") and t != "h2z_host"
        and os.path.exists(os.path.join(CKPT, t, "model.pth")))

    res = {"host": score(host["a2b"], host["b2a"], dl)}
    for t in tags:
        c = torch.load(os.path.join(CKPT, t, "model.pth"), map_location=DEV)
        ar = c["args"]
        m = MMCLASTcg(ar["ngf"], ar["n_blocks"], ar["n_flow"], ar["flow_hidden"],
                      bool(ar["pre_relu"]), img_ch=ar.get("img_ch", 1)).to(DEV)
        m.load_state_dict(c["model"]); m.eval()
        res[t] = score(m.cross_A2B, m.cross_B2A, dl)
        del m; torch.cuda.empty_cache()

    hdr = f"{'':16s} {'cyc A→B→A':>10s} {'cyc B→A→B':>10s} | {'SSIM(out,in) A→B':>17s} {'B→A':>8s}"
    print(hdr); print("-" * len(hdr))
    for k, d in res.items():
        print(f"{k:16s} {d['cyc_A']:10.4f} {d['cyc_B']:10.4f} | "
              f"{d['delta_A2B']:17.4f} {d['delta_B2A']:8.4f}")
    print("\ncycle SSIM: higher is better, and MMCLAST-cg wins by construction "
          "(f is an exact bijection).\nSSIM(out,in): NOT a quality metric — the "
          "host scores worst precisely because it\n  translates most, which "
          "inverts the FID ranking.  Shown as a warning, not a result.")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "ssim_table.json")
    json.dump(res, open(p, "w"), indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
