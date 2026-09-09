#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plumbing for the cross-modal baselines.

The baselines were written against the ADNI cache and hard-wired to it.  This
module is the seam that lets the same training code run on the paired
image-folder datasets (MNIST-PET/CT) as well, so the two testbeds report
numbers from one implementation rather than two.

Nothing here is method-specific; each method supplies its own loss and its own
sampler and calls `evaluate`.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def loaders(args):
    """(train, test, n_ch) for either testbed, under each one's own protocol."""
    if args.data == "adni":
        from data import build_cache, PairedADNISliceDataset, subject_level_split
        build_cache()
        tr_idx, te_idx = subject_level_split(seed=42, test_frac=0.20,
                                             label_scheme="label_4",
                                             z_lo=args.z_lo, z_hi=args.z_hi)
        tr = PairedADNISliceDataset(tr_idx, label_scheme="label_4")
        te = PairedADNISliceDataset(te_idx, label_scheme="label_4")
        n_ch = 1
    else:
        from data.unpaired_dataset import UnpairedFolderDataset
        # pair=True asserts filename alignment and applies ONE geometric draw to
        # both members, without which every paired loss compares misaligned
        # images.  no_flip because a mirrored digit is not a valid sample.
        common = dict(load_size=args.load_size, crop_size=args.crop_size,
                      img_ch=args.img_ch, pair=True, flip=False)
        tr = UnpairedFolderDataset(args.data_root, "train", train=True, **common)
        te = UnpairedFolderDataset(args.data_root, "test", train=False, **common)
        n_ch = args.img_ch
    return (DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4,
                       drop_last=True, pin_memory=True),
            DataLoader(te, batch_size=64, shuffle=False, num_workers=4,
                       pin_memory=True),
            n_ch)


def ssim_psnr(pred, gt):
    """Both arrays (B,C,H,W) in [0,1].  Multichannel handled via channel_axis."""
    from skimage.metrics import structural_similarity as ssim
    s, q = [], []
    for i in range(pred.shape[0]):
        p, g = pred[i], gt[i]
        if p.shape[0] == 1:
            s.append(ssim(g[0], p[0], data_range=1.0))
        else:
            s.append(ssim(g.transpose(1, 2, 0), p.transpose(1, 2, 0),
                          data_range=1.0, channel_axis=2))
        mse = float(np.mean((g - p) ** 2))
        q.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return s, q


@torch.no_grad()
def evaluate(gen_ab, gen_ba, loader, device, max_batches=None):
    """gen_*: callable(source_batch) -> translated batch, both in [0,1]."""
    acc = {k: [] for k in ["ssim_A2B", "psnr_A2B", "ssim_B2A", "psnr_B2A"]}
    for bi, (a, b, _y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        a = a.to(device); b = b.to(device)
        fb = gen_ab(a).clamp(0, 1).cpu().numpy()
        fa = gen_ba(b).clamp(0, 1).cpu().numpy()
        gb = b.cpu().numpy(); ga = a.cpu().numpy()
        s, q = ssim_psnr(fb, gb); acc["ssim_A2B"] += s; acc["psnr_A2B"] += q
        s, q = ssim_psnr(fa, ga); acc["ssim_B2A"] += s; acc["psnr_B2A"] += q
    return {k: float(np.mean(v)) for k, v in acc.items()}


def add_data_args(ap):
    ap.add_argument("--data", default="adni", choices=["adni", "folder"])
    ap.add_argument("--data_root", default="")
    ap.add_argument("--img_ch", type=int, default=3)
    ap.add_argument("--load_size", type=int, default=72)
    ap.add_argument("--crop_size", type=int, default=64)
    ap.add_argument("--z_lo", type=int, default=40)
    ap.add_argument("--z_hi", type=int, default=49)
    return ap
