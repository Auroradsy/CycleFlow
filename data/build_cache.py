#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the paired T1/FA slice cache, and report what an axial band contains.

Two jobs, because they are the two things you do to this dataset before you can
train on it:

    python -m data.build_cache                 # NIfTI -> data/cache/paired_112.pt
    python -m data.build_cache --report        # what the middle-10 band selects

The cache holds the FULL axial range (z = 32..58, 27 slices/subject).  The band
is applied later, at split time, by `subject_level_split(z_lo=, z_hi=)` — so
switching testbeds never means rebuilding 554 MB.

Why the middle 10 (z = 40..49).  Across the full range, slice POSITION is the
dominant source of variance: two slices of the SAME subject at different z
differ more than two DIFFERENT subjects at the same z.  Any clustering or SSIM
number computed over the full range is then substantially reporting "which
slice is this", which is not the question.

`--report` measures that directly on your own cache, as a ratio of mean |ΔT1|
between the two kinds of pair.  On ours:

    z = 32..58 (full)      0.0842 / 0.0486  =  1.73
    z = 40..49 (middle 10) 0.0639 / 0.0496  =  1.29

So the band cuts the confound by a quarter but does NOT eliminate it — slice
position still moves the image more than subject identity does, even here.  It
is a reduction, not a fix, and cross-subject numbers on this data should be
read with that in mind.  (This is the same reason eval.py reports the
inter-subject and template SSIM floors instead of raw SSIM alone.)
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.paired_dataset import (                                    # noqa: E402
    build_cache, subject_level_split, CACHE, Z_LO, Z_HI, IMG)


def report(z_lo, z_hi, label_scheme, seed, test_frac):
    d = torch.load(CACHE, map_location="cpu")
    T1, FA, z = d["T1"], d["FA"], d["z_idx"].numpy()
    subj = d["subj_idx"].numpy()
    print(f"cache: {CACHE}")
    print(f"  {len(T1)} slices  {len(d['subjects'])} subjects  "
          f"{tuple(T1.shape[1:])}  z = {z.min()}..{z.max()}\n")

    print(f"band z = {z_lo}..{z_hi}  ({z_hi - z_lo + 1} slices/subject)")
    tr, te = subject_level_split(seed, test_frac, label_scheme, z_lo, z_hi)

    # Which axis dominates inside this band — slice position, or who the subject
    # is?  Two matched samples of the same size, built by CONSTRUCTION rather
    # than by filtering random pairs (rejection sampling almost never lands two
    # draws on the same subject, and a 1-pair "average" is not an average).
    keep = np.where((z >= z_lo) & (z <= z_hi))[0]
    by_subj, by_z = {}, {}
    for i in keep:
        by_subj.setdefault(subj[i], []).append(i)
        by_z.setdefault(z[i], []).append(i)

    rng = np.random.default_rng(0)
    n_pairs = 200

    def mad(pairs):
        return float(np.mean([torch.abs(T1[i] - T1[j]).mean() for i, j in pairs]))

    # same subject, two different slice positions
    subj_pool = [s for s, v in by_subj.items() if len(v) >= 2]
    same_subj = []
    for s in rng.choice(subj_pool, n_pairs, replace=True):
        i, j = rng.choice(by_subj[s], 2, replace=False)
        same_subj.append((i, j))

    # same slice position, two different subjects
    z_pool = [zz for zz, v in by_z.items() if len({subj[i] for i in v}) >= 2]
    same_z = []
    for zz in rng.choice(z_pool, n_pairs, replace=True):
        i, j = rng.choice(by_z[zz], 2, replace=False)
        while subj[i] == subj[j]:
            i, j = rng.choice(by_z[zz], 2, replace=False)
        same_z.append((i, j))

    a, b = mad(same_subj), mad(same_z)
    print(f"\n  mean|ΔT1| same subject, different z : {a:.4f}   ({len(same_subj)} pairs)")
    print(f"  mean|ΔT1| same z, different subject : {b:.4f}   ({len(same_z)} pairs)")
    print(f"  ratio {a/b:.2f}   " + ("<1 -> subject identity dominates; the band is narrow enough"
                                     if a < b else
                                     ">1 -> slice position still dominates; consider a narrower band"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the cache already exists")
    ap.add_argument("--report", action="store_true",
                    help="do not build; describe the cache and the selected band")
    ap.add_argument("--z_lo", type=int, default=40, help="axial band low, inclusive")
    ap.add_argument("--z_hi", type=int, default=49, help="axial band high, inclusive")
    ap.add_argument("--label_scheme", default="label_4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    a = ap.parse_args()

    if a.report:
        if not os.path.exists(CACHE):
            raise SystemExit(f"no cache at {CACHE} — run `python -m data.build_cache` first")
        report(a.z_lo, a.z_hi, a.label_scheme, a.seed, a.test_frac)
        return

    print(f"full axial range z = {Z_LO}..{Z_HI - 1}, padded to {IMG}x{IMG}")
    path = build_cache(force=a.force)
    print(f"cache ready: {path}")
    print(f"\nnow: python -m data.build_cache --report --z_lo {a.z_lo} --z_hi {a.z_hi}")


if __name__ == "__main__":
    main()
