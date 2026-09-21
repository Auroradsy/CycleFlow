#!/usr/bin/env python3
"""Is the ADNI cache actually paired?  Aggregate checks only, no subject IDs.

1. every cache row is one (subject, z) and T1/FA share it by construction;
2. both NIfTIs exist for every subject, same shape and same affine (same space);
3. a random sample of rows re-extracted from the NIfTIs equals the cache;
4. anatomy agrees more within a pair than across pairs (brain-mask Dice / NCC);
5. train and test subjects are disjoint.
Checks 2-3 need the registered NIfTIs, which only exist where the cache was
built; on a server that holds only paired_112.pt they are skipped.
Run: srun -M htc ... bash -c 'source configs/_env.sh && python inpaper_utils/probe_adni_pairing.py'
"""
import os, sys, csv
import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.paired_dataset import (CACHE, T1D, FAD, Z_LO, Z_HI, pad_to_square, norm01,
                                 subject_level_split)

print(f"cache: {CACHE}")
d = torch.load(CACHE, map_location="cpu", weights_only=False)
T1, FA, sidx, zidx, subjects = d["T1"], d["FA"], d["subj_idx"], d["z_idx"], d["subjects"]
print(f"T1 {tuple(T1.shape)}  FA {tuple(FA.shape)}  subjects {len(subjects)}")

# 1. one row per (subject, z)
keys = sidx.numpy() * 1000 + zidx.numpy()
print(f"[1] rows {len(keys)}, unique (subj,z) {len(np.unique(keys))}, "
      f"expected {len(subjects)}x{Z_HI - Z_LO} = {len(subjects) * (Z_HI - Z_LO)}")

rng = np.random.default_rng(0)
if not (os.path.isdir(T1D) and os.path.isdir(FAD)):
    print(f"[2] skipped: NIfTI dirs absent on this machine (T1 {os.path.isdir(T1D)}, "
          f"FA {os.path.isdir(FAD)})")
    print("[3] skipped: same reason")
else:
    # 2. files + geometry
    miss_t1 = miss_fa = shape_bad = aff_bad = 0
    max_aff = 0.0
    for s in subjects:
        ft, ff = f"{T1D}/{s}_T1_mni2.nii.gz", f"{FAD}/{s}_FA_mni2.nii.gz"
        miss_t1 += not os.path.exists(ft); miss_fa += not os.path.exists(ff)
        if not (os.path.exists(ft) and os.path.exists(ff)):
            continue
        ht, hf = nib.load(ft), nib.load(ff)
        shape_bad += ht.shape != hf.shape
        diff = float(np.abs(ht.affine - hf.affine).max())
        max_aff = max(max_aff, diff); aff_bad += diff > 1e-3
    print(f"[2] missing T1 {miss_t1}  missing FA {miss_fa}  shape mismatch {shape_bad}  "
          f"affine mismatch {aff_bad}  (max |dA| {max_aff:.2e})")
    fa_man = f"{FAD}/manifest.csv"
    if os.path.exists(fa_man):
        rows = list(csv.DictReader(open(fa_man)))
        ok = {r["subject"] for r in rows if r.get("status") == "ok"}
        print(f"    FA manifest: {len(rows)} rows, {len(ok)} ok, "
              f"cache subjects not ok in FA manifest: {sum(s not in ok for s in subjects)}")

    # 3. cache == re-extraction on a random sample
    worst, skipped = 0.0, 0
    for k in rng.choice(len(T1), 40, replace=False):
        s, z = subjects[int(sidx[k])], int(zidx[k])
        ft, ff = f"{T1D}/{s}_T1_mni2.nii.gz", f"{FAD}/{s}_FA_mni2.nii.gz"
        if not (os.path.exists(ft) and os.path.exists(ff)):
            skipped += 1; continue
        t = pad_to_square(norm01(np.rot90(nib.load(ft).get_fdata().astype(np.float32)[:, :, z])))
        f = pad_to_square(norm01(np.rot90(nib.load(ff).get_fdata().astype(np.float32)[:, :, z])))
        worst = max(worst, float(np.abs(t - T1[k, 0].numpy()).max()),
                    float(np.abs(f - FA[k, 0].numpy()).max()))
    print(f"[3] {40 - skipped} random rows re-extracted from NIfTI: "
          f"max |cache - fresh| = {worst:.2e}  ({skipped} skipped)")

# 4. anatomical agreement, on the paper's z=40..49 band
tr, te = subject_level_split(42, 0.20, "label_4", 40, 49)
idx = torch.cat([tr, te]).numpy()
mt = (T1[idx, 0] > 0.05).numpy().reshape(len(idx), -1)
mf = (FA[idx, 0] > 0.05).numpy().reshape(len(idx), -1)
at, af = T1[idx, 0].numpy().reshape(len(idx), -1), FA[idx, 0].numpy().reshape(len(idx), -1)

def dice(a, b): return 2 * (a & b).sum(1) / np.maximum(a.sum(1) + b.sum(1), 1)
def ncc(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)

s_, z_ = sidx.numpy()[idx], zidx.numpy()[idx]
same = np.arange(len(idx))
# other subject, same z  (the hard negative: only individual anatomy differs)
other = np.array([rng.choice(np.where((z_ == z_[i]) & (s_ != s_[i]))[0]) for i in same])
# same subject, z +/- 3  (slice misalignment)
shift = np.array([np.where((s_ == s_[i]) & (z_ == (z_[i] + 3 if z_[i] <= 46 else z_[i] - 3)))[0][0]
                  for i in same])
d_same, d_other = dice(mt, mf), dice(mt, mf[other])
for name, j in (("matched pair (same subj, same z)", same),
                ("other subject, same z", other),
                ("same subject, z +/- 3", shift)):
    print(f"[4] {name:34s} mask Dice {dice(mt, mf[j]).mean():.4f}   "
          f"T1-FA NCC {ncc(at, af[j]).mean():.4f}")
print(f"    matched Dice beats other-subject Dice on {100 * (d_same > d_other).mean():.1f}% "
      f"of {len(idx)} slices")

# 5. leakage
st, ss = set(sidx.numpy()[tr.numpy()]), set(sidx.numpy()[te.numpy()])
print(f"[5] train subj {len(st)}  test subj {len(ss)}  overlap {len(st & ss)}")

# 6. identification.  After MNI registration every brain has nearly the same
# outline, so check 4 cannot tell a true pair from a same-z stranger.  Remove the
# population template per z and ask whether a subject's T1 residual picks out
# its OWN FA residual among all subjects at that z (chance = 1/n).
subs = sorted(set(s_.tolist()))
top1, ranks, C_sum = [], [], 0
for z in range(40, 50):
    rows = [idx[np.where((s_ == s) & (z_ == z))[0][0]] for s in subs]
    t = T1[rows, 0].numpy().reshape(len(rows), -1).astype(np.float64)
    f = FA[rows, 0].numpy().reshape(len(rows), -1).astype(np.float64)
    t -= t.mean(0); f -= f.mean(0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    C = t @ f.T                                   # [n_T1, n_FA]
    C_sum = C_sum + C
    top1.append((C.argmax(1) == np.arange(len(rows))).mean())
    ranks.append((C > np.diag(C)[:, None]).sum(1))  # 0 = own FA ranked first
n = len(subs)
diag = np.diag(C_sum) / 10
off = (C_sum.sum() - np.trace(C_sum)) / (n * (n - 1)) / 10
print(f"[6] residual T1-FA correlation, {n} subjects, z=40..49")
print(f"    matched {diag.mean():.4f}   mismatched {off:.4f}")
print(f"    per-slice top-1 identification {100 * np.mean(top1):.1f}%  "
      f"(chance {100 / n:.1f}%), median rank of own FA {np.median(np.concatenate(ranks)):.0f}/{n - 1}")
print(f"    per-subject (10 slices summed) top-1 {100 * (C_sum.argmax(1) == np.arange(n)).mean():.1f}%")

# 7. slice level.  Check 6 fixes z and asks about the subject; this fixes the
# subject and asks about z: does T1(s, z) match FA(s, z) better than FA(s, z')?
# "raw" is dominated by the slice's anatomy; "resid" subtracts each z's own
# population template first, so only this subject's deviation at that z is left.
S_all, Z_all = sidx.numpy(), zidx.numpy()
valid = np.isin(S_all, subs)
tmpl_T = {z: T1[np.where(valid & (Z_all == z))[0], 0].numpy().mean(0) for z in range(Z_LO, Z_HI)}
tmpl_F = {z: FA[np.where(valid & (Z_all == z))[0], 0].numpy().mean(0) for z in range(Z_LO, Z_HI)}
offs = np.arange(-8, 9)
def unit(v):
    v = v.reshape(v.shape[0], -1).astype(np.float64); v = v - v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
prof = {"raw": np.zeros(len(offs)), "resid": np.zeros(len(offs))}
argoff = {"raw": [], "resid": []}
for s in subs:
    r = np.where(S_all == s)[0]; r = r[np.argsort(Z_all[r])]; zs = Z_all[r]
    F_raw = unit(FA[r, 0].numpy())
    F_res = unit(np.stack([FA[k, 0].numpy() - tmpl_F[int(Z_all[k])] for k in r]))
    for z in range(40, 50):
        k = r[zs == z][0]
        for name, F_, t in (("raw", F_raw, T1[k, 0].numpy()),
                            ("resid", F_res, T1[k, 0].numpy() - tmpl_T[z])):
            c = F_ @ unit(t[None])[0]
            argoff[name].append(int(zs[c.argmax()] - z))
            for j, o in enumerate(offs):
                prof[name][j] += c[zs == z + o][0]
N7 = len(subs) * 10
print(f"[7] same subject, T1 at z vs FA at z+dz  ({N7} T1 slices, z=40..49)")
print("    dz      " + " ".join(f"{o:+6d}" for o in offs))
for name in ("raw", "resid"):
    print(f"    {name:6s}  " + " ".join(f"{v / N7:6.3f}" for v in prof[name]))
for name in ("raw", "resid"):
    a = np.array(argoff[name])
    print(f"    best-matching FA slice ({name}): dz=0 {100 * (a == 0).mean():.1f}%  "
          f"|dz|=1 {100 * (np.abs(a) == 1).mean():.1f}%  |dz|>=2 {100 * (np.abs(a) >= 2).mean():.1f}%  "
          f"mean dz {a.mean():+.2f}")
