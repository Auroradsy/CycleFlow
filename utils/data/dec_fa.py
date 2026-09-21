#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEC-FA (directionally-encoded colour FA) for any slice of the paired cache.

One cache, one script:

    python -m data.dec_fa build                         # V1 NIfTIs -> data/cache/v1_112.pt
    python -m data.dec_fa show --subj 002_S_0413 --z 45  # -> PNG
    python -m data.dec_fa show --idx 1234 5678 --out x.png

    from data.dec_fa import dec_fa
    rgb = dec_fa(k)            # [112,112,3] float in [0,1], k = index into paired_112.pt
    rgb = dec_fa(k, fa=pred)   # colour a GENERATED FA with the subject's true V1

v1_112.pt is aligned 1:1 with paired_112.pt (same N, same k -> same subject/z) and
stores |V1| as uint8 after the SAME rot90 + pad as data/paired_dataset.py, so
DEC = |V1| * FA is a lookup.  Subjects whose V1 has not been computed yet have
has_v1[k] = False.  Colour: red = L-R, green = A-P, blue = S-I.

V1 comes from data/preprocess/dec_fa_pilot.sh (dtifit on the raw DWI, rotated
into template space with the existing FA_to_template.mat by vecreg).  Brightness
always comes from the training FA, so a mismatch between recomputed and original
FA changes nothing but colour.
"""
import os, sys, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.paired_dataset import CACHE, Z_LO, Z_HI, pad_to_square      # noqa: E402

_D = os.path.dirname(os.path.abspath(__file__))
V1_CACHE = os.environ.get("ADNI_V1_CACHE", os.path.join(_D, "cache", "v1_112.pt"))
V1_DIR = os.path.join(_D, "cache", "dec_fa_pilot")        # <subj>/V1_<variant>_reg.nii.gz
MNI2 = "/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz"


def build(variant="raw", v1_dir=V1_DIR):
    import nibabel as nib
    from scipy.ndimage import map_coordinates
    # linear resample onto the MNI 2mm grid via the two affines (no nilearn: its
    # pandas import breaks once torch has loaded the system libstdc++)
    ref = nib.load(MNI2)
    ijk2 = np.stack(np.meshgrid(*[np.arange(n) for n in ref.shape[:3]], indexing="ij"), 0).reshape(3, -1)

    def resample(img):
        m = np.linalg.inv(img.affine) @ ref.affine
        ijk1 = m[:3, :3] @ ijk2 + m[:3, 3:]
        vol = img.get_fdata(dtype=np.float32)
        return np.stack([map_coordinates(np.abs(vol[..., c]), ijk1, order=1, cval=0.0)
                         .reshape(ref.shape[:3]) for c in range(3)], -1)

    d = torch.load(CACHE, map_location="cpu")
    subjects, sidx, zidx = d["subjects"], d["subj_idx"].numpy(), d["z_idx"].numpy()
    N = len(sidx)
    if os.path.exists(V1_CACHE):                      # incremental: keep what is there
        old = torch.load(V1_CACHE)
        assert old["V1"].shape[0] == N, "v1 cache does not match paired cache"
        V1, has = old["V1"].numpy(), old["has_v1"].numpy()
    else:
        V1, has = np.zeros((N, 3, 112, 112), np.uint8), np.zeros(N, bool)
    s2i = {s: i for i, s in enumerate(subjects)}
    added = []
    for f in sorted(glob.glob(f"{v1_dir}/*/V1_{variant}_reg.nii.gz")):
        subj = os.path.basename(os.path.dirname(f))
        if subj not in s2i:
            continue
        # abs BEFORE interpolation (inside resample): V1 sign is arbitrary, +v and -v would cancel
        v = resample(nib.load(f))
        v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
        for k in np.where(sidx == s2i[subj])[0]:
            z = zidx[k]
            V1[k] = np.stack([pad_to_square(np.rot90(v[:, :, z, c])) for c in range(3)]) \
                      .clip(0, 1).__mul__(255).round().astype(np.uint8)
            has[k] = True
        added.append(subj)
    os.makedirs(os.path.dirname(V1_CACHE), exist_ok=True)
    torch.save(dict(V1=torch.from_numpy(V1), has_v1=torch.from_numpy(has),
                    variant=variant, z_range=(Z_LO, Z_HI - 1)), V1_CACHE)
    n_subj = len(np.unique(sidx[has]))
    print(f"added {len(added)} subjects from {v1_dir}  ->  {V1_CACHE}\n"
          f"  {n_subj}/{len(subjects)} subjects, {has.sum()}/{N} slices have V1")


_cache = {}
def _load():
    if not _cache:
        _cache["p"] = torch.load(CACHE, map_location="cpu")
        _cache["v"] = torch.load(V1_CACHE, map_location="cpu")
    return _cache["p"], _cache["v"]


def index_of(subj, z):
    p, _ = _load()
    s = p["subjects"].index(subj)
    k = torch.where((p["subj_idx"] == s) & (p["z_idx"] == z))[0]
    if len(k) == 0:
        raise KeyError(f"{subj} z={z} not in cache (z range {Z_LO}..{Z_HI - 1})")
    return int(k[0])


def dec_fa(k, fa=None):
    """DEC-FA for cache index k -> [H,W,3] float in [0,1].
    fa: optional FA image ([H,W] / [1,H,W], numpy or tensor, in [0,1]) to use as
    brightness instead of the cached one, e.g. a model's T1->FA output."""
    p, v = _load()
    if not bool(v["has_v1"][k]):
        subj = p["subjects"][int(p["subj_idx"][k])]
        raise KeyError(f"no V1 for {subj} — run dec_fa_pilot.sh on it, then `build`")
    fa = p["FA"][k, 0] if fa is None else torch.as_tensor(fa).squeeze().cpu()
    rgb = v["V1"][k].float().div(255) * fa.float().clamp(0, 1)[None]
    return rgb.permute(1, 2, 0).numpy()


def show(idx, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    p, _ = _load()
    fig, ax = plt.subplots(len(idx), 3, figsize=(7.5, 2.6 * len(idx)), squeeze=False)
    for r, k in enumerate(idx):
        subj, z = p["subjects"][int(p["subj_idx"][k])], int(p["z_idx"][k])
        for c, (img, t) in enumerate([(p["T1"][k, 0], "T1"), (p["FA"][k, 0], "FA"),
                                      (dec_fa(k), "DEC-FA")]):
            ax[r, c].imshow(img, cmap="gray" if c < 2 else None, vmin=0, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(t)
        ax[r, 0].set_ylabel(f"{subj}\nz={z}  k={k}", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--variant", default="raw", choices=["raw", "eddy"])
    b.add_argument("--v1_dir", default=V1_DIR)
    s = sub.add_parser("show")
    s.add_argument("--idx", type=int, nargs="*", default=[])
    s.add_argument("--subj", nargs="*", default=[]); s.add_argument("--z", type=int, nargs="*", default=[])
    s.add_argument("--out", default="dec_fa.png")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.variant, a.v1_dir)
    else:
        zs = a.z * len(a.subj) if len(a.z) == 1 else a.z
        idx = a.idx + [index_of(sj, z) for sj, z in zip(a.subj, zs)]
        if not idx: raise SystemExit("give --idx or --subj/--z")
        show(idx, a.out)


if __name__ == "__main__":
    main()
