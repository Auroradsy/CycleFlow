#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resample all 4 DTI scalars (FA/MD/RD/AD) from MNI 1mm to MNI 2mm grid for the
214 T1∩DTI subjects, mirroring registrated_T1_sy.
No registration needed — both grids share the same MNI152 physical space.
"""
import os, glob, csv, time
import nibabel as nib
from nilearn.image import resample_to_img
from concurrent.futures import ProcessPoolExecutor, as_completed

B    = "/data_new3/nfs_share/public/Imaging_genetic"
SRC  = f"{B}/registered_DTI"
OUT  = f"{B}/registrated_DTI_2mm"
MNI2 = "/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz"
SCALARS = ["FA", "MD", "RD", "AD"]
N_WORKERS = 12

# get common subjects from the T1 manifest (ensures same 214)
manifest_T1 = f"{B}/registrated_T1_sy/manifest.csv"
SUBJECTS = sorted({r["subject"] for r in csv.DictReader(open(manifest_T1))
                   if r["status"] == "ok"})

def first_session(subj):
    for d in sorted(glob.glob(f"{SRC}/{subj}_I*")):
        # require all 4 scalars present in this session
        if all(glob.glob(f"{d}/*_{s}_reg.nii.gz") for s in SCALARS):
            return d
    return None

def process(subj):
    t0 = time.time()
    sess = first_session(subj)
    if sess is None:
        return dict(subject=subj, dti_session=None, status="no_session", seconds=0)
    sess_id = os.path.basename(sess)
    try:
        for s in SCALARS:
            src = glob.glob(f"{sess}/*_{s}_reg.nii.gz")[0]
            dst = f"{OUT}/{subj}_{s}_mni2.nii.gz"
            img2 = resample_to_img(src, MNI2, interpolation="continuous")
            img2.to_filename(dst)
        return dict(subject=subj, dti_session=sess_id, status="ok",
                    seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(subject=subj, dti_session=sess_id, status=f"fail:{type(e).__name__}",
                    seconds=round(time.time() - t0, 1))

def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Resampling 4-scalar DTI -> MNI 2mm for {len(SUBJECTS)} subjects, "
          f"{N_WORKERS} workers -> {OUT}", flush=True)
    rows = []; done = 0; t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process, s): s for s in SUBJECTS}
        for fut in as_completed(futs):
            r = fut.result(); rows.append(r); done += 1
            print(f"[{done:3d}/{len(SUBJECTS)}] {r['subject']} {r['status']} ({r['seconds']}s)",
                  flush=True)
    with open(f"{OUT}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject","dti_session","status","seconds"])
        w.writeheader(); w.writerows(sorted(rows, key=lambda x: x["subject"]))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDONE. {ok}/{len(rows)} ok in {(time.time()-t_start)/60:.1f} min. "
          f"manifest -> {OUT}/manifest.csv", flush=True)

if __name__ == "__main__":
    main()
