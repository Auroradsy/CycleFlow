#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-register all T1 (brain-in-rawavg) -> MNI152 2mm with flirt+fnirt, in parallel.

Per subject (first T1 session of the 214 Processed_T1 ∩ registered_DTI subjects):
    flirt 12-DOF affine  ->  fnirt nonlinear (FSL std config)  ->  applywarp
Outputs to registrated_T1_sy/:
    <subj>_T1_mni2.nii.gz   registered T1 in MNI152 2mm (91x109x91)
    <subj>_warp.nii.gz      fnirt warp coefficients (reusable, e.g. to warp aseg)
    <subj>_affine.mat       flirt affine matrix
Also writes manifest.csv (subject, t1_session, fa_session, status, seconds).
"""
import os, sys, glob, time, csv, tempfile, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

B = os.environ["ADNI_RAW_ROOT"]
FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
REF_BRAIN = f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"
REFMASK   = f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz"
CFG       = f"{FSLDIR}/etc/flirtsch/T1_2_MNI152_2mm.cnf"
OUTDIR    = f"{B}/registrated_T1_sy"
ENV = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ")
N_WORKERS = 16

def first_session(root, subj, pattern):
    for d in sorted(glob.glob(os.path.join(root, f"{subj}_I*"))):
        h = glob.glob(os.path.join(d, pattern))
        if h:
            return os.path.basename(d), h[0]
    return None, None

def run(cmd):
    subprocess.run(cmd, env=ENV, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process(subj):
    t0 = time.time()
    t1_sess, t1 = first_session(f"{B}/Processed_T1", subj, "brain-in-rawavg.nii")
    fa_sess, fa = first_session(f"{B}/registered_DTI", subj, "*_FA_reg.nii.gz")
    if not t1 or not fa:
        return dict(subject=subj, t1_session=t1_sess, fa_session=fa_sess,
                    status="missing_input", seconds=0)
    aff  = f"{OUTDIR}/{subj}_affine.mat"
    warp = f"{OUTDIR}/{subj}_warp.nii.gz"
    out  = f"{OUTDIR}/{subj}_T1_mni2.nii.gz"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_aff = os.path.join(tmp, "aff.nii.gz")
            # 1) affine
            run(["flirt", "-in", t1, "-ref", REF_BRAIN, "-out", tmp_aff,
                 "-omat", aff, "-dof", "12", "-interp", "trilinear"])
            # 2) nonlinear (std config; ref overridden to brain template)
            run(["fnirt", f"--in={t1}", f"--config={CFG}",
                 f"--ref={REF_BRAIN}", f"--refmask={REFMASK}",
                 f"--aff={aff}", f"--cout={warp}"])
            # 3) apply warp to get registered T1
            run(["applywarp", f"--in={t1}", f"--ref={REF_BRAIN}",
                 f"--warp={warp}", f"--out={out}", "--interp=trilinear"])
        ok = os.path.exists(out)
        return dict(subject=subj, t1_session=t1_sess, fa_session=fa_sess,
                    status="ok" if ok else "no_output", seconds=round(time.time()-t0, 1))
    except subprocess.CalledProcessError as e:
        return dict(subject=subj, t1_session=t1_sess, fa_session=fa_sess,
                    status=f"fail:{e.returncode}", seconds=round(time.time()-t0, 1))

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    subjects = [l.strip() for l in open("/tmp/common_all.txt") if l.strip()]
    print(f"Batch fnirt: {len(subjects)} subjects, {N_WORKERS} workers -> {OUTDIR}", flush=True)
    rows = []; done = 0; t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process, s): s for s in subjects}
        for fut in as_completed(futs):
            r = fut.result(); rows.append(r); done += 1
            el = time.time() - t_start
            print(f"[{done:3d}/{len(subjects)}] {r['subject']} {r['status']} "
                  f"({r['seconds']}s)  elapsed={el/60:.1f}min", flush=True)
            # incremental manifest write
            with open(f"{OUTDIR}/manifest.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["subject","t1_session","fa_session","status","seconds"])
                w.writeheader(); w.writerows(sorted(rows, key=lambda x: x["subject"]))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDONE. {ok}/{len(rows)} ok in {(time.time()-t_start)/60:.1f} min. "
          f"manifest -> {OUTDIR}/manifest.csv", flush=True)
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print("Non-ok:", [(r['subject'], r['status']) for r in bad], flush=True)

if __name__ == "__main__":
    main()
