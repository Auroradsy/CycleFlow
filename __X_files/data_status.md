# ADNI data status

Inventory of `/data_new3/nfs_share/public/Imaging_genetic/` (ADNI cohort) for
CLAST multimodal experiments. Subject ID format `<site>_S_<subject>` (e.g. `002_S_0413`).

## 1. Summary at a glance

| Source modality | Unique subjects | Sessions | Storage | Has labels? | Status |
|---|---:|---:|---|---|---|
| **T1 raw** `T1/t1_nii/*/I*/T1.nii.gz` | 1,146 | 5,419 | ~6 MB ea | ✅ 1,144 / 1,146 | Multi-shape, multi-vox |
| **Processed T1 (FreeSurfer brain + aseg)** `Processed_T1/<sess>/{brain,aseg}-in-rawavg.nii` | 1,144 | 5,408 | ~13 MB / session | ✅ all 1,144 | 2mm rawavg space, ~10 shapes |
| **DTI raw 4D** `DTI/dti_nii/*/I*/DTI.nii.gz` | 734 | 2,205 | ~75 MB ea | ✅ 228 (other 506 not in label table) | Wildly variable shape & gradients |
| **registered DTI (FA/MD/RD/AD)** `registered_DTI/<sess>/*_{FA,MD,RD,AD}_reg.nii.gz` | 695 | 2,037 | ~24 MB / session (4 maps) | ✅ 214 | (182,218,182) MNI152 1mm ✓ uniform |
| **rs-fMRI** `fMRI/rs_fMRI_nii/<subj>/I*/rest.nii.gz` | 192 | 625 | ~70 MB ea | ✅ 191 / 192 | (64,64,48,140), TR=3s |
| **ext rs-fMRI** `fMRI/ext_rs_fMRI_nii/<subj>/I*/rest.nii.gz` | 95 | 127 | ~100 MB ea | ✅ ~all | (64,64,48,200), TR≈3s |
| **T1 morphometric features** `T1_features/*.txt` | 5,409 (rows) | – | 18 tables | – | FreeSurfer thickness/area/volume |
| **Gene encoding + DX label table** `kunzhao/TinyLLaVA_Factory-main/dataset/genetic_image/all_infomation_v2.json` | 1,144 | 5,407 | 25 MB | self | per-scan gene string + DX label |
| **registrated_T1_sy ✦ (this project)** `<root>/registrated_T1_sy/*_T1_mni2.nii.gz` | **214** | 214 | ~1 MB ea | from kunzhao | (91,109,91) MNI152 2mm,fnirt-aligned |
| **registrated_DTI_2mm_sy ✦ (this project)** `<root>/registrated_DTI_2mm_sy/*_{FA,MD,RD,AD}_mni2.nii.gz` | **214** | 214 × 4 | ~2.5 MB ea | from kunzhao | (91,109,91) MNI152 2mm,4 scalars |

✦ = derived/produced by this project. Both registered to a common MNI152 2mm grid
so paired T1↔DTI loading is `np.array`-trivial.

## 2. Image dimensions (per modality)

| Modality | Common shape | Voxel size | 4D? | Notes |
|---|---|---|---|---|
| T1 raw | (176,240,256), (160,192,192), (170,256,256), … (≥5 distinct) | 1.0–1.25 mm,杂 | – | needs registration |
| Processed_T1 brain/aseg | (105,120,128), (96,120,120), (102,128,128), … (~10) | 2.0 mm 统一 | – | rawavg space (per-subject FOV) |
| DTI raw | (256,256,59,46), (116,116,80,55), (128,128,80,31), … 多 | anisotropic 1.4–2.7 mm | ✅ | 31–55 gradient dirs/scan |
| registered DTI (FA/MD/RD/AD) | **(182,218,182) 全统一** | 1.0 mm | – | ✅ MNI152 1mm 模板空间 |
| rs-fMRI | **(64,64,48,140) 主流** | 3.31 mm, TR=3.0s | ✅ | 少数 80×80 / TR=6.02s |
| ext rs-fMRI | **(64,64,48,200) 主流** | 3.31 mm, TR≈3.0s | ✅ | 时间帧 200 |
| registrated_T1_sy | (91,109,91) | 2.0 mm | – | flirt+fnirt → MNI 2mm |
| registrated_DTI_2mm_sy | (91,109,91) | 2.0 mm | – | nilearn resample 1mm→2mm |

## 3. Multi-modal pairing (subject-level intersections)

|  | T1raw | DTIraw | rs-fMRI | Notes |
|---|---:|---:|---:|---|
| **T1raw** ∩ … | 1,146 | 228 | 192 | base counts |
| **Processed_T1** ∩ … | 1,144 | **214** | 191 | regDTI more restrictive than DTIraw |
| **registered_DTI** ∩ … | 214 | 695 | 61 | |
| **rs-fMRI** ∩ … | 191 | 61 | 192 | fMRI almost always has T1 |
| **Pro_T1 ∩ regDTI ∩ rs-fMRI** | – | – | **61** | three-modality cohort |
| **rs-fMRI ∪ ext_rs-fMRI** | 197 | – | – | combined functional pool |

## 4. Label distribution per modality cohort

From `kunzhao/.../all_infomation_v2.json` (1,144 subjects with labels). Subject-level
majority vote across that subject's scans.

| Cohort | n | CN | SMC | EMCI | MCI | LMCI | AD | labelled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **T1 (Processed_T1)** | 1,144 | 295 | 54 | 226 | 254 | 120 | **195** | 1,144 |
| DTI raw | 734 | 96 | 17 | 63 | 27 | 24 | 1 | 228 / 734 ⚠ |
| **registered DTI** | 695 | 92 | 15 | 59 | 24 | 23 | **1** ⚠ | 214 / 695 ⚠ |
| **rs-fMRI** | 192 | 49 | 19 | 51 | 9 | 32 | **31** | 191 |
| **Pro_T1 ∩ regDTI** (our 214) | 214 | 92 | 15 | 59 | 24 | 23 | **1** ⚠ | 214 |
| **Pro_T1 ∩ rs-fMRI** | 191 | 49 | 19 | 51 | 9 | 32 | **31** | 191 |
| Pro_T1 ∩ regDTI ∩ rsfMRI | 61 | 26 | 4 | 18 | 4 | 9 | 0 | 61 |

⚠ critical insight: **DTI heavily under-samples AD patients** (AD subjects can't
tolerate long DTI scans). The 214-subject T1↔DTI cohort has only **1 AD subject**,
making 3-class CN/MCI/AD evaluation impossible.

**rs-fMRI** keeps 31 AD subjects — much better diagnosis spread for clustering
evaluation, but fMRI is 4D and needs feature extraction (ALFF / ReHo / connectivity)
to compare against structural T1.

## 5. Recommended pairs for CLAST experiments

| Goal | Best pair | Why |
|---|---|---|
| **Pilot pipeline validation** (current) | Pro_T1 ↔ regDTI (FA), 214 subj | All in MNI 2mm,uniform shape,easy load |
| **Better diagnosis-class spread (3-class CN/MCI/AD)** | Pro_T1 ↔ rs-fMRI, 191 subj | 31 AD subjects available |
| **Three-modality stress test** | Pro_T1 ↔ regDTI ↔ rs-fMRI, 61 subj | Small, sanity / viz only |
| **Imaging-genetic** | Pro_T1 ↔ gene, 1,144 subj | Largest pool,gene from kunzhao json |
| **Pure structural baseline** | Pro_T1 alone, 1,144 subj | Full cohort with full label spread (CN 295 / AD 195) |

For the current proposal: **T1 ↔ DTI(FA)** is what we're running. If the bimodal
CLAST gets stuck on class imbalance (we saw label_4 needs to drop AD), consider
**T1 ↔ rsfMRI(extracted 3D map)** as the second testbed — same engineering, much
better AD coverage.

## 6. Gene data (bonus modality)

`all_infomation_v2.json` per-scan entry:

```json
{
  "subject_id": "002_S_0413",
  "scan_id"   : "I30119",
  "label"     : "The subject is CN",
  "gene"      : "<A2M:10>, <ABCA7:001>, <ABI3:0>, ... ~80+ AD-risk genes, mutation pattern encoded per gene>"
}
```

- **1,144 subjects** with gene encoding (= full T1 cohort)
- ~80+ AD-risk gene loci encoded as digit strings per gene (length = # mutations)
- Could serve as a **third modality** for true imaging-genetic CLAST
- Untouched so far in our pipeline

## 7. Pre-registered files we produced

| Path | Files | Size | Purpose |
|---|---|---|---|
| `/data_new3/nfs_share/public/Imaging_genetic/registrated_T1_sy/` | 214 × `_T1_mni2.nii.gz` + 214 × `_warp.nii.gz` + 214 × `_affine.mat` + `manifest.csv` | 219 MB | T1 in MNI152 2mm,flirt+fnirt aligned |
| `/data_new3/nfs_share/public/Imaging_genetic/registrated_DTI_2mm_sy/` | 214 × 4 × `_{FA,MD,RD,AD}_mni2.nii.gz` + `manifest.csv` | 2.1 GB | DTI scalars in MNI152 2mm (resampled from 1mm) |
| `clast_base/adni_pilot/labels.csv` | 214 rows × {label_2, label_3, label_4, label_6} | tiny | subject-level diagnosis labels |
| `clast_base/adni_pilot/cache/paired_112.pt` | 5778 paired slices,T1+FA pad to 112² | 580 MB | training cache (T1∩DTI cohort,z=32..58) |

## 8. What's NOT covered

- **No diagnosis label CSV in `/Imaging_genetic` itself** — pulled from kunzhao's project (per-scan json,not the canonical ADNIMERGE)
- **Per-scan dates** — `T1.json` sidecars have `AcquisitionDate` but we don't use it. Currently we pair "first session" of T1 with "first session" of DTI,which might be different visits ⚠ (acceptable for representation learning but not for true longitudinal analysis)
- **fMRI not preprocessed** — raw 4D nii only,no ALFF / ReHo / connectome computed
- **Gene data not loaded** — gene string in json; needs parsing into a numeric feature vector

## 9. Open data questions

- Should we mirror `registrated_T1_sy` for the **rs-fMRI cohort** too (T1 already in MNI 2mm, but fMRI needs feature extraction)?
- Want to expand the bimodal cohort to **228** (DTIraw ∩ T1raw)? The extra 14 subjects exist in unregistered DTI; we'd need to register their FA ourselves.
- Should we materialize a `registrated_fMRI_2mm_sy/` (mean ALFF + mean ReHo at MNI 2mm,one volume per subject) to enable T1↔fMRI CLAST?
