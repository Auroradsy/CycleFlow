#!/usr/bin/env bash
# V1 (principal diffusion direction) for all 214 subjects, then v1_112.pt for DEC-FA.
# Runs the bundle the workstation shipped (raw_dti/ + v1_inputs/, see v1_inputs/README.md):
#   v1_all.sh  : fslroi b0 -> bet -> dtifit on the raw DWI -> vecreg into FMRIB58 1mm
#                with the existing FA_to_template.mat (skips subjects already done)
#   dec_fa.py  : resample |V1| to MNI 2mm, same rot90 + pad as paired_112.pt
# The result goes next to paired_112.pt; the repo's 3-subject data/v1_112.pt is untouched.
#   sbatch inpaper_utils/v1_all.sbatch
#SBATCH --job-name=v1-all
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -uo pipefail
R=/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow
module load fsl/6.0.7.18
echo "FSLDIR=$FSLDIR  $(which dtifit vecreg bet | tr '\n' ' ')"
export FSLOUTPUTTYPE=NIFTI_GZ
ADNI_ROOT=$R/raw_dti V1_OUT=$R/v1_nifti FMRIB58=$R/v1_inputs/FMRIB58_FA_1mm.nii.gz \
  bash $R/v1_inputs/v1_all.sh "$SLURM_CPUS_PER_TASK" | tee $R/v1_all.log
n_ok=$(grep -c '\[done\]\|\[skip\]' $R/v1_all.log)
echo "subjects done or skipped: $n_ok / 214; failures:"; grep '\[FAIL\]\|\[MISSING\]' $R/v1_all.log || echo "  none"
[ "$n_ok" -eq 214 ] || { echo "not all subjects have V1; re-submit to retry the failures"; exit 1; }

set -e
source configs/_env.sh
# dec_fa.py imports data.paired_dataset from the repo; its V1 cache goes next to paired_112.pt.
PYTHONPATH=$MMCLAST_ROOT ADNI_CACHE=$R/paired_112.pt ADNI_V1_CACHE=$R/v1_112.pt \
V1_OUT=$R/v1_nifti MNI152_2MM=$R/v1_inputs/MNI152_T1_2mm.nii.gz \
  python -u -W ignore $R/v1_inputs/dec_fa.py build

# The three pilot subjects were computed on the workstation with the same recipe:
# their V1 must agree with the new build.
python -u -W ignore - <<'EOF'
import numpy as np, torch
new = torch.load("/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow/v1_112.pt", weights_only=False)
old = torch.load("/ihome/lzhan/sid51/projects/Brain/CycleFlow/data/v1_112.pt", weights_only=False)
hn, ho = new["has_v1"].numpy(), old["has_v1"].numpy()
print(f"coverage: {hn.sum()}/{len(hn)} slices")
k = np.where(ho)[0]
a, b = new["V1"][k].float(), old["V1"][k].float()
brain = (b.sum(1) > 30)
print(f"pilot slices {len(k)}: mean |new-old| on brain voxels {float((a - b).abs().sum(1)[brain].mean()):.2f} (uint8 units, 3 channels)")
EOF
