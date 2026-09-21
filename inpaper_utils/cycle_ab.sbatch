#!/usr/bin/env bash
# Figures 5 and 7 of the paper: the cycles and the affine terms, each with both directions
# marked (a) T1->FA and (b) FA->T1, two held-out slices per direction -- the slices of Figure 2
# (drawn from Figure 1). DEC-FA throughout. affine_ab writes one file per direction, stacked
# in one figure environment in the paper.
#   sbatch inpaper_utils/cycle_ab.sbatch
#SBATCH --job-name=cycle-ab
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
# Random reads of the two caches over /ix leave the job in D state for ~20 min; one
# sequential copy to node-local scratch is far cheaper (both paths are env-overridable).
cp "$ADNI_ROOT/paired_112.pt" "$ADNI_ROOT/v1_112.pt" "$SLURM_SCRATCH"/
export ADNI_CACHE=$SLURM_SCRATCH/paired_112.pt ADNI_V1_CACHE=$SLURM_SCRATCH/v1_112.pt
python -m py_compile inpaper_utils/make_flow_viz.py
python -u -W ignore -m inpaper_utils.make_flow_viz --dataset adni --variant morph_smooth --dec-fa \
  --figures cycle_ab,affine_ab --ab-pick "416,111;379,309" --skip-summary
