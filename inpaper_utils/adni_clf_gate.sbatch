#!/usr/bin/env bash
# E1a: the ADNI downstream-classification gate -- can a classifier read the label off the
# real slices at all, before anything is claimed about augmenting them.
#   sbatch --partition=<whatever is free> inpaper_utils/adni_clf_gate.sbatch
#SBATCH --job-name=adni-gate
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m py_compile inpaper_utils/adni_clf_gate.py
# Random reads of the 580 MB cache over /ix park the job in D state; copy it once.
cp "$ADNI_ROOT/paired_112.pt" "$SLURM_SCRATCH"/
export ADNI_CACHE=$SLURM_SCRATCH/paired_112.pt
python -u -W ignore -m inpaper_utils.adni_clf_gate --repeats "${REPEATS:-3}" --epochs "${EPOCHS:-30}"
