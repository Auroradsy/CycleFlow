#!/usr/bin/env bash
# E1b (cross-modal probe + retrieval) and E1c (MNIST augmentation). Both read the
# held-out split and the stage-3 checkpoints; nothing is retrained except the small
# downstream classifiers.
#   sbatch --partition=l40s --qos=gpu-l40s-s inpaper_utils/repr_probe.sbatch
#   sbatch --export=ALL,STEPS=probe ... inpaper_utils/repr_probe.sbatch
#SBATCH --job-name=repr-probe
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
M=inpaper_utils
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m py_compile $M/cross_modal_probe.py $M/mnist_aug.py

# 10k small PNGs and a 580 MB tensor read at random over /ix park the job in D state.
cp "$ADNI_ROOT/paired_112.pt" "$SLURM_SCRATCH"/
export ADNI_CACHE=$SLURM_SCRATCH/paired_112.pt
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
export MNIST_PAIRED_ROOT=$SLURM_SCRATCH/mnist_petct_paired

case "${STEPS:-probe aug}" in *probe*)
  for ds in mnist adni; do
    echo "=== E1b cross-modal probe: $ds ==="
    python -u -W ignore -m $M.cross_modal_probe --dataset "$ds" --variant morph_smooth
  done
;; esac
case "${STEPS:-probe aug}" in *aug*)
  echo "=== E1c MNIST augmentation ==="
  # Two sweeps. sbatch --export splits on commas, so the lists live here, not in the
  # environment. The corpus sweep exists because 6,000 translated images saturate the
  # task: every augmented arm hits the oracle and the path/interp contrast cannot show.
  # SKIP_LABELS=1 reruns only the corpus sweeps, which is what an added direction needs.
  [ "${SKIP_LABELS:-0}" = 1 ] || \
    python -u -W ignore -m $M.mnist_aug --labels "${LABELS:-10,25,50,100}" --seeds "${SEEDS:-3}"
  for dir in a2b b2a; do
    python -u -W ignore -m $M.mnist_aug --labels 10 --seeds "${SEEDS:-3}" --direction $dir \
      --corpus-sweep "${CORPUS_SWEEP:-100,300,1000,3000}"
  done
;; esac
