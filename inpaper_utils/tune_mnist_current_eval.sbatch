#!/usr/bin/env bash
# Score the two current Table 1 checkpoints of ours on the validation split, so they
# enter the same validation-only selection as the tune_mnist_* configs
# (inpaper_utils/tune_mnist_collect.py). The test split is re-scored alongside.
#SBATCH --job-name=tune-cur-eval
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
X=/ix/lzhan/siyuan/exps/CycleFlow
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
R=$SLURM_SCRATCH/mnist_petct_paired
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for C in $X/mnist_p_base/model.pth $X/mnist/checkpoints/mnist_p_morph_smooth/model.pth; do
  python -u -m inpaper_utils.eval_mmclast --dataset mnist --data-root "$R" --split val --ckpt "$C"
  python -u -m inpaper_utils.eval_mmclast --dataset mnist --data-root "$R" --ckpt "$C"
done
