#!/usr/bin/env bash
# Stage-3 epoch time of MMCLAST-cg base ("Ours" in the result tables), to complete
# inpaper_utils/bench_train.sbatch: resumes that run's stage-2 checkpoint (S1/S2 are the
# same for base and morph) and trains 4 stage-3 epochs with the base objective.
#   sbatch --export=ALL,DATASET=adni --dependency=afterok:<bench-train adni> inpaper_utils/bench_train_base.sbatch
#SBATCH --job-name=bench-base
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
X=/ix/lzhan/siyuan/exps/CycleFlow
nvidia-smi --query-gpu=name --format=csv,noheader
case "${DATASET:?set DATASET=adni|mnist}" in
  adni)
    python -u train.py --variant base --tag bench_base_adni --warm $X/adni_host/last.pth \
      --resume_stage 2 --resume_from $X/adni/checkpoints/bench_morph_adni/stage2.pth --e3 4 --p3 99 ;;
  mnist)
    tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
    python -u train.py --config configs/mnist_p_base.yaml --tag bench_base_mnist \
      --data_root $SLURM_SCRATCH/mnist_petct_paired --warm $X/mnist_host/last.pth \
      --resume_stage 2 --resume_from $X/mnist/checkpoints/bench_morph_mnist/stage2.pth --e3 4 --p3 99 ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac
