#!/usr/bin/env bash
# Inference milliseconds of the two ADDED rows, on the A100 the cost column reports,
# merged into the existing EXPS/bench_cost/infer.json rather than re-timing the rows
# already in the paper (--only; see inpaper_utils/bench_infer.py).
#   sbatch inpaper_utils/bench_infer_newbaselines.sbatch
#SBATCH --job-name=bench-infer-newbase
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
X=/ix/lzhan/siyuan/exps/CycleFlow
nvidia-smi --query-gpu=name --format=csv,noheader
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
export MNIST_PAIRED_ROOT=$SLURM_SCRATCH/mnist_petct_paired
# keep the file that holds the published timings, in case the merge has to be undone
[ -f $X/bench_cost/infer.json ] && cp -p $X/bench_cost/infer.json \
  $X/bench_cost/infer.json.before-newbaselines
python -u -m inpaper_utils.bench_infer --n 256 --batch 64 --only RevGAN,RectFlow
