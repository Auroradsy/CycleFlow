#!/usr/bin/env bash
# Training cost of the two ADDED rows only, on one A100, with the same 4-epoch
# rule as inpaper_utils/bench_train.sbatch (which now also carries these two lines,
# for the case where the whole table is re-measured at once).
#
# This exists as a separate job because bench_train.sbatch re-runs every method:
# that would allocate new bench_* runs for the published rows as well and move
# their numbers in the paper. Measuring a new row must not disturb an old one.
#
#   sbatch --export=ALL,DATASET=adni  inpaper_utils/bench_newbaselines.sbatch
#   sbatch --export=ALL,DATASET=mnist inpaper_utils/bench_newbaselines.sbatch
#SBATCH --job-name=bench-newbase
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
E=4
nvidia-smi --query-gpu=name --format=csv,noheader
case "${DATASET:?set DATASET=adni|mnist}" in
  adni)
    python -u -m baselines.train_revgan --tag bench_revgan_adni --epochs $E --batch 12 --workers 8
    # --reflow_sim_steps is the real run's: the simulation between the stages costs what
    # it costs whatever the epoch count is, and it is part of RectFlow's training.
    python -u -m baselines.train --method recflow --tag bench_recflow_adni --coupling paired \
      --epochs $E --reflow_epochs $E --reflow_sim_steps 100 --batch 64 --eval_every 1000 ;;
  mnist)
    tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
    F="--data folder --data_root $SLURM_SCRATCH/mnist_petct_paired --img_ch 3"
    python -u -m baselines.train_revgan --tag bench_revgan_mnist $F --load_size 72 --crop_size 64 \
      --no_flip --epochs $E --batch 64 --workers 8
    python -u -m baselines.train --method recflow --tag bench_recflow_mnist $F --coupling paired \
      --epochs $E --reflow_epochs $E --reflow_sim_steps 100 --batch 64 --eval_every 1000 ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac
