#!/usr/bin/env bash
# The two baselines added after the first submission: RevGAN (reversible core
# shared by the two directions, CycleGAN's objective) and RectFlow (rectified
# flow with one reflow, one Euler step at inference).
#
# Each one inherits the schedule of the row it is closest to, so the table stays
# a comparison of methods rather than of budgets:
#   RevGAN   <- CycleGAN's   (ADNI 160 epochs batch 12 decay 80; MNIST 100/64/50)
#   RectFlow <- CFM's        (ADNI 120 epochs batch 64; MNIST 40/64), twice --
#                            once for the 1-rectified flow and once for the reflow.
#
# RectFlow is trained under both couplings. The reported row uses the PAIRED one,
# because that is the supervision every other flow/diffusion row in the tables gets
# (only CycleGAN and our model train unpaired), so the row isolates what the reflow
# buys over CFM rather than re-measuring the value of the pairing. `recflow_ind_*`
# is the same recipe on the independent coupling -- the setting the paper itself
# runs -- and is kept as the secondary number.
#
#   sbatch --export=ALL,METHOD=revgan,DATASET=adni    inpaper_utils/train_newbaselines.sbatch
#   sbatch --export=ALL,METHOD=revgan,DATASET=mnist   inpaper_utils/train_newbaselines.sbatch
#   sbatch --export=ALL,METHOD=recflow,DATASET=adni   inpaper_utils/train_newbaselines.sbatch
#   sbatch --export=ALL,METHOD=recflow,DATASET=mnist  inpaper_utils/train_newbaselines.sbatch
#   sbatch --export=ALL,METHOD=recflow_ind,DATASET=adni  inpaper_utils/train_newbaselines.sbatch
#   sbatch --export=ALL,METHOD=recflow_ind,DATASET=mnist inpaper_utils/train_newbaselines.sbatch
#SBATCH --job-name=newbase
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
: "${METHOD:?set METHOD=revgan|recflow|recflow_ind}"
: "${DATASET:?set DATASET=adni|mnist}"

if [ "$DATASET" = mnist ]; then
  # node-local copy: the small PNGs off /ix starve the loaders (train_cyclegan_rev.sbatch)
  tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
  F="--data folder --data_root $SLURM_SCRATCH/mnist_petct_paired --img_ch 3 --load_size 72 --crop_size 64"
else
  F=""
fi

case "$METHOD:$DATASET" in
  revgan:adni)
    python -u -m baselines.train_revgan --tag revgan_adni \
      --epochs 160 --decay_start 80 --batch 12 --workers 8 --eval_every 20 ;;
  revgan:mnist)
    python -u -m baselines.train_revgan --tag revgan_mnist $F --no_flip \
      --epochs 100 --decay_start 50 --batch 64 --workers 8 --eval_every 20 ;;
  recflow:adni)
    python -u -m baselines.train --method recflow --tag recflow_adni --coupling paired \
      --epochs 120 --reflow_epochs 120 --batch 64 --reflow_sim_steps 100 --eval_every 20 ;;
  recflow:mnist)
    python -u -m baselines.train --method recflow --tag recflow_mnist $F --coupling paired \
      --epochs 40 --reflow_epochs 40 --batch 64 --reflow_sim_steps 100 --eval_every 10 ;;
  recflow_ind:adni)
    python -u -m baselines.train --method recflow --tag recflow_ind_adni --coupling independent \
      --epochs 120 --reflow_epochs 120 --batch 64 --reflow_sim_steps 100 --eval_every 20 ;;
  recflow_ind:mnist)
    python -u -m baselines.train --method recflow --tag recflow_ind_mnist $F --coupling independent \
      --epochs 40 --reflow_epochs 40 --batch 64 --reflow_sim_steps 100 --eval_every 10 ;;
  *) echo "unknown METHOD=$METHOD DATASET=$DATASET" >&2; exit 2 ;;
esac
