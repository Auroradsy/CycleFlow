#!/usr/bin/env bash
# The ADNI baselines of Table 2 that were never migrated from the old server
# (only DiT was). Same testbed as everything else: z=40--49, 112^2, the seed-42
# 170/43 subject split, which are the defaults of baselines/common.add_data_args.
# Sampler settings follow the MNIST runs: CFM 10 Euler steps, DDPM 50 DDIM steps,
# MeanFlow one step. CycleGAN uses train_host.py's defaults, i.e. exactly the
# recipe of the reverse host in train_cyclegan_rev.sbatch without --swap_domains.
#   sbatch --export=ALL,METHOD=cfm       inpaper_utils/train_adni_baselines.sbatch
#   sbatch --export=ALL,METHOD=meanflow  inpaper_utils/train_adni_baselines.sbatch
#   sbatch --export=ALL,METHOD=ddpm      inpaper_utils/train_adni_baselines.sbatch
#   sbatch --export=ALL,METHOD=cyclegan  inpaper_utils/train_adni_baselines.sbatch
#SBATCH --job-name=adni-baseline
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
case "${METHOD:?set METHOD=cfm|meanflow|ddpm|cyclegan}" in
  cfm)      python -u -m baselines.train --method cfm      --tag cfm_adni      --epochs 120 --batch 64 --n_steps 10 ;;
  meanflow) python -u -m baselines.train --method meanflow --tag meanflow_adni --epochs 120 --batch 64 ;;
  ddpm)     python -u -m baselines.train --method ddpm     --tag ddpm_adni     --epochs 120 --batch 64 --n_steps 50 ;;
  cyclegan) HOST_TAG=host python -u train_host.py ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac
