#!/usr/bin/env bash
# CycleGAN baselines trained natively in the B->A direction (FA->T1, PET->CT).
# Same recipe as the forward hosts; --swap_domains exchanges the domains at the
# data source, so the generator saved as G_T1toFA is the native B->A model.
#   sbatch --export=ALL,DATASET=adni  inpaper_utils/train_cyclegan_rev.sbatch
#   sbatch --export=ALL,DATASET=mnist inpaper_utils/train_cyclegan_rev.sbatch
#SBATCH --job-name=cyclegan-rev
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
case "${DATASET:?set DATASET=adni|mnist}" in
  adni)   # forward host: train_host.py defaults (160 epochs, batch 12), all slices
    HOST_TAG=host_rev python -u train_host.py --swap_domains ;;
  mnist)  # forward host args (mnist_host/train.log) on the paired build used by Table 2.
    # With 8 cores and /ix reads, the two 8-worker PNG loaders starved the GPU
    # (0% util, 420 s/epoch): 16 cores and a node-local copy give 58 s/epoch.
    # Untar rather than cp -r: copying the ~39k small files off /ix took 30 min.
    tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz \
        -C "$SLURM_SCRATCH"
    HOST_TAG=host_rev python -u train_host.py --swap_domains --data folder \
      --data_root "$SLURM_SCRATCH/mnist_petct_paired" \
      --img_ch 3 --load_size 72 --crop_size 64 --no_flip \
      --epochs 100 --decay_start 50 --batch 64 --workers 8 ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac
