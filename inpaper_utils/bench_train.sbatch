#!/usr/bin/env bash
# Training cost of every Table 2 method on ONE A100, with each method's real training
# script, batch size and data. Every run is cut to 4 epochs (per stage for MMCLAST-cg);
# inpaper_utils/bench_cost_collect.py takes the steady epoch time (dropping the warm-up
# epoch and the last one, which evaluates) and multiplies it by the epochs of the real run.
#   sbatch --export=ALL,DATASET=adni  inpaper_utils/bench_train.sbatch
#   sbatch --export=ALL,DATASET=mnist inpaper_utils/bench_train.sbatch
#SBATCH --job-name=bench-train
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
X=/ix/lzhan/siyuan/exps/CycleFlow
E=4
nvidia-smi --query-gpu=name --format=csv,noheader
case "${DATASET:?set DATASET=adni|mnist}" in
  adni)
    HOST_TAG=bench_host_adni python -u train_host.py --epochs $E --batch 12
    for m in cfm meanflow ddpm; do
      b=64; [ $m = cfm ] || b=32          # the ADNI checkpoints' batch sizes
      python -u -m baselines.train --method $m --tag bench_${m}_adni --epochs $E --batch $b --eval_every 1000
    done
    python -u -m baselines.train_dit --tag bench_dit_adni --ae_epochs $E --epochs $E --eval_n 8
    python -u -m baselines.train_revgan --tag bench_revgan_adni --epochs $E --batch 12 --workers 8
    # --reflow_sim_steps is the real run's: the simulation between the stages costs
    # what it costs whatever the epoch count is, and it is part of RectFlow's training.
    python -u -m baselines.train --method recflow --tag bench_recflow_adni --epochs $E \
      --reflow_epochs $E --reflow_sim_steps 100 --batch 64 --eval_every 1000
    python -u train.py --variant morph --tag bench_morph_adni --warm $X/adni_host/last.pth \
      --e1 $E --e2 $E --e3 $E --p1 99 --p2 99 --p3 99
    python -u train.py --variant morph --tag bench_msmooth_adni --warm $X/adni_host/last.pth \
      --resume_stage 2 --resume_from $X/adni/checkpoints/bench_morph_adni/stage2.pth \
      --w_path_gan 0 --w_path_smooth 1.0 --e3 $E --p3 99 ;;
  mnist)
    # node-local copy: small PNGs off /ix starve the loaders (train_cyclegan_rev.sbatch)
    tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
    R=$SLURM_SCRATCH/mnist_petct_paired
    F="--data folder --data_root $R --img_ch 3"
    HOST_TAG=bench_host_mnist python -u train_host.py $F --load_size 72 --crop_size 64 --no_flip \
      --epochs $E --decay_start 50 --batch 64 --workers 8
    for m in cfm meanflow ddpm; do
      python -u -m baselines.train --method $m --tag bench_${m}_mnist $F --epochs $E --batch 64 --eval_every 1000
    done
    python -u -m baselines.train_dit --tag bench_dit_mnist $F --ae_epochs $E --epochs $E --eval_n 8
    python -u -m baselines.train_revgan --tag bench_revgan_mnist $F --load_size 72 --crop_size 64 \
      --no_flip --epochs $E --batch 64 --workers 8
    python -u -m baselines.train --method recflow --tag bench_recflow_mnist $F --epochs $E \
      --reflow_epochs $E --reflow_sim_steps 100 --batch 64 --eval_every 1000
    python -u train.py --config configs/mnist_p_morph.yaml --tag bench_morph_mnist --data_root $R \
      --warm $X/mnist_host/last.pth --e1 $E --e2 $E --e3 $E --p1 99 --p2 99 --p3 99
    python -u train.py --config configs/mnist_p_morph.yaml --tag bench_msmooth_mnist --data_root $R \
      --warm $X/mnist_host/last.pth --resume_stage 2 \
      --resume_from $X/mnist/checkpoints/bench_morph_mnist/stage2.pth \
      --w_path_gan 0 --w_path_smooth 1.0 --e3 $E --p3 99 ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 2 ;;
esac
