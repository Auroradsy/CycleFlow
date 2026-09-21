#!/usr/bin/env bash
# MNIST-PET/CT sensitivity analysis, step 2 of 2: one training step on one build.
#   STEP=host    CycleGAN host (the mnist_host recipe) that MMCLAST-cg warm-starts from
#   STEP=base    MMCLAST-cg base, all three stages, then the Table 2 evaluation
#   STEP=smooth  morph-smooth: resumes base's stage 2 and redoes stage 3 with the
#                smoothness path term only (w_path_gan 0). S1/S2 have the same
#                objective in both variants, so the two differ in stage 3 alone.
#   sbatch --export=ALL,HW=0.8,NOISE=0.05,STEP=host inpaper_utils/sens_mnist_train.sbatch
# inpaper_utils/sens_mnist_submit.sh chains data -> host -> base -> smooth per setting.
#SBATCH --job-name=sens-train
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
: "${HW:?set HW}" "${NOISE:?set NOISE}" "${STEP:?set STEP=host|base|smooth}"
S=hw${HW}_ns${NOISE}
NAME=mnist_petct_paired_$S
X=/ix/lzhan/siyuan/exps/CycleFlow/mnist/checkpoints
# Small PNGs off /ix starve the loaders; node-local copy, as in train_cyclegan_rev.sbatch.
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/sensitivity/$NAME.tar.gz -C "$SLURM_SCRATCH"
R=$SLURM_SCRATCH/$NAME
HOST=$X/sens_${S}_host/last.pth
echo "setting $S  step $STEP  data $R"
case "$STEP" in
  host)
    HOST_TAG=sens_${S}_host python -u train_host.py --data folder --data_root "$R" \
      --img_ch 3 --load_size 72 --crop_size 64 --no_flip \
      --epochs 100 --decay_start 50 --batch 64 --workers 8 ;;
  base)
    python -u train.py --config configs/mnist_p_base.yaml --tag sens_${S}_base \
      --data_root "$R" --warm "$HOST" --workers 8
    python -u -m inpaper_utils.eval_mmclast --dataset mnist --data-root "$R" \
      --ckpt "$X/sens_${S}_base/model.pth" ;;
  smooth)
    python -u train.py --config configs/mnist_p_morph.yaml --tag sens_${S}_morph_smooth \
      --data_root "$R" --warm "$HOST" --workers 8 \
      --resume_stage 2 --resume_from "$X/sens_${S}_base/stage2.pth" \
      --w_path_gan 0 --w_path_smooth 1.0
    python -u -m inpaper_utils.eval_mmclast --dataset mnist --data-root "$R" \
      --ckpt "$X/sens_${S}_morph_smooth/model.pth" ;;
  *) echo "unknown STEP=$STEP" >&2; exit 2 ;;
esac
