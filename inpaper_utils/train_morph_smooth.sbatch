#!/usr/bin/env bash
# MNIST-PET/CT morph without D_mix (only the smoothness term of L_path) -- the
# counterpart of ADNI's morph_smooth. Like that run, it resumes the morph run's
# stage-2 checkpoint and redoes stage 3 with w_path_gan = 0, so it differs from
# morph only in D_mix. Ends with the Table 2 evaluation.
#   sbatch inpaper_utils/train_morph_smooth.sbatch
#SBATCH --job-name=morph-smooth
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
X=/ix/lzhan/siyuan/exps/CycleFlow
python -u train.py --config configs/mnist_p_morph.yaml --tag mnist_p_morph_smooth \
  --resume_stage 2 --resume_from "$X/mnist_p_morph/stage2.pth" --warm "$X/mnist_host/last.pth" \
  --w_path_gan 0 --w_path_smooth 1.0
python -u -m inpaper_utils.eval_mmclast --dataset mnist \
  --ckpt "$X/mnist/checkpoints/mnist_p_morph_smooth/model.pth"
