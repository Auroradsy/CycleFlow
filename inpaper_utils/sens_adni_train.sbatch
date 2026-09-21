#!/usr/bin/env bash
# ADNI FA-noise sensitivity: one training step for one noise level. The noise is added
# inside PairedADNISliceDataset (env ADNI_FA_NOISE), so the host, MMCLAST-cg and the
# Table 2 evaluation all see the same noisy FA.
#   STEP=host    CycleGAN host with the Table 2 recipe (train_host.py defaults)
#   STEP=base    MMCLAST-cg base, all three stages, then eval_mmclast
#   STEP=smooth  morph-smooth: base's stage 2, stage 3 with the smoothness path term only
#   sbatch --export=ALL,NOISE=0.05,STEP=host inpaper_utils/sens_adni_train.sbatch
# inpaper_utils/sens_adni_submit.sh chains host -> base -> smooth per noise level.
#SBATCH --job-name=sens-adni
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
: "${NOISE:?set NOISE}" "${STEP:?set STEP=host|base|smooth}"
export ADNI_FA_NOISE=$NOISE ADNI_FA_NOISE_SEED=0
S=ns${NOISE}
X=/ix/lzhan/siyuan/exps/CycleFlow/adni/checkpoints
HOST=$X/sens_adni_${S}_host/last.pth
echo "ADNI FA noise $NOISE  step $STEP"
case "$STEP" in
  host)
    HOST_TAG=sens_adni_${S}_host python -u train_host.py ;;
  base)
    python -u train.py --variant base --tag sens_adni_${S}_base --warm "$HOST"
    python -u -m inpaper_utils.eval_mmclast --dataset adni --ckpt "$X/sens_adni_${S}_base/model.pth" ;;
  smooth)
    python -u train.py --variant morph --tag sens_adni_${S}_morph_smooth --warm "$HOST" \
      --resume_stage 2 --resume_from "$X/sens_adni_${S}_base/stage2.pth" \
      --w_path_gan 0 --w_path_smooth 1.0
    python -u -m inpaper_utils.eval_mmclast --dataset adni --ckpt "$X/sens_adni_${S}_morph_smooth/model.pth" ;;
  *) echo "unknown STEP=$STEP" >&2; exit 2 ;;
esac
