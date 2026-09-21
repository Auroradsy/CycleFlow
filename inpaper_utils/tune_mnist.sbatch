#!/usr/bin/env bash
# MNIST-PET/CT stage-3 tuning of our two table rows. Both resume the existing
# mnist_p_morph stage-2 checkpoint (S1/S2 share one objective across variants), so a
# config differs only in stage 3. Each run is scored on the validation split (model
# selection, inpaper_utils/tune_mnist_collect.py) and on the test split (reporting).
#   VARIANT=base   G=<w_gan>                    -> Ours w/o Morph-smooth
#   VARIANT=smooth G=<w_gan> S=<w_path_smooth>   -> Ours w/ Morph-smooth (w_path_gan 0)
#   VARIANT=base_own G=<w_gan> C=<w_cross>       -> Ours w/o, resumed from mnist_p_base's own stage 2
#   sbatch --export=ALL,VARIANT=smooth,G=0.3,S=0.3 inpaper_utils/tune_mnist.sbatch
#SBATCH --job-name=tune-mnist
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
: "${VARIANT:?set VARIANT=base|smooth}" "${G:?set G}"
X=/ix/lzhan/siyuan/exps/CycleFlow
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
R=$SLURM_SCRATCH/mnist_petct_paired
case "$VARIANT" in
  base)   TAG=tune_mnist_base_g${G};             CFG=configs/mnist_p_base.yaml;  EXTRA=() ;;
  # base on its OWN stage 2 (mnist_p_base/stage2.pth), with the L_cross weight C as well
  base_own) TAG=tune_mnist_base_own_g${G}_c${C:?set C}; CFG=configs/mnist_p_base.yaml
            S2=$X/mnist_p_base/stage2.pth; EXTRA=(--w_cross "$C") ;;
  smooth) TAG=tune_mnist_smooth_g${G}_s${S:?set S}; CFG=configs/mnist_p_morph.yaml; EXTRA=(--w_path_gan 0 --w_path_smooth "$S") ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac
echo "tag $TAG"
python -u train.py --config $CFG --tag $TAG --data_root "$R" --warm $X/mnist_host/last.pth --workers 8 \
  --resume_stage 2 --resume_from "${S2:-$X/mnist_p_morph/stage2.pth}" --w_gan "$G" "${EXTRA[@]}"
C=$X/mnist/checkpoints/$TAG/model.pth
python -u -m inpaper_utils.eval_mmclast --dataset mnist --data-root "$R" --split val --ckpt "$C"
python -u -m inpaper_utils.eval_mmclast --dataset mnist --data-root "$R" --ckpt "$C"
