#!/usr/bin/env bash
# MNIST-PET/CT sensitivity analysis, step 1 of 2: build one paired dataset with one
# rendering knob changed, and its reference rows (copy floor, stochastic ceiling).
# Written as a tarball, because training jobs untar it to node-local scratch.
#   sbatch --export=ALL,HW=0.8,NOISE=0.05 inpaper_utils/sens_mnist_data.sbatch
# Everything else is the build Table 2 uses: seed 0, 12000 paired train / 5000 test, 64 px.
#SBATCH --job-name=sens-data
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
: "${HW:?set HW}" "${NOISE:?set NOISE}"
NAME=mnist_petct_paired_hw${HW}_ns${NOISE}
D=/ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/sensitivity
# Raw MNIST as fetched and verified by inpaper_utils/mnist_raw_check.sbatch.
RAW="--mnist_raw ${MNIST_RAW:-/ix/lzhan/siyuan/datasets/raw/MNIST/raw}"
mkdir -p "$D"
python -u -m data.make_mnist_petct --out "$SLURM_SCRATCH/$NAME" --paired_train \
  --hotspot_weight "$HW" --pet_noise "$NOISE" $RAW
# Same digits, same hotspots, another noise draw (--seed 1 moves only the rng).
python -u -m data.make_mnist_petct --out "$SLURM_SCRATCH/alt" --paired_train --n_train 10 \
  --hotspot_weight "$HW" --pet_noise "$NOISE" --seed 1 $RAW
python -u -m inpaper_utils.sens_mnist_refs --root "$SLURM_SCRATCH/$NAME" --alt "$SLURM_SCRATCH/alt" \
  --out "$D/$NAME.refs.json"
tar -czf "$D/$NAME.tar.gz.part" -C "$SLURM_SCRATCH" "$NAME"
mv "$D/$NAME.tar.gz.part" "$D/$NAME.tar.gz"
echo "done -> $D/$NAME.tar.gz"
