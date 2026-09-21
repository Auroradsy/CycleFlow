#!/usr/bin/env bash
# Score the two added baselines on the FULL held-out split of each testbed, into
# the same cache the comparison figures and the result tables are read from.
#
# --reuse copies the predictions of the methods that were already inferred (by
# checkpoint tag) instead of re-sampling them, which matters: DiT's 1000-step
# ancestral sampler over 5,000 MNIST pairs is most of this job's cost otherwise.
# Our own model is re-run regardless, because `--reuse` only covers baselines.
#
#   sbatch --export=ALL,DATASET=mnist inpaper_utils/compare_newbaselines.sbatch
#   sbatch --export=ALL,DATASET=adni  inpaper_utils/compare_newbaselines.sbatch
#SBATCH --job-name=cmp-newbase
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=l40s
#SBATCH --qos=gpu-l40s-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
X=/ix/lzhan/siyuan/exps/CycleFlow
: "${DATASET:?set DATASET=adni|mnist}"
REUSE=$(ls -d $X/$DATASET/paper_compare/*/cache 2>/dev/null | tail -1)
echo "reusing baseline predictions from: ${REUSE:-<none>}"
if [ "$DATASET" = mnist ]; then
  tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
  export MNIST_PAIRED_ROOT=$SLURM_SCRATCH/mnist_petct_paired
fi
python -u -m inpaper_utils.make_paper_compare infer --dataset "$DATASET" \
  ${REUSE:+--reuse "$REUSE"}
