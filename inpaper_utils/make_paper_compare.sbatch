#!/usr/bin/env bash
#SBATCH --job-name=cycleflow-papercmp
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%j.log
set -euo pipefail
source configs/_env.sh
# Override per submission, e.g.
#   sbatch --export=ALL,DATASETS=mnist,BATCH=1000 --time=02:30:00 inpaper_utils/make_paper_compare.sbatch
for ds in ${DATASETS:-adni mnist}; do
  python -u -m inpaper_utils.make_paper_compare infer --dataset "$ds" --batch "${BATCH:-500}"
done
