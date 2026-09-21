#!/usr/bin/env bash
# Every flow figure with the morph-smooth model: grayscale (with the full-test summary
# the paper text quotes) and DEC-FA. CPU.
#   sbatch inpaper_utils/flowviz_smooth.sbatch
#SBATCH --job-name=flowviz-smooth
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
python -m py_compile inpaper_utils/make_flow_viz.py
python -u -W ignore -m inpaper_utils.make_flow_viz --dataset adni --variant morph_smooth
python -u -W ignore -m inpaper_utils.make_flow_viz --dataset adni --variant morph_smooth --dec-fa \
  --figures features,features_bi,affine,cycle --skip-summary
