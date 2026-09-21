#!/usr/bin/env bash
# DEC-FA preview of every ADNI paper figure that shows FA. V1 exists for three
# TRAINING subjects only, so the examples are their slices (--samples v1) and the
# renders go to snapshot_results/adni/decfa_preview/, never over the paper figures.
# GPU for the comparison figure's samplers (DiT is 1000 ancestral steps); the rest is cheap.
#   sbatch inpaper_utils/decfa_preview.sbatch
#SBATCH --job-name=decfa-preview
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
export ADNI_V1_CACHE="$MMCLAST_ROOT/data/v1_112.pt"
python -u -m inpaper_utils.make_paper_compare infer  --dataset adni --samples v1 --batch 64
python -u -m inpaper_utils.make_paper_compare render --dataset adni --samples v1 --dec-fa
python -u -m inpaper_utils.make_paper_ablation      --dataset adni --samples v1 --dec-fa
python -u -m inpaper_utils.make_paper_path_ablation --dataset adni --samples v1 --dec-fa
for v in morph morph_bi; do
  python -u -m inpaper_utils.make_flow_viz --dataset adni --variant $v --samples v1 --dec-fa \
    --figures features,features_bi,affine,cycle --n-bi 3 --skip-summary
done
