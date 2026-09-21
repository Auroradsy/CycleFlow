#!/usr/bin/env bash
# Two-direction feature figure (00_inpaper_flow_features_bi_<variant>) on the
# held-out z=44 examples of the existing flow figures. Renders only that figure,
# so the paper's current flow figures are not rewritten. CPU.
#   sbatch inpaper_utils/flowviz_bi.sbatch
#SBATCH --job-name=flowviz-bi
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
python -m py_compile inpaper_utils/decfa.py inpaper_utils/make_flow_viz.py inpaper_utils/make_paper_compare.py \
  inpaper_utils/make_paper_ablation.py inpaper_utils/make_paper_path_ablation.py
for v in morph morph_bi; do
  python -u -m inpaper_utils.make_flow_viz --dataset adni --variant $v --figures features_bi --n-bi 3 --skip-summary
done
