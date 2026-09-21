#!/usr/bin/env bash
# DEC-FA versions of the ADNI paper figures, on the same held-out examples as the
# grayscale ones. Needs v1_112.pt with V1 for the test subjects (inpaper_utils/v1_all.sbatch).
# Writes *_decfa.{pdf,png} next to the grayscale figures in snapshot_results/adni/;
# nothing grayscale is overwritten. CPU (the comparison figure reuses the infer cache).
#   sbatch inpaper_utils/decfa_paper.sbatch
#SBATCH --job-name=decfa-paper
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
export ADNI_V1_CACHE=/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow/v1_112.pt
python -u -W ignore -m inpaper_utils.make_paper_compare render --dataset adni --dec-fa
# Ablation and path ablation on exactly the examples the DEC-FA comparison just drew.
PICK=$(python - <<'EOF'
import glob, json, os
m = sorted(glob.glob('/ix/lzhan/siyuan/exps/CycleFlow/adni/paper_compare/*/00_inpaper_fig_compare_decfa_manifest.json'),
           key=os.path.getmtime)[-1]
rows = list(json.load(open(m))['rows'].values())
print(';'.join(','.join(str(r['test_index']) for r in v) for v in rows))
EOF
)
echo "picks: $PICK"
python -u -W ignore -m inpaper_utils.make_paper_ablation      --dataset adni --dec-fa --pick "$PICK"
python -u -W ignore -m inpaper_utils.make_paper_path_ablation --dataset adni --dec-fa --pick "$PICK"
python -u -W ignore -m inpaper_utils.make_flow_viz --dataset adni --variant morph --dec-fa \
  --figures features,features_bi,affine,cycle --skip-summary
