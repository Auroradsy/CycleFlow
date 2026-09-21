#!/usr/bin/env bash
# Comparison figures with "Ours" = morph-smooth (make_paper_compare.OURS), plus the ADNI
# ablation / path-ablation figures on the examples the new comparison figure draws, in
# grayscale and DEC-FA. MNIST keeps its fixed first-three-pairs selection.
#   sbatch inpaper_utils/compare_smooth.sbatch
#SBATCH --job-name=compare-smooth
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
M=inpaper_utils
python -u -W ignore -m $M.make_paper_compare infer  --dataset adni
python -u -W ignore -m $M.make_paper_compare render --dataset adni
python -u -W ignore -m $M.make_paper_compare render --dataset adni --dec-fa
PICK=$(python - <<'EOF'
import glob, json, os
m = sorted(glob.glob('/ix/lzhan/siyuan/exps/CycleFlow/adni/paper_compare/*/00_inpaper_fig_compare_manifest.json'),
           key=os.path.getmtime)[-1]
print(';'.join(','.join(str(r['test_index']) for r in v) for v in json.load(open(m))['rows'].values()))
EOF
)
echo "ADNI picks: $PICK"
for dec in "" "--dec-fa"; do
  python -u -W ignore -m $M.make_paper_ablation      --dataset adni --pick "$PICK" $dec
  python -u -W ignore -m $M.make_paper_path_ablation --dataset adni --pick "$PICK" $dec
done
python -u -W ignore -m $M.make_paper_compare infer  --dataset mnist
python -u -W ignore -m $M.make_paper_compare render --dataset mnist --pick '0,1,2;3,4,5'
