#!/usr/bin/env bash
# Figures 3 and 4 of the paper (MNIST-PET/CT), mirroring Figures 1 and 2 (ADNI):
#   Fig 3  comparison, "Ours" = FIG_TAG, examples picked automatically (top margin
#          over the best baseline, distinct digits);
#   Fig 4  per-block path of FIG_TAG on the first two examples per direction of Fig 3.
# FIG_TAG is the checkpoint the figures draw. It may differ from Table 1's morph-smooth row:
# the figures show a model that reproduces the PET acquisition grain (grain_probe.py).
# The MNIST ablation figure (Table 1 variants) is re-rendered on the same examples.
# Baseline predictions are copied from a previous comparison cache (REUSE).
#   sbatch --export=ALL,REUSE=<cache dir>,FIG_TAG=<tag> inpaper_utils/fig34_mnist.sbatch
#SBATCH --job-name=fig34-mnist
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
: "${REUSE:?set REUSE=<previous mnist paper_compare cache dir>}" "${FIG_TAG:?set FIG_TAG=<checkpoint tag>}"
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
export MNIST_PAIRED_ROOT=$SLURM_SCRATCH/mnist_petct_paired
M=inpaper_utils
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "figure model: $FIG_TAG"
python -u -W ignore -m $M.make_paper_compare infer  --dataset mnist --reuse "$REUSE" --ours-tag "$FIG_TAG"
python -u -W ignore -m $M.make_paper_compare render --dataset mnist
PICK=$(python - <<'PY'
import glob, json, os
m = sorted(glob.glob('/ix/lzhan/siyuan/exps/CycleFlow/mnist/paper_compare/*/00_inpaper_fig_compare_manifest.json'),
           key=os.path.getmtime)[-1]
print(';'.join(','.join(str(r['test_index']) for r in v) for v in json.load(open(m))['rows'].values()))
PY
)
echo "MNIST picks: $PICK"
python -u -W ignore -m $M.make_paper_ablation      --dataset mnist --pick "$PICK"
python -u -W ignore -m $M.make_paper_path_ablation --dataset mnist --pick "$PICK" --tags "Ours=$FIG_TAG" --n 2 \
  --suffix _ab --no-row-label --panel-labels a,b
