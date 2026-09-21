#!/usr/bin/env bash
# E2.1/E2.2/E2.4 (latent_geometry) and E2.3/E2.5 (domain_probe): where the intermediate
# flow states sit between the two representation manifolds, on the full held-out split.
#   sbatch inpaper_utils/latent_geometry.sbatch
#SBATCH --job-name=latent-geom
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
M=inpaper_utils
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m py_compile $M/latent_geometry.py $M/domain_probe.py

# Both caches are read at random over /ix, which parks the job in D state for many
# minutes; one sequential copy to node-local scratch is far cheaper.
cp "$ADNI_ROOT/paired_112.pt" "$SLURM_SCRATCH"/
export ADNI_CACHE=$SLURM_SCRATCH/paired_112.pt
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
export MNIST_PAIRED_ROOT=$SLURM_SCRATCH/mnist_petct_paired

for ds in ${DATASETS:-adni mnist}; do
  echo "=== $ds ==="
  python -u -W ignore -m $M.latent_geometry --dataset "$ds" --variant morph_smooth --plot
  python -u -W ignore -m $M.domain_probe    --dataset "$ds"
done
