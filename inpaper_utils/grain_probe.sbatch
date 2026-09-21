#!/usr/bin/env bash
# Which MNIST checkpoint shows the PET acquisition grain appearing along CT->PET?
# Grain ratios per checkpoint and per flow state (inpaper_utils/grain_probe.py), plus a
# candidate path figure (rows = checkpoints) on the first two CT->PET pairs of Figure 3.
#   sbatch inpaper_utils/grain_probe.sbatch
#SBATCH --job-name=grain-probe
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -euo pipefail
source configs/_env.sh
tar -xzf /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz -C "$SLURM_SCRATCH"
export MNIST_PAIRED_ROOT=$SLURM_SCRATCH/mnist_petct_paired
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
M=inpaper_utils
C=mnist/checkpoints
python -u -W ignore -m $M.grain_probe --dataset mnist --n 1000
python -u -W ignore -m $M.make_paper_path_ablation --dataset mnist --n 2 --pick "1593,2254;3768,2539" \
  --directions a2b --suffix _grain_candidates --out-dir /ix/lzhan/siyuan/exps/CycleFlow/mnist/grain_probe/figures \
  --tags "g1 s1 (orig)=$C/mnist_p_morph_smooth;g0.1 s0.3 (tab)=$C/tune_mnist_smooth_g0.1_s0.3;g0.3 s0.3=$C/tune_mnist_smooth_g0.3_s0.3;g1 s0.3=$C/tune_mnist_smooth_g1.0_s0.3;morph=mnist_p_morph;base=mnist_p_base"
