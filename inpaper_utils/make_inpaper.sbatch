#!/usr/bin/env bash
#SBATCH --job-name=cycleflow-inpaper
#SBATCH --clusters=gpu
#SBATCH --account=lzhan
#SBATCH --partition=a100
#SBATCH --qos=gpu-a100-s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%j.log
set -euo pipefail
source configs/_env.sh
python -u -m inpaper_utils.make_inpaper --dataset adni --with-dit
python -u -m inpaper_utils.make_inpaper --dataset mnist --with-dit
