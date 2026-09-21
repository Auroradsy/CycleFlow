#!/usr/bin/env bash
# The attribution check: split the host, leave f at its identity init, and
# confirm the result scores EXACTLY what plain CycleGAN scored.  If these two
# numbers do not match $MMCLAST_EXPS/adni/checkpoints/host/final_eval.txt, every later gain is
# unattributable and nothing below is worth running.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python train.py --config configs/morph.yaml --check_init "$@"
