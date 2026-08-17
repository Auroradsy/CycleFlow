#!/usr/bin/env bash
# All three variants.  The ablation only means anything if they are trained
# identically, so this is the script that should produce the paper's numbers —
# never three hand-typed commands with drifting flags.
#
#   bash configs/run_all.sh              # sequential on one GPU
#   PARALLEL=1 bash configs/run_all.sh   # one variant per GPU (needs 3)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

bash configs/check_init.sh

if [ "${PARALLEL:-0}" = "1" ]; then
  i=0
  for t in base latcyc morph; do
    CUDA_VISIBLE_DEVICES=$i bash "configs/run_$t.sh" &
    i=$((i + 1))
  done
  wait
else
  for t in base latcyc morph; do bash "configs/run_$t.sh"; done
fi

bash configs/make_figures.sh
