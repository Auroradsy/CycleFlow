#!/usr/bin/env bash
# Everything downstream of training: the calibrated probe, then the figures.
# Writes snapshot_results/{30,31,32}_*.png and checkpoints/<tag>/probe_eval.txt.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

for t in base latcyc morph; do
  [ -f "checkpoints/$t/model.pth" ] || { echo "skip $t (not trained)"; continue; }
  python eval.py --tag "$t"
  python -m utils.plot_morph --tag "$t"
done
python -m utils.plot_ablation --tags base latcyc morph
