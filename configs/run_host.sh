#!/usr/bin/env bash
# Train the plain CycleGAN host that train.py splits into E/D.
# Only needed if you do not already have exps/checkpoints/host/last.pth.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
export HOST_TAG="${HOST_TAG:-host}"
mkdir -p "$MMCLAST_EXPS/logs/$HOST_TAG"
python train_host.py "$@" 2>&1 | tee "$MMCLAST_EXPS/logs/$HOST_TAG/train.log"
