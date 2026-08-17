#!/usr/bin/env bash
# Train the "morph" variant.  Extra flags pass straight through:
#     bash configs/run_morph.sh --e3 200
#     CUDA_VISIBLE_DEVICES=1 bash configs/run_morph.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run morph "$@"
