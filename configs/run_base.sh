#!/usr/bin/env bash
# Train the "base" variant.  Extra flags pass straight through:
#     bash configs/run_base.sh --e3 200
#     CUDA_VISIBLE_DEVICES=1 bash configs/run_base.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run base "$@"
