#!/usr/bin/env bash
# Train the "morph_bi" variant (bidirectional L_path).  Extra flags pass straight through:
#     bash configs/run_morph_bi.sh --e3 200
#     CUDA_VISIBLE_DEVICES=1 bash configs/run_morph_bi.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run morph_bi "$@"
