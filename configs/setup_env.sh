#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CYCLEFLOW_ENV:-/ix/lzhan/siyuan/envs/cycleflow}"
BASE_PYTHON="${CYCLEFLOW_BASE_PYTHON:-/ix/lzhan/siyuan/envs/baselines_ultra/bin/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/ix/lzhan/siyuan/envs/.uv-cache}"
# Reuse the existing CUDA stack read-only; install small project dependencies
# into this dedicated venv. The base environment must remain available.
if [ ! -x "$ENV_DIR/bin/python" ]; then
  uv venv --python "$BASE_PYTHON" --system-site-packages "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --cache-dir "$UV_CACHE_DIR/pip" -r "$ROOT/requirements.txt"
"$ENV_DIR/bin/python" -c 'import torch, torchvision, numpy, yaml, nibabel, scipy, skimage, matplotlib, pytorch_fid; print("torch", torch.__version__, "CUDA build", torch.version.cuda, "GPU visible", torch.cuda.is_available())'
"$ENV_DIR/bin/python" -m pip --cache-dir "$UV_CACHE_DIR/pip" check
"$ENV_DIR/bin/python" -m pip freeze > "$ENV_DIR/requirements.freeze.txt"
