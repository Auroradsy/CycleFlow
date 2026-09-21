#!/usr/bin/env bash
# Before the sensitivity grid: find (or fetch) raw MNIST, rebuild the Table 2 build
# with default knobs, and check it matches the existing mnist_petct_paired pixel for
# pixel. If it does not, sensitivity variants would not be comparable to Table 2.
#   sbatch inpaper_utils/mnist_raw_check.sbatch
#SBATCH --job-name=mnist-raw-check
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -uo pipefail
source configs/_env.sh
RAWD=/ix/lzhan/siyuan/datasets/raw/MNIST/raw
FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
if [ ! -f "$RAWD/t10k-images-idx3-ubyte" ]; then
  # Only the places a copy would plausibly be; a sweep of the 4 TB /ix tree takes hours.
  hit=$(find /ix/lzhan/siyuan/datasets /ix/lzhan/siyuan/exps/CycleFlow/_cache /vast/lzhan/siyuan \
        -maxdepth 5 -name "t10k-images-idx3-ubyte*" -not -path "$RAWD/*" 2>/dev/null | head -1)
  echo "search hit: ${hit:-none}"
  mkdir -p "$RAWD"
  if [ -n "$hit" ]; then
    src=$(dirname "$hit")
    for f in $FILES; do
      if [ -f "$src/$f" ]; then cp "$src/$f" "$RAWD/"; else cp "$src/$f.gz" "$RAWD/" && gunzip -f "$RAWD/$f.gz"; fi
    done
  else
    for f in $FILES; do
      curl -fsSL --retry 3 -o "$RAWD/$f.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" \
        && gunzip -f "$RAWD/$f.gz" || echo "download failed: $f"
    done
  fi
fi
ls -la "$RAWD"; sha256sum "$RAWD"/* 2>/dev/null
[ -f "$RAWD/t10k-images-idx3-ubyte" ] || { echo "no raw MNIST available"; exit 1; }

T=$SLURM_SCRATCH/rebuild
python -u -m data.make_mnist_petct --mnist_raw "$RAWD" --out "$T" --paired_train --aux_nohot
python -u - "$T" <<'EOF'
import os, sys
import numpy as np
from PIL import Image
D = "/ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired"
T = sys.argv[1]
ok = True
for s in ("trainA", "trainB", "testA", "testB", "testB_nohot"):
    a, b = sorted(os.listdir(f"{D}/{s}")), sorted(os.listdir(f"{T}/{s}"))
    if a != b:
        print(f"{s}: file lists differ ({len(a)} vs {len(b)})"); ok = False; continue
    nd, worst = 0, 0
    for n in a:
        d = np.abs(np.asarray(Image.open(f"{D}/{s}/{n}"), int) - np.asarray(Image.open(f"{T}/{s}/{n}"), int)).max()
        nd += int(d > 0); worst = max(worst, int(d))
    print(f"{s}: {len(a)} files, {nd} differ, max pixel diff {worst}")
    ok &= nd == 0
print("REBUILD MATCHES" if ok else "REBUILD DIFFERS")
EOF
