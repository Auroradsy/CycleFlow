#!/usr/bin/env bash
# Submit the MNIST-PET/CT sensitivity grid: one knob at a time around the Table 2
# build (hotspot_weight 0.8, pet_noise 0.05), which is re-run through the same
# chain so every point of a curve comes from one pipeline. Only sbatch runs here.
#
# Two phases, because this Slurm has no cross-cluster dependencies (htc -> gpu):
#   PHASE=data  bash inpaper_utils/sens_mnist_submit.sh   # htc: build every dataset
#   PHASE=train bash inpaper_utils/sens_mnist_submit.sh   # gpu: host -> base -> smooth, chained
# SETTINGS="hw:noise ..." restricts either phase to some settings.
set -euo pipefail
cd "$(dirname "$0")/.."
SETTINGS=${SETTINGS:-"0.8:0.05 0.0:0.05 0.4:0.05 1.6:0.05 0.8:0.025 0.8:0.1"}
D=/ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/sensitivity
for s in $SETTINGS; do
  HW=${s%:*}; NOISE=${s#*:}
  env="ALL,HW=$HW,NOISE=$NOISE"
  case "${PHASE:?set PHASE=data|train}" in
    data)
      d=$(sbatch --parsable --export=$env inpaper_utils/sens_mnist_data.sbatch | cut -d';' -f1)
      echo "hw=$HW noise=$NOISE  data $d (htc)" ;;
    train)
      tarball=$D/mnist_petct_paired_hw${HW}_ns${NOISE}.tar.gz
      [ -f "$tarball" ] || { echo "hw=$HW noise=$NOISE  SKIP: $tarball not built yet"; continue; }
      h=$(sbatch --parsable --export=$env,STEP=host inpaper_utils/sens_mnist_train.sbatch | cut -d';' -f1)
      b=$(sbatch --parsable --export=$env,STEP=base   --dependency=afterok:$h inpaper_utils/sens_mnist_train.sbatch | cut -d';' -f1)
      m=$(sbatch --parsable --export=$env,STEP=smooth --dependency=afterok:$b inpaper_utils/sens_mnist_train.sbatch | cut -d';' -f1)
      echo "hw=$HW noise=$NOISE  host $h  base $b  smooth $m (gpu)" ;;
    *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
  esac
done
