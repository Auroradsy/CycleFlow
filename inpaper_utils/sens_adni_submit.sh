#!/usr/bin/env bash
# Submit the ADNI FA-noise sensitivity grid: the reference rows (htc) and, per noise
# level, host -> base -> morph-smooth chained on the gpu cluster. sigma = 0 is the
# Table 2 data re-run through the same chain. Only sbatch runs here.
#   bash inpaper_utils/sens_adni_submit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SETTINGS=${SETTINGS:-"0.0 0.025 0.05 0.1"}
r=$(sbatch --parsable -M htc -A lzhan -c 4 --mem=16G -t 01:00:00 -J sens-adni-refs \
      -o /ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log \
      --wrap "source configs/_env.sh && python -u -W ignore -m inpaper_utils.sens_adni_refs --noise $SETTINGS" | cut -d';' -f1)
echo "refs $r (htc)"
for s in $SETTINGS; do
  env="ALL,NOISE=$s"
  h=$(sbatch --parsable --export=$env,STEP=host inpaper_utils/sens_adni_train.sbatch | cut -d';' -f1)
  b=$(sbatch --parsable --export=$env,STEP=base   --dependency=afterok:$h inpaper_utils/sens_adni_train.sbatch | cut -d';' -f1)
  m=$(sbatch --parsable --export=$env,STEP=smooth --dependency=afterok:$b inpaper_utils/sens_adni_train.sbatch | cut -d';' -f1)
  echo "noise=$s  host $h  base $b  smooth $m (gpu)"
done
