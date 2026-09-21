# Shared server setup. Every Python entry records its own timestamped run.
export MMCLAST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MMCLAST_ROOT"
export CYCLEFLOW_ENV="${CYCLEFLOW_ENV:-/ix/lzhan/siyuan/envs/cycleflow}"
if [ ! -x "$CYCLEFLOW_ENV/bin/python" ]; then
  echo "Missing environment: run bash configs/setup_env.sh" >&2
  return 1
fi
export PATH="$CYCLEFLOW_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MMCLAST_EXPS="${MMCLAST_EXPS:-/ix/lzhan/siyuan/exps/CycleFlow}"
export ADNI_ROOT="${ADNI_ROOT:-/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow}"
export ADNI_CACHE="${ADNI_CACHE:-$ADNI_ROOT/paired_112.pt}"
export ADNI_LABELS="${ADNI_LABELS:-$ADNI_ROOT/labels.csv}"
export TORCH_HOME="${TORCH_HOME:-$MMCLAST_EXPS/_cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$MMCLAST_EXPS/_cache/matplotlib}"
# Preserve scheduler-provided GPU visibility; do not assign a GPU on login nodes.
run () {
  local tag="$1"; shift
  python train.py --config "configs/$tag.yaml" "$@"
}
