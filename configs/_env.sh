# Sourced by every run script.  Edit this one file, not the six below.
#
# The repo root, whatever directory you launch from:
export MMCLAST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MMCLAST_ROOT"

# Conda env with a CUDA build of torch.  Comment out if you manage envs some
# other way; nothing else here depends on conda.
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  set +u                      # conda.sh reads PS1, which `set -u` treats as fatal
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${MMCLAST_ENV:-brain}"
  set -u
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# run <tag> <extra args...>  — train, teeing stdout next to the epoch csv.
run () {
  local tag="$1"; shift
  mkdir -p "logs/$tag"
  echo "[$(date '+%F %T')] $tag on GPU $CUDA_VISIBLE_DEVICES -> logs/$tag/train.log"
  python train.py --config "configs/$tag.yaml" "$@" 2>&1 | tee "logs/$tag/train.log"
}
