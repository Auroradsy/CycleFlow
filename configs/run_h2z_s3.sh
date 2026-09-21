#!/usr/bin/env bash
# horse2zebra S3 ablation — three ways to supervise the path, one shared S1+S2.
#
#   abs     D_mix with an absolute target (fake -> 1).  The ADNI baseline.
#   ra      relativistic D_mix.  An intermediate frame can never enter either
#           real set, so `fake -> 1` is unreachable; ranking near the reals is.
#   smooth  no adversarial term at all.  On ADNI this arm did not move
#           (mean|d| 0.027 at the last frame against 0.153 for abs) because
#           w_path_smooth is a one-sided penalty whose optimum is "all frames
#           identical".  It is run here as the control that makes the ablation
#           honest -- and because D_mix is better posed on horse2zebra than on
#           T1/FA, so the arm deserves a fresh test rather than an assumption.
#
# All three resume the SAME stage-2 checkpoint, so S3 is the only difference.
# That is the control the ADNI ablation lacked: there, each variant ran its own
# S1 with its own early stop, so "morph vs latcyc" also compared two encoders.
#
# One arm per GPU at a time (S3 at batch 8 with D_mix needs ~24 GB); the third
# starts as soon as a card frees.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
FROM="${FROM:-$MMCLAST_EXPS/horse2zebra/checkpoints/h2z_latcyc/stage2.pth}"
GPUS="${GPUS:-0 1}"

if [ ! -f "$FROM" ]; then
  echo "no shared S2 at $FROM -- run round 1 first:" >&2
  echo "  python train.py --config configs/h2z_latcyc.yaml" >&2
  exit 1
fi

arm () {  # arm <gpu> <tag> <extra...>
  local gpu="$1" tag="$2"; shift 2
  echo "[$(date +%H:%M:%S)] start $tag on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train.py --config configs/h2z_morph.yaml \
      --tag "$tag" --resume_stage 2 --resume_from "$FROM" "$@"
  echo "[$(date +%H:%M:%S)] done  $tag"
}

set -- $GPUS
A="$1"; B="${2:-$1}"

# ra and abs go first, one per card; smooth follows on whichever finishes first.
( arm "$A" h2z_morph_ra  --path_gan_mode ra
  arm "$A" h2z_morph_smooth --w_path_gan 0 --w_path_smooth 1.0 ) &
P1=$!
( arm "$B" h2z_morph_abs ) &
P2=$!
wait $P1 $P2
echo "done -> $MMCLAST_EXPS/horse2zebra/checkpoints/{h2z_morph_abs,h2z_morph_ra,h2z_morph_smooth}"
