#!/usr/bin/env bash
# DEC-FA pilot: recompute V1 from raw DWI for a few subjects and bring it into the
# FMRIB58 1mm template space with the EXISTING FA_to_template.mat (vecreg rotates
# vectors).  Two variants (raw / eddy_correct) so we can tell which one reproduces
# the FA we trained on.
#   bash data/preprocess/dec_fa_pilot.sh 002_S_0413_I863064 002_S_1155_I843517 ...
set -euo pipefail
B=/data_new3/nfs_share/public/Imaging_genetic
OUT=$(cd "$(dirname "$0")/../cache" && pwd)/dec_fa_pilot
REF=$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz

one() {
  sess=$1; subj=${sess%_I*}; img=${sess##*_}
  raw=$B/DTI/dti_nii/$subj/$img; mat=$B/registered_DTI/$sess/${sess}_FA_to_template.mat
  d=$OUT/$subj; mkdir -p "$d"; cd "$d"
  fslroi "$raw/DTI.nii.gz" b0 0 1
  bet b0 b0_brain -f 0.3 -m
  for v in raw eddy; do
    if [ $v = eddy ]; then
      [ -f dwi_eddy.nii.gz ] || eddy_correct "$raw/DTI.nii.gz" dwi_eddy 0 > /dev/null
      dwi=dwi_eddy
    else dwi=$raw/DTI.nii.gz; fi
    dtifit -k $dwi -o dti_$v -m b0_brain_mask -r "$raw/DTI.bvec" -b "$raw/DTI.bval" > /dev/null
    flirt -in dti_${v}_FA -ref $REF -applyxfm -init "$mat" -out FA_${v}_reg
    vecreg -i dti_${v}_V1 -o V1_${v}_reg -r $REF -t "$mat"
  done
  echo "[done] $subj"
}
export -f one; export B OUT REF
printf "%s\n" "$@" | xargs -P 3 -I{} bash -c 'one {}'
