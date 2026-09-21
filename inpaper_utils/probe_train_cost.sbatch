#!/usr/bin/env bash
# Sum the logged per-epoch wall time of every run behind Table 2, and list the
# paper figures that contain FA.  Read-only; CPU.
#SBATCH --job-name=probe-cost
#SBATCH --clusters=htc
#SBATCH --account=lzhan
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/ihome/lzhan/sid51/projects/Brain/CycleFlow
#SBATCH --output=/ix/lzhan/siyuan/exps/CycleFlow/_inpaper_jobs/%x_%j.log
set -uo pipefail
source configs/_env.sh
E=/ix/lzhan/siyuan/exps/CycleFlow
echo "== warm/host in MMCLAST logs"
grep -iE "warm|resumed|params" $E/mnist_p_morph/train.log | head -6
grep -iE "warm|resumed|params" $E/adni_morph_bi/train.log | head -6
echo "== DiT latent-AE time (s)"
for t in dit_adni_scratch dit_mnist; do grep -oE "\[latent-ae\] ep +[0-9]+/[0-9]+ +loss [0-9.]+ +[0-9]+s" $E/$t/train.log | awk -v t=$t '{gsub("s","",$NF); s+=$NF} END {printf "%s AE total %.2f h\n", t, s/3600}'; done
python - <<'EOF'
import csv, os
E = "/ix/lzhan/siyuan/exps/CycleFlow"
logs = {
 "adni_morph (S1-S3)": "adni_morph/train_log.csv", "adni_base": "adni_base/train_log.csv",
 "adni_latcyc": "adni_latcyc/train_log.csv", "adni_morph_bi": "adni_morph_bi/train_log.csv",
 "adni_cyclegan fwd": "_baselines_unpacked/baselines/adni_cyclegan/train_log.csv",
 "adni_cfm": "_baselines_unpacked/baselines/adni_cfm/train_log.csv",
 "adni_meanflow": "_baselines_unpacked/baselines/adni_meanflow/train_log.csv",
 "adni_ddpm": "_baselines_unpacked/baselines/adni_ddpm/train_log.csv",
 "adni_dit": "dit_adni_scratch/train_log.csv",
 "mnist_p_morph (S1-S3)": "mnist_p_morph/train_log.csv", "mnist_p_base": "mnist_p_base/train_log.csv",
 "mnist_p_latcyc": "mnist_p_latcyc/train_log.csv",
 "mnist_host fwd": "mnist_host/train_log.csv",
 "mnist_cfm": "cfm_mnist/train_log.csv", "mnist_meanflow": "meanflow_mnist/train_log.csv",
 "mnist_ddpm": "ddpm_mnist/train_log.csv", "mnist_dit": "dit_mnist/train_log.csv",
}
for name, rel in logs.items():
    p = os.path.join(E, rel)
    if not os.path.exists(p): print(f"MISSING {name}"); continue
    rows = list(csv.DictReader(open(p)))
    sec = [float(r["sec"]) for r in rows]
    by = {}
    if "stage" in rows[0]:
        for r in rows: by[r["stage"]] = by.get(r["stage"], 0) + float(r["sec"])
    print(f"{name:24s} epochs {len(rows):4d}  total {sum(sec)/3600:6.2f} h  median {sorted(sec)[len(sec)//2]:6.1f} s/ep"
          + (f"  by stage (h): { {k: round(v/3600, 2) for k, v in by.items()} }" if by else ""))
EOF
host_rev_adni=$(ls -d $E/adni/host_rev/*/ | tail -1); host_rev_mnist=$(ls -d $E/mnist/host_rev/*/ | tail -1)
echo "== host_rev epoch times (A100 here)"
for d in $host_rev_adni $host_rev_mnist; do grep -oE "\([0-9]+s\)" $d/console.log | tr -d '()s' | awk -v d="$d" '{s+=$1; n++} END {printf "%s epochs %d total %.2f h\n", d, n, s/3600}'; done
