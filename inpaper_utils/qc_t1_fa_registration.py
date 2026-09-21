#!/usr/bin/env python3
"""Per-subject QC of the T1 <-> FA correspondence in paired_112.pt.

  python -m inpaper_utils.qc_t1_fa_registration

Both modalities are supposed to sit in MNI 2mm space, so on every axial slice the
T1 brain mask should overlap the FA brain mask and the population's mask. A failed
registration shows up as a subject whose T1 (or FA) mask departs from both.

Per subject, over z = 40..49 (the band every model trains on):
  dice_t1_fa   Dice(T1 mask, FA mask)            -- the pair itself
  dice_t1_pop  Dice(T1 mask, population T1 mask) -- is the T1 the odd one out?
  dice_fa_pop  Dice(FA mask, population FA mask) -- or the FA?
Flagged: robust z-score (median/MAD) below -4 on any of the three.
The table with subject IDs goes to EXPS/adni/qc/; stdout carries no IDs.
"""
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))


def main():
    sys.path.insert(0, str(ROOT))
    import torch
    from data.paired_dataset import CACHE, subject_level_split
    d = torch.load(CACHE, map_location='cpu', weights_only=False)
    S, Z = d['subj_idx'].numpy(), d['z_idx'].numpy()
    band = (Z >= 40) & (Z <= 49)
    T = (d['T1'][band, 0] > .05).numpy()
    F = (d['FA'][band, 0] > .05).numpy()
    s_b, z_b = S[band], Z[band]
    pop_t = {z: T[z_b == z].mean(0) > .5 for z in range(40, 50)}
    pop_f = {z: F[z_b == z].mean(0) > .5 for z in range(40, 50)}
    dice = lambda a, b: 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    tr, te = subject_level_split(42, .2, 'label_4', 40, 49)
    split = {int(s): 'train' for s in S[tr.numpy()]} | {int(s): 'test' for s in S[te.numpy()]}

    rows = []
    for s in range(len(d['subjects'])):
        k = np.where(s_b == s)[0]
        rows.append({'subj_idx': s, 'subject': d['subjects'][s], 'split': split.get(s, 'excluded'),
                     'dice_t1_fa': float(np.mean([dice(T[i], F[i]) for i in k])),
                     'dice_t1_pop': float(np.mean([dice(T[i], pop_t[z_b[i]]) for i in k])),
                     'dice_fa_pop': float(np.mean([dice(F[i], pop_f[z_b[i]]) for i in k]))})
    keys = ('dice_t1_fa', 'dice_t1_pop', 'dice_fa_pop')
    for key in keys:
        v = np.array([r[key] for r in rows])
        med, mad = np.median(v), np.median(np.abs(v - np.median(v))) * 1.4826
        for r, x in zip(rows, v):
            r['z_' + key] = float((x - med) / mad)
        print(f'{key:12s} median {med:.4f}  MAD-sd {mad:.4f}  min {v.min():.4f}')
    for r in rows:
        r['flag'] = any(r['z_' + k] < -4 for k in keys)
    out = EXPS / 'adni' / 'qc'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 't1_fa_registration.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: min(r['z_' + k] for k in keys)))
    flagged = [r for r in rows if r['flag']]
    print(f'\nflagged {len(flagged)}/{len(rows)} subjects '
          f'(train {sum(r["split"] == "train" for r in flagged)}, test {sum(r["split"] == "test" for r in flagged)}, '
          f'excluded {sum(r["split"] == "excluded" for r in flagged)})')
    for r in sorted(flagged, key=lambda r: min(r['z_' + k] for k in keys)):
        print(f'  subj_idx {r["subj_idx"]:3d} {r["split"]:8s} '
              + '  '.join(f'{k} {r[k]:.3f} (z {r["z_" + k]:+.1f})' for k in keys))
    print('table ->', out / 't1_fa_registration.csv')

    # Montage for eyeballing: one row per flagged subject, T1 then FA at z = 40, 44, 48.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = sorted(flagged, key=lambda r: min(r['z_' + k] for k in keys))
    zs = (40, 44, 48)
    fig, ax = plt.subplots(len(order), 6, figsize=(6 * 1.2, len(order) * 1.25), squeeze=False)
    for r, row in enumerate(order):
        for c, (mod, z) in enumerate([('T1', z) for z in zs] + [('FA', z) for z in zs]):
            k = int(np.where((S == row['subj_idx']) & (Z == z))[0][0])
            ax[r, c].imshow(d[mod][k, 0].numpy(), cmap='gray', vmin=0, vmax=1, interpolation='none')
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f'{mod} z={z}', fontsize=7)
        ax[r, 0].set_ylabel(f'idx {row["subj_idx"]}\n{row["split"]}', fontsize=7)
    fig.tight_layout(pad=.2)
    fig.savefig(out / 't1_fa_registration_flagged.png', dpi=150)
    print('montage ->', out / 't1_fa_registration_flagged.png')


if __name__ == '__main__':
    main()
