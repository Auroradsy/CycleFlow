#!/usr/bin/env python3
"""DEC-FA colouring for the ADNI paper figures.

An FA panel is drawn as |V1| * FA: red = left-right, green = anterior-posterior,
blue = superior-inferior, where V1 is the principal diffusion direction stored in
data/v1_112.pt (built by utils/data/dec_fa.py, aligned 1:1 with paired_112.pt).

The same lookup colours a GENERATED FA. A model that outputs FA has no direction
to show, so the hue always comes from the subject's measured V1 and only the
brightness is the model's: a DEC-FA panel of a prediction shows where the model
put anisotropy, not whether it got the fibre orientation right.

V1 exists for a few subjects only (has_v1). `preview_indices` returns the covered
slices; those subjects are in the TRAINING split, so figures built on them are
previews, not held-out results.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The full build (inpaper_utils/v1_all.sbatch) sits next to paired_112.pt; the
# 3-subject pilot copy in the repo is only the fallback.
_FULL = Path(os.environ.get('ADNI_ROOT', '/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow')) / 'v1_112.pt'
V1_CACHE = Path(os.environ.get('ADNI_V1_CACHE', _FULL if _FULL.exists() else ROOT / 'data' / 'v1_112.pt'))
_c = {}


def _v1():
    if not _c:
        import torch
        d = torch.load(V1_CACHE, map_location='cpu', weights_only=False)
        _c['V1'], _c['has'] = d['V1'].numpy(), d['has_v1'].numpy()
    return _c['V1'], _c['has']


def has_v1(k):
    return k is not None and bool(_v1()[1][int(k)])


def colour(fa, k):
    """FA image -> (H,W,3) float in [0,1].

    fa: (H,W), (H,W,1) or (1,H,W); float in [0,1] or uint8; numpy or torch.
    k: index into paired_112.pt."""
    V1, has = _v1()
    if not has_v1(k):
        raise KeyError(f'cache index {k} has no V1; run dec_fa_pilot.sh on that subject and rebuild v1_112.pt')
    f = fa.detach().cpu().numpy() if hasattr(fa, 'detach') else np.asarray(fa)
    f = np.squeeze(f.astype(np.float32) / (255. if f.dtype == np.uint8 else 1.))
    assert f.ndim == 2, f'expected a single-channel FA image, got shape {f.shape}'
    return (V1[int(k)].astype(np.float32).transpose(1, 2, 0) / 255. * np.clip(f, 0, 1)[..., None]).clip(0, 1)


def preview_indices(z_lo=40, z_hi=49):
    """Cache indices that have V1 inside the axial band, ordered by (subject, z)."""
    import torch
    from data.paired_dataset import CACHE
    p = torch.load(CACHE, map_location='cpu', weights_only=False)
    _, has = _v1()
    S, Z = p['subj_idx'].numpy(), p['z_idx'].numpy()
    return [int(k) for k in np.lexsort((Z, S)) if has[k] and z_lo <= Z[k] <= z_hi]


def missing(cache_indices):
    """The indices in `cache_indices` that cannot be drawn as DEC-FA."""
    return [k for k in cache_indices if not has_v1(k)]


def needed(out):
    """Write the held-out subjects that need V1 before the paper figures can be DEC-FA.

    One line per test subject (seed-42 split, label_4, z=40..49): subject ID, whether
    it is shown in a current paper figure (flow figures' z=44 picks, comparison /
    ablation picks), and whether V1 already exists. IDs go to the file only."""
    import json
    import torch
    from data.paired_dataset import CACHE, subject_level_split
    exps = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
    p = torch.load(CACHE, map_location='cpu', weights_only=False)
    S, Z = p['subj_idx'].numpy(), p['z_idx'].numpy()
    _, te = subject_level_split(42, .2, 'label_4', 40, 49)
    te = te.numpy()
    shown = set()
    z44 = [i for i, k in enumerate(te) if Z[k] == 44]            # make_flow_viz / make_inpaper rule
    shown |= {int(S[te[z44[int(j)]]]) for j in np.linspace(0, len(z44) - 1, 5)}
    runs = sorted((exps / 'adni' / 'paper_compare').glob('*/00_inpaper_fig_compare_manifest.json'))
    if runs:
        for rows in json.loads(runs[-1].read_text())['rows'].values():
            shown |= {int(S[te[r['test_index']]]) for r in rows}
    _, has = _v1()
    subs = sorted({int(s) for s in S[te]})
    lines = ['# subject\tin_paper_figure\thas_v1']
    for s in subs:
        lines.append(f"{p['subjects'][s]}\t{int(s in shown)}\t{int(bool(has[S == s].any()))}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text('\n'.join(lines) + '\n')
    print(f'{len(subs)} test subjects, {len(shown)} in paper figures, '
          f'{sum(bool(has[S == s].any()) for s in subs)} with V1 -> {out}')


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['needed'])
    ap.add_argument('--out', default='/ix/lzhan/siyuan/exps/CycleFlow/adni/qc/decfa_needed_subjects.tsv')
    a = ap.parse_args()
    needed(a.out)
