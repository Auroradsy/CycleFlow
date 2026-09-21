#!/usr/bin/env python3
"""E1b: does the translated code land where the target modality's classifier can read it?

  python -m inpaper_utils.cross_modal_probe --dataset mnist --variant morph_smooth
  python -m inpaper_utils.cross_modal_probe --dataset adni  --variant morph_smooth

A classifier is fitted on the *real* target codes $E_B(y)$ and then applied, without any
retraining, to four sets of codes on the held-out fold:

  E_B(y)                 real target code           -- the upper bound
  f(E_A(x))              our translation            -- the claim
  E_A(x)                 source code, no flow       -- the control: if this also works,
                                                       the two spaces were already aligned
                                                       and the flow is not what did it
  E_B(D_B(f(E_A(x))))    our translation, decoded and re-encoded

One further row, `E_A(x) own probe`, fits a *separate* classifier on the source codes and
scores it on source codes. It closes the argument: if `E_A(x)` scores well under its own
probe and at chance under the target's, the source code carries the class perfectly well
and what it lacks is alignment with the target space -- which is exactly what f supplies.

Cross-validation is grouped by subject on ADNI, so no subject's slices straddle a fold;
everything is measured on the paper's held-out split, so no extra data is touched.

A second, label-free measurement runs alongside and does not depend on any label being
predictable at all: **retrieval**. For each translated code, how far up the ranking of all
real target codes does its own paired target sit? A model that only performs a
modality-level style change cannot retrieve the right subject.

Result JSON -> EXPS/<dataset>/cross_modal_probe/<run>/probe.json.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
ARMS = ('E_B(y)', 'f(E_A(x))', 'E_A(x)', 'E_B(D_B(f(E_A(x))))')


def codes(m, pair, dev, bs, pool):
    """The four code sets, as pooled features for the probe and as flat codes for retrieval."""
    import torch
    import torch.nn.functional as F
    feat = {k: [] for k in ARMS}
    flat = {k: [] for k in ARMS}
    for i in range(0, len(pair[0]), bs):
        x, y = pair[0][i:i + bs].to(dev), pair[1][i:i + bs].to(dev)
        zA, zB = m.enc_A(x * 2 - 1), m.enc_B(y * 2 - 1)
        fzA = m.a_to_b(zA)
        reenc = m.enc_B(m.dec_B(fzA))
        for k, z in zip(ARMS, (zB, fzA, zA, reenc)):
            feat[k].append(F.adaptive_avg_pool2d(z, pool).flatten(1).cpu())
            flat[k].append(F.normalize(z.flatten(1), dim=1).cpu())
    return ({k: torch.cat(v).numpy() for k, v in feat.items()},
            {k: torch.cat(v) for k, v in flat.items()})


def probe(feat, y, groups, folds, seed):
    """Fit on the real target codes of the training folds; score every arm on the held out."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    n = len(y)
    if groups is None:
        splits = StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), y)
    else:
        splits = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), y, groups)
    acc = {k: [] for k in (*ARMS, 'E_A(x) own probe')}
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, n_jobs=-1))
        clf.fit(feat['E_B(y)'][tr], y[tr])
        for k in ARMS:
            acc[k].append(float((clf.predict(feat[k][te]) == y[te]).mean()))
        own = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, n_jobs=-1))
        own.fit(feat['E_A(x)'][tr], y[tr])
        acc['E_A(x) own probe'].append(float((own.predict(feat['E_A(x)'][te]) == y[te]).mean()))
    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'per_fold': v} for k, v in acc.items()}


def retrieval(flat, subj, chunk=512):
    """Rank of each sample's own paired target code among all real target codes."""
    import torch
    ref = flat['E_B(y)']
    out = {}
    for k in ('f(E_A(x))', 'E_A(x)', 'E_B(D_B(f(E_A(x))))'):
        q, ranks, same = flat[k], [], []
        for i in range(0, len(q), chunk):
            sim = q[i:i + chunk] @ ref.T                      # both are L2-normalised
            idx = torch.arange(i, min(i + chunk, len(q)))
            own = sim[torch.arange(len(idx)), idx]
            r = (sim > own[:, None]).sum(1)                   # 0 = retrieved first
            ranks.append(r)
            if subj is not None:
                best = sim.argmax(1).numpy()
                same.append(subj[best] == subj[idx.numpy()])
        r = torch.cat(ranks).numpy()
        out[k] = {'top1': float((r == 0).mean()), 'top5': float((r < 5).mean()),
                  'median_rank': float(np.median(r)), 'n_gallery': int(len(ref))}
        if subj is not None:
            out[k]['top1_same_subject'] = float(np.concatenate(same).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--variant', default='morph_smooth')
    ap.add_argument('--pool', type=int, default=4, help='spatial size the codes are pooled to')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--scheme', default='label_2', help='ADNI label scheme')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    from inpaper_utils.make_flow_viz import load_model
    from inpaper_utils.make_paper_compare import load_test

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m, ck = load_model(a.dataset, a.variant)
    m = m.to(dev).eval()
    (xa, xb), meta = load_test(a.dataset)
    if a.dataset == 'mnist':
        y = np.array([mm['digit'] for mm in meta]); groups = subj = None
        task = 'digit (10-way)'
    else:
        from data.paired_dataset import load_subject_labels
        lut = load_subject_labels(a.scheme)
        subj = np.array([mm['subject'] for mm in meta])
        y = np.array([lut.get(s, -1) for s in subj]); groups = subj
        task = f'{a.scheme} ({len(set(y))}-way)'
        assert (y >= 0).all(), 'unlabelled subject in the test split'
    print(f'{a.dataset}: {len(xa)} held-out pairs, task {task}, on {dev}', flush=True)

    with torch.no_grad():
        feat, flat = codes(m, (xa, xb), dev, a.batch, a.pool)
    out = {'dataset': a.dataset, 'variant': a.variant, 'checkpoint': str(ck), 'task': task,
           'n': int(len(xa)), 'pool': a.pool, 'folds': a.folds, 'feature_dim': int(feat[ARMS[0]].shape[1]),
           'chance': float(max(np.bincount(y)) / len(y)),
           'probe': probe(feat, y, groups, a.folds, a.seed),
           'retrieval': retrieval(flat, subj)}
    d = EXPS / a.dataset / 'cross_modal_probe' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                                                  + '-' + uuid.uuid4().hex[:8])
    d.mkdir(parents=True, exist_ok=True)
    (d / 'probe.json').write_text(json.dumps(out, indent=1))
    print(f'\n== probe, trained on E_B(y) only (chance {out["chance"]:.3f}) ==')
    for k in (*ARMS, 'E_A(x) own probe'):
        p = out['probe'][k]
        print(f'  {k:24s} {p["mean"]:.3f} +- {p["std"]:.3f}')
    print(f'\n== retrieval among {out["retrieval"]["f(E_A(x))"]["n_gallery"]} real target codes ==')
    for k, v in out['retrieval'].items():
        extra = f'  same-subject top-1 {v["top1_same_subject"]:.3f}' if 'top1_same_subject' in v else ''
        print(f'  {k:24s} top-1 {v["top1"]:.3f}  top-5 {v["top5"]:.3f}  median rank {v["median_rank"]:.0f}{extra}')
    print('wrote', d / 'probe.json')


if __name__ == '__main__':
    main()
