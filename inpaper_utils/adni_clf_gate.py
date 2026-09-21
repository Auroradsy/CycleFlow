#!/usr/bin/env python3
"""E1a: is the ADNI label predictable from these slices at all?

  python -m inpaper_utils.adni_clf_gate --repeats 3

A gate, not a result. Before any claim that translated or augmented data helps a
downstream classifier, the classifier has to work on the real thing. Three inputs --
real T1, real FA, and both stacked -- are scored under one protocol:

  * all 214 labelled subjects, ten axial slices each (z = 40..49), label_2
    (107 healthy vs 107 impaired, so chance is 0.5 and the classes need no reweighting);
  * StratifiedGroupKFold by subject, so no subject's slices straddle a fold, repeated
    with several shuffles -- the paper's single 43-subject test split cannot resolve
    anything below about 15 points and is deliberately not used here;
  * per-slice probabilities averaged into one decision per subject; balanced accuracy
    and ROC AUC are computed over subjects, reported as mean +- std over repeats;
  * a label-permutation control, subject-level, which any honest pipeline must put at
    chance;
  * a positive control -- the same network, the same folds, predicting the axial slice
    index z instead of the diagnosis. That is certainly readable off the image, so it
    separates "the label is not in these slices" from "this pipeline cannot learn".
    Without it a chance result is uninterpretable.

Result JSON -> EXPS/adni/clf_gate/<run>/gate.json.
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
INPUTS = ('T1', 'FA', 'T1+FA')


def small_cnn(in_ch, n_cls=2):
    import torch.nn as nn
    ch = (32, 64, 128, 128)
    layers, c = [], in_ch
    for out in ch:
        layers += [nn.Conv2d(c, out, 3, 2, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
        c = out
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(.3), nn.Linear(c, n_cls))


def augment(x, g):
    """Left-right flip and a small translation; both are label-preserving on MNI slices."""
    import torch
    import torch.nn.functional as F
    if torch.rand((), generator=g) < .5:
        x = torch.flip(x, dims=[3])
    dx, dy = (int(torch.randint(-4, 5, (), generator=g)) for _ in range(2))
    return torch.roll(F.pad(x, (4, 4, 4, 4))[:, :, 4:-4, 4:-4], shifts=(dy, dx), dims=(2, 3))


def fit_predict(xtr, ytr, xte, dev, epochs, seed, batch=64, n_cls=2):
    """Train the CNN on one fold and return per-slice probabilities for the held-out fold."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = small_cnn(xtr.shape[1], n_cls).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(xtr)
    for _ in range(epochs):
        net.train()
        for i in torch.randperm(n, generator=g).split(batch):
            xb = augment(xtr[i].to(dev), g)
            loss = F.cross_entropy(net(xb), ytr[i].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xte), 256):
            out.append(F.softmax(net(xte[i:i + 256].to(dev)), 1).cpu())
    return torch.cat(out).numpy()


def subject_scores(prob, subj, lab):
    """Average the slice probabilities of each subject into one decision."""
    order = sorted(set(subj.tolist()))
    p = np.array([prob[subj == s].mean() for s in order])
    y = np.array([lab[subj == s][0] for s in order])
    return p, y


def evaluate(p, y):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    return {'balanced_acc': float(balanced_accuracy_score(y, (p > .5).astype(int))),
            'auc': float(roc_auc_score(y, p)), 'n_subjects': int(len(y))}


def one_pass(data, dev, epochs, folds, seed, permute=False):
    """One shuffle of the subject-level k-fold, over all three inputs."""
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    x, subj, lab = data['x'], data['subj'], data['lab']
    y = lab.copy()
    if permute:  # permute at subject level, so the slice structure is untouched
        rng = np.random.default_rng(seed)
        uniq = np.array(sorted(set(subj.tolist())))
        m = dict(zip(uniq, rng.permutation([lab[subj == s][0] for s in uniq])))
        y = np.array([m[s] for s in subj])
    res = {}
    for name in INPUTS:
        xs = x[name]
        prob = np.zeros(len(xs), dtype=np.float64)
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(xs)), y, groups=subj)):
            prob[te] = fit_predict(xs[tr], torch.from_numpy(y[tr]), xs[te], dev, epochs, seed * 100 + f)[:, 1]
        p, yy = subject_scores(prob, subj, y)
        res[name] = evaluate(p, yy)
        print(f'  {name:6s} balanced acc {res[name]["balanced_acc"]:.3f}  AUC {res[name]["auc"]:.3f}',
              flush=True)
    return res


def positive_control(data, z, dev, epochs, folds, seed):
    """Same network, same folds, predicting the axial slice index instead of the label.

    z is trivially present in the image, so high accuracy here means a chance result on
    the diagnosis is a fact about the diagnosis, not about the classifier."""
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    x, subj = data['x']['T1+FA'], data['subj']
    y = (z - z.min()).astype(np.int64)
    n_cls = int(y.max() + 1)
    pred = np.zeros(len(y), dtype=np.int64)
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups=subj)):
        p = fit_predict(x[tr], torch.from_numpy(y[tr]), x[te], dev, epochs, seed * 100 + f, n_cls=n_cls)
        pred[te] = p.argmax(1)
    # Slice level, not subject level: z varies within a subject, which is the point.
    return {'accuracy': float((pred == y).mean()), 'chance': float(1 / n_cls), 'n_classes': n_cls,
            'within_one_slice': float((np.abs(pred - y) <= 1).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--scheme', default='label_2')
    ap.add_argument('--controls-only', action='store_true',
                    help='skip the repeats and the permutation; run the positive control alone')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    from data.paired_dataset import PairedADNISliceDataset, subject_level_split

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Every labelled subject, not just the paper's test split: the gate needs the power.
    tr, te = subject_level_split(42, .2, a.scheme, 40, 49)
    ds = PairedADNISliceDataset(torch.cat([tr, te]), a.scheme)
    items = [ds[i] for i in range(len(ds))]
    t1 = torch.stack([t[0] for t in items]); fa = torch.stack([t[1] for t in items])
    lab = np.array([t[2] for t in items])
    subj = np.array([ds.subjects[int(ds.subj_idx[int(k)])] for k in ds.indices])
    zs = np.array([int(ds.z_idx[int(k)]) for k in ds.indices])
    data = {'x': {'T1': t1, 'FA': fa, 'T1+FA': torch.cat([t1, fa], 1)},
            'subj': subj, 'lab': lab}
    print(f'{len(lab)} slices, {len(set(subj))} subjects, '
          f'class balance {np.bincount(lab).tolist()}, on {dev}', flush=True)

    out = {'scheme': a.scheme, 'folds': a.folds, 'epochs': a.epochs, 'device': str(dev),
           'n_slices': int(len(lab)), 'n_subjects': int(len(set(subj))), 'repeats': []}
    if not a.controls_only:
        for r in range(a.repeats):
            print(f'repeat {r}', flush=True)
            out['repeats'].append(one_pass(data, dev, a.epochs, a.folds, seed=r))
        print('permutation control', flush=True)
        out['permuted'] = one_pass(data, dev, a.epochs, a.folds, seed=1234, permute=True)
    print('positive control (axial slice index)', flush=True)
    out['positive_control'] = positive_control(data, zs, dev, a.epochs, a.folds, seed=0)
    print(f"  slice index: accuracy {out['positive_control']['accuracy']:.3f} "
          f"(chance {out['positive_control']['chance']:.3f}, "
          f"within one slice {out['positive_control']['within_one_slice']:.3f})", flush=True)

    out['summary'] = {} if not out['repeats'] else {name: {k: {'mean': float(np.mean([r[name][k] for r in out['repeats']])),
                                 'std': float(np.std([r[name][k] for r in out['repeats']]))}
                             for k in ('balanced_acc', 'auc')} for name in INPUTS}
    d = EXPS / 'adni' / 'clf_gate' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                                      + '-' + uuid.uuid4().hex[:8])
    d.mkdir(parents=True, exist_ok=True)
    (d / 'gate.json').write_text(json.dumps(out, indent=1))
    print('\n== summary (mean +- std over repeats) ==')
    for name in INPUTS if out['summary'] else ():
        s = out['summary'][name]
        print(f'  {name:6s} balanced acc {s["balanced_acc"]["mean"]:.3f} +- {s["balanced_acc"]["std"]:.3f}'
              f'   AUC {s["auc"]["mean"]:.3f} +- {s["auc"]["std"]:.3f}')
    q = out['positive_control']
    if out.get('permuted'):
        p = out['permuted']['T1+FA']
        print(f'  permuted T1+FA: balanced acc {p["balanced_acc"]:.3f}  AUC {p["auc"]:.3f}  (must be ~0.5)')
    print(f'  positive control (slice index, {q["n_classes"]}-way): accuracy {q["accuracy"]:.3f} '
          f'(chance {q["chance"]:.3f})')
    print('wrote', d / 'gate.json')


if __name__ == '__main__':
    main()
