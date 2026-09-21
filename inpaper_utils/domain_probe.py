#!/usr/bin/env python3
"""E2.3 and E2.5: how far along the two manifolds each intermediate state has travelled.

  python -m inpaper_utils.domain_probe --dataset adni
  python -m inpaper_utils.domain_probe --dataset mnist --run <latent_geometry run dir>

Reads the pooled features written by latent_geometry.py: every state and both endpoint
codes, average-pooled to 4x4 per channel. Not to 1x1 -- the encoder ends in an
InstanceNorm with affine=False, so a global spatial mean is identically zero for every
sample and a probe on it is reading pure noise. `endpoint_probe_acc` below exists to catch
exactly that: a probe that cannot separate the two endpoint code sets says nothing about
what lies between them, and this script refuses to report a curve when it fails.

  E2.3  a logistic probe fitted on the two endpoint code sets, E_A(x) against E_B(y), and
        applied to the held-out fold's intermediate states: P(B) as a function of the block
        index. Its own held-out accuracy on the endpoints is reported, because a probe that
        cannot separate the endpoints says nothing about what lies between them. The
        non-parametric version runs alongside: the fraction of each state's k nearest
        endpoint codes that belong to B.
  E2.5  PCA fitted on the endpoint codes only -- a linear map, so straightness and relative
        distance survive the projection, unlike t-SNE or UMAP -- with the mean trajectory
        of each direction drawn in it.

Folds are grouped by subject on ADNI. Numbers -> the same run directory, figure ->
snapshot_results/<sub>/: <dataset>_domain_margin.pdf (E2.3) and <dataset>_pca_path.pdf
(E2.5). An earlier version wrote the PCA figure under the E2.3 name, which left E2.3 with
no figure at all.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
SNAP = ROOT / 'snapshot_results'
SUB = {'adni': 'adni', 'mnist': 'mnist_petct'}
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}


def latest_run(dataset):
    runs = sorted(glob.glob(str(EXPS / dataset / 'latent_geometry' / '*' / 'features.npz')), key=os.path.getmtime)
    if not runs:
        raise SystemExit(f'no latent_geometry run for {dataset}; run inpaper_utils.latent_geometry first')
    return Path(runs[-1]).parent


def splits(n, y, groups, folds, seed):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if groups is None:
        return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), y))
    return list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), y, groups))


def probe(states, xa, xb, groups, folds, seed, k):
    """P(B) per block, from a logistic probe and from k nearest endpoint codes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n, L1 = states.shape[0], states.shape[1]
    y = np.concatenate([np.zeros(n), np.ones(n)])
    g = None if groups is None else np.concatenate([groups, groups])
    lg = np.zeros((n, L1)); kn = np.zeros((n, L1)); mg = np.zeros((n, L1)); endpoint_acc = []
    for tr, te in splits(2 * n, y, g, folds, seed):
        X = np.concatenate([xa, xb])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        endpoint_acc.append(float((clf.predict(X[te]) == y[te]).mean()))
        # Rows of `states` whose sample index is held out in this fold.
        held = np.unique(te % n)
        for l in range(L1):
            lg[held, l] = clf.predict_proba(states[held, l])[:, 1]
        # P(B) saturates to 0/1 once the endpoints are perfectly separable, which they
        # are here; the signed distance to the boundary, in units of the gap between the
        # two endpoint clouds, stays graded and says *how far* across each state is.
        dec = clf.decision_function(X[tr])
        lo, hi = dec[y[tr] == 0].mean(), dec[y[tr] == 1].mean()
        for l in range(L1):
            mg[held, l] = (clf.decision_function(states[held, l]) - lo) / (hi - lo)
        nn = NearestNeighbors(n_neighbors=k).fit(X[tr])
        for l in range(L1):
            _, idx = nn.kneighbors(states[held, l])
            kn[held, l] = y[tr][idx].mean(1)
    if np.mean(endpoint_acc) < .9:
        raise SystemExit(f'endpoint probe accuracy {np.mean(endpoint_acc):.3f}: the probe cannot '
                         'separate E_A from E_B, so the intermediate curve is meaningless')
    return {'margin_to_b': mg.mean(0).tolist(), 'margin_to_b_std': mg.std(0).tolist(),
            'p_b_logistic': lg.mean(0).tolist(), 'p_b_logistic_std': lg.std(0).tolist(),
            'p_b_knn': kn.mean(0).tolist(), 'p_b_knn_std': kn.std(0).tolist(),
            'endpoint_probe_acc': {'mean': float(np.mean(endpoint_acc)), 'per_fold': endpoint_acc},
            'k': k, 'folds': folds}


def figure(dataset, pooled, out_json):
    """E2.5: the trajectories in a PCA fitted on the endpoint codes only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from inpaper_utils.make_flow_viz import setup, SURFACE, INK, INK2, HAIR, AXIS, SERIES
    setup(8)
    dom = DOMAINS[dataset]
    L1 = pooled[0].shape[1] - 1                       # states are 0..L, then the target code
    xa, xb = pooled[0][:, 0], pooled[0][:, -1]        # E_A(x) and E_B(y)
    p = PCA(2).fit(np.concatenate([xa, xb]))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), dpi=200)
    for side, ax in enumerate(axes):
        st = np.stack([p.transform(pooled[side][:, l]) for l in range(L1)], 1)   # (n, L+1, 2)
        for pts, colour, label in ((p.transform(xa), '#c0392b', f'$E_{{{dom[0]}}}(x)$'),
                                   (p.transform(xb), SERIES, f'$E_{{{dom[1]}}}(y)$')):
            ax.scatter(pts[:, 0], pts[:, 1], s=3, c=colour, alpha=.25, lw=0, label=label, zorder=2)
        mu = st.mean(0)
        ax.plot(mu[:, 0], mu[:, 1], color=INK, lw=1.6, zorder=4)
        ax.scatter(mu[:, 0], mu[:, 1], s=14, c=INK, zorder=5)
        for l, (px, py) in enumerate(mu):
            ax.annotate(str(l), (px, py), textcoords='offset points', xytext=(3, 4),
                        fontsize=6, color=INK)
        for j in range(0, len(st), max(1, len(st) // 25)):      # a few individual paths
            ax.plot(st[j, :, 0], st[j, :, 1], color=INK2, lw=.4, alpha=.35, zorder=3)
        # (a)/(b) to match every other two-panel figure in the paper.
        ax.set_xlabel(f'({"ab"[side]}) {dom[side]}' + r'$\rightarrow$' + f'{dom[1 - side]}'
                      + f'   (PC1 {p.explained_variance_ratio_[0]:.0%}, PC2 {p.explained_variance_ratio_[1]:.0%})',
                      color=INK, fontsize=7)
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, length=2, labelsize=6)
        ax.grid(color=HAIR, lw=.6, zorder=0)
        if side == 0:
            ax.legend(frameon=False, fontsize=6, labelcolor=INK2, markerscale=2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=.4)
    out = SNAP / SUB[dataset] / f'{dataset}_pca_path.pdf'
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.with_suffix('.png'), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print('wrote', out)


def margin_figure(dataset, out_json):
    """E2.3: how far across the A/B discriminant each state sits.

    Both directions are drawn as progress towards *their own* target, so each runs 0 -> 1;
    the stored `margin_to_b` always points at domain B, which would make the reverse
    direction run downwards and the two curves incomparable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from inpaper_utils.make_flow_viz import setup, SURFACE, INK, INK2, HAIR, AXIS, SERIES
    setup(8)
    dom = DOMAINS[dataset]
    keys = [k for k in out_json if '->' in k]
    fig, ax = plt.subplots(figsize=(2.9, 2.1), dpi=200)
    # The two curves very nearly coincide -- both traversals cross the discriminant at the
    # same rate, which is the point -- so they need different dashes to both stay visible.
    for k, colour, marker, ls, fill in zip(keys, (SERIES, '#c0392b'), ('o', 's'),
                                           ('-', (0, (4, 2))), ('full', 'none')):
        m = np.array(out_json[k]['margin_to_b'])
        if not k.startswith(dom[0]):
            m = 1 - m
        x = np.arange(len(m))
        ax.plot(x, m, color=colour, lw=1.6, ls=ls, marker=marker, ms=4.5, fillstyle=fill,
                mew=1.2, zorder=3, label=k.replace('->', r'$\rightarrow$'))
    for yv in (0, 1):
        ax.axhline(yv, color=AXIS, lw=.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(np.arange(len(m)))
    ax.set_xticklabels(['$z^{(0)}$'] + [str(i) for i in range(1, len(m))])
    ax.set_ylabel('progress towards the target side', color=INK)
    # Block k of the reverse traversal is f_{L+1-k}^{-1}, so the axis counts blocks applied
    # rather than naming them, and the two directions stay comparable.
    ax.set_xlabel('blocks applied', color=INK)
    ax.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(AXIS)
    ax.grid(axis='y', color=HAIR, lw=.6, zorder=0)
    ax.tick_params(colors=INK2, length=2)
    ax.legend(frameon=False, fontsize=6, labelcolor=INK2, loc='upper left')
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=.4)
    out = SNAP / SUB[dataset] / f'{dataset}_domain_margin.pdf'
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.with_suffix('.png'), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--run', help='latent_geometry run directory (default: the most recent)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    d = Path(a.run) if a.run else latest_run(a.dataset)
    z = np.load(d / 'features.npz', allow_pickle=True)
    pooled = {s: z[f'pooled_{s}'].astype(np.float64) for s in (0, 1)}
    assert np.abs(pooled[0]).max() > 0, 'pooled features are all zero'
    subj = z['subject']
    groups = subj if a.dataset == 'adni' and subj.size and str(subj[0]) else None
    dom = DOMAINS[a.dataset]
    L1 = pooled[0].shape[1] - 1
    print(f'{a.dataset}: {pooled[0].shape[0]} pairs, {L1} states per direction, '
          f'{pooled[0].shape[2]} pooled channels, grouped={groups is not None}')

    xa, xb = pooled[0][:, 0], pooled[0][:, -1]
    out = {'dataset': a.dataset, 'run': str(d), 'n': int(pooled[0].shape[0]), 'grouped': groups is not None}
    for side in (0, 1):
        out[f'{dom[side]}->{dom[1 - side]}'] = probe(pooled[side][:, :L1], xa, xb, groups, a.folds, a.seed, a.k)
    (d / 'domain_probe.json').write_text(json.dumps(out, indent=1))
    for side in (0, 1):
        r = out[f'{dom[side]}->{dom[1 - side]}']
        print(f'== {dom[side]}->{dom[1 - side]}  (endpoint probe acc {r["endpoint_probe_acc"]["mean"]:.3f})')
        print('   margin 0=A,1=B', ' '.join(f'{v:.3f}' for v in r['margin_to_b']))
        print('   P(B) logistic ', ' '.join(f'{v:.3f}' for v in r['p_b_logistic']))
        print(f'   P(B) {a.k}-NN     ', ' '.join(f'{v:.3f}' for v in r['p_b_knn']))
    margin_figure(a.dataset, out)
    figure(a.dataset, pooled, out)
    print('wrote', d / 'domain_probe.json')


if __name__ == '__main__':
    main()
