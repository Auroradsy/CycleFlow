#!/usr/bin/env python3
"""The PCA-path figure (E2.5), reduced to the two arrays it actually draws.

  # once, on the cluster: 245 MB of pooled features -> a 0.5 MB projection
  python -m inpaper_utils.replot_pca_path export --run <latent_geometry run dir> \
      --out snapshot_results/mnist_petct/mnist_pca_proj.npz
  # anywhere, from the small file alone
  python inpaper_utils/replot_pca_path.py plot --proj mnist_pca_proj.npz --out mnist_pca_path.pdf

`domain_probe.figure()` fits a 2-component PCA on the endpoint codes $E_A(x)$ and
$E_B(y)$ and draws every state in it. The fit and the transform are the only things it
needs the $(n, L+2, 4096)$ pooled features for, and those are 245 MB per direction --
too large to keep beside the paper. `export` performs exactly that fit and transform and
stores the resulting $(n, L+2, 2)$ coordinates; `plot` redraws the published figure from
them and nothing else, so the figure travels with the paper without its inputs.

This file is deliberately standalone: it repeats the six style constants of
make_flow_viz.py rather than importing them, so that a copy of it under
__X_files/iclr2027/figs_scripts/ still runs outside the repository.
"""
import argparse
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, HAIR, AXIS, SERIES = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#2a78d6'
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}


def export(run, out, dataset):
    """Fit the PCA on the endpoint codes and store every state's coordinates in it."""
    from sklearn.decomposition import PCA
    z = np.load(Path(run) / 'features.npz', allow_pickle=True)
    pooled = {s: z[f'pooled_{s}'].astype(np.float64) for s in (0, 1)}
    assert np.abs(pooled[0]).max() > 0, 'pooled features are all zero'
    L1 = pooled[0].shape[1] - 1                       # states 0..L, then the target code
    p = PCA(2).fit(np.concatenate([pooled[0][:, 0], pooled[0][:, -1]]))
    proj = {f'states_{s}': np.stack([p.transform(pooled[s][:, l]) for l in range(L1)], 1).astype(np.float32)
            for s in (0, 1)}
    proj['endpoint_a'] = p.transform(pooled[0][:, 0]).astype(np.float32)
    proj['endpoint_b'] = p.transform(pooled[0][:, -1]).astype(np.float32)
    proj['explained_variance_ratio'] = p.explained_variance_ratio_.astype(np.float32)
    proj['dataset'] = np.array(dataset)
    proj['source_run'] = np.array(str(run))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **proj)
    print(f'{out}: {proj["states_0"].shape[0]} pairs, {L1} states per direction, '
          f'PC1 {p.explained_variance_ratio_[0]:.1%} PC2 {p.explained_variance_ratio_[1]:.1%}')


def plot(proj_path, out):
    """Redraw the published figure from the projection. Mirrors domain_probe.figure()."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(proj_path, allow_pickle=True)
    dataset = str(z['dataset'])
    dom = DOMAINS[dataset]
    evr = z['explained_variance_ratio']
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), dpi=200)
    for side, ax in enumerate(axes):
        st = z[f'states_{side}']
        for pts, colour, label in ((z['endpoint_a'], '#c0392b', f'$E_{{{dom[0]}}}(x)$'),
                                   (z['endpoint_b'], SERIES, f'$E_{{{dom[1]}}}(y)$')):
            ax.scatter(pts[:, 0], pts[:, 1], s=3, c=colour, alpha=.25, lw=0, label=label, zorder=2)
        mu = st.mean(0)
        ax.plot(mu[:, 0], mu[:, 1], color=INK, lw=1.6, zorder=4)
        ax.scatter(mu[:, 0], mu[:, 1], s=14, c=INK, zorder=5)
        for l, (px, py) in enumerate(mu):
            ax.annotate(str(l), (px, py), textcoords='offset points', xytext=(3, 4), fontsize=6, color=INK)
        for j in range(0, len(st), max(1, len(st) // 25)):
            ax.plot(st[j, :, 0], st[j, :, 1], color=INK2, lw=.4, alpha=.35, zorder=3)
        ax.set_xlabel(f'({"ab"[side]}) {dom[side]}' + r'$\rightarrow$' + f'{dom[1 - side]}'
                      + f'   (PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%})', color=INK, fontsize=7)
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
    out = Path(out)
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.with_suffix('.png'), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--run', required=True, help='latent_geometry run directory holding features.npz')
    e.add_argument('--out', required=True)
    e.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    p = sub.add_parser('plot')
    p.add_argument('--proj', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'export':
        export(a.run, a.out, a.dataset)
    else:
        plot(a.proj, a.out)


if __name__ == '__main__':
    main()
