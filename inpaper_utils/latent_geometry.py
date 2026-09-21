#!/usr/bin/env python3
"""Where the intermediate flow states sit between the two representation manifolds.

  python -m inpaper_utils.latent_geometry --dataset adni  --variant morph_smooth
  python -m inpaper_utils.latent_geometry --dataset mnist --variant morph_smooth --plot

Measured in the latent space itself -- no t-SNE/UMAP, whose inter-cluster distances are
not interpretable and whose picture is decided by a hyperparameter. For every held-out
pair and every block $\\ell$ of the flow this records

  E2.1  relative distance from z^(l) to the source code z^(0) and to the paired target
        code E_B(y), and the cosine; with the linear interpolation (1-t) z^(0) + t E_B(y)
        at t = l/L as the reference that is two straight lines by construction. No
        whitening is applied and none is needed: the encoder ends in an InstanceNorm with
        affine=False, so every channel of every sample is already zero-mean and unit
        variance and no channel can dominate the Euclidean distance. `channel_std` in the
        output records that -- it is ~1 across all 256 channels -- so the claim can be
        checked rather than taken on trust;
  E2.2  the per-block step norm, the path length and the chord, whose ratio says whether
        the flow is doing anything a straight line would not;
  E2.4  the same distance to a random other sample's target code, which is what the
        paired distance has to beat for the flow to be doing subject-specific work.

E2.3 (the domain probe) is in domain_probe.py: it needs the pooled features this script
writes, so run this one first.

Numbers -> EXPS/<dataset>/latent_geometry/<run>/geometry.json, pooled features ->
features.npz next to it, figure -> snapshot_results/<sub>/<dataset>_latent_distance.pdf.
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
SNAP = ROOT / 'snapshot_results'
SUB = {'adni': 'adni', 'mnist': 'mnist_petct'}
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}


def run_dir(dataset):
    d = EXPS / dataset / 'latent_geometry' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                                              + '-' + uuid.uuid4().hex[:8])
    d.mkdir(parents=True, exist_ok=True)
    return d


def channel_std(m, pair, dev, bs):
    """Per-channel std of each domain's endpoint codes, over the whole split.

    Recorded, not used: the encoder's final InstanceNorm (affine=False) already makes this
    1 for every channel, which is why the distances below need no whitening."""
    import torch
    enc, out = (m.enc_A, m.enc_B), []
    for side in (0, 1):
        n = s1 = s2 = 0
        for i in range(0, len(pair[side]), bs):
            z = enc[side](pair[side][i:i + bs].to(dev) * 2 - 1)
            s1 = s1 + z.sum((0, 2, 3)); s2 = s2 + (z ** 2).sum((0, 2, 3))
            n += z.shape[0] * z.shape[2] * z.shape[3]
        out.append(((s2 / n - (s1 / n) ** 2).clamp_min(1e-12)).sqrt())
    return out


def geometry(m, pair, dev, bs, seed=0):
    """Per-sample distance curves and step norms, for both directions."""
    import torch
    g = torch.Generator().manual_seed(seed)
    enc, std = (m.enc_A, m.enc_B), channel_std(m, pair, dev, bs)
    rel = lambda a, b: (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)
    out = {'channel_std': {str(s): {'min': float(std[s].min()), 'max': float(std[s].max()),
                                    'mean': float(std[s].mean())} for s in (0, 1)}}
    for side in (0, 1):
        src, tgt = pair[side], pair[1 - side]
        # A fixed derangement of the pairing: the random-target control of E2.4.
        perm = torch.randperm(len(tgt), generator=g)
        perm = torch.where(perm == torch.arange(len(tgt)), (perm + 1) % len(tgt), perm)
        acc = {k: [] for k in ('to_src', 'to_tgt', 'cos_tgt', 'to_tgt_rand',
                               'lin_to_src', 'lin_to_tgt', 'step', 'pooled', 'src_norm')}
        for i in range(0, len(src), bs):
            x, y, yr = (t[i:i + bs].to(dev) for t in (src, tgt, tgt[perm]))
            z0, zt, ztr = enc[side](x * 2 - 1), enc[1 - side](y * 2 - 1), enc[1 - side](yr * 2 - 1)
            st = m.walk(z0, inverse=bool(side))
            L = len(st) - 1
            ts = [k / L for k in range(L + 1)]
            lin = [(1 - t) * z0 + t * zt for t in ts]
            acc['to_src'].append(torch.stack([rel(s, z0) for s in st], 1))
            acc['to_tgt'].append(torch.stack([rel(s, zt) for s in st], 1))
            acc['to_tgt_rand'].append(torch.stack([rel(s, ztr) for s in st], 1))
            acc['cos_tgt'].append(torch.stack([torch.nn.functional.cosine_similarity(
                s.flatten(1), zt.flatten(1), dim=1) for s in st], 1))
            acc['lin_to_src'].append(torch.stack([rel(s, z0) for s in lin], 1))
            acc['lin_to_tgt'].append(torch.stack([rel(s, zt) for s in lin], 1))
            acc['step'].append(torch.stack([(s1 - s0).flatten(1).norm(dim=1)
                                            for s0, s1 in zip(st[:-1], st[1:])], 1))
            acc['src_norm'].append(z0.flatten(1).norm(dim=1))
            # Pooled states + both endpoints, for the probe (E2.3). A *global* mean would
            # be identically zero: the encoder ends in InstanceNorm(affine=False), which
            # makes every channel of every sample zero-mean over space. Pooling to 4x4
            # keeps local structure and is not annihilated by that normalisation.
            pool = lambda z: torch.nn.functional.adaptive_avg_pool2d(z, 4).flatten(1)
            acc['pooled'].append(torch.stack([pool(s) for s in st] + [pool(zt)], 1).half())
        d = {k: torch.cat(v).cpu().numpy() for k, v in acc.items()}
        # E2.2: the chord is the straight line from z^(0) to f(z^(0)); to_src is relative.
        d['chord'] = d['to_src'][:, -1] * d['src_norm']
        d['path_len'] = d['step'].sum(1)
        d['ratio'] = d['path_len'] / np.maximum(d['chord'], 1e-12)
        out[side] = d
    return out


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    ax = 0 if a.ndim > 1 else None
    return {'mean': np.mean(a, axis=ax).tolist(), 'std': np.std(a, axis=ax).tolist(),
            'median': np.median(a, axis=ax).tolist(), 'n': int(len(a))}


def plot(res, dataset, tag):
    """E2.1 figure: departure from the source code and approach to the paired target code,
    against the straight line between the same two endpoints (dashed). The distance to a
    random other target is recorded in the JSON but not drawn: it is a marginal
    distribution, so next to the paired curve it reads as "barely different" when the
    paired comparison is in fact decided per sample (86-100% of the time)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from inpaper_utils.make_flow_viz import setup, SURFACE, INK, INK2, HAIR, AXIS, SERIES
    setup(8)
    dom = DOMAINS[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), dpi=200)
    for side, ax in enumerate(axes):
        d = res[side]
        L = d['to_tgt'].shape[1] - 1
        x = np.arange(L + 1)
        for key, colour, label in (('to_src', '#c0392b', f'to $E_{{{dom[side]}}}(x)$'),
                                   ('to_tgt', SERIES, f'to $E_{{{dom[1 - side]}}}(y)$')):
            mu, sd = d[key].mean(0), d[key].std(0)
            ax.plot(x, mu, color=colour, lw=2, label=label, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=colour, alpha=.15, lw=0, zorder=2)
        for key, colour in (('lin_to_src', '#c0392b'), ('lin_to_tgt', SERIES)):
            ax.plot(x, d[key].mean(0), color=colour, lw=1, ls=(0, (3, 2)), zorder=3)
        # One proxy entry so the dashed pair is named once rather than twice.
        ax.plot([], [], color=INK2, lw=1, ls=(0, (3, 2)), label='straight line between them')
        ax.set_xticks(x)
        ax.set_xticklabels(['$z^{(0)}$'] + [f'$f_{{{k}}}$' if not side else f'$f_{{{L + 1 - k}}}^{{-1}}$'
                                            for k in range(1, L + 1)])
        ax.set_xlabel(f'{dom[side]}' + r'$\rightarrow$' + f'{dom[1 - side]}', color=INK)
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.grid(axis='y', color=HAIR, lw=.6, zorder=0)
        ax.tick_params(colors=INK2, length=2)
        if side == 0:
            ax.set_ylabel('relative distance', color=INK)
            ax.legend(frameon=False, fontsize=6, labelcolor=INK2, loc='lower right',
                      borderaxespad=.2, handlelength=1.6)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=.4)
    out = SNAP / SUB[dataset] / f'{dataset}_latent_distance.pdf'
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.with_suffix('.png'), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--variant', default='morph_smooth')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--replot', help='redraw from a finished run directory and exit; '
                                     'reads features.npz, touches neither model nor data')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    if a.replot:
        z = np.load(Path(a.replot) / 'features.npz', allow_pickle=True)
        keys = ('to_src', 'to_tgt', 'cos_tgt', 'to_tgt_rand', 'lin_to_src', 'lin_to_tgt', 'step')
        plot({s: {k: z[f'{k}_{s}'] for k in keys} for s in (0, 1)}, a.dataset, a.variant)
        return
    import torch
    from inpaper_utils.make_flow_viz import load_model
    from inpaper_utils.make_paper_compare import load_test

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m, ck = load_model(a.dataset, a.variant)
    m = m.to(dev).eval()
    (xa, xb), meta = load_test(a.dataset)
    pair = (xa, xb)
    print(f'{a.dataset}: {len(xa)} held-out pairs on {dev}')

    with torch.no_grad():
        res = geometry(m, pair, dev, a.batch)
    cstd = res['channel_std']
    d = run_dir(a.dataset)
    # Pooled features for the probe, and the per-sample curves so the figure can be
    # redrawn (or restyled) without another forward pass over the split.
    curves = {f'{k}_{s}': res[s][k] for s in (0, 1)
              for k in ('to_src', 'to_tgt', 'cos_tgt', 'to_tgt_rand', 'lin_to_src',
                        'lin_to_tgt', 'step', 'chord', 'path_len', 'ratio')}
    np.savez_compressed(d / 'features.npz',
                        **{f'pooled_{s}': res[s]['pooled'] for s in (0, 1)}, **curves,
                        subject=np.array([mm.get('subject', '') for mm in meta]))
    out = {'dataset': a.dataset, 'variant': a.variant, 'n': len(xa), 'device': str(dev),
           'checkpoint': str(ck)}
    out['channel_std'] = res.pop('channel_std')
    for side in (0, 1):
        r = res[side]
        out[f'{DOMAINS[a.dataset][side]}->{DOMAINS[a.dataset][1 - side]}'] = {
            k: stats(r[k]) for k in ('to_src', 'to_tgt', 'cos_tgt', 'to_tgt_rand',
                                     'lin_to_src', 'lin_to_tgt', 'step',
                                     'path_len', 'chord', 'ratio')}
    (d / 'geometry.json').write_text(json.dumps(out, indent=1))
    print('wrote', d / 'geometry.json')
    print(f"channel std of the endpoint codes: {cstd['0']['min']:.3f}..{cstd['0']['max']:.3f} (A), "
          f"{cstd['1']['min']:.3f}..{cstd['1']['max']:.3f} (B)")
    if a.plot:
        plot(res, a.dataset, a.variant)


if __name__ == '__main__':
    main()
