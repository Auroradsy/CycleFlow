#!/usr/bin/env python3
"""Inside the shared flow f: block-wise features, the affine terms, and cycles.

  python -m inpaper_utils.make_flow_viz --dataset adni --variant morph
  python -m inpaper_utils.make_flow_viz --dataset adni --figures features_bi --skip-summary
  python -m inpaper_utils.make_flow_viz --dataset adni --samples v1 --dec-fa   # DEC-FA preview

Each flow block is ActNorm (y = a*x + b, one a, b per channel) followed by an
affine coupling (y2 = a(x1)*x2 + b(x1), a = exp(tanh s), spatial maps).
Travelling B -> A applies the inverse, whose effective terms are a' = 1/a and
b' = -b/a; that is what the B -> A panels show.

Figures (paper-facing) -> snapshot_results/<dataset>/00_inpaper_{flow_features,
flow_affine,cycle}_{a2b,b2a}_<variant>, 00_inpaper_flow_features_bi_<variant>
(both directions on one grid, aligned by depth in the flow) and
00_inpaper_flow_actnorm_<variant>.

--dec-fa draws every FA panel as DEC-FA (inpaper_utils/decfa.py). V1 exists for a
few TRAINING subjects only, so --samples v1 takes the examples from those slices
and writes to snapshot_results/adni/decfa_preview/: previews, not held-out results.
Manifest with summary numbers -> EXPS/<dataset>/flow_viz/<run>/. CPU only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
DATA = Path('/ix/lzhan/siyuan/datasets/processed_datas')
SUB = {'adni': 'adni', 'mnist': 'mnist_petct'}
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}
PREFIX = {'adni': 'adni_', 'mnist': 'mnist_p_'}
FIGURES = ('features', 'features_bi', 'features_sym', 'affine', 'affine_ab', 'actnorm', 'cycle', 'cycle_ab')

# Reference palette of the dataviz skill: one-hue blue ramp for magnitude
# (receding to the surface at zero), blue <-> red with a gray midpoint for sign.
SURFACE, INK, INK2, HAIR, AXIS, SERIES = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#2a78d6'
SEQ = ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
DIV = ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f2a7a6', '#e34948', '#8e1f1f']


# ------------------------------------------------------------------ data/model
def load_data(dataset, n, samples='test'):
    import torch
    if dataset == 'adni':
        from data.paired_dataset import subject_level_split, PairedADNISliceDataset
        if samples == 'v1':
            from inpaper_utils.decfa import preview_indices
            ds = PairedADNISliceDataset(torch.tensor(preview_indices()), 'label_4')
        else:
            _, test = subject_level_split(42, .2, 'label_4', 40, 49)
            ds = PairedADNISliceDataset(test, 'label_4')
        A = torch.stack([ds[i][0] for i in range(len(ds))]); B = torch.stack([ds[i][1] for i in range(len(ds))])
        # Same rule as make_inpaper: evenly spaced subjects at axial z=44.
        eligible = [i for i, k in enumerate(ds.indices) if int(ds.z_idx[k]) == 44]
        pick = [eligible[int(j)] for j in np.linspace(0, len(eligible) - 1, min(n, len(eligible)))]
        key = 'test_index' if samples == 'test' else 'preview_index'
        info = lambda i: {key: i, 'cache_index': int(ds.indices[i]),
                          'subject': ds.subjects[int(ds.subj_idx[ds.indices[i]])], 'z': int(ds.z_idx[ds.indices[i]])}
        meta = [info(i) for i in pick]
    else:
        from PIL import Image
        root = DATA / 'MNIST_CycleFlow/mnist_petct_paired'
        names = sorted(p.name for p in (root / 'testA').glob('*.png'))
        read = lambda side, nm: np.asarray(Image.open(root / side / nm).convert('RGB'), dtype=np.float32) / 255
        A = torch.from_numpy(np.stack([read('testA', nm) for nm in names])).permute(0, 3, 1, 2).contiguous()
        B = torch.from_numpy(np.stack([read('testB', nm) for nm in names])).permute(0, 3, 1, 2).contiguous()
        pick, seen = [], set()
        for i, nm in enumerate(names):
            d = int(nm.split('_d')[1][0])
            if d not in seen:
                pick.append(i); seen.add(d)
            if len(pick) == n:
                break
        info = lambda i: {'test_index': i, 'filename': names[i]}
        meta = [info(i) for i in pick]
    return (A, B), pick, meta, info


def load_model(dataset, variant):
    import torch
    from model import MMCLASTcg
    # Checkpoints live either at EXPS/<tag> (migrated runs) or at
    # EXPS/<dataset>/checkpoints/<tag> (runs trained here).
    name = PREFIX[dataset] + variant
    path = next((c / 'model.pth' for c in (EXPS / name, EXPS / dataset / 'checkpoints' / name)
                 if (c / 'model.pth').exists()), EXPS / name / 'model.pth')
    ck = torch.load(path, map_location='cpu', weights_only=False); ar = ck['args']
    m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']),
                  img_ch=ar.get('img_ch', 1)).eval()
    m.load_state_dict(ck['model'], strict=True)
    return m, str(path)


def trace(m, z, inverse):
    """States after every block plus the affine terms each block applied (in travel order)."""
    import torch
    states, log_a, b, blocks = [z], [], [], list(range(m.n_blocks))[::-1] if inverse else list(range(m.n_blocks))
    x = z
    for k in blocks:
        an, cp = m.flow.layers[2 * k], m.flow.layers[2 * k + 1]
        if not inverse:
            x, _ = an(x)
        # The conditioning half is left unchanged by the coupling, so s, t are the
        # same whether they are read before (forward) or after (inverse) it.
        s, t = cp.net(cp._split(x)[0]).chunk(2, 1)
        s = torch.tanh(s)
        if inverse:
            log_a.append(-s); b.append(-t * torch.exp(-s))
            x, _ = cp.inverse(x); x, _ = an.inverse(x)
        else:
            log_a.append(s); b.append(t)
            x, _ = cp(x)
        states.append(x)
    for mine, ref in zip(states, m.walk(z, inverse=inverse)):
        torch.testing.assert_close(mine, ref)
    return states, log_a, b, blocks


def img(t):  # (n,C,H,W) in [0,1] -> (n,H,W) or (n,H,W,3)
    a = t.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return a[..., 0] if a.shape[-1] == 1 else a


# ------------------------------------------------------------------ rendering
def setup(fontsize):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': fontsize, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
    return plt


def cmap(colors, bad='white'):
    from matplotlib.colors import LinearSegmentedColormap
    c = LinearSegmentedColormap.from_list('c', colors)
    c.set_bad(bad)
    return c


class Grid:
    """Exact physical layout (inches), so 8pt text stays 8pt at the chosen width."""

    def __init__(self, plt, nrow, ncol, W, left, top, bottom=.02, gap=.025, extra=.07,
                 col_gaps=(), row_gaps=(), right=0.):
        self.s = (W - left - right - (ncol - 1) * gap - len(col_gaps) * extra) / ncol
        self.xs, x = [], left
        for c in range(ncol):
            self.xs.append(x); x += self.s + gap + (extra if c in col_gaps else 0)
        self.H = top + nrow * self.s + (nrow - 1) * gap + len(row_gaps) * extra + bottom
        self.ys, y = [], self.H - top
        for r in range(nrow):
            y -= self.s; self.ys.append(y); y -= gap + (extra if r in row_gaps else 0)
        self.W, self.fig = W, plt.figure(figsize=(W, self.H))

    def ax(self, r, c):
        a = self.fig.add_axes([self.xs[c] / self.W, self.ys[r] / self.H, self.s / self.W, self.s / self.H])
        a.set_axis_off()
        return a

    def col_label(self, c, text, c2=None, dy=.04, **kw):
        x = (self.xs[c] + self.xs[c if c2 is None else c2] + self.s) / 2
        self.fig.text(x / self.W, (self.ys[0] + self.s + dy) / self.H, text, ha='center', va='bottom', **kw)

    def row_label(self, r, text, r2=None, dx=.06, rotation=90):
        y = (self.ys[r] + self.s + self.ys[r if r2 is None else r2]) / 2
        self.fig.text((self.xs[0] - dx) / self.W, y / self.H, text, rotation=rotation,
                      ha='center' if rotation else 'right', va='center')

    def colorbar(self, im, c0, c1, label, y=None, h=.06):
        x0, x1 = self.xs[c0], self.xs[c1] + self.s
        cax = self.fig.add_axes([x0 / self.W, (self.ys[-1] - .16 if y is None else y) / self.H,
                                 (x1 - x0) / self.W, h / self.H])
        cb = self.fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.outline.set_linewidth(.4); cb.ax.tick_params(width=.4, length=2, pad=1)
        cb.set_label(label, labelpad=1)
        return cb


def show(ax, im, **kw):
    if im.ndim == 2 and 'cmap' not in kw:
        kw.update(cmap='gray', vmin=0, vmax=1)
    return ax.imshow(im, interpolation='none', **kw)


def save(fig, out, name, files):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=400, facecolor='white')
        files.append(f'{name}.{ext}'); print('Saved', out / f'{name}.{ext}', flush=True)


def arrow(a, b):
    return rf'{a}$\,\rightarrow\,${b}'


# ------------------------------------------------------------------ figures
def pca_rgb(tr, mask):
    """Colouring shared by both feature figures: one PCA basis for every state of
    both directions (and the target codes), so a colour means the same feature
    direction in every panel of either figure."""
    import torch
    maps = [mp for t in tr for mp in t['states'] + [t['target']]]
    X = torch.cat([mp.permute(0, 2, 3, 1)[mask] for mp in maps])
    mu = X.mean(0)
    # Exact eigendecomposition, not the randomised pca_lowrank, with a fixed sign
    # per component, so the colours are identical on every re-render.
    Xc = (X - mu).double()
    V = torch.linalg.eigh(Xc.T @ Xc)[1][:, -3:].flip(1)
    V = (V * torch.sign(V[V.abs().argmax(0), torch.arange(3)])).float()
    P = torch.cat([((mp.permute(0, 2, 3, 1) - mu) @ V)[mask] for mp in maps])
    lo, hi = torch.quantile(P, .01, 0), torch.quantile(P, .99, 0)
    # mp and msk must cover the same examples: a (1,C,h,w) map with the full
    # (n,h,w) mask would broadcast to n copies, each under another example's mask.
    return lambda mp, msk: (((mp.permute(0, 2, 3, 1) - mu) @ V - lo) / (hi - lo)).clamp(0, 1) * msk[..., None]


def features(plt, tr, xs, mask, dom, out, tag, W, files, paint):
    import torch
    n = xs[0].shape[0]
    rgb = pca_rgb(tr, mask)
    rms = lambda a, b: ((a - b) ** 2).mean(1).sqrt().masked_fill(~mask, float('nan'))
    deltas = [[rms(s1, s0) for s0, s1 in zip(t['states'][:-1], t['states'][1:])] + [rms(t['target'], t['states'][-1])]
              for t in tr]
    vmax = float(torch.nanquantile(torch.stack([d for dd in deltas for d in dd]).flatten(), .99))
    seq = cmap(SEQ)
    for side, t in enumerate(tr):
        src, tgt = dom[side], dom[1 - side]
        steps = [rf'$f_{{{k + 1}}}$' if side == 0 else rf'$f_{{{k + 1}}}^{{-1}}$' for k in t['blocks']]
        cols = ['Input', rf'$E_{{\mathrm{{{src}}}}}(x)$'] + steps + [rf'$E_{{\mathrm{{{tgt}}}}}(y)$', 'Target']
        g = Grid(plt, 2 * n, len(cols), W, left=.30, top=.22, bottom=.55,
                 row_gaps=tuple(2 * i + 1 for i in range(n - 1)))
        for i in range(n):
            r = 2 * i
            show(g.ax(r, 0), paint(xs[side][i], src, i)); show(g.ax(r, len(cols) - 1), paint(xs[1 - side][i], tgt, i))
            for c, mp in enumerate(t['states'] + [t['target']], start=1):
                show(g.ax(r, c), rgb(mp[i:i + 1], mask[i:i + 1])[0].numpy())
            for c, d in enumerate(deltas[side], start=2):
                im = show(g.ax(r + 1, c), d[i].numpy(), cmap=seq, vmin=0, vmax=vmax)
            g.row_label(r, 'PCA', dx=.04, rotation=0); g.row_label(r + 1, r'$\|\Delta\|$', dx=.04, rotation=0)
        for c, name in enumerate(cols):
            g.col_label(c, name)
        g.colorbar(im, 2, len(cols) - 2, 'RMS change from previous column', y=.36)
        save(g.fig, out, f'00_inpaper_flow_features_{"a2b" if side == 0 else "b2a"}_{tag}', files)
        plt.close(g.fig)
    return {'pca_basis': 'joint over both directions, masked pixels', 'delta_vmax': vmax}


def features_bi(plt, m, tr, xs, mask, dom, out, tag, W, files, paint, n):
    """Both directions on one grid, aligned by depth in the flow.

    Column h_k holds, in the top row, the state after f_1 ... f_k applied to E_A(x)
    and, in the middle row, the state after f_K^-1 ... f_{k+1}^-1 applied to E_B(y):
    the same depth reached from the two ends. The bottom row is their per-pixel
    RMS distance, i.e. where the two trajectories meet. The image columns hold
    each row's input and its translation."""
    import torch
    n = min(n, xs[0].shape[0])
    K = len(tr[0]['blocks'])
    rgb = pca_rgb(tr, mask)
    fwd = tr[0]['states']              # h_0 = E_A(x) ... h_K = f(E_A(x))
    inv = tr[1]['states'][::-1]        # h_0 = f^-1(E_B(y)) ... h_K = E_B(y)
    gap = [((p - q) ** 2).mean(1).sqrt().masked_fill(~mask, float('nan')) for p, q in zip(fwd, inv)]
    vmax = float(torch.nanquantile(torch.stack(gap)[:, :n].flatten(), .99))
    y_hat = ((m.dec_B(fwd[-1]) + 1) / 2).clamp(0, 1)     # A -> B translation
    x_hat = ((m.dec_A(inv[0]) + 1) / 2).clamp(0, 1)      # B -> A translation
    seq = cmap(SEQ)
    A, B = dom
    ncol = K + 3
    g = Grid(plt, 3 * n, ncol, W, left=.58, top=.22, bottom=.55, col_gaps=(0, K + 1),
             row_gaps=tuple(3 * i + 2 for i in range(n - 1)))
    for i in range(n):
        r = 3 * i
        show(g.ax(r, 0), paint(xs[0][i], A, i)); show(g.ax(r, ncol - 1), paint(y_hat[i], B, i))
        show(g.ax(r + 1, 0), paint(x_hat[i], A, i)); show(g.ax(r + 1, ncol - 1), paint(xs[1][i], B, i))
        for k in range(K + 1):
            show(g.ax(r, k + 1), rgb(fwd[k][i:i + 1], mask[i:i + 1])[0].numpy())
            show(g.ax(r + 1, k + 1), rgb(inv[k][i:i + 1], mask[i:i + 1])[0].numpy())
            im = show(g.ax(r + 2, k + 1), gap[k][i].numpy(), cmap=seq, vmin=0, vmax=vmax)
        g.row_label(r, arrow(A, B), dx=.04, rotation=0)
        g.row_label(r + 1, rf'{A}$\,\leftarrow\,${B}', dx=.04, rotation=0)
        g.row_label(r + 2, 'gap', dx=.04, rotation=0)
    g.col_label(0, A); g.col_label(ncol - 1, B)
    for k in range(K + 1):
        g.col_label(k + 1, rf'$h_{{{k}}}$')
    g.colorbar(im, 1, K + 1, 'RMS distance between the two directions at the same depth', y=.36)
    save(g.fig, out, f'00_inpaper_flow_features_bi_{tag}', files)
    plt.close(g.fig)
    return {'n': n, 'gap_vmax': vmax,
            'mean_gap_per_depth': [float(torch.nanmean(gp[:n])) for gp in gap]}


def features_sym(plt, tr, mask, dom, out, tag, W, files, n):
    """Per example: A -> B through the flow, then B -> A back through it, one row.

    Left half: E_A(x), then the state after f_1 ... f_K. Right half: E_B(y), then the
    state after f_K^-1 ... f_1^-1. Each half starts at one modality's code and ends
    near the other's, so a row reads A ... B | B ... A, mirror-symmetric about its
    middle. Features only (no images); one PCA basis for every panel (pca_rgb).
    Below each row, 'gap': the per-pixel RMS distance between the two directions at
    the same depth of the flow, i.e. under a panel of either half the distance to its
    mirror panel in the other half -- so the gap row is itself mirror-symmetric."""
    import torch
    n = min(n, tr[0]['states'][0].shape[0])
    K = len(tr[0]['blocks'])
    rgb = pca_rgb(tr, mask)
    fwd, inv = tr[0]['states'], tr[1]['states'][::-1]      # both indexed by depth 0..K
    gap = [((p - q) ** 2).mean(1).sqrt().masked_fill(~mask, float('nan')) for p, q in zip(fwd, inv)]
    vmax = float(torch.nanquantile(torch.stack(gap)[:, :n].flatten(), .99))
    seq = cmap(SEQ)
    A, B = dom
    ncol = 2 * (K + 1)
    g = Grid(plt, 2 * n, ncol, W, left=.30, top=.42, bottom=.55, gap=.02, extra=.14, col_gaps=(K,),
             row_gaps=tuple(2 * i + 1 for i in range(n - 1)))
    for i in range(n):
        r = 2 * i
        for side in (0, 1):
            for k, st in enumerate(tr[side]['states']):
                c = side * (K + 1) + k
                show(g.ax(r, c), rgb(st[i:i + 1], mask[i:i + 1])[0].numpy())
                depth = k if side == 0 else K - k          # right half runs from depth K down to 0
                im = show(g.ax(r + 1, c), gap[depth][i].numpy(), cmap=seq, vmin=0, vmax=vmax)
        g.row_label(r, 'PCA', dx=.04, rotation=0); g.row_label(r + 1, 'gap', dx=.04, rotation=0)
    names = ([rf'$E_{{\mathrm{{{A}}}}}(x)$'] + [rf'$f_{{{k + 1}}}$' for k in tr[0]['blocks']]
             + [rf'$E_{{\mathrm{{{B}}}}}(y)$'] + [rf'$f_{{{k + 1}}}^{{-1}}$' for k in tr[1]['blocks']])
    for c, name in enumerate(names):
        g.col_label(c, name)
    g.col_label(0, arrow(A, B), K, dy=.22)
    g.col_label(K + 1, arrow(B, A), ncol - 1, dy=.22)
    g.colorbar(im, 1, ncol - 2, 'RMS distance between the two directions at the same depth', y=.36)
    save(g.fig, out, f'00_inpaper_flow_features_sym_{tag}', files)
    plt.close(g.fig)
    return {'n': n, 'pca_basis': 'joint over both directions, masked pixels', 'gap_vmax': vmax,
            'mean_gap_per_depth': [float(torch.nanmean(gp[:n])) for gp in gap]}


def affine(plt, tr, xs, mask, dom, out, tag, W, reduce, files, paint):
    import torch
    # auto: log a keeps its sign under the channel mean (channels mostly agree on
    # expand vs. contract), while b cancels to ~1/10 of its RMS, so b shows magnitude.
    how = {'auto': ('mean', 'rms'), 'mean': ('mean', 'mean'), 'rms': ('rms', 'rms')}[reduce]
    red = {'mean': lambda v: v.mean(1), 'rms': lambda v: v.pow(2).mean(1).sqrt()}
    A = [[red[how[0]](v).masked_fill(~mask, float('nan')) for v in t['log_a']] for t in tr]
    Bm = [[red[how[1]](v).masked_fill(~mask, float('nan')) for v in t['b']] for t in tr]
    lim = lambda L: float(torch.nanquantile(torch.stack([v for vv in L for v in vv]).abs().flatten(), .99))
    la, lb = lim(A), lim(Bm)
    style = lambda h, l: (cmap(DIV), dict(vmin=-l, vmax=l)) if h == 'mean' else (cmap(SEQ), dict(vmin=0, vmax=l))
    (cma, ka), (cmb, kb) = style(how[0], la), style(how[1], lb)
    n = xs[0].shape[0]
    what = {'mean': 'channel mean', 'rms': 'channel RMS'}
    for side, t in enumerate(tr):
        steps = [rf'$f_{{{k + 1}}}$' if side == 0 else rf'$f_{{{k + 1}}}^{{-1}}$' for k in t['blocks']]
        ncol = 1 + 2 * len(steps)
        g = Grid(plt, n, ncol, W, left=.17, top=.36, bottom=.55, col_gaps=tuple(range(0, ncol - 1, 2)))
        for i in range(n):
            show(g.ax(i, 0), paint(xs[side][i], dom[side], i))
            for j in range(len(steps)):
                ima = show(g.ax(i, 1 + 2 * j), A[side][j][i].numpy(), cmap=cma, **ka)
                imb = show(g.ax(i, 2 + 2 * j), Bm[side][j][i].numpy(), cmap=cmb, **kb)
        g.col_label(0, 'Input')
        for j, st in enumerate(steps):
            g.col_label(1 + 2 * j, st, 2 + 2 * j, dy=.17)
            g.col_label(1 + 2 * j, r'$\log a$'); g.col_label(2 + 2 * j, r'$b$')
        g.row_label(0, arrow(dom[side], dom[1 - side]), n - 1)
        # The two scales hold for every block, so keep them clear of any one block's columns.
        g.colorbar(ima, 1, 3, rf'$\log a$ ({what[how[0]]})', y=.36)
        g.colorbar(imb, ncol - 3, ncol - 1, rf'$b$ ({what[how[1]]})', y=.36)
        save(g.fig, out, f'00_inpaper_flow_affine_{"a2b" if side == 0 else "b2a"}_{tag}', files)
        plt.close(g.fig)
    return {'reduce_log_a': how[0], 'reduce_b': how[1], 'log_a_limit': la, 'b_limit': lb}


def affine_ab(plt, tr, xs, masks, dom, out, tag, W, reduce, files, paint_ab, labels=('a', 'b')):
    """The affine terms with the cycle figure's framing: one file per direction, a few rows
    each, marked (a)/(b) instead of the direction. Both files share the two colour scales,
    so the panels stay comparable when the two are stacked in one figure."""
    import torch
    how = {'auto': ('mean', 'rms'), 'mean': ('mean', 'mean'), 'rms': ('rms', 'rms')}[reduce]
    red = {'mean': lambda v: v.mean(1), 'rms': lambda v: v.pow(2).mean(1).sqrt()}
    A = [[red[how[0]](v).masked_fill(~masks[s], float('nan')) for v in t['log_a']] for s, t in enumerate(tr)]
    Bm = [[red[how[1]](v).masked_fill(~masks[s], float('nan')) for v in t['b']] for s, t in enumerate(tr)]
    lim = lambda L: float(torch.nanquantile(torch.stack([v for vv in L for v in vv]).abs().flatten(), .99))
    la, lb = lim(A), lim(Bm)
    style = lambda h, l: (cmap(DIV), dict(vmin=-l, vmax=l)) if h == 'mean' else (cmap(SEQ), dict(vmin=0, vmax=l))
    (cma, ka), (cmb, kb) = style(how[0], la), style(how[1], lb)
    what = {'mean': 'channel mean', 'rms': 'channel RMS'}
    for side, t in enumerate(tr):
        n = xs[side].shape[0]
        steps = [rf'$f_{{{k + 1}}}$' if side == 0 else rf'$f_{{{k + 1}}}^{{-1}}$' for k in t['blocks']]
        ncol = 1 + 2 * len(steps)
        bars = side == len(tr) - 1        # the two files are stacked: one pair of scales, under (b)
        g = Grid(plt, n, ncol, W, left=.17, top=.36, bottom=.55 if bars else .02,
                 col_gaps=tuple(range(0, ncol - 1, 2)))
        for i in range(n):
            show(g.ax(i, 0), paint_ab(xs[side][i], dom[side], side, i))
            for j in range(len(steps)):
                ima = show(g.ax(i, 1 + 2 * j), A[side][j][i].numpy(), cmap=cma, **ka)
                imb = show(g.ax(i, 2 + 2 * j), Bm[side][j][i].numpy(), cmap=cmb, **kb)
        g.col_label(0, 'Input')
        for j, st in enumerate(steps):
            g.col_label(1 + 2 * j, st, 2 + 2 * j, dy=.17)
            g.col_label(1 + 2 * j, r'$\log a$'); g.col_label(2 + 2 * j, r'$b$')
        g.row_label(0, f'({labels[side]})', n - 1, dx=.03, rotation=0)
        if bars:
            g.colorbar(ima, 1, 3, rf'$\log a$ ({what[how[0]]})', y=.36)
            g.colorbar(imb, ncol - 3, ncol - 1, rf'$b$ ({what[how[1]]})', y=.36)
        save(g.fig, out, f'00_inpaper_flow_affine_{"a2b" if side == 0 else "b2a"}_ab_{tag}', files)
        plt.close(g.fig)
    return {'reduce_log_a': how[0], 'reduce_b': how[1], 'log_a_limit': la, 'b_limit': lb}


def actnorm(plt, m, out, tag, files):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.75))
    rng = np.random.default_rng(0)
    for ax, key, label in ((axes[0], 'log_scale', r'$\log a$'), (axes[1], 'bias', r'$b$')):
        for k in range(m.n_blocks):
            v = getattr(m.flow.layers[2 * k], key).detach().flatten().numpy()
            ax.scatter(k + 1 + rng.uniform(-.18, .18, v.size), v, s=2.5, color=SERIES, alpha=.55, linewidths=0)
            ax.hlines(np.median(v), k + .7, k + 1.3, color=INK, lw=.8)
        ax.axhline(0, color=AXIS, lw=.5, zorder=0)
        ax.set_xticks(range(1, m.n_blocks + 1), [rf'$f_{{{k}}}$' for k in range(1, m.n_blocks + 1)])
        ax.set_xlim(.4, m.n_blocks + .6); ax.set_ylabel(label, labelpad=2)
        ax.grid(axis='y', color=HAIR, lw=.4); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(.5)
        ax.tick_params(width=.4, length=2)
    fig.tight_layout(pad=.3, w_pad=1.5)
    save(fig, out, f'00_inpaper_flow_actnorm_{tag}', files)
    plt.close(fig)


def cycles(plt, m, xs, dom, out, tag, W, files, paint):
    import torch
    enc, dec = (m.enc_A, m.enc_B), (m.dec_A, m.dec_B)
    fwd = (m.a_to_b, m.b_to_a)
    res = []
    for side in (0, 1):
        x = xs[side]
        z = enc[side](x * 2 - 1)
        u = fwd[side](z)
        y_hat = ((dec[1 - side](u) + 1) / 2).clamp(0, 1)
        lat = ((dec[side](fwd[1 - side](u)) + 1) / 2).clamp(0, 1)
        imgc = ((dec[side](fwd[1 - side](enc[1 - side](y_hat * 2 - 1))) + 1) / 2).clamp(0, 1)
        res.append((x, y_hat, lat, imgc))
    err = lambda a, b: (a - b).abs().mean(1).numpy()
    vmax = float(np.quantile(np.concatenate([err(r[3], r[0]).ravel() for r in res]), .995))
    seq = cmap(SEQ)
    cols = ['Input', 'Translated', 'Latent cycle', 'Error', 'Image cycle', 'Error']
    n = xs[0].shape[0]
    for side, (x, y_hat, lat, imgc) in enumerate(res):
        a, b = dom[side], dom[1 - side]
        g = Grid(plt, n, len(cols), W, left=.17, top=.22, bottom=.08, col_gaps=(1, 3), right=.55)
        for i in range(n):
            show(g.ax(i, 0), paint(x[i], a, i)); show(g.ax(i, 1), paint(y_hat[i], b, i))
            show(g.ax(i, 2), paint(lat[i], a, i))
            show(g.ax(i, 3), err(lat[i:i + 1], x[i:i + 1])[0], cmap=seq, vmin=0, vmax=vmax)
            show(g.ax(i, 4), paint(imgc[i], a, i))
            im = show(g.ax(i, 5), err(imgc[i:i + 1], x[i:i + 1])[0], cmap=seq, vmin=0, vmax=vmax)
        for c, name in enumerate(cols):
            g.col_label(c, name)
        g.row_label(0, rf'{a}$\,\rightarrow\,${b}$\,\rightarrow\,${a}', n - 1)
        x0 = g.xs[-1] + g.s + .08
        cax = g.fig.add_axes([x0 / g.W, g.ys[-1] / g.H, .07 / g.W, (g.ys[0] + g.s - g.ys[-1]) / g.H])
        cb = g.fig.colorbar(im, cax=cax); cb.outline.set_linewidth(.4)
        cb.ax.tick_params(width=.4, length=2, pad=1); cb.set_label('Absolute error', labelpad=2)
        save(g.fig, out, f'00_inpaper_cycle_{"a2b" if side == 0 else "b2a"}_{tag}', files)
        plt.close(g.fig)
    return {'error_vmax': vmax}


def cycles_ab(plt, m, xab, dom, out, tag, W, files, paint_ab, labels=('a', 'b')):
    """Both cycle directions in one figure: rows of xab[0] go A->B->A, marked (a); rows of
    xab[1] go B->A->B, marked (b). One header row and one error scale for both."""
    enc, dec, fwd = (m.enc_A, m.enc_B), (m.dec_A, m.dec_B), (m.a_to_b, m.b_to_a)
    res = []
    for side in (0, 1):
        x = xab[side]
        u = fwd[side](enc[side](x * 2 - 1))
        y_hat = ((dec[1 - side](u) + 1) / 2).clamp(0, 1)
        lat = ((dec[side](fwd[1 - side](u)) + 1) / 2).clamp(0, 1)
        imgc = ((dec[side](fwd[1 - side](enc[1 - side](y_hat * 2 - 1))) + 1) / 2).clamp(0, 1)
        res.append((x, y_hat, lat, imgc))
    err = lambda a, b: (a - b).abs().mean(1).numpy()
    vmax = float(np.quantile(np.concatenate([err(r[3], r[0]).ravel() for r in res]), .995))
    seq = cmap(SEQ)
    cols = ['Input', 'Translated', 'Latent cycle', 'Error', 'Image cycle', 'Error']
    n0, n1 = len(res[0][0]), len(res[1][0])
    g = Grid(plt, n0 + n1, len(cols), W, left=.17, top=.22, bottom=.08, col_gaps=(1, 3),
             row_gaps=(n0 - 1,), right=.55)
    r = 0
    for side, (x, y_hat, lat, imgc) in enumerate(res):
        a, b = dom[side], dom[1 - side]
        first = r
        for i in range(len(x)):
            show(g.ax(r, 0), paint_ab(x[i], a, side, i)); show(g.ax(r, 1), paint_ab(y_hat[i], b, side, i))
            show(g.ax(r, 2), paint_ab(lat[i], a, side, i))
            show(g.ax(r, 3), err(lat[i:i + 1], x[i:i + 1])[0], cmap=seq, vmin=0, vmax=vmax)
            show(g.ax(r, 4), paint_ab(imgc[i], a, side, i))
            im = show(g.ax(r, 5), err(imgc[i:i + 1], x[i:i + 1])[0], cmap=seq, vmin=0, vmax=vmax)
            r += 1
        g.row_label(first, f'({labels[side]})', r - 1, dx=.03, rotation=0)
    for c, name in enumerate(cols):
        g.col_label(c, name)
    x0 = g.xs[-1] + g.s + .08
    cax = g.fig.add_axes([x0 / g.W, g.ys[-1] / g.H, .07 / g.W, (g.ys[0] + g.s - g.ys[-1]) / g.H])
    cb = g.fig.colorbar(im, cax=cax); cb.outline.set_linewidth(.4)
    cb.ax.tick_params(width=.4, length=2, pad=1); cb.set_label('Absolute error', labelpad=2)
    save(g.fig, out, f'00_inpaper_cycle_ab_{tag}', files)
    plt.close(g.fig)
    return {'error_vmax': vmax, 'rows': [n0, n1]}


def summary(m, pair, bs=64):
    """Full-set numbers for the manifest (not drawn)."""
    import torch
    from skimage.metrics import structural_similarity as ssim
    enc, dec, fwd = (m.enc_A, m.enc_B), (m.dec_A, m.dec_B), (m.a_to_b, m.b_to_a)
    acc = {s: {'latent_max_abs_err': 0., 'ssim_latent_cycle': [], 'ssim_image_cycle': [], 'mae_image_cycle': [],
               'rms_change_per_block': [], 'rel_dist_to_target_code': []} for s in (0, 1)}
    for i in range(0, len(pair[0]), bs):
        for side in (0, 1):
            x, y = pair[side][i:i + bs], pair[1 - side][i:i + bs]
            z, zt = enc[side](x * 2 - 1), enc[1 - side](y * 2 - 1)
            st = m.walk(z, inverse=bool(side))
            u = st[-1]
            back = fwd[1 - side](u)
            a = acc[side]
            a['latent_max_abs_err'] = max(a['latent_max_abs_err'], float((back - z).abs().max()))
            y_hat = ((dec[1 - side](u) + 1) / 2).clamp(0, 1)
            lat = ((dec[side](back) + 1) / 2).clamp(0, 1)
            imc = ((dec[side](fwd[1 - side](enc[1 - side](y_hat * 2 - 1))) + 1) / 2).clamp(0, 1)
            g, pl, pc = [t.permute(0, 2, 3, 1).numpy() for t in (x, lat, imc)]
            kw = dict(data_range=1, channel_axis=-1)
            a['ssim_latent_cycle'] += [ssim(g[j], pl[j], **kw) for j in range(len(g))]
            a['ssim_image_cycle'] += [ssim(g[j], pc[j], **kw) for j in range(len(g))]
            a['mae_image_cycle'] += (imc - x).abs().flatten(1).mean(1).tolist()
            a['rms_change_per_block'].append(torch.stack([((s1 - s0) ** 2).mean((1, 2, 3)).sqrt()
                                                          for s0, s1 in zip(st[:-1], st[1:])], 1))
            a['rel_dist_to_target_code'].append(torch.stack([(s - zt).flatten(1).norm(dim=1) / zt.flatten(1).norm(dim=1)
                                                             for s in st], 1))
    out = {}
    for side, a in acc.items():
        out[side] = {'latent_max_abs_err': a['latent_max_abs_err'],
                     **{k: float(np.mean(a[k])) for k in ('ssim_latent_cycle', 'ssim_image_cycle', 'mae_image_cycle')},
                     **{k: torch.cat(a[k]).mean(0).tolist() for k in ('rms_change_per_block', 'rel_dist_to_target_code')}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='adni', choices=['adni', 'mnist'])
    ap.add_argument('--variant', default='morph', choices=['morph', 'morph_bi', 'morph_smooth'])
    ap.add_argument('--figures', default=','.join(FIGURES), help=f'comma-separated subset of {FIGURES}')
    ap.add_argument('--n', type=int, default=5, help='examples per figure')
    ap.add_argument('--n-bi', type=int, default=3, help='examples in the two-direction feature figure')
    ap.add_argument('--n-sym', type=int, default=3, help='rows in the mirror-symmetric feature figure')
    ap.add_argument('--ab-pick', help='cycle_ab / affine_ab: test indices "i,j;k,l" for (a) the A->B '
                                      'direction and (b) the B->A one (default: the first two of --n for both)')
    ap.add_argument('--samples', default='test', choices=['test', 'v1'],
                    help='test: held-out subjects; v1: the V1-covered TRAINING slices (DEC-FA preview)')
    ap.add_argument('--dec-fa', action='store_true', help='draw FA panels as DEC-FA (needs V1 for every example)')
    ap.add_argument('--skip-summary', action='store_true', help='do not compute the full-set manifest numbers')
    ap.add_argument('--reduce', default='auto', choices=['auto', 'mean', 'rms'],
                    help='per-pixel summary of the 128 coupling channels; auto = mean for log a, RMS for b')
    ap.add_argument('--width', type=float, default=5.5, help='figure width in inches (ICLR \\linewidth = 5.5)')
    ap.add_argument('--fontsize', type=float, default=8)
    a = ap.parse_args()
    figs = set(a.figures.split(','))
    assert figs <= set(FIGURES), f'unknown figure(s): {figs - set(FIGURES)}'
    if (a.dec_fa or a.samples == 'v1') and a.dataset != 'adni':
        raise SystemExit('--dec-fa / --samples v1 are ADNI only')
    sys.path.insert(0, str(ROOT))
    os.environ['CYCLEFLOW_PURPOSE'] = 'flow_viz'
    from server_paths import experiment_root
    run = Path(experiment_root())
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    pair, pick, meta, info_of = load_data(a.dataset, a.n, a.samples)
    ab_pick = [[int(v) for v in g.split(',')] for g in a.ab_pick.split(';')] if a.ab_pick else [pick[:2], pick[:2]]
    ab_meta = [[info_of(i) for i in p] for p in ab_pick]
    if a.dec_fa:
        from inpaper_utils import decfa
        need = meta + (sum(ab_meta, []) if 'cycle_ab' in figs else [])
        miss = decfa.missing([x['cache_index'] for x in need])
        if miss:
            raise SystemExit(f'{len(miss)}/{len(meta)} examples have no V1. Use --samples v1 for a preview, '
                             f'or compute V1 for those subjects and rebuild v1_112.pt.')
    m, ck = load_model(a.dataset, a.variant)
    suffix = ('_decfa' if a.dec_fa else '') + ('_preview' if a.samples == 'v1' else '')
    dom, tag = DOMAINS[a.dataset], a.variant.replace('_', '-') + suffix
    out = ROOT / 'snapshot_results' / SUB[a.dataset] / ('decfa_preview' if a.samples == 'v1' else '')
    out.mkdir(parents=True, exist_ok=True)
    ks = [x.get('cache_index') for x in meta]

    def paint(t, name, i):   # t: (C,H,W) in [0,1] -> what imshow gets
        im = img(t[None])[0]
        return decfa.colour(im, ks[i]) if a.dec_fa and name == 'FA' else im

    def paint_ab(t, name, side, i):   # cycle_ab: examples differ per direction
        im = img(t[None])[0]
        return decfa.colour(im, ab_meta[side][i]['cache_index']) if a.dec_fa and name == 'FA' else im

    plt = setup(a.fontsize)
    files, info = [], {}
    with torch.inference_mode():
        xs = (pair[0][pick], pair[1][pick])
        brain = ((xs[0] > .02) | (xs[1] > .02)).any(1, keepdim=True).float()
        z0 = m.enc_A(xs[0] * 2 - 1)
        size = z0.shape[-1]
        mask = (F.adaptive_max_pool2d(brain, size)[:, 0] > 0) if a.dataset == 'adni' else \
            torch.ones(len(pick), size, size, dtype=torch.bool)
        tr = []
        for side, (enc, enc_t) in enumerate(((m.enc_A, m.enc_B), (m.enc_B, m.enc_A))):
            states, log_a, b, blocks = trace(m, enc(xs[side] * 2 - 1), inverse=bool(side))
            tr.append({'states': states, 'log_a': log_a, 'b': b, 'blocks': blocks,
                       'target': enc_t(xs[1 - side] * 2 - 1)})
        if 'features' in figs:
            info['features'] = features(plt, tr, xs, mask, dom, out, tag, a.width, files, paint)
        if 'features_bi' in figs:
            info['features_bi'] = features_bi(plt, m, tr, xs, mask, dom, out, tag, a.width, files, paint, a.n_bi)
        if 'features_sym' in figs:
            info['features_sym'] = features_sym(plt, tr, mask, dom, out, tag, a.width, files, a.n_sym)
        if 'affine' in figs:
            info['affine'] = affine(plt, tr, xs, mask, dom, out, tag, a.width, a.reduce, files, paint)
        if 'actnorm' in figs:
            actnorm(plt, m, out, tag, files)
        if 'cycle' in figs:
            info['cycle'] = cycles(plt, m, xs, dom, out, tag, a.width, files, paint)
        if 'affine_ab' in figs:
            # (a) and (b) draw different examples, so each direction gets its own trace and mask
            tr_ab, masks, xs_ab = [], [], []
            for side, enc in enumerate((m.enc_A, m.enc_B)):
                x_in, x_t = pair[side][ab_pick[side]], pair[1 - side][ab_pick[side]]
                st, log_a, b, blocks = trace(m, enc(x_in * 2 - 1), inverse=bool(side))
                tr_ab.append({'states': st, 'log_a': log_a, 'b': b, 'blocks': blocks})
                br = ((x_in > .02) | (x_t > .02)).any(1, keepdim=True).float()
                masks.append(F.adaptive_max_pool2d(br, size)[:, 0] > 0 if a.dataset == 'adni'
                             else torch.ones(len(x_in), size, size, dtype=torch.bool))
                xs_ab.append(x_in)
            info['affine_ab'] = {**affine_ab(plt, tr_ab, xs_ab, masks, dom, out, tag, a.width, a.reduce,
                                             files, paint_ab),
                                 'samples': {f'(a) {dom[0]}->{dom[1]}': ab_meta[0],
                                             f'(b) {dom[1]}->{dom[0]}': ab_meta[1]}}
        if 'cycle_ab' in figs:
            xab = (pair[0][ab_pick[0]], pair[1][ab_pick[1]])
            info['cycle_ab'] = {**cycles_ab(plt, m, xab, dom, out, tag, a.width, files, paint_ab),
                                'samples': {f'(a) {dom[0]}->{dom[1]}->{dom[0]}': ab_meta[0],
                                            f'(b) {dom[1]}->{dom[0]}->{dom[1]}': ab_meta[1]}}
        full = None
        if not a.skip_summary:
            print('Summary over the whole sample set...', flush=True)
            full = summary(m, pair)
    over = 'held-out test set' if a.samples == 'test' else 'V1-covered slices (training subjects)'
    manifest = {'dataset': a.dataset, 'variant': a.variant, 'checkpoint': ck, 'samples': meta, 'args': vars(a),
                'output_dir': str(out), 'figures': files, 'render': info, 'summary_over': over,
                'summary': None if full is None else {f'{dom[s]}->{dom[1 - s]}': v for s, v in full.items()},
                'notes': ['B->A panels show the inverse terms actually applied: log a\' = -log a, b\' = -b/a.',
                          'Latent cycle = D_A(f^-1(f(E_A x))); image cycle re-encodes the translated image first.',
                          'DEC-FA panels: hue from the subject\'s measured V1, brightness from the drawn FA '
                          '(ground truth or model output).']}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    print(json.dumps({k: manifest[k] for k in ('render', 'summary')}, indent=1), '\nManifest ->',
          run / 'manifest.json', flush=True)


if __name__ == '__main__':
    main()
