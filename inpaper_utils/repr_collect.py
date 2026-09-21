#!/usr/bin/env python3
"""Collect the representation experiments into the figures and tables the paper uses.

  python -m inpaper_utils.repr_collect

Reads the most recent run of each experiment and writes

  snapshot_results/mnist_petct/mnist_missing_modality.pdf  E1b + E1c, the case that a
                                                    translation can stand in for a
                                                    modality that was not acquired
  snapshot_results/mnist_petct/mnist_aug.pdf        E1c, both sweeps
  snapshot_results/mnist_petct/mnist_aug_table.tex  E1c, as a table
  snapshot_results/adni/repr_probe_table.tex        E1b, both datasets
  snapshot_results/adni/latent_geometry_table.tex   E2.1/E2.2/E2.3/E2.4, both datasets

Series identity never rests on colour alone: every line is also given a marker and a
direct label at its right-hand end, and the table carries the same numbers.
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
ARMS = ('pet_only', '+translate', '+path', '+interp', '+mixup', 'oracle')
# The synthetic domain A was called CT while the experiments ran and is called MRI in the
# paper. The runs' JSON keys keep the old name; only what is printed is renamed.
SHOW = {'CT': 'MRI'}
# Fixed order, never cycled; distinct in lightness as well as hue, and each series also
# carries a marker and a direct label, so identity survives a greyscale print.
ARM_STYLE = {'pet_only':   ('#52514e', 'o', 'PET only'),
             '+translate': ('#2a78d6', 's', '+ translation'),
             '+path':      ('#c0392b', '^', '+ flow path'),
             '+interp':    ('#e08214', 'v', '+ straight line'),
             '+mixup':     ('#7b52ab', 'D', '+ mixup'),
             'oracle':     ('#2e8b57', '*', 'real PET (oracle)')}


def spread(ys, gap):
    """Nudge label positions apart, keeping their order, for a minimum vertical gap.

    +translate, +path and +interp land on top of each other -- that coincidence is the
    result -- so their labels have to be separated or the figure is unreadable."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k, i in enumerate(order[1:], 1):
        j = order[k - 1]
        out[i] = max(out[i], out[j] + gap)
    return out


def latest(pattern):
    hits = sorted(glob.glob(str(EXPS / pattern)), key=os.path.getmtime)
    return Path(hits[-1]) if hits else None


def aug_runs():
    """The labelled-count sweep and the corpus-size sweep(s), whichever runs exist.

    Keyed by (kind, direction). A newer run replaces an older one of the same key only if
    it is at least as complete: a one-cell rerun must not clobber a four-cell sweep.
    """
    out = {}
    for f in sorted(glob.glob(str(EXPS / 'mnist/aug/*/aug.json')), key=os.path.getmtime):
        d = json.loads(Path(f).read_text())
        cells = d.get('cells') or []
        if not cells:
            continue
        kind = 'corpus' if len({c['corpus'] for c in cells}) > 1 else 'labels'
        key = (kind, d.get('direction', 'a2b'))
        if key not in out or len(cells) >= len(out[key]['cells']):
            out[key] = d
    return out


def aug_figure(runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from inpaper_utils.make_flow_viz import setup, SURFACE, INK, INK2, HAIR, AXIS
    setup(8)
    # The path-vs-straight-line question is asked in the MRI->PET direction only.
    kinds = [k for k in (('labels', 'a2b'), ('corpus', 'a2b')) if k in runs]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5, 2.3), dpi=200, squeeze=False)
    for ax, (kind, _) in zip(axes[0], kinds):
        d = runs[(kind, 'a2b')]
        key = 'labels_per_class' if kind == 'labels' else 'corpus'
        cells = sorted(d['cells'], key=lambda c: c[key])
        xs = [c[key] for c in cells]
        ends = []
        for arm in ARMS:
            colour, marker, label = ARM_STYLE[arm]
            mu = np.array([d['results'][f"n{c['labels_per_class']}_c{c['corpus']}"][arm]['mean'] for c in cells])
            sd = np.array([d['results'][f"n{c['labels_per_class']}_c{c['corpus']}"][arm]['std'] for c in cells])
            ax.plot(xs, mu, color=colour, lw=1.6, marker=marker, ms=3.5, zorder=3)
            ax.fill_between(xs, mu - sd, mu + sd, color=colour, alpha=.15, lw=0, zorder=2)
            ends.append((arm, colour, label, float(mu[-1])))
        lo, hi = ax.get_ylim()
        pos = spread([e[3] for e in ends], (hi - lo) * .045)
        # Spreading can push the top label past the frame, where it is silently clipped.
        ax.set_ylim(min(lo, min(pos) - (hi - lo) * .03), max(hi, max(pos) + (hi - lo) * .03))
        for (arm, colour, label, y), ly in zip(ends, pos):
            ax.annotate(label, (xs[-1], ly), textcoords='offset points', xytext=(5, 0),
                        fontsize=5.5, color=colour, va='center', annotation_clip=False)
        ax.set_xscale('log')
        ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel('labelled PET per class' if kind == 'labels' else 'translated MRI images',
                      color=INK)
        ax.set_xlim(min(xs) / 1.3, max(xs) * 4.2)
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.grid(axis='y', color=HAIR, lw=.6, zorder=0)
        ax.tick_params(colors=INK2, length=2)
        if ax is axes[0][0]:
            ax.set_ylabel('PET digit accuracy', color=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=.4)
    out = SNAP / 'mnist_petct' / 'mnist_aug.pdf'
    fig.savefig(out, facecolor=SURFACE); fig.savefig(out.with_suffix('.png'), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print('wrote', out)


def missing_modality_figure(runs):
    """One figure for the claim that a translation can replace a missing modality.

    (a) is the cross-modal probe: a classifier fitted on real PET codes, applied without
    retraining. (b) is the augmentation curve reduced to the three series that bear on the
    same claim -- no labelled target data, translated data, and real target data. The
    path/interp/mixup arms belong to the separate question of whether the *trajectory*
    helps, which it does not, and they are left in mnist_aug.pdf.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from inpaper_utils.make_flow_viz import setup, SURFACE, INK, INK2, HAIR, AXIS
    f = latest('mnist/cross_modal_probe/*/probe.json')
    if not f or ('corpus', 'a2b') not in runs:
        print('missing_modality: need the mnist probe and the corpus sweep')
        return
    pr = json.loads(f.read_text())
    setup(8)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5), dpi=200,
                             gridspec_kw={'width_ratios': (1.3, 1, 1)})

    # (a) the probe
    ax = axes[0]
    # Short labels: the panel is a third of the figure and the full expressions collide.
    # The code each bar stands for is spelled out in the caption.
    bars = [('real PET', 'E_B(y)', '#2e8b57'),
            ('translated', 'f(E_A(x))', '#2a78d6'),
            ('MRI code,\nno flow', 'E_A(x)', '#52514e'),
            ('MRI code,\nown probe', 'E_A(x) own probe', '#b9b8b0')]
    xs = np.arange(len(bars))
    for i, (label, key, colour) in enumerate(bars):
        v = pr['probe'][key]
        ax.bar(i, v['mean'], .62, color=colour, zorder=3)
        ax.errorbar(i, v['mean'], yerr=v['std'], color=INK, lw=.9, capsize=2, zorder=4)
        ax.annotate(f"{v['mean']:.3f}", (i, v['mean']), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=6, color=INK, zorder=5)
    ax.axhline(pr['chance'], color='#c0392b', lw=.9, ls=(0, (4, 3)), zorder=2)
    ax.annotate('chance', (len(bars) - 1, pr['chance']), textcoords='offset points',
                xytext=(0, 4), ha='center', fontsize=6, color='#c0392b')
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=5)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel('PET digit accuracy', color=INK)
    ax.set_xlabel('(a) read by a classifier fitted on real PET codes', color=INK, fontsize=7)

    # (b), (c): the same claim as training data, in both directions. Every point trains on
    # the same 10 labelled images per class of the target modality; the x axis is how many
    # extra images are added, and the three series differ only in what those images are.
    for ax, (direction, tgt) in zip(axes[1:], (('a2b', 'PET'), ('b2a', 'MRI'))):
        key = ('corpus', direction)
        if key not in runs:
            ax.set_visible(False)
            continue
        d = runs[key]
        cells = sorted(d['cells'], key=lambda c: c['corpus'])
        cx = [c['corpus'] for c in cells]
        ends = []
        for arm, colour, label in (('oracle', '#2e8b57', 'real'),
                                   ('+translate', '#2a78d6', 'translated'),
                                   ('pet_only', '#52514e', 'none')):
            mu = np.array([d['results'][f"n{c['labels_per_class']}_c{c['corpus']}"][arm]['mean'] for c in cells])
            sd = np.array([d['results'][f"n{c['labels_per_class']}_c{c['corpus']}"][arm]['std'] for c in cells])
            ax.plot(cx, mu, color=colour, lw=1.8, marker='o', ms=3.5, zorder=3)
            ax.fill_between(cx, mu - sd, mu + sd, color=colour, alpha=.15, lw=0, zorder=2)
            ends.append((colour, label, float(mu[-1])))
        lo, hi = ax.get_ylim()
        pos = spread([e[2] for e in ends], (hi - lo) * .05)
        ax.set_ylim(min(lo, min(pos) - (hi - lo) * .03), max(hi, max(pos) + (hi - lo) * .03))
        for (colour, label, _), ly in zip(ends, pos):
            ax.annotate(label, (cx[-1], ly), textcoords='offset points', xytext=(5, 0),
                        fontsize=6, color=colour, va='center', annotation_clip=False)
        ax.set_xscale('log'); ax.set_xticks(cx); ax.set_xticklabels([str(v) for v in cx])
        ax.set_xlim(min(cx) / 1.3, max(cx) * 4.0)
        ax.set_xlabel(f'({"bc"[direction == "b2a"]}) images added, target {tgt}\n'
                      f'10 labelled {tgt} per class', color=INK, fontsize=7)
        ax.set_ylabel(f'{tgt} digit accuracy', color=INK)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(AXIS)
        ax.grid(axis='y', color=HAIR, lw=.6, zorder=0)
        ax.tick_params(colors=INK2, length=2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=.4)
    out = SNAP / 'mnist_petct' / 'mnist_missing_modality.pdf'
    fig.savefig(out, facecolor=SURFACE); fig.savefig(out.with_suffix('.png'), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print('wrote', out)


def aug_table(runs):
    rows = []
    for (kind, direction), d in sorted(runs.items()):
        key = 'labels_per_class' if kind == 'labels' else 'corpus'
        tgt = d.get('target', 'PET')
        for c in sorted(d['cells'], key=lambda c: c[key]):
            r = d['results'][f"n{c['labels_per_class']}_c{c['corpus']}"]
            rows.append((tgt, c['labels_per_class'], c['corpus'],
                         [(r[a]['mean'], r[a]['std']) for a in ARMS]))
    # 'pet_only' is the labelled target modality alone, which is MRI in the reverse
    # direction, so the table -- which mixes directions -- cannot call it "PET only".
    head = ' & '.join('target only' if a == 'pet_only' else ARM_STYLE[a][2] for a in ARMS)
    body = '\n'.join(
        f'{tgt} & {n} & {cc} & ' + ' & '.join(f'${m:.4f}$' for m, _ in vals) + r' \\'
        for tgt, n, cc, vals in rows)
    tex = (f'% E1c. Every arm gets the same number of gradient steps; see inpaper_utils/mnist_aug.py.\n'
           f'% "labels/class" counts labelled images of the target modality; "corpus" counts\n'
           f'% source-modality images translated into it.\n'
           f'\\begin{{tabular}}{{lll{"c" * len(ARMS)}}}\n\\toprule\n'
           f'target & labels/class & corpus & {head} \\\\\n\\midrule\n{body}\n'
           f'\\bottomrule\n\\end{{tabular}}\n')
    out = SNAP / 'mnist_petct' / 'mnist_aug_table.tex'
    out.write_text(tex)
    print('wrote', out)


def probe_table():
    tex = ['% E1b. A classifier fitted on real target codes only, applied without retraining.']
    for ds in ('mnist', 'adni'):
        f = latest(f'{ds}/cross_modal_probe/*/probe.json')
        if not f:
            continue
        d = json.loads(f.read_text())
        tex.append(f'% {ds}: task {d["task"]}, n={d["n"]}, chance {d["chance"]:.3f}')
        for k, v in d['probe'].items():
            tex.append(f'%   probe {k:24s} {v["mean"]:.4f} +- {v["std"]:.4f}')
        for k, v in d['retrieval'].items():
            extra = f', same-subject top-1 {v["top1_same_subject"]:.3f}' if 'top1_same_subject' in v else ''
            tex.append(f'%   retrieval {k:20s} top-1 {v["top1"]:.4f}, top-5 {v["top5"]:.4f}, '
                       f'median rank {v["median_rank"]:.0f}{extra}')
    out = SNAP / 'adni' / 'repr_probe_table.tex'
    out.write_text('\n'.join(tex) + '\n')
    print('wrote', out)


def geometry_table():
    tex = ['% E2. Latent geometry over the full held-out split.']
    for ds in ('adni', 'mnist'):
        g = latest(f'{ds}/latent_geometry/*/geometry.json')
        if not g:
            continue
        d = json.loads(g.read_text())
        p = g.parent / 'domain_probe.json'
        dp = json.loads(p.read_text()) if p.exists() else {}
        cs = d.get('channel_std', {}).get('0', {})
        tex.append(f'% {ds}: n={d["n"]}, variant {d["variant"]}, '
                   f'endpoint channel std {cs.get("min", float("nan")):.3f}..{cs.get("max", float("nan")):.3f}')
        for key in [k for k in d if '->' in k]:
            r = d[key]
            fmt = lambda v: ' '.join(f'{x:.3f}' for x in v)
            tex.append(f'%   {key}')
            tex.append(f'%     to target   {fmt(r["to_tgt"]["mean"])}')
            tex.append(f'%     linear ref  {fmt(r["lin_to_tgt"]["mean"])}')
            tex.append(f'%     random tgt  {fmt(r["to_tgt_rand"]["mean"])}')
            tex.append(f'%     cosine      {fmt(r["cos_tgt"]["mean"])}')
            tex.append(f'%     path/chord  {r["ratio"]["mean"]:.3f} +- {r["ratio"]["std"]:.3f}')
            if key in dp:
                q = dp[key]
                if 'margin_to_b' in q:   # the graded measure; P(B) below saturates to 0/1
                    tex.append(f'%     margin A->B {fmt(q["margin_to_b"])}  '
                               f'(endpoint probe acc {q["endpoint_probe_acc"]["mean"]:.3f})')
                tex.append(f'%     P(B) logistic {fmt(q["p_b_logistic"])}  '
                           f'(endpoint probe acc {q["endpoint_probe_acc"]["mean"]:.3f})')
                tex.append(f'%     P(B) {q["k"]}-NN     {fmt(q["p_b_knn"])}')
    # A usable tabular for the paper, with the full per-block curves kept above as a
    # comment record. One row per direction; the curves themselves belong in the figure.
    rows = []
    for ds in ('adni', 'mnist'):
        g = latest(f'{ds}/latent_geometry/*/geometry.json')
        if not g:
            continue
        d = json.loads(g.read_text())
        pth = g.parent / 'domain_probe.json'
        dp = json.loads(pth.read_text()) if pth.exists() else {}
        keys = [k for k in d if '->' in k]
        dom = keys[0].split('->') if keys else []
        # Both files write the A->B direction first. They are matched by position, not by
        # key, because a run made after the domain was renamed spells the same direction
        # differently from one made before it.
        dpk = [k for k in dp if '->' in k]
        for i, key in enumerate(keys):
            r = d[key]
            m = dp[dpk[i]]['margin_to_b'] if i < len(dpk) and 'margin_to_b' in dp[dpk[i]] else None
            src, tgt = (SHOW.get(v, v) for v in key.split('->'))
            # margin_to_b is always measured towards domain B, so for the B->A direction
            # it runs downwards; flip it so every row reads "how far towards its own
            # target", and all four become comparable.
            mv = float('nan') if not m else (m[-1] if key.startswith(dom[0]) else 1 - m[-1])
            rows.append((f'${src}\\rightarrow {tgt}$',
                         r['to_tgt']['mean'][-1], r['to_tgt_rand']['mean'][-1],
                         r['cos_tgt']['mean'][-1], r['ratio']['mean'], r['ratio']['std'], mv))
    body = '\n'.join(
        f'{name} & ${a:.3f}$ & ${b:.3f}$ & ${c:.3f}$ & ${e:.3f} \\pm {f:.3f}$ & ${g:.3f}$ \\\\'
        for name, a, b, c, e, f, g in rows)
    tab = ('\\begin{tabular}{lccccc}\n\\toprule\n'
           '& \\multicolumn{2}{c}{relative distance at $f(z)$} & & & \\\\\n'
           '\\cmidrule(lr){2-3}\n'
           'Direction & paired target & random target & cosine & path\\,/\\,chord & '
           'margin at $f(z)$ \\\\\n'
           '\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n')
    out = SNAP / 'adni' / 'latent_geometry_table.tex'
    out.write_text('\n'.join(tex) + '\n\n' + tab)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    sys.path.insert(0, str(ROOT))
    runs = aug_runs()
    if runs:
        aug_figure(runs); aug_table(runs); missing_modality_figure(runs)
    else:
        print('no mnist_aug run yet')
    probe_table()
    geometry_table()


if __name__ == '__main__':
    main()
