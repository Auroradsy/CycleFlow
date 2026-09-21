#!/usr/bin/env python3
"""Ablation figure: Input | Target | MMCLAST-cg variants, one figure per dataset.

  python -m inpaper_utils.make_paper_ablation --dataset adni

Uses the held-out examples of the baseline comparison figure (read from its
manifest under EXPS/<dataset>/paper_compare/<run>/), so the two figures read
side by side; --pick "i,j,k;l,m,n" overrides. Variants without a checkpoint are
skipped. Deterministic; CPU is enough.
Figure -> snapshot_results/<sub>/00_inpaper_fig_ablation.{pdf,png};
manifest -> EXPS/<dataset>/paper_ablation/<run>/.
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
SUB = {'adni': 'adni', 'mnist': 'mnist_petct'}
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}
VARIANTS = {   # (column label, checkpoint directory under EXPS), in Table 2 order
    'adni': [('base', 'adni_base'), ('latcyc', 'adni_latcyc'), ('morph', 'adni_morph'),
             ('morph-smooth', 'adni_morph_smooth'), ('morph-bi', 'adni_morph_bi')],
    'mnist': [('base', 'mnist_p_base'), ('latcyc', 'mnist_p_latcyc'), ('morph', 'mnist_p_morph'),
              ('morph-smooth', 'mnist/checkpoints/tune_mnist_smooth_g0.1_s0.3')],
}


def compare_picks(dataset):
    runs = sorted((EXPS / dataset / 'paper_compare').glob('*/00_inpaper_fig_compare_manifest.json'))
    if not runs:
        raise FileNotFoundError(f'no comparison-figure manifest for {dataset}; render it first or pass --pick')
    rows = json.loads(runs[-1].read_text())['rows']
    return [[r['test_index'] for r in v] for v in rows.values()], str(runs[-1])


def v1_picks(meta):
    """DEC-FA preview: the z=44 slice of every V1-covered subject, both directions, unranked."""
    z44 = [i for i, m in enumerate(meta) if m['z'] == 44]
    return [z44, z44], 'z=44 slice of every V1-covered subject (training split), not ranked'


def dec_check(a, picks, meta):
    """The decfa module when --dec-fa is on and every shown example has V1, else None."""
    if not a.dec_fa:
        return None
    from inpaper_utils import decfa
    miss = decfa.missing([meta[i].get('cache_index') for idx in picks for i in idx])
    if miss:
        raise SystemExit(f'{len(miss)} shown examples have no V1; use --samples v1 for a preview.')
    return decfa


def out_path(a, stem):
    """Paper figures go to snapshot_results/<sub>/; V1 previews to its decfa_preview/."""
    out = ROOT / 'snapshot_results' / SUB[a.dataset] / ('decfa_preview' if a.samples == 'v1' else '')
    out.mkdir(parents=True, exist_ok=True)
    return out, stem + ('_decfa' if a.dec_fa else '') + ('_preview' if a.samples == 'v1' else '')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--pick', help='explicit test indices "i,j,k;l,m,n" (first direction; second)')
    ap.add_argument('--samples', default='test', choices=['test', 'v1'],
                    help='test: held-out pairs; v1: V1-covered TRAINING slices (ADNI DEC-FA preview)')
    ap.add_argument('--dec-fa', action='store_true', help='draw FA panels as DEC-FA')
    ap.add_argument('--width', type=float, default=5.5, help='figure width in inches (ICLR \\linewidth = 5.5)')
    ap.add_argument('--fontsize', type=float, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    os.environ['CYCLEFLOW_PURPOSE'] = 'paper_ablation'
    from server_paths import experiment_root
    run = Path(experiment_root())
    import torch
    from model import MMCLASTcg
    from inpaper_utils.make_paper_compare import load_test, ssim_all, psnr_all
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))

    pair, meta = load_test(a.dataset, a.samples)
    if a.pick:
        picks, source = [[int(v) for v in g.split(',')] for g in a.pick.split(';')], 'manual --pick'
    elif a.samples == 'v1':
        picks, source = v1_picks(meta)
    else:
        picks, source = compare_picks(a.dataset)
    decfa = dec_check(a, picks, meta)
    dom = DOMAINS[a.dataset]
    variants, skipped, preds, scores = [], [], {}, {}
    for label, tag in VARIANTS[a.dataset]:
        path = EXPS / tag / 'model.pth'
        if not path.exists():
            print(f'SKIP {label}: no checkpoint at {path}', flush=True)
            skipped.append({'label': label, 'checkpoint': str(path)}); continue
        ck = torch.load(path, map_location='cpu', weights_only=False); ar = ck['args']
        m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']),
                      img_ch=ar.get('img_ch', 1)).eval()
        m.load_state_dict(ck['model'], strict=True)
        variants.append({'label': label, 'checkpoint': str(path.resolve())})
        with torch.inference_mode():
            for side, idx in enumerate(picks):
                fn = m.cross_A2B if side == 0 else m.cross_B2A
                y = ((fn(pair[side][idx] * 2 - 1) + 1) / 2).clamp(0, 1)
                preds[(label, side)] = y
                scores[f'{label} {dom[side]}->{dom[1 - side]}'] = {
                    'ssim': ssim_all(y, pair[1 - side][idx]).round(4).tolist(),
                    'psnr': psnr_all(y, pair[1 - side][idx]).round(2).tolist()}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': a.fontsize, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    img = lambda t: (lambda x: x[..., 0] if x.shape[-1] == 1 else x)(t.permute(1, 2, 0).numpy())
    cols = ['Input', 'Target'] + [v['label'] for v in variants]
    ncol, nrow = len(cols), sum(len(p) for p in picks)
    # Exact physical layout (inches) so the text is true size at the chosen width.
    W, left, top, gap, group_gap = a.width, .17, .20, .025, .07
    s = (W - left - (ncol - 1) * gap) / ncol
    H = top + nrow * s + (nrow - 1) * gap + (group_gap - gap) + .01
    fig = plt.figure(figsize=(W, H))
    y = H - top
    for side, idx in enumerate(picks):
        y_top = y
        for r, i in enumerate(idx):
            y -= s
            ims = [pair[side][i], pair[1 - side][i]] + [preds[(v['label'], side)][r] for v in variants]
            doms = [dom[side]] + [dom[1 - side]] * (len(ims) - 1)
            for c, (im, dn) in enumerate(zip(ims, doms)):
                ax = fig.add_axes([(left + c * (s + gap)) / W, y / H, s / W, s / H])
                if decfa and dn == 'FA':
                    ax.imshow(decfa.colour(img(im), meta[i]['cache_index']), interpolation='none')
                else:
                    ax.imshow(img(im), cmap='gray', vmin=0, vmax=1, interpolation='none')
                ax.set_axis_off()
            y -= gap
        y += gap
        fig.text((left - .06) / W, (y + y_top) / 2 / H, rf'{dom[side]}$\,\rightarrow\,${dom[1 - side]}',
                 rotation=90, ha='center', va='center')
        y -= group_gap
    for c, name in enumerate(cols):
        fig.text((left + c * (s + gap) + s / 2) / W, (H - top + .04) / H, name, ha='center', va='bottom')

    out, stem = out_path(a, '00_inpaper_fig_ablation')
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=400, facecolor='white')
        shutil.copy2(out / f'{stem}.{ext}', run / f'{stem}.{ext}')
        print('Saved', out / f'{stem}.{ext}', flush=True)
    manifest = {'dataset': a.dataset, 'samples_mode': a.samples, 'dec_fa': a.dec_fa,
                'columns': cols, 'variants': variants, 'skipped': skipped,
                'sample_source': source, 'picks': {f'{dom[s]}->{dom[1 - s]}': [meta[i] for i in picks[s]]
                                                   for s in (0, 1)},
                'per_image_scores': scores, 'figure_width_in': W}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'per_image_scores'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
