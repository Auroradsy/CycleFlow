#!/usr/bin/env python3
"""Path ablation: what each flow block does, variant by variant.

  python -m inpaper_utils.make_paper_path_ablation --dataset adni

One row per MMCLAST-cg variant; along a row: the source image, the encoded
representation before the first block, the state after every flow block -- all
decoded with the TARGET decoder -- and the ground-truth target. Variants without
a checkpoint are skipped. Deterministic; CPU is enough.

By default the example is the first held-out pair of the ablation/comparison
figure (manifest under EXPS/<dataset>/paper_compare/<run>/); --pick overrides.
Figures -> snapshot_results/<sub>/00_inpaper_fig_path_ablation_{a2b,b2a}.{pdf,png};
manifest -> EXPS/<dataset>/paper_path_ablation/<run>/.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
SUB = {'adni': 'adni', 'mnist': 'mnist_petct'}
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='adni', choices=['adni', 'mnist'])
    ap.add_argument('--n', type=int, default=1, help='examples per direction (rows = variants x n)')
    ap.add_argument('--pick', help='explicit test indices "i,j;k,l" (first direction; second)')
    ap.add_argument('--samples', default='test', choices=['test', 'v1'],
                    help='test: held-out pairs; v1: V1-covered TRAINING slices (ADNI DEC-FA preview)')
    ap.add_argument('--dec-fa', action='store_true', help='draw FA panels as DEC-FA')
    ap.add_argument('--variants', help='comma-separated subset of the variant labels (default: all)')
    ap.add_argument('--tags', help='"label=tag;label=tag": rows from these checkpoints (EXPS/<tag>/model.pth) '
                                   'instead of the ablation variants')
    ap.add_argument('--directions', default='a2b,b2a', help='which directions to draw')
    ap.add_argument('--out-dir', help='write the figure here instead of snapshot_results')
    ap.add_argument('--suffix', default='', help='appended to the figure name, to keep a subset figure '
                                                 'beside the full one')
    ap.add_argument('--row-label', help='one label for every row (e.g. "Ours" when a single variant is shown)')
    ap.add_argument('--no-row-label', action='store_true',
                    help='no row labels; the direction is written vertically on the left instead')
    ap.add_argument('--panel-labels',
                    help='e.g. "a,b": write (a) / (b) on the left instead of the direction, '
                         'which the caption then names')
    ap.add_argument('--width', type=float, default=5.5, help='figure width in inches (ICLR \\linewidth = 5.5)')
    ap.add_argument('--fontsize', type=float, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    os.environ['CYCLEFLOW_PURPOSE'] = 'paper_path_ablation'
    from server_paths import experiment_root
    run = Path(experiment_root())
    import torch
    from model import MMCLASTcg
    from inpaper_utils.make_paper_compare import load_test
    from inpaper_utils.make_paper_ablation import VARIANTS, compare_picks, v1_picks, dec_check, out_path
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))

    pair, meta = load_test(a.dataset, a.samples)
    if a.pick:
        picks, source = [[int(v) for v in g.split(',')] for g in a.pick.split(';')], 'manual --pick'
    elif a.samples == 'v1':
        picks, source = v1_picks(meta)
    else:
        picks, source = compare_picks(a.dataset)
    picks = [p[:a.n] for p in picks]
    decfa = dec_check(a, picks, meta)
    dom = DOMAINS[a.dataset]

    want = {v.replace('-', '_') for v in a.variants.split(',')} if a.variants else None
    todo = [tuple(t.split('=', 1)) for t in a.tags.split(';')] if a.tags else \
        [v for v in VARIANTS[a.dataset] if want is None or v[0].replace('-', '_') in want]
    rows, skipped = [], []
    for label, tag in todo:
        path = EXPS / tag / 'model.pth'
        if not path.exists():
            print(f'SKIP {label}: no checkpoint at {path}', flush=True)
            skipped.append({'label': label, 'checkpoint': str(path)}); continue
        ck = torch.load(path, map_location='cpu', weights_only=False); ar = ck['args']
        m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']),
                      img_ch=ar.get('img_ch', 1)).eval()
        m.load_state_dict(ck['model'], strict=True)
        states = {}
        with torch.inference_mode():
            for side, idx in enumerate(picks):
                enc = m.enc_A if side == 0 else m.enc_B
                dec = m.dec_B if side == 0 else m.dec_A          # always the TARGET decoder
                walk = m.walk(enc(pair[side][idx] * 2 - 1), inverse=bool(side))
                states[side] = [((dec(s) + 1) / 2).clamp(0, 1) for s in walk]
        rows.append({'label': a.row_label or label, 'checkpoint': str(path.resolve()), 'n_flow': ar['n_flow'],
                     'states': states})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': a.fontsize, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    show = lambda t: (lambda x: x[..., 0] if x.shape[-1] == 1 else x)(t.permute(1, 2, 0).numpy())
    files = []
    n_flow = rows[0]['n_flow']
    for side, idx in enumerate(picks):
        if ('a2b', 'b2a')[side] not in a.directions.split(','):
            continue
        cols = ['Input', '$z$'] + [rf'$f_{{{k + 1}}}$' if side == 0 else rf'$f_{{{n_flow - k}}}^{{-1}}$'
                                   for k in range(n_flow)] + ['Target']
        ncol, nrow = len(cols), len(rows) * len(idx)
        # left fits "morph-smooth"; without row labels only the vertical direction label needs room
        W, left, top, gap, group_gap = a.width, (.17 if a.no_row_label else .90), .20, .025, .07
        s = (W - left - (ncol - 1) * gap) / ncol
        H = top + nrow * s + (nrow - 1) * gap + (len(idx) - 1) * (group_gap - gap) + .01
        fig = plt.figure(figsize=(W, H))
        y = H - top
        for e, i in enumerate(idx):
            for v in rows:
                y -= s
                ims = [pair[side][i]] + [st[e] for st in v['states'][side]] + [pair[1 - side][i]]
                # every decoded state goes through the TARGET decoder, so only the input is dom[side]
                doms = [dom[side]] + [dom[1 - side]] * (len(ims) - 1)
                for c, (im, dn) in enumerate(zip(ims, doms)):
                    ax = fig.add_axes([(left + c * (s + gap)) / W, y / H, s / W, s / H])
                    if decfa and dn == 'FA':
                        ax.imshow(decfa.colour(show(im), meta[i]['cache_index']), interpolation='none')
                    else:
                        ax.imshow(show(im), cmap='gray', vmin=0, vmax=1, interpolation='none')
                    ax.set_axis_off()
                if not a.no_row_label:
                    fig.text((left - .04) / W, (y + s / 2) / H, v['label'], ha='right', va='center')
                y -= gap
            y -= group_gap - gap
        for c, name in enumerate(cols):
            fig.text((left + c * (s + gap) + s / 2) / W, (H - top + .04) / H, name, ha='center', va='bottom')
        direction = rf'{dom[side]}$\,\rightarrow\,${dom[1 - side]}'
        if a.panel_labels:
            # after the loop y sits one group gap below the last row
            fig.text((left - .09) / W, (H - top + y + group_gap) / 2 / H,
                     f'({a.panel_labels.split(",")[side]})', ha='center', va='center')
        elif a.no_row_label:
            # after the loop y sits one group gap below the last row
            fig.text((left - .06) / W, (H - top + y + group_gap) / 2 / H, direction,
                     rotation=90, ha='center', va='center')
        else:
            fig.text(.02 / W, (H - top + .04) / H, direction, ha='left', va='bottom')
        out, name = out_path(a, f'00_inpaper_fig_path_ablation_{"a2b" if side == 0 else "b2a"}{a.suffix}')
        if a.out_dir:   # candidate / diagnostic figures stay out of snapshot_results
            out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=400, facecolor='white')
            shutil.copy2(out / f'{name}.{ext}', run / f'{name}.{ext}')
            files.append(f'{name}.{ext}'); print('Saved', out / f'{name}.{ext}', flush=True)
        plt.close(fig)

    manifest = {'dataset': a.dataset, 'columns': 'Input, z, per-block states decoded by the target decoder, Target',
                'variants': [{k: v[k] for k in ('label', 'checkpoint')} for v in rows], 'skipped': skipped,
                'sample_source': source, 'files': files,
                'picks': {f'{dom[s]}->{dom[1 - s]}': [meta[i] for i in picks[s]] for s in (0, 1)}}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == '__main__':
    main()
