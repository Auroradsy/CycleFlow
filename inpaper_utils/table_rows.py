#!/usr/bin/env python3
"""Every row of Tables 1 and 2, assembled from the three places they come from.

  python -m inpaper_utils.table_rows                 # both datasets, LaTeX rows
  python -m inpaper_utils.table_rows --check         # only: do the published rows still reproduce?

  quality  EXPS/<ds>/paper_compare/<run>/cache/meta.json   (full held-out split, both directions)
  size     inpaper_utils.count_params                      (the constructors)
  time     EXPS/bench_cost/{train,infer}.json              (one A100, batch 64)

Printing the published rows beside the new ones is the point: if a number that is
already in the paper does not come back the same, the pipeline moved under it and
the new rows cannot be trusted either.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
ORDER = ['DDPM', 'CycleGAN', 'RevGAN', 'MeanFlow', 'DiT', 'CFM', 'RectFlow']
TEX = ROOT / '__X_files/iclr2027/iclr2027_conference.tex'


def published():
    """What the fidelity table says right now, read out of the tex rather than copied.

    A hard-coded copy of the numbers is a second source of truth that goes stale
    the moment a row is edited, which is the failure this check exists to catch.
    The paper now has one table for both datasets (sections/translation_table.tex,
    tab:translation): Method, then SSIM/PSNR for MNIST MRI->PET, PET->MRI, then for
    ADNI T1->FA, FA->T1. Size and cost are no longer tabulated, so only those four
    columns per dataset are checked.
    """
    import re
    from inpaper_utils.paper_tex import flat
    tex = flat()
    i = tex.index('\\label{tab:translation}')
    body = tex[i:tex.index('\\end{tabular', i)]
    out = {'mnist': {}, 'adni': {}}
    for line in body.split('\n'):
        if not line.rstrip().endswith('\\\\'):
            continue
        cells = [re.sub(r'\\(?:textbf|underline){([^}]*)}', r'\1', c).strip()
                 for c in line.rstrip()[:-2].split('&')]
        if len(cells) != 9:
            continue
        try:
            v = [float(c) for c in cells[1:]]
        except ValueError:
            continue                     # a header row
        out['mnist'][cells[0]], out['adni'][cells[0]] = tuple(v[:4]), tuple(v[4:])
    return out


# The table prints SSIM to 4 decimals and PSNR to 2, so a reproduced number must
# match to that rounding; the looser relative bound is for columns that are timed.
def tolerance(col, value):
    return {0: 1.1e-4, 2: 1.1e-4, 1: 0.011, 3: 0.011}.get(col, 0.02 * abs(value) + 1e-4)


# ------------------------------------------------------------------ size and cost
# The table rows of our two variants, and the checkpoint behind each.
OURS_ROWS = {'mnist': [('FlowCycle', 'mnist_p_base'),
                       ('FlowCycle + MS', 'mnist/checkpoints/tune_mnist_smooth_g0.1_s0.3')],
             'adni': [('FlowCycle', 'adni_base'), ('FlowCycle + MS', 'adni_morph_smooth')]}
BENCH = {'CycleGAN': 'CycleGAN', 'RevGAN': 'RevGAN', 'DDPM': 'DDPM', 'MeanFlow': 'MeanFlow',
         'CFM': 'CFM', 'RectFlow': 'RectFlow', 'DiT': 'DiT',
         'FlowCycle': 'MMCLAST-cg, base', 'FlowCycle + MS': 'MMCLAST-cg, morph-smooth'}
COST_ORDER = ['CycleGAN', 'RevGAN', 'DDPM', 'MeanFlow', 'CFM', 'RectFlow', 'DiT']


def trainable(sd_modules):
    return sum(p.numel() for m in sd_modules for p in m.parameters())


def dit_params(tag, img_ch, size):
    """Trainable parameters of the trained DiT pair and its latent autoencoder.

    Counted on modules loaded strict=True from the checkpoint, so the architecture
    is the trained one; parameters() leaves out the fixed positional-embedding
    buffers that a raw state-dict count would include.
    """
    import torch
    from baselines.latent_ae import LatentAE
    from baselines.nets_dit import DiT
    ae = LatentAE(img_ch=img_ch)
    ae.load_state_dict(torch.load(EXPS / tag / 'latent_ae.pth', map_location='cpu', weights_only=False)['ae'])
    ck = torch.load(EXPS / tag / 'last.pth', map_location='cpu', weights_only=False)
    ar, nets = ck['args'], []
    for key in ('net_ab', 'net_ba'):
        net = DiT(latent_size=size // 4, latent_ch=ae.latent_ch, patch=ar['patch'], hidden=ar['hidden'],
                  depth=ar['depth'], heads=ar['heads'])
        net.load_state_dict(ck[key], strict=True)
        nets.append(net)
    n = trainable(nets + [ae])
    return n, n


def ours_params(tag):
    """(inference, trained) parameters of one of our checkpoints.

    Trained = the model and the critics its own training actually updates. train.py
    BUILDS the mixed-domain path critic whenever any path term is on, but calls and
    steps it only when w_path_gan > 0; morph-smooth runs with w_path_gan = 0 (L1
    smoothness between consecutive decoded states, no critic), so for them that
    critic is an untouched initialisation and is not counted. The CycleGAN run the
    model is warm-started from is not counted either, by the author's choice.
    """
    import torch
    from model import MMCLASTcg, make_discriminators
    a = torch.load(EXPS / tag / 'model.pth', map_location='cpu', weights_only=False)['args']
    ch = a.get('img_ch', 1)
    m = MMCLASTcg(a['ngf'], a['n_blocks'], a['n_flow'], a['flow_hidden'], bool(a['pre_relu']), img_ch=ch)
    path_critic = (a.get('w_path_gan') or 0) > 0
    D = make_discriminators(a.get('ndf', 64), mix=path_critic,
                            mix_b=bool(path_critic and a.get('path_bidir') and a.get('path_critics') == 'separate'),
                            img_ch=ch)
    inf = trainable([m])
    return inf, inf + trainable(D.values())


def cost_rows(legacy=False):
    """{dataset: {row: (params infer, params train, train GPU-h, infer ms)}}.

    Every run a baseline row needs, both directions: a two-network baseline that
    steps both networks in one run counts that run once, CycleGAN counts its
    forward and native-reverse runs. Our rows count their three stages; the
    CycleGAN warm start is not included. legacy=True doubles the single-run
    two-network baselines, as the submission did.
    """
    import json
    from inpaper_utils.count_params import rows as param_rows
    train = json.loads((EXPS / 'bench_cost/train.json').read_text())
    infer = json.loads((EXPS / 'bench_cost/infer.json').read_text())
    out = {}
    for ds, ch, size in (('mnist', 3, 64), ('adni', 1, 112)):
        pars, tr, inf, rows = param_rows(ch, size), train[ds], infer[ds], {}
        for name in COST_ORDER:
            pi, pt = dit_params({'mnist': 'dit_mnist', 'adni': 'dit_adni_scratch'}[ds], ch, size) \
                if name == 'DiT' else pars[name]
            t = tr[name]
            single_run = name not in ('CycleGAN', 'RevGAN')
            h = t['gpu_h'] if legacy or not single_run else t.get('gpu_h_measured', t['gpu_h'] / t['runs'])
            rows[name] = (pi / 1e6, pt / 1e6, h, inf[name]['ms_per_image_mean'])
        ms_ours = inf['MMCLAST-cg, morph-smooth']['ms_per_image_mean']    # one architecture, both rows
        for name, tag in OURS_ROWS[ds]:
            pi, pt = ours_params(tag)
            rows[name] = (pi / 1e6, pt / 1e6, tr[BENCH[name]]['gpu_h'], ms_ours)
        out[ds] = rows
    return out


def fmt_ms(v):
    return f'{v:.0f}' if v >= 100 else f'{v:.1f}' if v >= 10 else f'{v:.2f}'


def latest_meta(ds):
    runs = sorted((EXPS / ds / 'paper_compare').glob('*/cache/meta.json'))
    if not runs:
        raise SystemExit(f'no paper_compare cache for {ds}')
    return json.loads(runs[-1].read_text()), runs[-1].parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='only report which published rows moved')
    ap.add_argument('--cost', action='store_true', help='print the size-and-cost block of tab:translation')
    ap.add_argument('--legacy', action='store_true', help='with --cost: the submission\'s accounting')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    if a.cost:
        c = cost_rows(a.legacy)
        for name in COST_ORDER + ['FlowCycle', 'FlowCycle + MS']:
            cells = []
            for ds in ('mnist', 'adni'):
                pi, pt, h, ms = c[ds][name]
                cells += [f'{pi:.2f}', f'{pt:.2f}', f'{h:.2f}', fmt_ms(ms)]
            print(f'{name} & ' + ' & '.join(cells) + ' \\\\')
        return
    from inpaper_utils.count_params import rows as param_rows
    train = json.loads((EXPS / 'bench_cost/train.json').read_text()) if \
        (EXPS / 'bench_cost/train.json').exists() else {}
    infer = json.loads((EXPS / 'bench_cost/infer.json').read_text()) if \
        (EXPS / 'bench_cost/infer.json').exists() else {}
    PUB = published()
    for ds in ('mnist', 'adni'):
        meta, cache = latest_meta(ds)
        pars = param_rows(3 if ds == 'mnist' else 1, 64 if ds == 'mnist' else 112)
        print(f'\n% {ds}: {meta["n"]} held-out pairs, cache {cache}')
        for name in ORDER:
            if name not in meta['mean_ssim']:
                print(f'% {name}: not in the cache yet')
                continue
            s, q = meta['mean_ssim'][name], meta['mean_psnr'][name]
            pi, pt = (v / 1e6 for v in pars[name])
            th = train.get(ds, {}).get(name, {}).get('gpu_h', float('nan'))
            ms = infer.get(ds, {}).get(name, {}).get('ms_per_image_mean', float('nan'))
            got = (round(s[0], 4), round(q[0], 2), round(s[1], 4), round(q[1], 2),
                   round(pi, 2), round(pt, 2), round(th, 2), round(ms, 2))
            pub = PUB[ds].get(name)
            mark = ''
            if pub:
                diff = [i for i, (g, p) in enumerate(zip(got, pub)) if abs(g - p) > tolerance(i, p)]
                mark = '  % reproduces' if not diff else f'  % MOVED at columns {diff}: was {pub}'
            if a.check:
                print(f'{name:10s} {got}{mark}')
            else:
                print(f'{name:21s} & {s[0]:.4f} & {q[0]:.2f} & {s[1]:.4f} & {q[1]:.2f} & '
                      f'{pi:.2f} & {pt:.2f} & {th:.2f} & {ms:.2f} \\\\{mark}')
        if 'Ours' in meta['mean_ssim']:
            # paper_compare's "Ours" is the morph-smooth checkpoint, i.e. the FlowCycle + MS row
            s, q = meta['mean_ssim']['Ours'], meta['mean_psnr']['Ours']
            got = (round(s[0], 4), round(q[0], 2), round(s[1], 4), round(q[1], 2))
            pub = PUB[ds].get('FlowCycle + MS')
            ok = pub and all(abs(g - p) <= tolerance(i, p) for i, (g, p) in enumerate(zip(got, pub)))
            print(f'% FlowCycle + MS (cache "Ours"): {got}  '
                  + ('% reproduces' if ok else f'% MOVED: table says {pub}'))


if __name__ == '__main__':
    main()
