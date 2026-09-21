#!/usr/bin/env python3
"""Select the MNIST-PET/CT stage-3 configs of our two table rows on VALIDATION only.

  python -m inpaper_utils.tune_mnist_collect

Reads EXPS/mnist/checkpoints/tune_mnist_*/paper_eval_val.json (selection) and
paper_eval.json (reported). Rule, fixed before looking at test: within each variant,
the config with the lowest sum of its ranks over the four validation metrics
(SSIM and PSNR, both directions) wins; ties go to the higher CT->PET PSNR. The test
numbers of the winners are then set against the baselines of Table 1 to show whether
our two rows take top 1 and top 2 in every column. (They do not, in one: RectFlow's
one-step MRI->PET PSNR is 31.25 against our 30.78.)
"""
import json
import os
from pathlib import Path

EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
COLS = [('CT->PET', 'ssim'), ('CT->PET', 'psnr'), ('PET->CT', 'ssim'), ('PET->CT', 'psnr')]
BASELINES = {   # Table 1 test numbers
    'DDPM': (0.1953, 25.46, 0.4416, 22.82), 'CycleGAN': (0.8280, 28.47, 0.9463, 25.67),
    'RevGAN': (0.8266, 28.37, 0.9404, 25.29),
    'MeanFlow': (0.8158, 30.33, 0.4055, 18.66), 'DiT': (0.8314, 28.76, 0.9196, 23.71),
    'CFM': (0.8412, 31.12, 0.9028, 24.66), 'RectFlow': (0.8226, 31.25, 0.8960, 25.06)}
CURRENT = {'Ours w/o Morph-smooth': (0.8657, 30.55, 0.9525, 26.54),
           'Ours w/ Morph-smooth': (0.8405, 29.61, 0.9435, 25.99)}


def vec(ev):
    return tuple(ev[d][m] for d, m in COLS)


def main():
    runs = {}
    # The two current Table 1 checkpoints compete under the same rule
    # (scored by inpaper_utils/tune_mnist_current_eval.sbatch).
    current = {'tune_mnist_base_current': EXPS / 'mnist_p_base',
               'tune_mnist_smooth_current': EXPS / 'mnist' / 'checkpoints' / 'mnist_p_morph_smooth'}
    for name, d in current.items():
        d = (d / 'model.pth').resolve().parent
        val, test = d / 'paper_eval_val.json', d / 'paper_eval.json'
        if val.exists() and test.exists():
            runs[name] = (vec(json.loads(val.read_text())), vec(json.loads(test.read_text())))
    for d in sorted((EXPS / 'mnist' / 'checkpoints').glob('tune_mnist_*')):
        val, test = d / 'paper_eval_val.json', d / 'paper_eval.json'
        if val.exists() and test.exists():
            runs[d.name] = (vec(json.loads(val.read_text())), vec(json.loads(test.read_text())))
    fmt = lambda v: '  '.join(f'{x:.4f}' if i % 2 == 0 else f'{x:6.2f}' for i, x in enumerate(v))
    print(f'{"config":34s} {"val: CT>PET ssim psnr  PET>CT ssim psnr":44s} test')
    for k, (v, t) in runs.items():
        print(f'{k:34s} {fmt(v):44s} {fmt(t)}')
    chosen = {}
    for label, prefix in (('Ours w/o Morph-smooth', 'tune_mnist_base_'), ('Ours w/ Morph-smooth', 'tune_mnist_smooth_')):
        cand = {k: v for k, v in runs.items() if k.startswith(prefix)}
        if not cand:
            continue
        ranks = {k: 0 for k in cand}
        for j in range(4):
            for r, k in enumerate(sorted(cand, key=lambda k: -cand[k][0][j])):
                ranks[k] += r
        best = min(cand, key=lambda k: (ranks[k], -cand[k][0][1]))
        chosen[label] = (best, cand[best][1])
        print(f'\n{label}: {best}  (val rank sum {ranks[best]})')
    if len(chosen) == 2:
        table = {**BASELINES, **{lab: t for lab, (_, t) in chosen.items()}}
        print('\ntest, chosen configs against Table 1 baselines (top1 / top2 per column):')
        ok = True
        for j, (d, m) in enumerate(COLS):
            order = sorted(table, key=lambda k: -table[k][j])
            ours = set(chosen)
            ok &= set(order[:2]) == ours
            print(f'  {d} {m}: 1 {order[0]} {table[order[0]][j]:.4f}   2 {order[1]} {table[order[1]][j]:.4f}'
                  f'   {"OK" if set(order[:2]) == ours else "not both ours"}')
        print('ALL COLUMNS TOP-2 OURS' if ok else 'NOT YET')
        (EXPS / 'mnist' / 'tune_selection.json').write_text(json.dumps(
            {lab: {'config': c, 'test': dict(zip([f'{d} {m}' for d, m in COLS], t))} for lab, (c, t) in chosen.items()},
            indent=2) + '\n')


if __name__ == '__main__':
    main()
