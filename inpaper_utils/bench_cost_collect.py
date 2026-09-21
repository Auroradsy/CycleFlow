#!/usr/bin/env python3
"""Training GPU-hours on one A100: steady epoch time x the epochs of the real run.

  python -m inpaper_utils.bench_cost_collect

Reads the 4-epoch runs of inpaper_utils/bench_train.sbatch (tags bench_*) and the
epoch counts of the runs behind Table 2. Steady epoch time = median over epochs 2..3,
dropping the warm-up epoch and the last one (which runs the periodic evaluation).
Costs cover both translation directions: every baseline is counted as two
single-direction models, so its run time is x2 (author's accounting; note that
CFM/MeanFlow/DDPM/DiT runs already step both direction networks, see
inpaper_utils/bench_train_step.py). MMCLAST-cg is one model for both directions and
counts stages 1-3 only (its CycleGAN initialisation is not included).
JSON -> EXPS/bench_cost/train.json.
"""
import csv
import json
import os
from pathlib import Path
import re
import statistics

EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
SPEC = {   # real schedules behind the result tables
    'adni': {'host': 160, 'base_ep': 120, 'dit_ae': 60, 'dit': 200,
             'revgan_ep': 160, 'recflow_ep': 120,
             'ours_base_log': EXPS / 'adni_base/train_log.csv',
             'morph_log': EXPS / 'adni_morph/train_log.csv', 'smooth_log': EXPS / 'adni_morph_smooth/train_log.csv'},
    'mnist': {'host': 100, 'base_ep': 40, 'dit_ae': 30, 'dit': 200,
              'revgan_ep': 100, 'recflow_ep': 40,
              'ours_base_log': EXPS / 'mnist_p_base/train_log.csv',
              'morph_log': EXPS / 'mnist_p_morph/train_log.csv',
              # Table 1's morph-smooth row is the validation-selected tune_mnist_* config
              # (80 stage-3 epochs; the original mnist_p_morph_smooth ran 33)
              'smooth_log': (EXPS / 'mnist/checkpoints/tune_mnist_smooth_g0.1_s0.3').resolve().parents[1]
              / 'logs/tune_mnist_smooth_g0.1_s0.3/train_log.csv'},
}


def steady(secs):
    body = secs[1:-1] if len(secs) >= 3 else secs[1:]
    return statistics.median(body)


def added_baselines(ds, o, sp):
    """RevGAN and RectFlow, on the same bench_* proxy as every row above.

    RevGAN is ONE run for both directions -- its core is shared, so there is no
    second run the way CycleGAN has a forward host and a native reverse host --
    and is therefore not doubled. RectFlow is two networks stepped in one run,
    exactly as CFM is, so it follows the accounting of the rows above (see the
    note in the module docstring) and `gpu_h_measured` records the undoubled
    figure. Its two stages have different epoch costs, and the reflow simulation
    between them is training cost too: it is what the second stage learns from.
    """
    for tag, name in ((f'bench_revgan_{ds}', 'RevGAN'), (f'bench_recflow_{ds}', 'RectFlow')):
        log = EXPS / ds / 'checkpoints' / tag / 'train_log.csv'
        if not log.exists():
            print(f'{ds:5s} {name:26s} no benchmark run yet ({log})')
            continue
        rs = rows(log)
        if name == 'RevGAN':
            s_ = steady([float(r['sec']) for r in rs])
            o[name] = {'epoch_s': s_, 'epochs': sp['revgan_ep'], 'runs': 1,
                       'gpu_h': s_ * sp['revgan_ep'] / 3600}
        else:
            per = {st: steady([float(r['sec']) for r in rs if r['stage'] == st]) for st in ('1', '2')}
            con = (run_dir(ds, tag) / 'console.log').read_text()
            sim = re.search(r'reflow: \d+ generated pairs per direction, \d+ Euler steps \((\d+)s\)', con)
            sim_s = float(sim.group(1)) if sim else 0.0
            train_s = sum(per[st] * sp['recflow_ep'] for st in per) + sim_s
            o[name] = {'epoch_s': per, 'epochs': {st: sp['recflow_ep'] for st in per},
                       'reflow_sim_s': sim_s, 'runs': 2,
                       'gpu_h': 2 * train_s / 3600, 'gpu_h_measured': train_s / 3600}
    return o


def run_dir(ds, tag):
    return (EXPS / ds / 'checkpoints' / tag).resolve().parents[1]


def rows(p):
    return list(csv.DictReader(open(p)))


def stage_counts(p):
    c = {}
    for r in rows(p):
        c[r['stage']] = c.get(r['stage'], 0) + 1
    return c


def main():
    out = {}
    for ds, sp in SPEC.items():
        try:
            out[ds] = one_dataset(ds, sp)
        except (FileNotFoundError, OSError, ValueError, statistics.StatisticsError) as e:
            print(f'{ds}: benchmark incomplete ({type(e).__name__}: {e}); skipped')
    dest = EXPS / 'bench_cost'; dest.mkdir(parents=True, exist_ok=True)
    (dest / 'train.json').write_text(json.dumps(out, indent=2) + '\n')
    print('->', dest / 'train.json')


def one_dataset(ds, sp):
    if True:
        o = {}
        con = (run_dir(ds, f'bench_host_{ds}') / 'console.log').read_text()
        host = steady([float(s) for s in re.findall(r'\[ep \d+/\d+\].*\((\d+)s\)', con)])
        o['CycleGAN'] = {'epoch_s': host, 'epochs': sp['host'], 'runs': 2, 'gpu_h': 2 * host * sp['host'] / 3600}
        for m, name in (('cfm', 'CFM'), ('meanflow', 'MeanFlow'), ('ddpm', 'DDPM')):
            s = steady([float(r['sec']) for r in rows(EXPS / ds / 'checkpoints' / f'bench_{m}_{ds}' / 'train_log.csv')])
            o[name] = {'epoch_s': s, 'epochs': sp['base_ep'], 'runs': 2, 'gpu_h': 2 * s * sp['base_ep'] / 3600}
        con = (run_dir(ds, f'bench_dit_{ds}') / 'console.log').read_text()
        ae = steady([float(s) for s in re.findall(r'\[latent-ae\] ep +\d+/\d+ +loss \S+ +(\d+)s', con)])
        dit = steady([float(r['sec']) for r in rows(EXPS / ds / 'checkpoints' / f'bench_dit_{ds}' / 'train_log.csv')])
        o['DiT'] = {'ae_epoch_s': ae, 'epoch_s': dit, 'ae_epochs': sp['dit_ae'], 'epochs': sp['dit'], 'runs': 2,
                    'gpu_h': 2 * (ae * sp['dit_ae'] + dit * sp['dit']) / 3600}
        bm = rows(run_dir(ds, f'bench_morph_{ds}') / 'logs' / f'bench_morph_{ds}' / 'train_log.csv')
        bs = rows(run_dir(ds, f'bench_msmooth_{ds}') / 'logs' / f'bench_msmooth_{ds}' / 'train_log.csv')
        per_stage = {st: steady([float(r['sec']) for r in bm if r['stage'] == st]) for st in ('1', '2', '3')}
        per_stage_smooth3 = steady([float(r['sec']) for r in bs if r['stage'] == '3'])
        n_morph, n_smooth = stage_counts(sp['morph_log']), stage_counts(sp['smooth_log'])
        # base ("Ours"): S1/S2 epochs cost the same as morph's; its S3 has neither latcyc nor path terms
        bb = rows(run_dir(ds, f'bench_base_{ds}') / 'logs' / f'bench_base_{ds}' / 'train_log.csv')
        base3, n_base = steady([float(r['sec']) for r in bb if r['stage'] == '3']), stage_counts(sp['ours_base_log'])
        o['MMCLAST-cg, base'] = {
            'epoch_s': {'1': per_stage['1'], '2': per_stage['2'], '3': base3}, 'epochs': n_base,
            'gpu_h': (per_stage['1'] * n_base.get('1', 0) + per_stage['2'] * n_base.get('2', 0)
                      + base3 * n_base.get('3', 0)) / 3600}
        o['MMCLAST-cg, morph'] = {'epoch_s': per_stage, 'epochs': n_morph,
                                  'gpu_h': sum(per_stage[s] * n_morph.get(s, 0) for s in per_stage) / 3600}
        o['MMCLAST-cg, morph-smooth'] = {
            'epoch_s': {**{s: per_stage[s] for s in ('1', '2')}, '3': per_stage_smooth3},
            'epochs': {'1': n_morph.get('1', 0), '2': n_morph.get('2', 0), '3': n_smooth.get('3', 0)},
            'gpu_h': (per_stage['1'] * n_morph.get('1', 0) + per_stage['2'] * n_morph.get('2', 0)
                      + per_stage_smooth3 * n_smooth.get('3', 0)) / 3600}
        added_baselines(ds, o, sp)
        for k, v in o.items():
            print(f'{ds:5s} {k:26s} {v["gpu_h"]:6.2f} GPU-h   {json.dumps({kk: vv for kk, vv in v.items() if kk != "gpu_h"})}')
        return o


if __name__ == '__main__':
    main()
