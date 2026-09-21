#!/usr/bin/env python3
"""Pack the trained networks and both datasets, with their splits, into one archive.

  python -m inpaper_utils.package_release                  # -> /ix/lzhan/siyuan/releases/
  python -m inpaper_utils.package_release --list           # what would go in, with sizes
  python -m inpaper_utils.package_release --no-v1          # drop the 207 MB DEC-FA volume

Contents:

  checkpoints/<dataset>/<role>/   every network the paper reports or draws -- the seven
                                  baselines and ours, per dataset -- together with the
                                  small files beside them (final_eval.txt, train.log,
                                  train_log.csv, paper_eval*.json). DiT's train.log is
                                  not optional: its latent normalisation is parsed back
                                  out of it. The curriculum's stage1/2/3 snapshots are
                                  left out; model.pth is the trained network.
  data/adni/                      paired_112.pt (every T1/FA slice pair at 112x112),
                                  labels.csv, and v1_112.pt for the DEC-FA colouring.
  data/mnist/                     mnist_petct_paired.tar.gz exactly as built, with the
                                  MANIFEST.sha256 it shipped with.
  splits/                         who is in train / val / test, resolved to subject ids
                                  and file names, plus the code path that produces them.
  configs/                        the training configs; the model code is the repository.
  MANIFEST.json                   size, sha256 and role of every file in the archive.

The ADNI half is derived from ADNI data held under a Data Use Agreement: this archive is
for our own use and backup, and is not ours to redistribute. The MNIST half is synthetic
and carries no such restriction. README.md inside the archive says so as well.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
DATA = Path('/ix/lzhan/siyuan/datasets/processed_datas')
OUT = Path(os.environ.get('CYCLEFLOW_RELEASE', '/ix/lzhan/siyuan/releases'))
NAME = 'cycleflow_models_data'
AUX = ('final_eval.txt', 'train.log', 'train_log.csv', 'paper_eval.json', 'paper_eval_val.json',
       'probe_eval.txt')

# role -> (checkpoint directory under EXPS, the weight files, what the paper uses it for).
# Two names per dataset are ours: the table row and, on MNIST, the checkpoint the figures
# draw, which is a different one (it reproduces the PET acquisition grain; grain_probe.py).
CKPTS = {
    'adni': [
        ('ours_morph_smooth', 'adni_morph_smooth', ['model.pth'],
         'Ours w/ Morph-smooth in Table 2, and every ADNI figure (2, 4, 5, 6, 7)'),
        ('ours_base', 'adni_base', ['model.pth'], 'Ours w/o Morph-smooth in Table 2'),
        ('cyclegan_a2b', 'adni_host', ['last.pth'], 'CycleGAN, T1->FA'),
        ('cyclegan_b2a', 'adni/checkpoints/host_rev', ['last.pth'],
         'CycleGAN, FA->T1 -- a separately trained network, which is the point of the row'),
        ('ddpm', 'ddpm_adni', ['last.pth'], 'DDPM, 50 DDIM steps'),
        ('meanflow', 'meanflow_adni', ['last.pth'], 'MeanFlow, one step'),
        ('cfm', 'cfm_adni', ['last.pth'], 'CFM, 10 Euler steps'),
        ('revgan', 'adni/checkpoints/revgan_adni', ['last.pth'],
         'RevGAN -- one network for both directions, its reversible core shared'),
        ('recflow', 'adni/checkpoints/recflow_adni', ['last.pth'],
         'RectFlow, one Euler step; the file holds the reflowed field (net_ab/net_ba) '
         'and the 1-rectified flow it was reflowed from (net_ab_1rf/net_ba_1rf)'),
        ('dit', 'dit_adni_scratch', ['last.pth', 'latent_ae.pth'],
         'DiT, 1000 ancestral steps, with its latent autoencoder'),
    ],
    'mnist': [
        ('ours_morph_smooth_table', 'mnist/checkpoints/tune_mnist_smooth_g0.1_s0.3', ['model.pth'],
         'Ours w/ Morph-smooth in Table 1 (the validation-selected tuning run)'),
        ('ours_base_table', 'mnist_p_base', ['model.pth'], 'Ours w/o Morph-smooth in Table 1'),
        ('ours_morph_smooth_figures', 'mnist/checkpoints/mnist_p_morph_smooth', ['model.pth'],
         'the MNIST figures (1, 3, 8, 9) and every representation experiment'),
        ('cyclegan_a2b', 'mnist_host', ['last.pth'], 'CycleGAN, MRI->PET'),
        ('cyclegan_b2a', 'mnist/checkpoints/host_rev', ['last.pth'], 'CycleGAN, PET->MRI'),
        ('ddpm', 'ddpm_mnist', ['last.pth'], 'DDPM, 50 DDIM steps'),
        ('meanflow', 'meanflow_mnist', ['last.pth'], 'MeanFlow, one step'),
        ('cfm', 'cfm_mnist', ['last.pth'], 'CFM, 10 Euler steps'),
        ('revgan', 'mnist/checkpoints/revgan_mnist', ['last.pth'],
         'RevGAN -- one network for both directions, its reversible core shared'),
        ('recflow', 'mnist/checkpoints/recflow_mnist', ['last.pth'],
         'RectFlow, one Euler step; the file holds the reflowed field (net_ab/net_ba) '
         'and the 1-rectified flow it was reflowed from (net_ab_1rf/net_ba_1rf)'),
        ('dit', 'dit_mnist', ['last.pth', 'latent_ae.pth'],
         'DiT, 1000 ancestral steps, with its latent autoencoder'),
    ],
}
DATASETS = {
    'adni': [('ADNI_CycleFlow/paired_112.pt', 'every T1/FA slice pair, 112x112, and the '
                                              'subject/z index the splits are taken on'),
             ('ADNI_CycleFlow/labels.csv', 'the subject-level diagnosis labels'),
             ('ADNI_CycleFlow/v1_112.pt', 'the principal diffusion direction, for DEC-FA rendering')],
    'mnist': [('MNIST_CycleFlow/mnist_petct_paired.tar.gz',
               'the built dataset: trainA/trainB (disjoint indices, unpaired training) and '
               'testA/testB (the same indices in both, paired evaluation)'),
              ('MNIST_CycleFlow/MANIFEST.sha256', 'the checksums it was built with')],
}


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return f'{n:.0f} {u}' if u == 'B' else f'{n:.1f} {u}'
        n /= 1024


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def collect(no_v1):
    """(source, arcname, role) for every file that goes in, resolved and existing."""
    out = []
    for ds, entries in CKPTS.items():
        for role, tag, weights, what in entries:
            d = EXPS / tag
            if not d.is_dir():
                raise SystemExit(f'no checkpoint directory {d}')
            for w in weights:
                f = d / w
                if not f.exists():
                    raise SystemExit(f'missing {f}')
                out.append((f, f'checkpoints/{ds}/{role}/{w}', f'{what} [{tag}]'))
            for a in AUX:
                if (d / a).exists():
                    out.append((d / a, f'checkpoints/{ds}/{role}/{a}', f'training record of {tag}'))
    for ds, entries in DATASETS.items():
        for rel, what in entries:
            if no_v1 and rel.endswith('v1_112.pt'):
                continue
            f = DATA / rel
            if not f.exists():
                raise SystemExit(f'missing {f}')
            out.append((f, f'data/{ds}/{Path(rel).name}', what))
    for c in sorted((ROOT / 'configs').glob('*.yaml')):
        if c.name.startswith('h2z_'):
            continue                                  # horse2zebra, not a dataset in this paper
        out.append((c, f'configs/{c.name}', 'training config'))
    return out


def adni_split():
    """The paper's split, by subject: seed 42, 20% test, label_4, axial z = 40..49."""
    sys.path.insert(0, str(ROOT))
    import torch
    from data.paired_dataset import subject_level_split, CACHE
    tr, te = subject_level_split(42, .2, 'label_4', 40, 49)
    d = torch.load(CACHE, map_location='cpu')
    subj, subj_idx = d['subjects'], d['subj_idx'].numpy()
    who = lambda idx: sorted({subj[subj_idx[int(i)]] for i in idx})
    return {
        'call': "data.paired_dataset.subject_level_split(seed=42, test_frac=0.20, "
                "label_scheme='label_4', z_lo=40, z_hi=49)",
        'note': 'Split by subject, so no subject has slices on both sides. There is no '
                'validation split on ADNI: model selection was done on MNIST. The indices '
                'below index paired_112.pt directly.',
        'train': {'subjects': who(tr), 'n_subjects': len(who(tr)), 'n_slices': len(tr),
                  'indices': [int(i) for i in tr]},
        'test': {'subjects': who(te), 'n_subjects': len(who(te)), 'n_slices': len(te),
                 'indices': [int(i) for i in te]},
    }


def mnist_split():
    """train / val / test as train.py sees them; val is carved off the front of train."""
    root = DATA / 'MNIST_CycleFlow/mnist_petct_paired'
    names = {f: sorted(p.name for p in (root / f).glob('*.png'))
             for f in ('trainA', 'trainB', 'testA', 'testB')}
    n_val = max(1, round(.1 * len(names['trainA'])))
    return {
        'call': "data.unpaired_dataset.folder_loaders(root, ..., val_frac=0.1) -- the first "
                "n_val images of each training folder are the validation set and are skipped "
                "when training; test is testA/testB.",
        'note': 'trainA and trainB hold disjoint MNIST indices, so training never sees a '
                'pair; testA and testB hold the same indices in matching filename order, '
                'which is what makes the evaluation paired.',
        'n_val': n_val,
        'counts': {f'{f}': len(v) for f, v in names.items()}
        | {'trainA_used_for_training': len(names['trainA']) - n_val,
           'trainB_used_for_training': len(names['trainB']) - n_val},
        'val': {f: names[f][:n_val] for f in ('trainA', 'trainB')},
        'test': {f: names[f] for f in ('testA', 'testB')},
    }


def readme(files, commit):
    by = {}
    for src, arc, what in files:
        by.setdefault(arc.split('/')[0], []).append((arc, src.stat().st_size, what))
    total = sum(s.stat().st_size for s, _, _ in files)
    d = [f'# CycleFlow: trained networks and data', '',
         f'Packed {dt.date.today().isoformat()} from `{EXPS}` and `{DATA}`, '
         f'repository commit `{commit}`. {len(files)} files, {human(total)} unpacked.', '',
         '## ADNI is under a Data Use Agreement', '',
         'The `data/adni/` and `checkpoints/adni/` halves derive from ADNI data held under a '
         'DUA. This archive is for our own use and backup; it is not ours to redistribute, and '
         'it must not be uploaded as supplementary material. The MNIST half is synthetic and '
         'carries no such restriction.', '',
         '## Networks', '',
         'Every network the paper reports or draws. `model.pth` (ours) and `last.pth` '
         '(baselines) are the trained weights; the text files beside them are that run\'s own '
         'record. The curriculum\'s stage1/stage2/stage3 snapshots are not included -- they are '
         'intermediate optimiser states, not the trained network.', '',
         '| in the archive | what the paper uses it for |', '|---|---|']
    for ds, entries in CKPTS.items():
        for role, tag, weights, what in entries:
            d.append(f'| `checkpoints/{ds}/{role}/` | {what} (`{tag}`) |')
    d += ['', '## Data and splits', '',
          'ADNI: `paired_112.pt` holds every slice pair; the split is by subject and is '
          'reproduced exactly by the call in `splits/adni_split.json`, which also lists the '
          'subject ids and the slice indices on each side. There is no ADNI validation split.',
          '', 'MNIST: the tarball is the built dataset. `splits/mnist_split.json` gives the '
          'train / val / test counts and the exact file names of the validation set, which is '
          'carved off the front of each training folder and excluded from training.', '',
          '## Files', '', '| path | size | what it is |', '|---|---|---|']
    for group in ('checkpoints', 'data', 'splits', 'configs'):
        for arc, n, what in sorted(by.get(group, [])):
            d.append(f'| `{arc}` | {human(n)} | {what} |')
    d += ['', 'Sizes and sha256 for every file are in `MANIFEST.json`.']
    return '\n'.join(d) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='print the contents and stop')
    ap.add_argument('--no-v1', action='store_true', help='leave out v1_112.pt (DEC-FA only)')
    ap.add_argument('--out', default=str(OUT), help=f'destination directory (default {OUT})')
    ap.add_argument('--stage', default=os.environ.get('SLURM_SCRATCH', '/tmp'),
                    help='where the symlink tree is built')
    ap.add_argument('--jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args()
    files = collect(a.no_v1)
    total = sum(s.stat().st_size for s, _, _ in files)
    if a.list:
        for src, arc, what in files:
            print(f'{human(src.stat().st_size):>9}  {arc:<58} {what[:60]}')
        print(f'{len(files)} files, {human(total)} unpacked')
        return 0

    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip() or 'unknown'
    stage = Path(a.stage) / NAME
    subprocess.run(['rm', '-rf', str(stage)], check=True)
    for src, arc, _ in files:                       # symlinks; tar -h reads through them
        dst = stage / arc
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
    print(f'{len(files)} files staged, {human(total)} unpacked', flush=True)

    (stage / 'splits').mkdir(exist_ok=True)
    (stage / 'splits/mnist_split.json').write_text(json.dumps(mnist_split(), indent=1))
    print('mnist split written', flush=True)
    (stage / 'splits/adni_split.json').write_text(json.dumps(adni_split(), indent=1))
    print('adni split written', flush=True)

    manifest = {'packed_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'git_commit': commit,
                'exps': str(EXPS), 'data': str(DATA), 'n_files': len(files),
                'bytes_unpacked': total, 'files': []}
    for src, arc, what in sorted(files, key=lambda t: t[1]):
        manifest['files'].append({'path': arc, 'source': str(src.resolve()),
                                  'bytes': src.stat().st_size, 'sha256': sha256(src), 'role': what})
        print(f'  hashed {arc}', flush=True)
    for p in sorted((stage / 'splits').glob('*.json')):
        manifest['files'].append({'path': f'splits/{p.name}', 'source': 'computed here',
                                  'bytes': p.stat().st_size, 'sha256': sha256(p),
                                  'role': 'the split, resolved to ids and file names'})
    (stage / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    (stage / 'README.md').write_text(readme(files, commit))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tgz = out / f'{NAME}.tar.gz'
    subprocess.run(['tar', '-C', str(stage.parent), '-h', '-I', f'pigz -p {a.jobs}',
                    '-cf', str(tgz), NAME], check=True)
    print(f'\n{tgz}  {human(tgz.stat().st_size)}  ({human(total)} unpacked)')
    n = subprocess.run(['tar', '-tzf', str(tgz)], capture_output=True, text=True).stdout
    print(f'{len([l for l in n.splitlines() if not l.endswith("/")])} files in the archive')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
