#!/usr/bin/env python3
"""Stage, per paper figure, the code and the data that produced it.

  python -m inpaper_utils.collect_fig_sources            # rebuild the whole tree
  python -m inpaper_utils.collect_fig_sources --check    # report only, write nothing
  python -m inpaper_utils.collect_fig_sources --archive  # rebuild, then zip it for upload

Writes __X_files/iclr2027/figs_scripts/fig<N>/ in the paper's own figure order:

  figure/   the PDF(s) the paper includes, byte-identical to __X_files/iclr2027/figures/
  code/     the scripts that drew it. When the run directory kept its own snapshot of a
            script -- the runners copy themselves next to their output -- that copy wins
            over the repository's current file, because it is the version that actually
            ran, and README.md says so when the two differ.
  data/     everything small the run produced: manifests (which held-out slices, which
            checkpoints, which sampler), per-image SSIM/PSNR, the result JSONs of the
            representation experiments. Large inputs are not copied; they are listed in
            README.md with their path and size.
  README.md what the figure shows, the exact commands, the pinned run directory and its
            git commit, and a line per file.

What is deliberately *not* copied: model checkpoints, the dataset caches, and the
prediction caches under <run>/cache (42 MB on ADNI, 409 MB on MNIST). Those are inputs,
not products, they live on /ix, and the ADNI ones are not ours to redistribute -- no voxel
data of any subject is written into this tree, only the identifiers and the metrics that
the figures' own captions already quote.

Figures 8 and 9 are the exception that is fully self-contained: their inputs are small
enough to travel, so `data/` holds everything needed to redraw them with no model, no
dataset and no GPU.
"""
import argparse
import filecmp
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
SNAP = ROOT / 'snapshot_results'
PAPER = ROOT / '__X_files/iclr2027'
DEST = PAPER / 'figs_scripts'
DATASETS = {
    'adni': ['/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow/paired_112.pt',
             '/ix/lzhan/siyuan/datasets/processed_datas/ADNI_CycleFlow/v1_112.pt'],
    'mnist': ['/ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired.tar.gz'],
}
ENV = 'source configs/_env.sh   # /ix/lzhan/siyuan/envs/cycleflow, MMCLAST_EXPS, dataset roots'

# Every entry is pinned to the run that produced the figure the paper currently includes,
# not to "the latest run": re-running an experiment must not silently re-point the paper's
# provenance. `verify` names the copy of the figure inside the run directory; where the
# renderer writes straight into snapshot_results/ there is none, and the run's manifest
# lists the file names instead.
FIGS = [
    dict(n=1, label='fig:mnist-compare', dataset='mnist',
         title='MNIST-PET/MRI: CycleFlow against the baselines, both directions',
         figures=[('mnist_petct/mnist_compare.pdf', '00_inpaper_fig_compare.pdf')],
         run='mnist/paper_compare/20260916T225122.934735Z-fa164c7a',
         code=['make_paper_compare.py', 'fig34_mnist.sbatch'], code_from='repo',
         data=[('00_inpaper_fig_compare_manifest.json',
                'which held-out pairs, which checkpoint per column, and the full-test SSIM/PSNR'),
               ('run.json', 'argv, git commit, python, start/end of the inference pass'),
               ('console.log', 'the run as it happened'),
               ('cache/meta.json', 'per-image record of every cached prediction'),
               ('cache/ssim.npz', 'per-image SSIM, all 5000 held-out pairs x 6 methods x 2 directions'),
               ('cache/psnr.npz', 'the same for PSNR')],
         cmd=['python -m inpaper_utils.make_paper_compare infer  --dataset mnist \\\n'
              '    --reuse <earlier cache> --ours-tag mnist/checkpoints/mnist_p_morph_smooth',
              'python -m inpaper_utils.make_paper_compare render --dataset mnist'],
         note='The examples are the three largest SSIM margins of ours over the best baseline per '
              'direction, with distinct digits, chosen before anything was drawn; the figure was later '
              're-rendered from the same cache with those indices passed explicitly, when domain A was '
              'renamed from CT to MRI, so `selection` in the manifest reads "manual --pick". The run '
              'directory holds its own snapshot of make_paper_compare.py, but that is the inference '
              'stage; the copy here is the repository version that drew the figure.'),
    dict(n=2, label='fig:adni-ablation', dataset='adni',
         title='ADNI: CycleFlow against the baselines, both directions, DEC-FA rendering',
         figures=[('adni/adni_compare.pdf', '00_inpaper_fig_compare_decfa.pdf')],
         run='adni/paper_compare/20260915T203025.886304Z-b3a09545',
         code=['make_paper_compare.py', 'compare_smooth.sbatch', 'decfa.py'],
         data=[('00_inpaper_fig_compare_decfa_manifest.json',
                'which held-out slices (subject, z), which checkpoint per column, full-test SSIM/PSNR'),
               ('00_inpaper_fig_compare_manifest.json', 'the same figure in grayscale, same slices'),
               ('run.json', 'argv, git commit, python, start/end of the inference pass'),
               ('console.log', 'the run as it happened'),
               ('cache/meta.json', 'per-image record of every cached prediction'),
               ('cache/ssim.npz', 'per-image SSIM, all 430 held-out slices x 6 methods x 2 directions'),
               ('cache/psnr.npz', 'the same for PSNR')],
         cmd=['python -m inpaper_utils.make_paper_compare infer  --dataset adni',
              'python -m inpaper_utils.make_paper_compare render --dataset adni --dec-fa'],
         note='DEC-FA: brightness is the FA value, hue the measured principal diffusion direction '
              '(decfa.py), so v1_112.pt is needed for the FA columns.'),
    dict(n=3, label='fig:mnist-path-ours', dataset='mnist',
         title='MNIST-PET/MRI: the state after every flow block, both directions',
         figures=[('mnist_petct/mnist_path_a2b.pdf', '00_inpaper_fig_path_ablation_a2b_ab.pdf'),
                  ('mnist_petct/mnist_path_b2a.pdf', '00_inpaper_fig_path_ablation_b2a_ab.pdf')],
         run='mnist/paper_path_ablation/20260916T225720.386907Z-3abfecca',
         code=['make_paper_path_ablation.py', 'fig34_mnist.sbatch'],
         data=[('manifest.json', 'the two examples per direction and the checkpoint they run through'),
               ('run.json', 'argv, git commit, python'),
               ('console.log', 'the run as it happened')],
         cmd=['python -m inpaper_utils.make_paper_path_ablation --dataset mnist \\\n'
              '    --pick "2254,3708,3631;2860,437,3768" --tags Ours=mnist/checkpoints/mnist_p_morph_smooth \\\n'
              '    --n 2 --suffix _ab --no-row-label --panel-labels a,b'],
         note='--pick are the test indices Figure 1 drew, so the two figures show the same samples. '
              'Every panel is a decoded state the network traverses; nothing is interpolated.'),
    dict(n=4, label='fig:adni-path-ours', dataset='adni',
         title='ADNI: the state after every flow block, both directions, DEC-FA',
         figures=[('adni/adni_path_a2b.pdf', '00_inpaper_fig_path_ablation_a2b_ab_decfa.pdf'),
                  ('adni/adni_path_b2a.pdf', '00_inpaper_fig_path_ablation_b2a_ab_decfa.pdf')],
         run='adni/paper_path_ablation/20260916T205428.577712Z-f2513aa0',
         code=['make_paper_path_ablation.py', 'compare_smooth.sbatch', 'decfa.py'],
         data=[('manifest.json', 'the two slices per direction and the checkpoint they run through'),
               ('run.json', 'argv, git commit, python'),
               ('console.log', 'the run as it happened')],
         cmd=['python -m inpaper_utils.make_paper_path_ablation --dataset adni --dec-fa \\\n'
              '    --variants morph_smooth --n 2 --pick "416,111,16;379,309,77" \\\n'
              '    --suffix _ab --no-row-label --panel-labels a,b'],
         note='--pick are the test indices Figure 2 drew. Only the first two per direction are shown.'),
    dict(n=5, label='fig:adni-flow-features', dataset='adni',
         title='ADNI: the 256x28x28 state after every block, in PCA colour, with the per-location change',
         figures=[('adni/adni_features.pdf', None)],
         run='adni/flow_viz/20260915T203032.213866Z-90b1ad18',
         code=['make_flow_viz.py', 'flowviz_smooth.sbatch'],
         data=[('manifest.json', 'the five held-out subjects, the checkpoint, the render settings, and the '
                                 'full-test summary the paper quotes (cycle SSIM/MAE, latent error, '
                                 'relative distance to the target code per block)'),
               ('run.json', 'argv, git commit, python'),
               ('console.log', 'the run as it happened')],
         cmd=['python -m inpaper_utils.make_flow_viz --dataset adni --variant morph_smooth'],
         note='This is the grayscale run that also computes the held-out summary quoted in Section 4.3 '
              'and 4.4 (1.29 -> 0.83 relative distance; latent cycle error 8.3e-6; image cycle SSIM 0.986). '
              'The figure name in the run is 00_inpaper_flow_features_sym_morph-smooth; the renderer writes '
              'straight into snapshot_results/adni/, which is why no copy sits in the run directory.'),
    dict(n=6, label='fig:adni-flow-affine', dataset='adni',
         title='ADNI: the affine terms log a and RMS b of each coupling layer, both directions',
         figures=[('adni/adni_affine_a2b.pdf', None), ('adni/adni_affine_b2a.pdf', None)],
         run='adni/flow_viz/20260917T204933.601065Z-1082c8ed',
         code=['make_flow_viz.py', 'cycle_ab.sbatch'],
         data=[('manifest.json', 'the two slices per direction, the checkpoint, and the shared colour limits '
                                 'of log a and of b'),
               ('run.json', 'argv, git commit, python'),
               ('console.log', 'the run as it happened')],
         cmd=['python -m inpaper_utils.make_flow_viz --dataset adni --variant morph_smooth --dec-fa \\\n'
              '    --figures cycle_ab,affine_ab --ab-pick "416,111;379,309" --skip-summary'],
         note='One run draws Figures 6 and 7 on the same four slices, so both folders pin it. The reverse '
              'direction is the same bijection traversed backwards: a\' = 1/a, b\' = -b/a.'),
    dict(n=7, label='fig:adni-cycle-b2a', dataset='adni',
         title='ADNI: the latent cycle and the image cycle, both directions, on a shared error scale',
         figures=[('adni/adni_cycle.pdf', None)],
         run='adni/flow_viz/20260917T204933.601065Z-1082c8ed',
         code=['make_flow_viz.py', 'cycle_ab.sbatch'],
         data=[('manifest.json', 'the two slices per direction, the checkpoint, and the shared error vmax'),
               ('run.json', 'argv, git commit, python'),
               ('console.log', 'the run as it happened')],
         cmd=['python -m inpaper_utils.make_flow_viz --dataset adni --variant morph_smooth --dec-fa \\\n'
              '    --figures cycle_ab,affine_ab --ab-pick "416,111;379,309" --skip-summary'],
         note='The numbers the paragraph quotes come from the full held-out split, i.e. from the summary in '
              'fig5/data/manifest.json, not from these four slices.'),
    dict(n=8, label='fig:mnist-pca', dataset='mnist',
         title='MNIST-PET/MRI: every state in a PCA fitted on the two endpoint code sets',
         figures=[('mnist_petct/mnist_pca_path.pdf', None)],
         run='mnist/latent_geometry/20260917T231928.199530Z-8fa231bb',
         code=['latent_geometry.py', 'domain_probe.py', 'latent_geometry.sbatch', 'replot_pca_path.py'],
         data=[('geometry.json', 'E2.1/E2.2/E2.4 over all 5000 held-out pairs: distance to each endpoint, '
                                 'cosine, path/chord, and the endpoint channel std'),
               ('domain_probe.json', 'E2.3/E2.5: the A/B margin per block, P(B) from a logistic probe and '
                                     'from 10-NN, and the endpoint probe accuracy that licenses them'),
               ('mnist_pca_proj.npz', 'DERIVED: the (5000, 5, 2) coordinates the figure plots -- five states '
                                      'per direction, two components -- exported '
                                      'from the 455 MB features.npz by replot_pca_path.py export')],
         cmd=['python -m inpaper_utils.latent_geometry --dataset mnist --variant morph_smooth --plot',
              'python -m inpaper_utils.domain_probe    --dataset mnist',
              '# redraw from the bundled 0.5 MB projection alone, no model and no dataset:',
              'python code/replot_pca_path.py plot --proj data/mnist_pca_proj.npz --out mnist_pca_path.pdf'],
         note='Self-contained. The PCA is fitted on the endpoint codes only and is linear, so distance and '
              'straightness survive the projection, which is why it is a PCA and not t-SNE or UMAP.'),
    dict(n=9, label='fig:mnist-missing-modality', dataset='mnist',
         title='MNIST-PET/MRI: a translation standing in for a modality that was never acquired',
         figures=[('mnist_petct/mnist_missing_modality.pdf', None)],
         run=None,
         runs={'mnist/cross_modal_probe/20260917T233253.847240Z-d0f56fd5':
               [('probe.json', 'E1b, panel (a): a probe fitted on real PET codes only, applied without '
                               'retraining to the translated / source / re-encoded codes, plus retrieval')],
               'mnist/aug/20260918T233014.288730Z-d9dd74d1':
               [('aug.json', 'E1c, panel (b): the corpus sweep in the MRI->PET direction, all six arms')],
               'mnist/aug/20260918T233436.713669Z-1cdf81cc':
               [('aug.json', 'E1c, panel (c): the same sweep in the PET->MRI direction')]},
         code=['cross_modal_probe.py', 'mnist_aug.py', 'repr_collect.py', 'repr_probe.sbatch'],
         cmd=['python -m inpaper_utils.cross_modal_probe --dataset mnist --variant morph_smooth',
              'python -m inpaper_utils.mnist_aug --labels 10 --seeds 3 --direction a2b \\\n'
              '    --corpus-sweep 100,300,1000,3000        # and again with --direction b2a',
              '# redraw from the bundled JSONs alone (data/ is laid out as an MMCLAST_EXPS tree):',
              'MMCLAST_EXPS=$PWD/data python -c "from inpaper_utils.repr_collect import aug_runs, \\\n'
              '    missing_modality_figure as f; f(aug_runs())"'],
         note='Self-contained. Three runs feed one figure, so data/ keeps the <dataset>/<experiment>/<run>/ '
              'layout the collectors expect. Every arm gets the same 1500 gradient steps, not the same '
              'number of epochs: counting epochs would have given the unaugmented arm 25 steps against '
              '1200 and put it at chance for the wrong reason.'),
]


def size(p):
    n = p.stat().st_size
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return f'{n:.0f} {u}' if u == 'B' else f'{n:.1f} {u}'
        n /= 1024


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def copy(src, dst, check):
    """Copy unless identical; returns True when the destination changed."""
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return False
    if not check:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def code_source(fig, name):
    """The run's own snapshot of a script if it kept one, else the repository's copy.

    `code_from='repo'` overrides that, for a figure whose run directory snapshotted an
    earlier stage than the one that drew it.
    """
    run = EXPS / fig['run'] / name if fig.get('run') and fig.get('code_from') != 'repo' else None
    repo = ROOT / 'inpaper_utils' / name
    if run is not None and run.exists():
        return run, 'run' if not repo.exists() or not filecmp.cmp(run, repo, shallow=False) else 'both'
    if not repo.exists():
        raise SystemExit(f'fig{fig["n"]}: no such script {name}')
    return repo, 'repo'


def checkpoints(fig):
    """What the copied JSONs say the figure ran, as (path, description) pairs.

    A manifest either names one checkpoint outright (the flow and representation figures)
    or carries a `checkpoints` table with one entry per column (the comparison figures);
    in the second case the table is the answer and is pointed at rather than flattened.
    """
    out = []
    for run, entries in runs_of(fig).items():
        for name, _ in entries:
            f = EXPS / run / name
            if not name.endswith('.json') or not f.exists():
                continue
            d = json.loads(f.read_text())
            if isinstance(d.get('checkpoint'), str):
                out.append((Path(d['checkpoint']), 'the CycleFlow checkpoint this figure runs'))
            if isinstance(d.get('checkpoints'), dict):
                out.append((EXPS, f'one checkpoint per column, named under `checkpoints` in '
                                  f'`data/{name}`, relative to this root'))
    return out


def runs_of(fig):
    if fig.get('runs'):
        return fig['runs']
    return {fig['run']: fig['data']} if fig.get('run') else {}


def readme(fig, files, notes):
    d = [f'# Figure {fig["n"]} -- {fig["title"]}', '',
         f'`\\label{{{fig["label"]}}}` in `iclr2027_conference.tex`; '
         f'dataset: {"ADNI (T1 / DWI-FA)" if fig["dataset"] == "adni" else "MNIST-PET/MRI (synthetic)"}.', '',
         fig['note'], '', '## Regenerate', '', '```bash',
         'cd /ihome/lzhan/sid51/projects/Brain/CycleFlow', ENV, *fig['cmd'],
         'python -m inpaper_utils.sync_figures        # restage into ../figures/', '```', '',
         '## Runs', '']
    for run in runs_of(fig):
        r = EXPS / run / 'run.json'
        meta = json.loads(r.read_text()) if r.exists() else {}
        commit = f', git `{meta["git_commit"][:8]}`' + (' (dirty)' if meta.get('git_dirty') else '') \
            if meta.get('git_commit') else ''
        d.append(f'- `{EXPS / run}`{commit}')
    d += ['', '## Files', '', '| file | size | what it is |', '|---|---|---|']
    d += [f'| `{n}` | {s} | {w} |' for n, s, w in files]
    if notes:
        d += ['', '## Inputs not copied', '',
              'Too large for the paper directory' + (', and not ours to redistribute'
                                                     if fig['dataset'] == 'adni' else '')
              + '. Paths are on CRCD.', '', '| path | size | what it is |', '|---|---|---|']
        d += [f'| `{n}` | {s} | {w} |' for n, s, w in notes]
    return '\n'.join(d) + '\n'


def build(fig, check):
    out = DEST / f'fig{fig["n"]}'
    files, external, changed = [], [], 0
    for paper_rel, in_run in fig['figures']:
        src = SNAP / paper_rel
        name = Path(paper_rel).name
        staged = PAPER / 'figures' / name
        if staged.exists() and md5(staged) != md5(src):
            raise SystemExit(f'fig{fig["n"]}: {name} in figures/ differs from snapshot_results; '
                             'run sync_figures first')
        if in_run:                                   # the run kept its own copy: verify it
            r = EXPS / fig['run'] / in_run
            if md5(r) != md5(src):
                raise SystemExit(f'fig{fig["n"]}: {src.name} does not match {r}; the pinned run is wrong')
        changed += copy(src, out / 'figure' / name, check)
        files.append((f'figure/{name}', size(src), 'the PDF the paper includes'
                      + (f', identical to `{in_run}` in the run' if in_run else '')))
    for name in fig['code']:
        src, where = code_source(fig, name)
        changed += copy(src, out / 'code' / name, check)
        files.append((f'code/{name}', size(src),
                      {'run': 'the version that ran (the repository has since changed it)',
                       'both': 'the version that ran, unchanged in the repository since',
                       'repo': 'from `inpaper_utils/`'}[where]))
    for run, entries in runs_of(fig).items():
        multi = len(runs_of(fig)) > 1
        for name, what in entries:
            src = EXPS / run / name
            if name.endswith('.npz') and not src.exists():
                continue                             # derived files are made by --derive
            if not src.exists():
                raise SystemExit(f'fig{fig["n"]}: missing {src}')
            rel = f'{run}/{name}' if multi else name
            changed += copy(src, out / 'data' / rel, check)
            files.append((f'data/{rel}', size(src), what))
    for p, what in checkpoints(fig) + [(Path(p), 'the held-out split it reads')
                                       for p in DATASETS[fig['dataset']]]:
        if p.exists() and not any(e[0] == p for e in external):
            external.append((p, size(p) if p.is_file() else '-', what))
    for run, entries in runs_of(fig).items():
        f = EXPS / run / 'features.npz'
        if f.exists():
            external.append((f, size(f), 'the pooled codes the projection in `data/` was exported from'))
        c = EXPS / run / 'cache'
        if c.is_dir():
            n = sum(x.stat().st_size for x in c.glob('*.npy'))
            external.append((c, f'{n / 2**20:.0f} MB', 'the cached predictions of every method, '
                                                       'from which the panels are cropped'))
    if not check:
        (out / 'README.md').write_text(readme(fig, files, external))
    return changed, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='report only, write nothing')
    ap.add_argument('--derive', action='store_true',
                    help='re-export fig8\'s projection from the run\'s features.npz (needs the cluster)')
    ap.add_argument('--archive', action='store_true',
                    help='also write figs_scripts.zip next to the tree, for the supplementary upload')
    a = ap.parse_args()
    if a.derive:
        from inpaper_utils.replot_pca_path import export
        fig = next(f for f in FIGS if f['n'] == 8)
        export(EXPS / fig['run'], EXPS / fig['run'] / 'mnist_pca_proj.npz', fig['dataset'])
    total = 0
    index = ['# Figure sources', '',
             'One folder per figure of `iclr2027_conference.tex`, in the order the paper numbers them. '
             'Each holds the figure, the code that drew it, and the data it was drawn from; '
             '`fig<N>/README.md` has the commands and the provenance.', '',
             'Built by `python -m inpaper_utils.collect_fig_sources` (`--check` to see what is stale). '
             'The figures themselves are staged into `../figures/` by '
             '`python -m inpaper_utils.sync_figures`, and both read '
             '`snapshot_results/{adni,mnist_petct}/`, which is the store of record.', '',
             '| folder | figure | label | dataset | drawn by |', '|---|---|---|---|---|']
    for fig in FIGS:
        changed, files = build(fig, a.check)
        total += changed
        drew = ', '.join(f'`{c}`' for c in fig['code'] if c.endswith('.py'))
        index.append(f'| `fig{fig["n"]}` | {fig["title"]} | `{fig["label"]}` | '
                     f'{"ADNI" if fig["dataset"] == "adni" else "MNIST"} | {drew} |')
        print(f'fig{fig["n"]:<2} {len(files):2d} files, {changed} {"stale" if a.check else "updated"}'
              f'   {fig["title"][:60]}')
    index += ['', 'Model checkpoints, dataset caches and the prediction caches are **not** in this tree: '
                  'they are inputs, they live on `/ix`, and the ADNI ones are not ours to redistribute. '
                  'Each `README.md` lists the ones its figure needs, with their paths and sizes. No voxel '
                  'data of any ADNI subject is written here -- only identifiers, settings and metrics.', '',
              'Figures 8 and 9 are the exception: their inputs are small, so those two folders redraw '
              'their figure with no model, no dataset and no GPU.']
    if not a.check:
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / 'README.md').write_text('\n'.join(index) + '\n')
    print(f'{len(FIGS)} figures, {total} files {"stale" if a.check else "updated"} -> {DEST}')
    if a.archive and not a.check:
        z = shutil.make_archive(str(DEST), 'zip', root_dir=DEST.parent, base_dir=DEST.name)
        print(f'{z}  {Path(z).stat().st_size / 2**20:.1f} MB')
    return 1 if (a.check and total) else 0


if __name__ == '__main__':
    raise SystemExit(main())
