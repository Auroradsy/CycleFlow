#!/usr/bin/env python3
"""Stage the figures the paper includes, from snapshot_results into the paper directory.

  python -m inpaper_utils.sync_figures        # copy
  python -m inpaper_utils.sync_figures --check # only report what is stale

snapshot_results/{adni,mnist_petct}/ is the store: every figure keeps its full
provenance there (PDF for the paper, PNG for preview, <name>_caption.tex, and the
superseded renders under archived/). The paper needs one flat directory, because
\\includegraphics paths are flat and the two datasets would otherwise collide, so the
included PDFs are copied to __X_files/iclr2027/figures/ under the same names. Re-run
after re-rendering anything, or the paper keeps building the previous version.
"""
import argparse
import filecmp
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'snapshot_results'
DEST = ROOT / '__X_files/iclr2027/figures'
# name -> source folder; the name is also the \includegraphics path, minus "figures/".
FIGURES = {
    'mnist_compare': 'mnist_petct',       # fig:mnist-compare
    'mnist_path_a2b': 'mnist_petct',      # fig:mnist-path-ours (a)
    'mnist_path_b2a': 'mnist_petct',      # fig:mnist-path-ours (b)
    'adni_compare': 'adni',               # fig:adni-ablation
    'adni_path_a2b': 'adni',              # fig:adni-path-ours (a)
    'adni_path_b2a': 'adni',              # fig:adni-path-ours (b)
    'adni_cycle': 'adni',                 # fig:adni-cycle-b2a
    'adni_features': 'adni',              # fig:adni-flow-features
    'adni_affine_a2b': 'adni',            # fig:adni-flow-affine (a)
    'adni_affine_b2a': 'adni',            # fig:adni-flow-affine (b)
    'mnist_pca_path': 'mnist_petct',      # fig:mnist-pca
    'mnist_missing_modality': 'mnist_petct',   # fig:mnist-missing-modality
}


def included():
    """The figure names the paper actually includes, straight from the tex."""
    from inpaper_utils.paper_tex import flat
    tex = flat()     # the main file and every \input, as LaTeX reads them
    return set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\.pdf\}', tex))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report only, copy nothing')
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    # The map above is the documented one; the tex is the truth. They drifted once already,
    # when the paper was reorganised and went on including stems that no longer existed.
    used = included()
    if used != set(FIGURES):
        for n in sorted(used - set(FIGURES)):
            print(f'included by the paper but not in FIGURES: {n}')
        for n in sorted(set(FIGURES) - used):
            print(f'in FIGURES but no longer included by the paper: {n}')
        raise SystemExit('sync_figures: FIGURES and the tex disagree')
    stale = 0
    for name, folder in FIGURES.items():
        src, dst = SNAP / folder / f'{name}.pdf', DEST / f'{name}.pdf'
        if not src.exists():
            raise SystemExit(f'missing source: {src}')
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        stale += 1
        print(('stale' if args.check else 'copied'), f'{folder}/{name}.pdf')
        if not args.check:
            shutil.copy2(src, dst)
    print(f'{len(FIGURES)} figures, {stale} {"stale" if args.check else "updated"}')
    return 1 if (args.check and stale) else 0


if __name__ == '__main__':
    raise SystemExit(main())
