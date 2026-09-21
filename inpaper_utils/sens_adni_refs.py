#!/usr/bin/env python3
"""Reference rows for the ADNI FA-noise sensitivity settings.

  python -m inpaper_utils.sens_adni_refs --noise 0.0 0.025 0.05 0.1

Per sigma, on the 430 held-out slices (z=40..49, seed-42 split):
  copy                 : SSIM/PSNR of emitting the input unchanged (T1 vs noisy FA)
  FA vs re-noised FA   : the noisy FA against the same slice with another noise draw
                         (noise_seed 1). A model that reproduces the noise cannot beat it;
                         one that predicts the noise-free FA can.
JSON -> EXPS/adni/sensitivity/refs_ns<sigma>.json
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--noise', nargs='+', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    from data.paired_dataset import subject_level_split, PairedADNISliceDataset
    from inpaper_utils.make_paper_compare import ssim_all, psnr_all
    _, te = subject_level_split(42, .2, 'label_4', 40, 49)
    out_dir = EXPS / 'adni' / 'sensitivity'; out_dir.mkdir(parents=True, exist_ok=True)
    ds = PairedADNISliceDataset(te, 'label_4')       # one copy of the cache; noise set per call

    def get(sigma, seed):
        ds.fa_noise, ds.noise_seed = sigma, seed
        return [ds[i] for i in range(len(ds))]

    for s in a.noise:
        d0, d1 = get(float(s), 0), get(float(s), 1)
        t1 = torch.stack([x[0] for x in d0]); fa0 = torch.stack([x[1] for x in d0]); fa1 = torch.stack([x[1] for x in d1])
        res = {'fa_noise': float(s), 'n': len(te),
               'copy T1->FA': {'ssim': float(ssim_all(t1, fa0).mean()), 'psnr': float(psnr_all(t1, fa0).mean())},
               'copy FA->T1': {'ssim': float(ssim_all(fa0, t1).mean()), 'psnr': float(psnr_all(fa0, t1).mean())},
               'FA vs re-noised FA': {'ssim': float(ssim_all(fa1, fa0).mean()),
                                      'psnr': float(psnr_all(fa1, fa0).clip(max=99).mean())}}
        (out_dir / f'refs_ns{s}.json').write_text(json.dumps(res, indent=2) + '\n')
        print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
