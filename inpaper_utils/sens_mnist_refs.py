#!/usr/bin/env python3
"""Reference rows for one MNIST-PET/CT build: the copy floor and the stochastic ceiling.

  python -m inpaper_utils.sens_mnist_refs --root <build> --alt <same build, --seed 1> --out refs.json

copy floor         : SSIM/PSNR of emitting the input unchanged (CT vs PET), per direction.
stochastic ceiling : SSIM/PSNR of the real PET against the SAME PET re-rendered with a
                     different noise draw (--seed 1 changes only the noise of testB). No
                     CT->PET model can beat it, since the noise is not predictable from CT.
Both move with the rendering knobs, so every sensitivity setting needs its own.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--alt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from inpaper_utils.make_paper_compare import load_test, ssim_all, psnr_all
    (ct, pet), meta = load_test('mnist', root=a.root)
    (_, pet_alt), meta_alt = load_test('mnist', root=a.alt)
    assert [m['filename'] for m in meta] == [m['filename'] for m in meta_alt], 'test sets differ in content'
    out = {'root': a.root, 'n': len(meta),
           'copy CT->PET': {'ssim': float(ssim_all(ct, pet).mean()), 'psnr': float(psnr_all(ct, pet).mean())},
           'copy PET->CT': {'ssim': float(ssim_all(pet, ct).mean()), 'psnr': float(psnr_all(pet, ct).mean())},
           'ceiling CT->PET': {'ssim': float(ssim_all(pet_alt, pet).mean()),
                               'psnr': float(psnr_all(pet_alt, pet).mean())}}
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
