#!/usr/bin/env python3
"""SSIM/PSNR of the CycleGAN trained natively in the B->A direction.

  python -m inpaper_utils.eval_cyclegan_rev --dataset adni    # FA -> T1
  python -m inpaper_utils.eval_cyclegan_rev --dataset mnist   # PET -> CT

train_host.py --swap_domains stores that native B->A generator under the key
G_T1toFA. It is scored on the held-out pairs and with the per-image SSIM/PSNR
of make_paper_compare (ADNI: 430 slices, z=40-49; MNIST: 5000 pairs), so the
numbers go straight into Table 2. When the forward host is available, its
jointly trained reverse generator is scored on the same pairs for reference.
Result JSON -> next to the checkpoint under EXPS.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
FORWARD_HOST = {'adni': EXPS / 'adni_host/last.pth', 'mnist': EXPS / 'mnist_host/last.pth'}
DIRECTION = {'adni': 'FA->T1', 'mnist': 'PET->CT'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--ckpt', help='default: EXPS/<dataset>/checkpoints/host_rev/last.pth')
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    from model.backbone import ResnetGenerator
    from inpaper_utils.make_paper_compare import load_test, batched, ssim_all, psnr_all

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (x_a, x_b), _ = load_test(a.dataset)
    ckpt = Path(a.ckpt) if a.ckpt else EXPS / a.dataset / 'checkpoints' / 'host_rev' / 'last.pth'

    def score(path, key):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        ar = ck.get('args', {})
        g = ResnetGenerator(x_b.shape[1], x_b.shape[1], ar.get('ngf', 64), ar.get('n_blocks', 6)).to(dev).eval()
        g.load_state_dict(ck[key], strict=True)
        pred = batched(lambda x: (g(x * 2 - 1) + 1) / 2, x_b, dev, a.batch)     # B -> A
        s, q = ssim_all(pred, x_a), psnr_all(pred, x_a)
        return {'checkpoint': str(Path(path).resolve()), 'generator': key, 'epoch': ck.get('epoch'),
                'swap_domains': bool(ar.get('swap_domains')), 'n': int(len(s)),
                'ssim': float(s.mean()), 'psnr': float(q.mean())}, ar

    native, ar = score(ckpt, 'G_T1toFA')
    if not native['swap_domains']:
        raise SystemExit(f'{ckpt} was not trained with --swap_domains; G_T1toFA is not B->A')
    out = {'dataset': a.dataset, 'direction': DIRECTION[a.dataset], 'native_b2a': native}
    if FORWARD_HOST[a.dataset].exists():
        out['forward_host_reverse_generator'] = score(FORWARD_HOST[a.dataset], 'G_FAtoT1')[0]
    dest = ckpt.resolve().parent / 'paper_eval_native_b2a.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), '\n->', dest, flush=True)


if __name__ == '__main__':
    main()
