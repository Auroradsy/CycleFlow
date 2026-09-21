#!/usr/bin/env python3
"""Table 2 SSIM/PSNR for any MMCLAST-cg checkpoint, both translation directions.

  python -m inpaper_utils.eval_mmclast --dataset adni  --ckpt /ix/.../adni_morph_smooth/model.pth
  python -m inpaper_utils.eval_mmclast --dataset mnist --ckpt /ix/.../mnist/checkpoints/<tag>/model.pth

Same held-out pairs and per-image SSIM/PSNR as make_paper_compare (ADNI: 430
slices, z=40-49; MNIST: 5000 pairs). Result JSON -> next to the checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}
KEEP = ('variant', 'tag', 'w_latcyc', 'w_path_gan', 'w_path_smooth', 'path_gan_mode', 'path_bidir',
        'resume_stage', 'resume_from')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--ckpt', required=True, help='MMCLAST-cg model.pth')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--data-root', help='MNIST only: evaluate on another build of the dataset')
    ap.add_argument('--split', default='test', choices=['test', 'val'],
                    help='val (MNIST): the early-stopping split carved from training, for model selection')
    a = ap.parse_args()
    if a.split == 'val' and a.dataset != 'mnist':
        raise SystemExit('--split val is implemented for MNIST only')
    sys.path.insert(0, str(ROOT))
    import torch
    from model import MMCLASTcg
    from inpaper_utils.make_paper_compare import load_test, batched, ssim_all, psnr_all

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (x_a, x_b), _ = load_test(a.dataset, root=a.data_root, split=a.split)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    ar = ck['args']
    m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']),
                  img_ch=ar.get('img_ch', 1)).to(dev).eval()
    m.load_state_dict(ck['model'], strict=True)

    d = DOMAINS[a.dataset]
    out = {'dataset': a.dataset, 'split': a.split, 'data_root': a.data_root,
           'checkpoint': str(Path(a.ckpt).resolve()), 'n': int(len(x_a)),
           'args': {k: ar.get(k) for k in KEEP}}
    for name, fn, src, tgt in ((f'{d[0]}->{d[1]}', m.cross_A2B, x_a, x_b),
                               (f'{d[1]}->{d[0]}', m.cross_B2A, x_b, x_a)):
        pred = batched(lambda x: (fn(x * 2 - 1) + 1) / 2, src, dev, a.batch)
        s, q = ssim_all(pred, tgt), psnr_all(pred, tgt)
        out[name] = {'ssim': float(s.mean()), 'psnr': float(q.mean())}
    dest = Path(a.ckpt).resolve().parent / ('paper_eval.json' if a.split == 'test' else 'paper_eval_val.json')
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), '\n->', dest, flush=True)


if __name__ == '__main__':
    main()
