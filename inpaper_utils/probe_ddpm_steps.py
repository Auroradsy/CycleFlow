#!/usr/bin/env python3
"""Which sampling budget reproduces the published ADNI DDPM score?

The migrated ADNI DDPM carries no n_steps (its args are {T:1000, amp, ...}, a
different training script from the MNIST runs), so make_paper_compare must pick
a fallback. 50-step DDIM gives 0.2095 against the published 0.5586. Sweep the
budget on T1->FA and report which one lands on the published pair
(SSIM 0.5586 / PSNR 18.91, from adni_ddpm/eval_metrics.csv over the same 430
held-out slices).
"""
import time
import torch

from inpaper_utils.make_paper_compare import EXPS, load_test, batched, ssim_all, psnr_all
from baselines.train import build, ddpm_generate

TARGET = (0.5586, 18.91)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(x_a, x_b), _ = load_test('adni')
print(f'test pairs: {tuple(x_a.shape)} -> {tuple(x_b.shape)}  device={dev}', flush=True)

ck = torch.load(EXPS / 'ddpm_adni' / 'last.pth', map_location='cpu', weights_only=False)
ar = ck.get('args', {})
ma, _, diff = build('ddpm', x_a.shape[1], ar.get('base', 64), dev)
ma.load_state_dict(ck['model_T1toFA'], strict=True)
ma.eval()
print('published T1->FA: SSIM %.4f  PSNR %.2f' % TARGET, flush=True)

for n in (10, 50, 100, 250, 1000):
    t0 = time.time()
    with torch.no_grad():
        pred = batched(lambda x: ddpm_generate(ma, diff, x, n), x_a, dev, 256)
    s, q = float(ssim_all(pred, x_b).mean()), float(psnr_all(pred, x_b).mean())
    print(f'  n_steps={n:5d}  SSIM {s:.4f}  PSNR {q:.2f}   '
          f'(dSSIM {s - TARGET[0]:+.4f}, dPSNR {q - TARGET[1]:+.2f})  [{time.time() - t0:.0f}s]',
          flush=True)
