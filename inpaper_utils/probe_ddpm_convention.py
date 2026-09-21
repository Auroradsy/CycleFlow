#!/usr/bin/env python3
"""Is the migrated ADNI DDPM merely being fed the wrong convention?

The step sweep ruled out sampling budget: SSIM falls monotonically with more
steps (0.2435 at 10 -> 0.1719 at 250) instead of approaching the published
0.5586. The current code is self-consistent -- ddpm_loss and ddim_sample both
use cat([x_t, src]) with inputs mapped to [-1,1] -- so the suspect is the
convention the OLD training script used. strict-load only checks names and
shapes, not semantics.

Cross the two plausible conventions (concat order x value range) at 10 steps.
If one lands near 0.5586 that is the old convention; if all four sit near 0.2,
the weights are not usable with this forward pass and the number has to come
from elsewhere.
"""
import torch
import torch.nn as nn

from inpaper_utils.make_paper_compare import EXPS, load_test, batched, ssim_all, psnr_all
from baselines.train import build

TARGET = (0.5586, 18.91)
STEPS = 10
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(x_a, x_b), _ = load_test('adni')
ck = torch.load(EXPS / 'ddpm_adni' / 'last.pth', map_location='cpu', weights_only=False)
ar = ck.get('args', {})
ma, _, diff = build('ddpm', x_a.shape[1], ar.get('base', 64), dev)
ma.load_state_dict(ck['model_T1toFA'], strict=True)
ma.eval()
n_ch = x_a.shape[1]


class SwapCat(nn.Module):
    """Feed the net cat([src, x_t]) while ddim_sample still hands it cat([x_t, src])."""

    def __init__(self, net, n):
        super().__init__()
        self.net, self.n = net, n

    def forward(self, inp, t):
        a, b = inp[:, :self.n], inp[:, self.n:]
        return self.net(torch.cat([b, a], 1), t)


def run(net, to_pm1):
    def gen(x):
        src = x * 2 - 1 if to_pm1 else x
        out = diff.ddim_sample(net, src, n_steps=STEPS)
        x0 = out[0] if isinstance(out, tuple) else out
        return (x0 + 1) * 0.5 if to_pm1 else x0
    with torch.no_grad():
        pred = batched(gen, x_a, dev, 256)
    return float(ssim_all(pred, x_b).mean()), float(psnr_all(pred, x_b).mean())


swapped = SwapCat(ma, n_ch).to(dev).eval()
CASES = [('A  cat[x_t, src]  src in [-1,1]  (current)', ma, True),
         ('B  cat[src, x_t]  src in [-1,1]', swapped, True),
         ('C  cat[x_t, src]  src in [0,1]', ma, False),
         ('D  cat[src, x_t]  src in [0,1]', swapped, False)]

print('published T1->FA: SSIM %.4f  PSNR %.2f   (%d-step DDIM)' % (*TARGET, STEPS), flush=True)
for name, net, pm1 in CASES:
    s, q = run(net, pm1)
    hit = '   <== MATCHES PUBLISHED' if abs(s - TARGET[0]) < 0.02 else ''
    print(f'  {name:42s} SSIM {s:.4f}  PSNR {q:6.2f}   (dSSIM {s - TARGET[0]:+.4f}){hit}', flush=True)
