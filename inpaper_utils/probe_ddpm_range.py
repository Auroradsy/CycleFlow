#!/usr/bin/env python3
"""Pin down the last degree of freedom in the migrated ADNI DDPM's convention.

Established so far:
  * concat order is cat([x_t, src]) -- same as the current code (A>B and C>D).
  * the source must be fed in [0,1], not [-1,1]  (0.2405 -> 0.6205 SSIM).
  * but that run returned x0 as-is and landed at SSIM +0.06 / PSNR -3.25 dB
    against the published pair: structure right, values offset.

ddim_sample seeds x with randn_like(source), so the target side lives on the
standard-normal scale, i.e. [-1,1]. The likely old convention is therefore
asymmetric: source in [0,1], target in [-1,1], output mapped back by (x0+1)/2.
Test that across both directions and two step budgets. A match must hold for
BOTH directions to count.
"""
import torch

from inpaper_utils.make_paper_compare import EXPS, load_test, batched, ssim_all, psnr_all
from baselines.train import build

TARGET = {'T1->FA': (0.5586, 18.91), 'FA->T1': (0.4884, 19.64)}
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(x_a, x_b), _ = load_test('adni')
ck = torch.load(EXPS / 'ddpm_adni' / 'last.pth', map_location='cpu', weights_only=False)
ar = ck.get('args', {})
ma, mb, diff = build('ddpm', x_a.shape[1], ar.get('base', 64), dev)
ma.load_state_dict(ck['model_T1toFA'], strict=True); ma.eval()
mb.load_state_dict(ck['model_FAtoT1'], strict=True); mb.eval()


def run(net, src, tgt, steps, out_from_pm1):
    def gen(x):                                   # x arrives in [0,1]
        out = diff.ddim_sample(net, x, n_steps=steps)   # source fed as [0,1]
        x0 = out[0] if isinstance(out, tuple) else out
        return (x0 + 1) * 0.5 if out_from_pm1 else x0
    with torch.no_grad():
        pred = batched(gen, src, dev, 256)
    return float(ssim_all(pred, tgt).mean()), float(psnr_all(pred, tgt).mean())


for name, net, src, tgt in (('T1->FA', ma, x_a, x_b), ('FA->T1', mb, x_b, x_a)):
    ts, tq = TARGET[name]
    print(f'=== {name}   published SSIM {ts:.4f}  PSNR {tq:.2f} ===', flush=True)
    for steps in (10, 50):
        for out_pm1 in (True, False):
            s, q = run(net, src, tgt, steps, out_pm1)
            tag = '(x0+1)/2' if out_pm1 else 'x0 as-is'
            hit = '  <== MATCH' if abs(s - ts) < 0.02 and abs(q - tq) < 0.6 else ''
            print(f'  steps={steps:4d}  out={tag:9s}  SSIM {s:.4f}  PSNR {q:6.2f}   '
                  f'(dSSIM {s - ts:+.4f}, dPSNR {q - tq:+.2f}){hit}', flush=True)
