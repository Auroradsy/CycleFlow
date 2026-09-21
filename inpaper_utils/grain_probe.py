#!/usr/bin/env python3
"""How much acquisition grain does each MNIST checkpoint put into CT->PET, and where on the path?

  python -m inpaper_utils.grain_probe [--n 1000]

grain(x) = std of the high-pass residual x - GaussianBlur(x, sigma=1), per image, on the
channel mean. Reported as a ratio to the ground-truth PET's grain (1 = as grainy as the
target, ~0 = the smooth conditional mean), for the endpoint and for every decoded state
along the flow (z, f1..f4, all through the PET decoder), next to the endpoint SSIM.
Baselines are read from the latest comparison cache for reference.
JSON -> EXPS/mnist/grain_probe/<run>/grain.json (pass --dataset mnist so server_paths files it there)
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
CK = 'mnist/checkpoints/'
CANDIDATES = [   # (label, tag): stage-3 adversarial weight g, smoothness weight s
    ('base (Table 1 w/o)', 'mnist_p_base'),
    ('morph', 'mnist_p_morph'),
    ('smooth g1 s1 (original)', CK + 'mnist_p_morph_smooth'),
    ('smooth g0.1 s0.3 (Table 1 w/)', CK + 'tune_mnist_smooth_g0.1_s0.3'),
] + [(f'smooth g{g} s{s}', CK + f'tune_mnist_smooth_g{g}_s{s}')
     for g in ('0.1', '0.3', '1.0') for s in ('0.1', '0.3', '1.0') if (g, s) != ('0.1', '0.3')]


def grain(x):
    """x: (N,C,H,W) in [0,1] -> per-image high-pass std on the channel mean."""
    import torch
    from torchvision.transforms.functional import gaussian_blur
    m = x.float().mean(1, keepdim=True)
    return (m - gaussian_blur(m, 7, 1.0)).flatten(1).std(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='mnist', choices=['mnist'],
                    help='read by server_paths to file the run under EXPS/mnist/ (pass it explicitly)')
    ap.add_argument('--n', type=int, default=1000, help='first N test pairs')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    os.environ['CYCLEFLOW_PURPOSE'] = 'grain_probe'
    from server_paths import experiment_root
    run = Path(experiment_root())
    import torch
    from model import MMCLASTcg
    from inpaper_utils.make_paper_compare import load_test, ssim_all, latest_cache
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    (ct, pet), _ = load_test('mnist')
    ct, pet = ct[:a.n], pet[:a.n]
    g_gt = grain(pet)
    out = {'n': a.n, 'gt_grain_mean': float(g_gt.mean()), 'models': {}, 'baselines': {}}
    print(f'target PET grain {g_gt.mean():.4f}  (ratios below are per-image pred/target, averaged)', flush=True)
    for label, tag in CANDIDATES:
        path = EXPS / tag / 'model.pth'
        if not path.exists():
            print(f'SKIP {label}: {path}', flush=True); continue
        ck = torch.load(path, map_location='cpu', weights_only=False); ar = ck['args']
        m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']),
                      img_ch=ar.get('img_ch', 1)).eval()
        m.load_state_dict(ck['model'], strict=True)
        states = None
        with torch.inference_mode():
            for i in range(0, a.n, 250):
                walk = m.walk(m.enc_A(ct[i:i + 250] * 2 - 1), inverse=False)
                dec = [((m.dec_B(s) + 1) / 2).clamp(0, 1) for s in walk]
                states = dec if states is None else [torch.cat([p, q]) for p, q in zip(states, dec)]
        ratios = [float((grain(s) / g_gt).mean()) for s in states]
        ssim = float(ssim_all(states[-1], pet).mean())
        out['models'][label] = {'tag': tag, 'w_gan': ar.get('w_gan'), 'w_path_smooth': ar.get('w_path_smooth'),
                                'ssim_ct2pet': ssim, 'grain_ratio_path': ratios}
        print(f'{label:32s} SSIM {ssim:.4f}  grain z..f4: ' + ' '.join(f'{r:.2f}' for r in ratios), flush=True)
    cache = latest_cache('mnist')
    info = json.loads((cache / 'meta.json').read_text())
    for meth in info['methods']:
        if meth['role'] != 'baseline':
            continue
        pred = torch.from_numpy(np.load(cache / f"{meth['key']}_0.npy")[:a.n]).permute(0, 3, 1, 2).float() / 255
        r = float((grain(pred) / g_gt).mean())
        out['baselines'][meth['name']] = r
        print(f'{meth["name"]:32s} grain {r:.2f}', flush=True)
    (run / 'grain.json').write_text(json.dumps(out, indent=2) + '\n')
    print('->', run / 'grain.json', flush=True)


if __name__ == '__main__':
    main()
