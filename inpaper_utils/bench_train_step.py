#!/usr/bin/env python3
"""Does one logged epoch of a two-network baseline cost one direction or both?

  python -m inpaper_utils.bench_train_step     # GPU

baselines/train.py and baselines/train_dit.py build net_ab and net_ba and update
BOTH in every training step. This times one such step against a step that trains
net_ab alone, on the ADNI shapes and each method's batch size (synthetic tensors, so
data loading is left out). A ratio near 2 means the logged epoch time already holds
the compute of both directions.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def timeit(fn, n=40, warm=8):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    sys.path.insert(0, str(ROOT))
    import torch
    from baselines.train import build, cfm_loss, meanflow_loss, ddpm_loss
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    out = {'device': torch.cuda.get_device_name(0)}

    # batch sizes from the ADNI checkpoints' args (cfm 64, meanflow 32, ddpm 32)
    for method, batch in (('cfm', 64), ('meanflow', 32), ('ddpm', 32)):
        ma, mb, diff = build(method, 1, 64, dev)
        loss = {'cfm': cfm_loss, 'meanflow': meanflow_loss,
                'ddpm': lambda net, x0, x1: ddpm_loss(net, diff, x0, x1)}[method]
        opt2 = torch.optim.AdamW(list(ma.parameters()) + list(mb.parameters()), lr=2e-4)
        opt1 = torch.optim.AdamW(ma.parameters(), lr=2e-4)
        xa = torch.rand(batch, 1, 112, 112, device=dev); xb = torch.rand_like(xa)

        def both():   # exactly the train.py step
            l = loss(ma, xa, xb) + loss(mb, xb, xa)
            opt2.zero_grad(set_to_none=True); l.backward(); opt2.step()

        def one():
            l = loss(ma, xa, xb)
            opt1.zero_grad(set_to_none=True); l.backward(); opt1.step()

        tb, t1 = timeit(both), timeit(one)
        out[method] = {'batch': batch, 'both_nets_ms': round(tb * 1e3, 1), 'one_net_ms': round(t1 * 1e3, 1),
                       'ratio': round(tb / t1, 2)}
        print(method, out[method], flush=True)
        del ma, mb, opt1, opt2; torch.cuda.empty_cache()

    from baselines.nets_dit import DiT
    from baselines.diffusion_iddpm import IDDPM
    mk = lambda: DiT(latent_size=28, latent_ch=4, patch=2, hidden=384, depth=12, heads=6).to(dev)
    ma, mb, diff = mk(), mk(), IDDPM(T=1000, device=dev)
    oa, ob = torch.optim.Adam(ma.parameters(), lr=1e-4), torch.optim.Adam(mb.parameters(), lr=1e-4)
    za = torch.randn(64, 4, 28, 28, device=dev); zb = torch.randn_like(za)

    def step(m, o, src, tgt):   # train_dit.py's step, without the CFG dropout
        t = torch.randint(0, 1000, (tgt.shape[0],), device=dev)
        l = diff.training_losses(m, tgt, t, {'src': src})[0]
        o.zero_grad(set_to_none=True); l.backward(); o.step()

    tb = timeit(lambda: (step(ma, oa, za, zb), step(mb, ob, zb, za)))
    t1 = timeit(lambda: step(ma, oa, za, zb))
    out['dit'] = {'batch': 64, 'both_nets_ms': round(tb * 1e3, 1), 'one_net_ms': round(t1 * 1e3, 1),
                  'ratio': round(tb / t1, 2)}
    print('dit', out['dit'], flush=True)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
