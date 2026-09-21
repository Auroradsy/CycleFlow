#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RevGAN under CycleGAN's own recipe, on either testbed.

  python -m baselines.train_revgan --tag revgan_adni --epochs 160 --batch 12 --decay_start 80
  python -m baselines.train_revgan --tag revgan_mnist --data folder --data_root <paired build> \
      --img_ch 3 --load_size 72 --crop_size 64 --no_flip --epochs 100 --decay_start 50 --batch 64

The objective, the optimiser, the image pool and the schedule are train_host.py's
(LSGAN + cycle L1 at 10 + identity L1 at 5, Adam 2e-4 betas (0.5, 0.999), pool of
50, constant LR then linear decay to zero, no early stop, the FINAL model is the
reported one).  The only thing that differs is the generator: RevGAN's two
directions are one shared reversible core between per-domain encoders and
decoders (baselines/nets_revgan.py), so ONE run yields both directions, where the
CycleGAN row of the paper is two runs (a forward host and a native reverse host).

Training is unpaired, as CycleGAN's is: two independently shuffled loaders over
the training split, so the A batch and the B batch are unrelated.  The pairing is
used only by the held-out evaluation.

Note on what the cycle loss does here.  A->B->A is
dec_a(R^-1(enc_b(dec_b(R(enc_a(x)))))), and R cancels only to the extent that
enc_b . dec_b is the identity; the penalty therefore lands on the encoder/decoder
round trip, not on the core, which is exactly invertible to begin with.  RevGAN
still needs the penalty because its bottleneck is not shared between domains.
"""
import argparse
import csv
import itertools
import os
import random
import sys
import time
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines.common import add_data_args, evaluate                       # noqa: E402
from baselines.nets_revgan import RevGAN, count_params                     # noqa: E402
from model import PatchDiscriminator, init_weights                         # noqa: E402

from server_paths import experiment_root
EXPS = experiment_root()


class ImagePool:
    """CycleGAN's buffer of 50 previously generated images.

    A copy of train_host.ImagePool, not an import: importing train_host would
    allocate a second run directory at module level.
    """

    def __init__(self, pool_size=50):
        self.pool_size, self.images = pool_size, []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img); out.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                out.append(self.images[idx].clone()); self.images[idx] = img
            else:
                out.append(img)
        return torch.cat(out, dim=0)


def datasets(a):
    """(train, test, n_ch); the train split is read twice, unpaired, by the caller."""
    if a.data == "adni":
        from data.paired_dataset import build_cache, PairedADNISliceDataset, subject_level_split
        build_cache()
        tr_idx, te_idx = subject_level_split(seed=42, test_frac=0.20,
                                             label_scheme="label_4", z_lo=a.z_lo, z_hi=a.z_hi)
        return (PairedADNISliceDataset(tr_idx, label_scheme="label_4"),
                PairedADNISliceDataset(te_idx, label_scheme="label_4"), 1)
    from data.unpaired_dataset import UnpairedFolderDataset
    common = dict(load_size=a.load_size, crop_size=a.crop_size, img_ch=a.img_ch)
    # train: the reference augmentation, B drawn independently of A.
    # test : pair=True, so SSIM/PSNR are scored against the true counterpart.
    return (UnpairedFolderDataset(a.data_root, "train", train=True, flip=not a.no_flip, **common),
            UnpairedFolderDataset(a.data_root, "test", train=False, flip=False, pair=True, **common),
            a.img_ch)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--decay_start", type=int, default=80,
                    help="epoch at which linear LR decay to 0 begins")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=6, help="reversible blocks in the core")
    ap.add_argument("--core_hidden", type=int, default=0,
                    help="hidden width of each coupling body; 0 = ngf*4, which makes one "
                         "reversible block cost what the residual block it replaces costs")
    ap.add_argument("--lambda_cycle", type=float, default=10.0)
    ap.add_argument("--lambda_id", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_flip", action="store_true",
                    help="disable h-flip augmentation; required for chiral content (digits)")
    add_data_args(ap)
    a = ap.parse_args()
    if a.data == "folder" and not a.data_root:
        ap.error("--data folder requires --data_root")

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RES = os.path.join(EXPS, "checkpoints", a.tag)
    os.makedirs(RES, exist_ok=True)
    print(f"device={dev}  method=revgan\nargs={vars(a)}", flush=True)

    tr, te, n_ch = datasets(a)
    load = dict(batch_size=a.batch, shuffle=True, num_workers=a.workers,
                drop_last=True, pin_memory=True)
    loader_a = DataLoader(tr, **load)
    loader_b = DataLoader(tr, **load)              # independent shuffle -> unpaired
    el = DataLoader(te, batch_size=64, shuffle=False, num_workers=a.workers, pin_memory=True)

    net = RevGAN(n_ch, a.ngf, a.n_blocks, a.core_hidden or None).to(dev)
    init_weights(net)
    D_B = init_weights(PatchDiscriminator(n_ch, a.ndf)).to(dev)     # judges domain B
    D_A = init_weights(PatchDiscriminator(n_ch, a.ndf)).to(dev)
    print(f"{len(tr)} train / {len(te)} test  |  {n_ch}ch  |  "
          f"G {count_params(net) / 1e6:.2f}M (core {count_params(net.core) / 1e6:.2f}M)  "
          f"D {2 * count_params(D_A) / 1e6:.2f}M", flush=True)

    crit_gan, crit_l1 = nn.MSELoss(), nn.L1Loss()
    opt_G = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                             lr=a.lr, betas=(0.5, 0.999))

    def lr_lambda(epoch):
        if epoch < a.decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - a.decay_start) / float(max(1, a.epochs - a.decay_start)))

    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    pool_A, pool_B = ImagePool(50), ImagePool(50)

    f = open(os.path.join(RES, "train_log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["epoch", "loss_G", "loss_D", "loss_cyc", "loss_id", "loss_gan",
                "ssim_A2B", "ssim_B2A", "lr", "sec"])

    def requires_grad(nets, flag):
        for n in nets:
            for p in n.parameters():
                p.requires_grad = flag

    n_batches = min(len(loader_a), len(loader_b))
    for ep in range(1, a.epochs + 1):
        net.train(); D_A.train(); D_B.train()
        t0, acc = time.time(), {k: 0.0 for k in ("G", "D", "cyc", "id", "gan")}
        for (xa, _b_unused, _y1), (_a_unused, xb, _y2) in zip(loader_a, loader_b):
            real_A = xa.to(dev, non_blocking=True) * 2 - 1
            real_B = xb.to(dev, non_blocking=True) * 2 - 1

            # ---- generator ----
            requires_grad([D_A, D_B], False)
            opt_G.zero_grad(set_to_none=True)
            # identity, as in CycleGAN: the A->B map applied to a B image
            loss_id = (crit_l1(net.a2b(real_B), real_B)
                       + crit_l1(net.b2a(real_A), real_A)) * a.lambda_id
            fake_B, fake_A = net.a2b(real_A), net.b2a(real_B)
            pf_B, pf_A = D_B(fake_B), D_A(fake_A)
            loss_gan = (crit_gan(pf_B, torch.ones_like(pf_B))
                        + crit_gan(pf_A, torch.ones_like(pf_A)))
            loss_cyc = (crit_l1(net.b2a(fake_B), real_A)
                        + crit_l1(net.a2b(fake_A), real_B)) * a.lambda_cycle
            loss_G = loss_gan + loss_cyc + loss_id
            loss_G.backward(); opt_G.step()

            # ---- discriminators ----
            requires_grad([D_A, D_B], True)
            opt_D.zero_grad(set_to_none=True)
            loss_D = 0.0
            for D, real, fake, pool in ((D_B, real_B, fake_B, pool_B),
                                        (D_A, real_A, fake_A, pool_A)):
                pr = D(real)
                pf = D(pool.query(fake.detach()))
                d = 0.5 * (crit_gan(pr, torch.ones_like(pr)) + crit_gan(pf, torch.zeros_like(pf)))
                d.backward(); loss_D += d.item()
            opt_D.step()

            acc["G"] += loss_G.item(); acc["D"] += loss_D
            acc["cyc"] += loss_cyc.item(); acc["id"] += loss_id.item()
            acc["gan"] += loss_gan.item()
        sch_G.step(); sch_D.step()
        for k in acc:
            acc[k] /= n_batches

        sa = sb = float("nan")
        if ep % a.eval_every == 0 or ep == a.epochs:
            net.eval()
            r = evaluate(*generators(net), el, dev, max_batches=8)
            sa, sb = r["ssim_A2B"], r["ssim_B2A"]
        sec = time.time() - t0
        w.writerow([ep, f"{acc['G']:.4f}", f"{acc['D']:.4f}", f"{acc['cyc']:.4f}",
                    f"{acc['id']:.4f}", f"{acc['gan']:.4f}", f"{sa:.4f}", f"{sb:.4f}",
                    f"{opt_G.param_groups[0]['lr']:.2e}", f"{sec:.1f}"]); f.flush()
        print(f"[ep {ep}/{a.epochs}] G={acc['G']:.3f} D={acc['D']:.3f} cyc={acc['cyc']:.3f} "
              f"id={acc['id']:.3f} gan={acc['gan']:.3f} ssim A->B {sa:.4f} B->A {sb:.4f} "
              f"lr={opt_G.param_groups[0]['lr']:.2e} ({sec:.0f}s)", flush=True)
        torch.save({"epoch": ep, "args": vars(a), "revgan": net.state_dict(),
                    "D_A": D_A.state_dict(), "D_B": D_B.state_dict()},
                   os.path.join(RES, "last.pth"))
    f.close()

    # FAITHFUL (CycleGAN): fixed schedule, no early stop, the FINAL model is reported.
    net.eval()
    r = evaluate(*generators(net), el, dev)
    with torch.no_grad():
        # TF32 convolutions (cuDNN's default on Ampere and later) carry a 10-bit
        # mantissa, which turns this into a measurement of the GPU's precision
        # rather than of the coupling: it reads ~1e-3 relative under TF32 and
        # ~2e-6 in true float32, for a core that is invertible by construction.
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        z = net.enc_a(next(iter(el))[0][:8].to(dev) * 2 - 1)
        err = (net.core.inverse(net.core(z)) - z).abs().max().item()
        # Relative to the code's own scale: the absolute figure alone says nothing,
        # because the encoder's features are not normalised to any particular range.
        rel = err / z.abs().max().item()
        torch.backends.cudnn.allow_tf32 = tf32
    txt = (f"method=revgan\ndata={a.data_root or 'adni'}\nepochs={a.epochs}\n"
           f"params_generator={count_params(net)}\nparams_core={count_params(net.core)}\n"
           f"params_discriminators={2 * count_params(D_A)}\n"
           f"core_round_trip_max_abs={err:.3e}\ncore_round_trip_relative={rel:.3e}\n"
           f"ssim_A2B={r['ssim_A2B']:.4f}\npsnr_A2B={r['psnr_A2B']:.2f}\n"
           f"ssim_B2A={r['ssim_B2A']:.4f}\npsnr_B2A={r['psnr_B2A']:.2f}\n")
    open(os.path.join(RES, "final_eval.txt"), "w").write(txt)
    print("\n" + txt, flush=True)


def generators(net):
    """The two [0,1]->[0,1] translators, the convention baselines.common.evaluate wants."""
    return (lambda x: (net.a2b(x * 2 - 1) + 1) / 2,
            lambda y: (net.b2a(y * 2 - 1) + 1) / 2)


if __name__ == "__main__":
    main()
