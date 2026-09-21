#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latent DiT baseline, trained END TO END FROM SCRATCH on either testbed.

The original ADNI DiT ran in MedVAE's *pretrained* latent space.  That gave it
an advantage no other baseline had -- CFM, MeanFlow, DDPM and the CycleGAN host
are all trained from scratch -- and it is not portable to non-medical data.
Here the latent autoencoder (`baselines/latent_ae.py`) is trained from scratch
on the same training split as everything else, with the same 4x spatial
compression MedVAE provided, and is then frozen.  The DiT is trained in that
latent, and sampled with classifier-free guidance.

Three phases, all logged:  latent AE  ->  two DiTs  ->  CFG sampling + scoring.

  python -m baselines.train_dit --tag dit_mnist --data folder \
      --data_root /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired
  python -m baselines.train_dit --tag dit_adni_scratch --data adni
"""
import argparse
import csv
import os
import sys
import time
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines.common import loaders, ssim_psnr, add_data_args               # noqa: E402
from baselines.latent_ae import LatentAE                                     # noqa: E402
from baselines.nets_dit import DiT                                           # noqa: E402
from baselines.diffusion_iddpm import IDDPM                                  # noqa: E402

from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()


def train_latent_ae(tl, n_ch, dev, epochs, lr, kl_w, res):
    """Both domains in one latent space; that is what the DiT transports between."""
    ae = LatentAE(img_ch=n_ch).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)
    print(f"[latent-ae] {sum(p.numel() for p in ae.parameters())/1e6:.2f}M params",
          flush=True)
    for ep in range(1, epochs + 1):
        ae.train(); t0, run, nb = time.time(), 0.0, 0
        for xa, xb, _ in tl:
            x = torch.cat([xa, xb], 0).to(dev)          # one space for both
            rec, mu, logvar = ae(x)
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
            loss = Fn.l1_loss(rec, x) + kl_w * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        print(f"[latent-ae] ep {ep:3d}/{epochs}  loss {run/max(nb,1):.5f}  "
              f"{time.time()-t0:.0f}s", flush=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    torch.save({"ae": ae.state_dict()}, os.path.join(res, "latent_ae.pth"))
    return ae


@torch.no_grad()
def encode_split(ae, loader, dev):
    za, zb = [], []
    for xa, xb, _ in loader:
        za.append(ae.encode_norm(xa.to(dev), 0).cpu())
        zb.append(ae.encode_norm(xb.to(dev), 1).cpu())
    return torch.cat(za), torch.cat(zb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ae_epochs", type=int, default=30)
    ap.add_argument("--ae_lr", type=float, default=2e-4)
    ap.add_argument("--kl_w", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)      # DiT paper: constant, no wd
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--patch", type=int, default=2)        # DiT-S/2
    ap.add_argument("--hidden", type=int, default=384)     # DiT-S
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--cfg_dropout", type=float, default=0.1)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--eval_n", type=int, default=512,
                    help="test images to score; full DDPM sampling is expensive")
    ap.add_argument("--seed", type=int, default=42)
    add_data_args(ap)
    a = ap.parse_args()
    if a.data == "folder" and not a.data_root:
        ap.error("--data folder requires --data_root")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RES = os.path.join(EXPS, "checkpoints", a.tag); os.makedirs(RES, exist_ok=True)
    print(f"device={dev}\nargs={vars(a)}", flush=True)

    tl, el, n_ch = loaders(a)
    print(f"{len(tl.dataset)} train / {len(el.dataset)} test | {n_ch}ch", flush=True)

    # ---- phase 1: the latent space, from scratch --------------------------
    ae = train_latent_ae(tl, n_ch, dev, a.ae_epochs, a.ae_lr, a.kl_w, RES)
    ae.fit_stats(tl, device=dev)
    zA, zB = encode_split(ae, tl, dev)
    lat_size, lat_ch = zA.shape[-1], zA.shape[1]
    print(f"[dit] latent {tuple(zA.shape[1:])}  ->  "
          f"{(lat_size//a.patch)**2} tokens", flush=True)

    # ---- phase 2: two DiTs in that latent ---------------------------------
    mk = lambda: DiT(latent_size=lat_size, latent_ch=lat_ch, patch=a.patch,
                     hidden=a.hidden, depth=a.depth, heads=a.heads).to(dev)
    m_ab, m_ba = mk(), mk()
    print(f"[dit] {sum(p.numel() for p in m_ab.parameters())/1e6:.2f}M params/net",
          flush=True)
    diff = IDDPM(T=a.T, device=dev)
    o_ab = torch.optim.Adam(m_ab.parameters(), lr=a.lr)
    o_ba = torch.optim.Adam(m_ba.parameters(), lr=a.lr)

    def step(model, opt, src, tgt):
        model.train()
        src = src.clone()
        if a.cfg_dropout > 0:                       # CFG: drop the condition
            m = (torch.rand(src.shape[0], 1, 1, 1, device=dev) < a.cfg_dropout)
            src = torch.where(m, torch.zeros_like(src), src)
        t = torch.randint(0, a.T, (tgt.shape[0],), device=dev)
        loss = diff.training_losses(model, tgt, t, {"src": src})[0]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        return loss.item()

    N = zA.shape[0]; steps = N // a.batch
    f = open(os.path.join(RES, "train_log.csv"), "w", newline="")
    w = csv.writer(f); w.writerow(["epoch", "loss_A2B", "loss_B2A", "sec"])
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); la = lb = 0.0
        perm = torch.randperm(N)
        for s in range(steps):
            i = perm[s * a.batch:(s + 1) * a.batch]
            za, zb = zA[i].to(dev), zB[i].to(dev)
            la += step(m_ab, o_ab, za, zb); lb += step(m_ba, o_ba, zb, za)
        la /= steps; lb /= steps
        w.writerow([ep, f"{la:.6f}", f"{lb:.6f}", f"{time.time()-t0:.1f}"]); f.flush()
        if ep % 10 == 0 or ep == 1 or ep == a.epochs:
            print(f"[dit] ep {ep:3d}/{a.epochs} L A→B={la:.4f} B→A={lb:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        torch.save({"epoch": ep, "args": vars(a), "net_ab": m_ab.state_dict(),
                    "net_ba": m_ba.state_dict()}, os.path.join(RES, "last.pth"))
    f.close()

    # ---- phase 3: CFG sampling, decode, score -----------------------------
    m_ab.eval(); m_ba.eval()

    @torch.no_grad()
    def translate(model, src_img, src_m, tgt_m):
        z = ae.encode_norm(src_img.to(dev), src_m)
        out = diff.p_sample_loop(model, z.shape, {"src": z}, cfg_scale=a.cfg,
                                 null_kwargs={"src": torch.zeros_like(z)})
        return ae.decode_norm(out, tgt_m)

    acc = {k: [] for k in ["ssim_A2B", "psnr_A2B", "ssim_B2A", "psnr_B2A"]}
    done = 0
    for xa, xb, _ in el:
        if done >= a.eval_n:
            break
        fb = translate(m_ab, xa, 0, 1).cpu().numpy()
        fa = translate(m_ba, xb, 1, 0).cpu().numpy()
        s, q = ssim_psnr(fb, xb.numpy()); acc["ssim_A2B"] += s; acc["psnr_A2B"] += q
        s, q = ssim_psnr(fa, xa.numpy()); acc["ssim_B2A"] += s; acc["psnr_B2A"] += q
        done += xa.shape[0]
        print(f"[dit] scored {done}/{a.eval_n}", flush=True)
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    txt = (f"method=dit_scratch\ndata={a.data_root or 'adni'}\n"
           f"latent={lat_ch}x{lat_size}x{lat_size}\ncfg={a.cfg}\n"
           f"epochs={a.epochs}\nn_eval={done}\n"
           f"ssim_A2B={r['ssim_A2B']:.4f}\npsnr_A2B={r['psnr_A2B']:.2f}\n"
           f"ssim_B2A={r['ssim_B2A']:.4f}\npsnr_B2A={r['psnr_B2A']:.2f}\n")
    open(os.path.join(RES, "final_eval.txt"), "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
