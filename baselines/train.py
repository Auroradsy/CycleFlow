#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-modal baselines on either testbed, from one implementation.

  cfm       Rectified-flow Conditional Flow Matching.  Learns a velocity field
            v(x,t) with the linear interpolant and the paired (OT) coupling;
            translation is Euler ODE integration from source to target.
            Strongest baseline on ADNI.
  meanflow  MeanFlow (Geng et al. 2025).  Learns the AVERAGE velocity over an
            interval via the MeanFlow identity, so t=0->1 is a single forward.
  ddpm      Conditional DDPM; the source image is concatenated to the noisy
            target, sampled with DDIM.

Each method keeps the objective and the schedule of the original ADNI scripts
in __outdated_files/baseline/; the only thing this file changes is that the
data source and the channel count are arguments.

  python -m baselines.train --method cfm --data folder \
      --data_root /home/siyuan/datasets/mnist_petct_paired --tag cfm_mnist
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines.common import loaders, evaluate, add_data_args              # noqa: E402

EXPS = os.environ.get("MMCLAST_EXPS", os.path.join(_ROOT, "exps"))


# --------------------------------------------------------------------------
# CFM  (rectified flow, linear interpolant, paired coupling)
# --------------------------------------------------------------------------
def cfm_loss(net, x0, x1):
    t = torch.rand(x0.shape[0], device=x0.device)
    x_t = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    return ((net(x_t, t) - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def cfm_generate(net, x0, n_steps=10):
    x, dt = x0.clone(), 1.0 / n_steps
    for k in range(n_steps):
        x = x + dt * net(x, torch.full((x.shape[0],), k * dt, device=x.device))
    return x


# --------------------------------------------------------------------------
# MeanFlow  (average velocity; one-step generation)
# --------------------------------------------------------------------------
def sample_rt(B, device, p_eq=0.5):
    t = torch.rand(B, device=device)
    r = t.clone()
    m = torch.rand(B, device=device) > p_eq
    r[m] = torch.rand(int(m.sum()), device=device) * t[m]
    return r, t


def meanflow_loss(net, x0, x1, p_eq=0.5, adaptive_p=1.0, adaptive_c=1e-3):
    # torch 1.12 has no torch.func; autograd.functional.jvp is the only JVP
    # available here, and is what the original MeanFlow script used.
    from torch.autograd.functional import jvp
    B = x0.shape[0]
    r, t = sample_rt(B, x0.device, p_eq)
    z_t = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    v = x1 - x0
    u, du_dt = jvp(lambda z, tt: net(z, r, tt), (z_t, t),
                   (v, torch.ones_like(t)), create_graph=True)
    target = (v - (t - r)[:, None, None, None] * du_dt).detach()
    err2 = ((u - target) ** 2).mean(dim=[1, 2, 3])
    w = 1.0 / (err2.detach() + adaptive_c) ** adaptive_p
    return (w * err2).mean()


@torch.no_grad()
def meanflow_generate(net, x0):
    B = x0.shape[0]
    return x0 + net(x0, torch.zeros(B, device=x0.device),
                    torch.ones(B, device=x0.device))


# --------------------------------------------------------------------------
# DDPM  (conditional epsilon-prediction; source concatenated to the noisy target)
# --------------------------------------------------------------------------
# The loaders hand out [0,1]; DDPM operates in [-1,1], so the conversion happens
# here rather than in the data path, which the other two methods share.
def ddpm_loss(net, diff, src, tgt):
    src, tgt = src * 2 - 1, tgt * 2 - 1
    t = torch.randint(0, diff.T, (tgt.shape[0],), device=tgt.device)
    noise = torch.randn_like(tgt)
    x_t = diff.q_sample(tgt, t, noise)
    return nn.functional.mse_loss(net(torch.cat([x_t, src], 1), t), noise)


@torch.no_grad()
def ddpm_generate(net, diff, x, n_steps):
    out = diff.ddim_sample(net, x * 2 - 1, n_steps=n_steps)
    x0 = out[0] if isinstance(out, tuple) else out       # (x0, traj) or x0
    return (x0 + 1) * 0.5


# --------------------------------------------------------------------------
def build(method, n_ch, base, device):
    if method == "cfm":
        from baselines.nets_cfm import VelocityUNet
        return (VelocityUNet(in_ch=n_ch, base=base).to(device),
                VelocityUNet(in_ch=n_ch, base=base).to(device), None)
    if method == "meanflow":
        from baselines.nets_meanflow import MeanVelocityUNet
        return (MeanVelocityUNet(in_ch=n_ch, base=base).to(device),
                MeanVelocityUNet(in_ch=n_ch, base=base).to(device), None)
    from baselines.nets_ddpm import CondUNet, GaussianDiffusion as Diffusion
    # conditional: the source is concatenated channel-wise to the noisy target
    return (CondUNet(in_ch=2 * n_ch, out_ch=n_ch, base=base).to(device),
            CondUNet(in_ch=2 * n_ch, out_ch=n_ch, base=base).to(device),
            Diffusion(T=1000, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["cfm", "meanflow", "ddpm"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--n_steps", type=int, default=10, help="CFM Euler / DDIM steps")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    add_data_args(ap)
    a = ap.parse_args()
    if a.data == "folder" and not a.data_root:
        ap.error("--data folder requires --data_root")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RES = os.path.join(EXPS, "checkpoints", a.tag)
    os.makedirs(RES, exist_ok=True)
    print(f"device={dev}  method={a.method}\nargs={vars(a)}", flush=True)

    tl, el, n_ch = loaders(a)
    net_ab, net_ba, diff = build(a.method, n_ch, a.base, dev)
    npar = sum(p.numel() for p in net_ab.parameters())
    print(f"{len(tl.dataset)} train / {len(el.dataset)} test  |  "
          f"{n_ch}ch  |  params/net {npar/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(list(net_ab.parameters()) + list(net_ba.parameters()),
                            lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    if a.method == "cfm":
        loss_fn = cfm_loss
        gen = lambda net, x: cfm_generate(net, x, a.n_steps)
    elif a.method == "meanflow":
        loss_fn = meanflow_loss
        gen = lambda net, x: meanflow_generate(net, x)
    else:
        loss_fn = lambda net, x0, x1: ddpm_loss(net, diff, x0, x1)
        gen = lambda net, x: ddpm_generate(net, diff, x, a.n_steps)

    f = open(os.path.join(RES, "train_log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["epoch", "loss", "ssim_A2B", "ssim_B2A", "lr", "sec"])

    for ep in range(1, a.epochs + 1):
        net_ab.train(); net_ba.train()
        t0, run, nb = time.time(), 0.0, 0
        for xa, xb, _y in tl:
            xa, xb = xa.to(dev, non_blocking=True), xb.to(dev, non_blocking=True)
            loss = loss_fn(net_ab, xa, xb) + loss_fn(net_ba, xb, xa)
            opt.zero_grad(set_to_none=True); loss.backward()
            if a.method == "ddpm":       # as in the original DDPM script
                torch.nn.utils.clip_grad_norm_(
                    list(net_ab.parameters()) + list(net_ba.parameters()), 1.0)
            opt.step()
            run += loss.item(); nb += 1
        sch.step()
        tr = run / max(nb, 1)

        sa = sb = float("nan")
        if ep % a.eval_every == 0 or ep == a.epochs:
            net_ab.eval(); net_ba.eval()
            r = evaluate(lambda x: gen(net_ab, x), lambda x: gen(net_ba, x),
                         el, dev, max_batches=8)
            sa, sb = r["ssim_A2B"], r["ssim_B2A"]
        sec = time.time() - t0
        w.writerow([ep, f"{tr:.6f}", f"{sa:.4f}", f"{sb:.4f}",
                    f"{opt.param_groups[0]['lr']:.2e}", f"{sec:.1f}"]); f.flush()
        print(f"ep {ep:3d}  loss {tr:.5f}  ssim A→B {sa:.4f} B→A {sb:.4f}  "
              f"{sec:.1f}s", flush=True)
        torch.save({"epoch": ep, "args": vars(a), "net_ab": net_ab.state_dict(),
                    "net_ba": net_ba.state_dict()}, os.path.join(RES, "last.pth"))
    f.close()

    # FAITHFUL: fixed schedule, no early stop, report the FINAL model -- the
    # same rule the ADNI baseline scripts and the CycleGAN host use.
    net_ab.eval(); net_ba.eval()
    r = evaluate(lambda x: gen(net_ab, x), lambda x: gen(net_ba, x), el, dev)
    txt = (f"method={a.method}\ndata={a.data_root or 'adni'}\nepochs={a.epochs}\n"
           f"params_per_net={npar}\n"
           f"ssim_A2B={r['ssim_A2B']:.4f}\npsnr_A2B={r['psnr_A2B']:.2f}\n"
           f"ssim_B2A={r['ssim_B2A']:.4f}\npsnr_B2A={r['psnr_B2A']:.2f}\n")
    open(os.path.join(RES, "final_eval.txt"), "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
