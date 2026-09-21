#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-modal baselines on either testbed, from one implementation.

  cfm       Rectified-flow Conditional Flow Matching.  Learns a velocity field
            v(x,t) with the linear interpolant and the paired (OT) coupling;
            translation is Euler ODE integration from source to target.
            Strongest baseline on ADNI.
  recflow   Rectified Flow (Liu et al. 2023), i.e. the same interpolant taken
            through the paper's own procedure: fit 1-rectified flow on the
            INDEPENDENT coupling of the two domains, simulate it to draw
            (x0, ODE(x0)) pairs, refit on those pairs (reflow), and translate in
            ONE Euler step.  The reflow is the whole contribution, so it is what
            separates this row from cfm above.
  meanflow  MeanFlow (Geng et al. 2025).  Learns the AVERAGE velocity over an
            interval via the MeanFlow identity, so t=0->1 is a single forward.
  ddpm      Conditional DDPM; the source image is concatenated to the noisy
            target, sampled with DDIM.

Each method keeps the objective and the schedule of the original ADNI scripts
in __outdated_files/baseline/; the only thing this file changes is that the
data source and the channel count are arguments.

  python -m baselines.train --method cfm --data folder \
      --data_root /ix/lzhan/siyuan/datasets/processed_datas/MNIST_CycleFlow/mnist_petct_paired --tag cfm_mnist
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

from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()


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
# Rectified Flow  (independent coupling, then reflow)
# --------------------------------------------------------------------------
def rectflow_loss(net, x0, x1):
    """cfm_loss on the INDEPENDENT coupling: the target is a shuffled batch.

    "Flow Straight and Fast" transports one data distribution to another without
    assuming a coupling; drawing (x0, x1) from pi_0 x pi_1 is what makes the
    reflow below a non-trivial step, because the 1-rectified flow's own transport
    plan is the thing being rewired.  cfm above instead uses the ground-truth
    pairing, which no unpaired method has access to.
    """
    return cfm_loss(net, x0, x1[torch.randperm(x1.shape[0], device=x1.device)])


@torch.no_grad()
def reflow_pairs(nets, loader, dev, n_steps):
    """(x0, ODE_{v1}(x0)) for both directions, simulated once with the fitted flow.

    Reflow retrains on pairs the CURRENT flow itself produces; the new coupling is
    at least as good in transport cost and its paths are straighter, which is what
    buys the one-step inference.  The pairs are drawn once, with a frozen v1 and a
    fine (n_steps) solver, and then held fixed, as in the paper -- for the folder
    datasets this also freezes a single draw of the random crop, which is the same
    thing as treating the generated set as a dataset.
    """
    cols = [[], [], [], []]
    for xa, xb, *_ in loader:
        xa, xb = xa.to(dev, non_blocking=True), xb.to(dev, non_blocking=True)
        cols[0].append(xa.cpu()); cols[1].append(cfm_generate(nets[0], xa, n_steps).cpu())
        cols[2].append(xb.cpu()); cols[3].append(cfm_generate(nets[1], xb, n_steps).cpu())
    return torch.utils.data.TensorDataset(*[torch.cat(c) for c in cols])


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
    if method in ("cfm", "recflow"):
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


def train_stage(a, nets, loader, step, gen, epochs, stage, el, dev, w, save):
    """One cosine-annealed run over `loader`; the only loop any method uses.

    `step(nets, batch) -> loss` owns the device transfer and the objective, so a
    stage that trains on generated pairs differs from a stage that trains on the
    data only in that function.
    """
    opt = torch.optim.AdamW(list(nets[0].parameters()) + list(nets[1].parameters()),
                            lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        nets[0].train(); nets[1].train()
        t0, run, nb = time.time(), 0.0, 0
        for batch in loader:
            loss = step(nets, batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            if a.method == "ddpm":       # as in the original DDPM script
                torch.nn.utils.clip_grad_norm_(
                    list(nets[0].parameters()) + list(nets[1].parameters()), 1.0)
            opt.step()
            run += loss.item(); nb += 1
        sch.step()
        tr = run / max(nb, 1)

        sa = sb = float("nan")
        if ep % a.eval_every == 0 or ep == epochs:
            nets[0].eval(); nets[1].eval()
            r = evaluate(lambda x: gen(nets[0], x), lambda x: gen(nets[1], x),
                         el, dev, max_batches=8)
            sa, sb = r["ssim_A2B"], r["ssim_B2A"]
        sec = time.time() - t0
        w.writerow([stage, ep, f"{tr:.6f}", f"{sa:.4f}", f"{sb:.4f}",
                    f"{opt.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
        print(f"[stage {stage}] ep {ep:3d}  loss {tr:.5f}  ssim A→B {sa:.4f} B→A {sb:.4f}  "
              f"{sec:.1f}s", flush=True)
        save(stage, ep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["cfm", "recflow", "meanflow", "ddpm"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--n_steps", type=int, default=10, help="CFM Euler / DDIM steps")
    ap.add_argument("--reflow_epochs", type=int, default=0,
                    help="recflow: epochs of the second (reflowed) stage; 0 = --epochs")
    ap.add_argument("--reflow_sim_steps", type=int, default=100,
                    help="recflow: Euler steps used to SIMULATE the pairs the reflow trains on")
    ap.add_argument("--coupling", default="independent", choices=["independent", "paired"],
                    help="recflow: the coupling of the first stage; the paper's is independent")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    add_data_args(ap)
    a = ap.parse_args()
    if a.data == "folder" and not a.data_root:
        ap.error("--data folder requires --data_root")
    # RecFlow's claim is that the reflowed field is straight enough for ONE step;
    # the shared default (10) is CFM's, so override it unless it was given.
    if a.method == "recflow" and not any(s.split("=")[0] == "--n_steps" for s in sys.argv[1:]):
        a.n_steps = 1

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

    if a.method in ("cfm", "recflow"):
        loss_fn = cfm_loss if a.method == "cfm" or a.coupling == "paired" else rectflow_loss
        gen = lambda net, x: cfm_generate(net, x, a.n_steps)
    elif a.method == "meanflow":
        loss_fn = meanflow_loss
        gen = lambda net, x: meanflow_generate(net, x)
    else:
        loss_fn = lambda net, x0, x1: ddpm_loss(net, diff, x0, x1)
        gen = lambda net, x: ddpm_generate(net, diff, x, a.n_steps)

    def data_step(nets, batch):
        xa = batch[0].to(dev, non_blocking=True)
        xb = batch[1].to(dev, non_blocking=True)
        return loss_fn(nets[0], xa, xb) + loss_fn(nets[1], xb, xa)

    f = open(os.path.join(RES, "train_log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["stage", "epoch", "loss", "ssim_A2B", "ssim_B2A", "lr", "sec"])
    state = {}

    def save(stage, ep):
        f.flush()
        # Stage 1 of recflow is the 1-rectified flow; the reported model is the
        # reflowed one, so the keys the readers use (net_ab/net_ba) always name
        # the latest stage and the earlier flow is kept beside it.
        keys = ("net_ab_1rf", "net_ba_1rf") if a.method == "recflow" and stage == 1 \
            else ("net_ab", "net_ba")
        state.update({keys[0]: net_ab.state_dict(), keys[1]: net_ba.state_dict(),
                      "epoch": ep, "stage": stage, "args": vars(a)})
        torch.save(state, os.path.join(RES, "last.pth"))

    train_stage(a, (net_ab, net_ba), tl, data_step, gen, a.epochs, 1, el, dev, w, save)

    if a.method == "recflow":
        t0 = time.time()
        net_ab.eval(); net_ba.eval()
        pairs = reflow_pairs((net_ab, net_ba), tl, dev, a.reflow_sim_steps)
        print(f"reflow: {len(pairs)} generated pairs per direction, "
              f"{a.reflow_sim_steps} Euler steps ({time.time() - t0:.0f}s)", flush=True)
        # The 1-rectified flow is frozen; the second flow starts from scratch, as
        # the paper's reflow does (it refits, it does not fine-tune).
        state["reflow_pairs"] = len(pairs)
        net_ab, net_ba, _ = build(a.method, n_ch, a.base, dev)
        pl = torch.utils.data.DataLoader(pairs, batch_size=a.batch, shuffle=True,
                                         drop_last=True, pin_memory=True)

        def reflow_step(nets, batch):
            t = [x.to(dev, non_blocking=True) for x in batch]
            return cfm_loss(nets[0], t[0], t[1]) + cfm_loss(nets[1], t[2], t[3])

        train_stage(a, (net_ab, net_ba), pl, reflow_step, gen,
                    a.reflow_epochs or a.epochs, 2, el, dev, w, save)
    f.close()

    # FAITHFUL: fixed schedule, no early stop, report the FINAL model -- the
    # same rule the ADNI baseline scripts and the CycleGAN host use.
    net_ab.eval(); net_ba.eval()
    r = evaluate(lambda x: gen(net_ab, x), lambda x: gen(net_ba, x), el, dev)
    txt = (f"method={a.method}\ndata={a.data_root or 'adni'}\nepochs={a.epochs}\n"
           f"params_per_net={npar}\n"
           f"ssim_A2B={r['ssim_A2B']:.4f}\npsnr_A2B={r['psnr_A2B']:.2f}\n"
           f"ssim_B2A={r['ssim_B2A']:.4f}\npsnr_B2A={r['psnr_B2A']:.2f}\n")
    if a.method == "recflow":
        # The straightening is the claim; report what it buys at each step count.
        txt += f"coupling={a.coupling}\nreflow_sim_steps={a.reflow_sim_steps}\nn_steps={a.n_steps}\n"
        for k in (1, 2, 10):
            rk = evaluate(lambda x: cfm_generate(net_ab, x, k),
                          lambda x: cfm_generate(net_ba, x, k), el, dev)
            txt += (f"steps={k}: ssim_A2B={rk['ssim_A2B']:.4f} psnr_A2B={rk['psnr_A2B']:.2f} "
                    f"ssim_B2A={rk['ssim_B2A']:.4f} psnr_B2A={rk['psnr_B2A']:.2f}\n")
    open(os.path.join(RES, "final_eval.txt"), "w").write(txt)
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
