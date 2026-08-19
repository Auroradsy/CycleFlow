#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMCLAST-cg — three-stage training of the rewired CycleGAN (no GMM).

    S1  two INDEPENDENT autoencoders (+ self-GAN).   train E,D    f frozen = id
    S2  the bijection only.                          train f      E,D frozen
    S3  joint: + cycle, + latent-cycle, + morph.     train f,D    E   frozen

Why S1 does not train the cross paths: if it did, the two feature spaces would
align on their own and the flow would have nothing left to do (flow_work -> 0,
the identity-collapse we measured at 0.34 in the -bNLL run).  Keeping the two
autoencoders independent guarantees f has real work.

Why E stays frozen in S3: the representation is the object of study; letting the
encoders drift while the decoders adapt turns any morph result into a statement
about the decoders instead.

Losses (all GMM / flow-NLL / KL / pair terms are gone):
    L_self    L1(D_A(z), T1) + L1(D_B(u_B), FA)                        x10
    L_cross   L1(D_B(f z), FA) + L1(D_A(f^-1 u_B), T1)                 x10   <- paired data
    L_gan     LSGAN, host discriminators                               x1
    L_cyc     pixel cycle T1->FA->T1                                   x10
    L_latcyc  || E_B(D_B(f z)) - f z ||_1 / || f z ||_1                x1..5
    L_path    LSGAN(D_mix(D_B(s_t))) + || D_B(s_t) - D_B(s_{t-1}) ||_1

`pair` is deliberately absent: across the flow3 runs latent alignment and
cross-modal SSIM were anti-correlated (gap 9.07 -> 0.77 -> 0.091 while cross
went 0.749 -> 0.706 -> 0.738).

L_latcyc is the term that was missing from all eight earlier morph attempts: it
constrains the flow's output to be a FIXED POINT of the B autoencoder, i.e. to
be decodable, without constraining its distribution (which is what -bNLL did,
at the cost of collapsing the flow's work to 0.34).

  python train.py --config configs/morph.yaml
  bash configs/run_morph.sh
"""
import os
import sys
import csv
import time
import random
import argparse
import itertools

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.paired_dataset import (                                        # noqa: E402
    build_cache, PairedADNISliceDataset, subject_level_split, CACHE)
from model import MMCLASTcg, make_discriminators                         # noqa: E402
from utils.image import to_pm1, to_01                                    # noqa: E402
from utils.pool import ImagePool                                         # noqa: E402
from utils.config import parse_with_config                               # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Everything a run produces lives under exps/ (gitignored).  Override the whole
# tree with MMCLAST_EXPS to keep several experiment sets side by side.
EXPS = os.environ.get("MMCLAST_EXPS", os.path.join(_HERE, "exps"))
CKPT = os.path.join(EXPS, "checkpoints")
LOGS = os.path.join(EXPS, "logs")
CFM_T1FA, CFM_FAT1 = 0.729, 0.790          # best baseline, RECON_RESULTS.md §1


def val_split_from_train(train_idx, seed=42, frac=0.15):
    """Carve a validation set out of the TRAIN subjects (leakage-free)."""
    d = torch.load(CACHE, map_location="cpu")
    subj = d["subj_idx"].numpy()[train_idx.numpy()]
    uniq = np.unique(subj)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    n = max(1, int(round(frac * len(uniq))))
    va_s = set(uniq[perm[:n]].tolist())
    mask = np.array([s in va_s for s in subj])
    return train_idx[~mask], train_idx[mask]


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(m, loader, shuffle_probe=False):
    """SSIM for the four paths + latent diagnostics (+ optional u-shuffle)."""
    from skimage.metrics import structural_similarity as ssim
    m.eval()
    acc = {k: [] for k in ["self_T1", "self_FA", "T1toFA", "FAtoT1",
                           "shuf_FA", "shuf_T1", "ref_FA", "ref_T1"]}
    fw, gap = [], []
    for t1, fa, _ in loader:
        T = to_pm1(t1.to(DEV)); F = to_pm1(fa.to(DEV))
        zA = m.enc_A(T); uB = m.enc_B(F)
        u = m.a_to_b(zA); z = m.b_to_a(uB)
        im = {"self_T1": (to_01(m.dec_A(zA)), t1), "self_FA": (to_01(m.dec_B(uB)), fa),
              "T1toFA": (to_01(m.dec_B(u)), fa),   "FAtoT1": (to_01(m.dec_A(z)), t1)}
        if shuffle_probe and u.shape[0] > 1:
            r = torch.roll(torch.arange(u.shape[0]), 1)
            im["shuf_FA"] = (to_01(m.dec_B(u[r])), fa)
            im["shuf_T1"] = (to_01(m.dec_A(z[r])), t1)
            # Calibration: every slice here is MNI-registered, so two DIFFERENT
            # real brains already score high against each other.  ref_* is that
            # floor — the score a perfect model gets when handed the wrong code.
            # A load-bearing latent lands ON this floor; a decorative one (the
            # FiLM head of §3, Δ = 0.000) stays at its unshuffled score.
            im["ref_FA"] = (fa[r], fa); im["ref_T1"] = (t1[r], t1)
        for k, (pred, gt) in im.items():
            p = pred.cpu().numpy() if pred.is_cuda else pred.numpy()
            g = gt.numpy()
            for i in range(p.shape[0]):
                acc[k].append(ssim(g[i, 0], p[i, 0], data_range=1.0))
        fw.append(m.flow_work(zA)); gap.append(m.latent_gap(zA, uB))
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    out["flow_work"] = float(np.mean(fw)); out["latent_gap"] = float(np.mean(gap))
    m.train()
    return out


@torch.no_grad()
def val_monitor(m, loader, stage, w_latcyc):
    """Non-adversarial validation loss — the early-stopping criterion.

    loss_gan oscillates by construction, so it is excluded (same rule as the
    faithful CycleGAN runs).
    """
    m.eval(); tot, n = 0.0, 0
    l1 = nn.L1Loss()
    for t1, fa, _ in loader:
        T = to_pm1(t1.to(DEV)); F = to_pm1(fa.to(DEV))
        zA = m.enc_A(T); uB = m.enc_B(F)
        v = 0.0
        if stage in (1, 3):
            v = v + l1(m.dec_A(zA), T).item() + l1(m.dec_B(uB), F).item()
        if stage in (2, 3):
            u = m.a_to_b(zA); z = m.b_to_a(uB)
            fake_FA = m.dec_B(u); fake_T1 = m.dec_A(z)
            v = v + l1(fake_FA, F).item() + l1(fake_T1, T).item()
            if stage == 3:
                back = m.enc_B(fake_FA)
                v = v + w_latcyc * float((back - u).abs().mean() /
                                         (u.abs().mean() + 1e-8))
        tot += v; n += 1
    m.train()
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="MMCLAST-cg three-stage trainer.  See configs/ for ready-made runs.")
    ap.add_argument("--config", default=None,
                    help="YAML of hyperparameters; command-line flags still win")
    ap.add_argument("--variant", choices=["base", "latcyc", "morph", "morph_bi"],
                    default="morph")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--warm", default=os.path.join(CKPT, "host", "last.pth"),
                    help="plain-CycleGAN checkpoint to split into E/D ('' = from scratch)")
    # data
    ap.add_argument("--z_lo", type=int, default=40, help="axial band low, inclusive")
    ap.add_argument("--z_hi", type=int, default=49, help="axial band high, inclusive")
    ap.add_argument("--label_scheme", default="label_4")
    ap.add_argument("--resume_stage", type=int, default=0,
                    help="skip stages <= this and load weights from --resume_from. "
                         "Use 2 to share one S1+S2 across variants so that S3 is "
                         "the only thing that differs.")
    ap.add_argument("--resume_from", default="",
                    help="stage checkpoint to resume from (default: this tag's own)")
    ap.add_argument("--check_init", action="store_true",
                    help="verify f=id + the E/D split reproduces the host, then exit")
    # schedule
    ap.add_argument("--e1", type=int, default=120); ap.add_argument("--p1", type=int, default=20)
    ap.add_argument("--e2", type=int, default=80);  ap.add_argument("--p2", type=int, default=15)
    ap.add_argument("--e3", type=int, default=120); ap.add_argument("--p3", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_flow", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    # architecture
    ap.add_argument("--ngf", type=int, default=64); ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--n_flow", type=int, default=4)
    ap.add_argument("--flow_hidden", type=int, default=128)
    ap.add_argument("--pre_relu", type=int, default=1)
    # loss weights
    ap.add_argument("--w_self", type=float, default=10.0)
    ap.add_argument("--w_cross", type=float, default=10.0)
    ap.add_argument("--w_gan", type=float, default=1.0)
    ap.add_argument("--w_cyc", type=float, default=10.0)
    ap.add_argument("--w_latcyc", type=float, default=2.0)
    ap.add_argument("--w_path_gan", type=float, default=0.5)
    ap.add_argument("--w_path_smooth", type=float, default=1.0)
    ap.add_argument("--path_bidir", type=int, default=0,
                    help="also supervise the B->A path (walk f backwards, read with D_A)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = parse_with_config(ap)

    # A config may give `warm` as a repo-relative path; resolve it here so the
    # script works from any working directory, not just the repo root.
    if a.warm and not os.path.isabs(a.warm):
        a.warm = os.path.join(_HERE, a.warm)

    if a.variant == "base":
        a.w_latcyc = 0.0; a.w_path_gan = 0.0; a.w_path_smooth = 0.0; a.path_bidir = 0
    elif a.variant == "latcyc":
        a.w_path_gan = 0.0; a.w_path_smooth = 0.0; a.path_bidir = 0
    elif a.variant == "morph":
        a.path_bidir = 0
    elif a.variant == "morph_bi":
        a.path_bidir = 1
    use_path = a.w_path_gan > 0 or a.w_path_smooth > 0
    tag = a.tag or a.variant
    RESULTS = os.path.join(CKPT, tag)                 # weights + final_eval.txt
    LOGDIR = os.path.join(LOGS, tag)                  # per-epoch csv
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    print(f"device={DEV}  variant={a.variant}\n  weights -> {RESULTS}\n  log     -> {LOGDIR}\nargs={vars(a)}", flush=True)

    # --- data -------------------------------------------------------------
    build_cache()
    tr_idx, te_idx = subject_level_split(a.seed, 0.20, a.label_scheme, a.z_lo, a.z_hi)
    tr_idx, va_idx = val_split_from_train(tr_idx, a.seed, 0.15)
    tl = DataLoader(PairedADNISliceDataset(tr_idx, a.label_scheme), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(PairedADNISliceDataset(va_idx, a.label_scheme), batch_size=32,
                    shuffle=False, num_workers=2)
    el = DataLoader(PairedADNISliceDataset(te_idx, a.label_scheme), batch_size=32,
                    shuffle=False, num_workers=2)
    print(f"train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} slices  "
          f"({len(tl)} batches/epoch)", flush=True)

    # --- init check: does the split + f=id reproduce plain CycleGAN? ------
    if a.check_init:
        mm = MMCLASTcg(a.ngf, a.n_blocks, a.n_flow, a.flow_hidden, pre_relu=False).to(DEV)
        mm.load_cyclegan(a.warm, map_location=DEV)
        r = evaluate(mm, el)
        ref = os.path.join(CKPT, "host", "final_eval.txt")
        print(f"\n[check_init] pre_relu=0, f=identity")
        print(f"  T1->FA {r['T1toFA']:.4f}   FA->T1 {r['FAtoT1']:.4f}")
        print(f"  flow_work {r['flow_work']:.2e} (must be ~0)")
        if os.path.exists(ref):
            print("  host reference:\n   ", open(ref).read().replace("\n", "\n    ").strip())
        return

    # --- model ------------------------------------------------------------
    m = MMCLASTcg(a.ngf, a.n_blocks, a.n_flow, a.flow_hidden, bool(a.pre_relu)).to(DEV)
    if a.warm and os.path.exists(a.warm):
        ep = m.load_cyclegan(a.warm, map_location=DEV)
        print(f"warm start from {a.warm} (host epoch {ep})", flush=True)
    else:
        print("no warm start — training the host from scratch", flush=True)
    D = make_discriminators(a.ndf, mix=use_path)
    for k in D:
        D[k] = D[k].to(DEV)

    if a.resume_stage:
        src = a.resume_from or os.path.join(RESULTS, f"stage{a.resume_stage}.pth")
        if not os.path.isabs(src):
            src = os.path.join(_HERE, src)
        ck = torch.load(src, map_location=DEV)
        m.load_state_dict(ck["model"])
        # Stage checkpoints written before this flag existed carry no
        # discriminators; those runs restart S3 with fresh critics.  Say so out
        # loud — it is a real difference between two otherwise-identical runs,
        # so every arm of a comparison must resume the same way.
        have_d = [k for k in D if f"d_{k}" in ck]
        for k in have_d:
            D[k].load_state_dict(ck[f"d_{k}"])
        print(f"resumed from {src} (through stage {ck.get('stage')}); "
              f"discriminators: {'restored ' + ','.join(have_d) if have_d else 'FRESH (not in checkpoint)'}",
              flush=True)

    n_host = sum(p.numel() for mod in (m.enc_A, m.enc_B, m.dec_A, m.dec_B)
                 for p in mod.parameters())
    print(f"params: host {n_host/1e6:.2f} M + flow {sum(p.numel() for p in m.flow.parameters())/1e6:.2f} M",
          flush=True)

    opt_G = torch.optim.Adam([
        {"params": itertools.chain(m.enc_A.parameters(), m.enc_B.parameters(),
                                   m.dec_A.parameters(), m.dec_B.parameters()), "lr": a.lr},
        {"params": m.flow.parameters(), "lr": a.lr_flow}], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(*[D[k].parameters() for k in D]),
                             lr=a.lr, betas=(0.5, 0.999))
    crit_gan, l1 = nn.MSELoss(), nn.L1Loss()
    pools = {k: ImagePool(50) for k in D}

    def req(mods, flag):
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad = flag

    def set_stage(s):
        enc, dec, flow = [m.enc_A, m.enc_B], [m.dec_A, m.dec_B], [m.flow]
        if s == 1:   req(enc, True);  req(dec, True);  req(flow, False)
        elif s == 2: req(enc, False); req(dec, False); req(flow, True)
        else:        req(enc, False); req(dec, True);  req(flow, True)
        print(f"  [stage{s}] trainable: enc={s==1} dec={s!=2} flow={s!=1}", flush=True)

    def snapshot():
        return {k: {kk: vv.detach().cpu().clone() for kk, vv in mod.state_dict().items()}
                for k, mod in [("m", m)] + [(f"d_{k}", D[k]) for k in D]}

    def restore(sn):
        m.load_state_dict(sn["m"])
        for k in D:
            D[k].load_state_dict(sn[f"d_{k}"])

    log = open(os.path.join(LOGDIR, "train_log.csv"), "w", newline="")
    W = csv.writer(log)
    W.writerow(["epoch", "stage", "G", "D", "self", "cross", "gan", "cyc", "latcyc",
                "path_gan", "path_sm", "val", "self_T1", "self_FA", "T1toFA", "FAtoT1",
                "flow_work", "latent_gap", "sec"])

    ep_global = 0
    for stage, n_ep, patience in [(1, a.e1, a.p1), (2, a.e2, a.p2), (3, a.e3, a.p3)]:
        if stage <= a.resume_stage:
            print(f"  [stage{stage}] skipped (resumed)", flush=True)
            continue
        set_stage(stage)
        if stage == 2:                       # the flow alone gets the full LR
            opt_G.param_groups[1]["lr"] = a.lr_flow * 2
        if stage == 3:
            opt_G.param_groups[0]["lr"] = a.lr * 0.5
            opt_G.param_groups[1]["lr"] = a.lr_flow
        best, bad, best_sn = float("inf"), 0, None

        for e in range(n_ep):
            t0 = time.time(); m.train()
            acc = dict(G=0., D=0., self=0., cross=0., gan=0., cyc=0.,
                       latcyc=0., path_gan=0., path_sm=0.); nb = 0

            for t1, fa, _ in tl:
                T = to_pm1(t1.to(DEV, non_blocking=True))
                F = to_pm1(fa.to(DEV, non_blocking=True))
                nb += 1
                req(list(D.values()), False); opt_G.zero_grad()

                zA = m.enc_A(T); uB = m.enc_B(F)
                fakes = {}                         # name -> (image, discriminator key)
                L_self = torch.zeros((), device=DEV)
                L_cross = torch.zeros((), device=DEV)
                L_cyc = torch.zeros((), device=DEV)
                L_lat = torch.zeros((), device=DEV)
                L_pg = torch.zeros((), device=DEV)
                L_ps = torch.zeros((), device=DEV)

                if stage in (1, 3):
                    rec_A = m.dec_A(zA); rec_B = m.dec_B(uB)
                    L_self = l1(rec_A, T) + l1(rec_B, F)
                    fakes["self_T1"] = (rec_A, "T1"); fakes["self_FA"] = (rec_B, "FA")

                if stage in (2, 3):
                    u = m.a_to_b(zA); z = m.b_to_a(uB)
                    fake_FA = m.dec_B(u); fake_T1 = m.dec_A(z)
                    L_cross = l1(fake_FA, F) + l1(fake_T1, T)
                    fakes["cross_FA"] = (fake_FA, "FA"); fakes["cross_T1"] = (fake_T1, "T1")

                    if stage == 3:
                        # cycle T1->FA->T1 and FA->T1->FA.  The flow cancels
                        # analytically ONLY if E_B(D_B(.)) = id, so this is a real
                        # constraint on the encoder/decoder pair.
                        back_u = m.enc_B(fake_FA); back_z = m.enc_A(fake_T1)
                        rec_T1 = m.dec_A(m.b_to_a(back_u))
                        rec_FA = m.dec_B(m.a_to_b(back_z))
                        L_cyc = l1(rec_T1, T) + l1(rec_FA, F)
                        # latent cycle: f's output must be a fixed point of the
                        # B autoencoder (scale-free, so it cannot be minimised by
                        # shrinking the code).
                        L_lat = ((back_u - u).abs().mean() / (u.abs().mean() + 1e-8)
                                 + (back_z - z).abs().mean() / (z.abs().mean() + 1e-8))

                    if stage == 3 and use_path:
                        # Trajectory supervision: one random adjacent pair per
                        # step, per supervised direction.
                        #
                        # A->B walks f forward and reads the states with D_B;
                        # B->A walks f backward and reads them with D_A.  With
                        # path_bidir=0 only the first exists, which is why every
                        # FA->T1 intermediate is unsupervised (fig 33).
                        #
                        # Both halves are AVERAGED, not summed, so w_path_* keeps
                        # the same effective scale in either mode and the
                        # ablation is about direction rather than weight.
                        legs = [(m.walk(zA), m.dec_B)]
                        if a.path_bidir:
                            legs.append((m.walk(uB, inverse=True), m.dec_A))

                        path_imgs = []
                        for states, dec in legs:
                            t = random.randint(1, len(states) - 1)
                            prev = dec(states[t - 1]); cur = dec(states[t])
                            L_ps = L_ps + (cur - prev).abs().mean()
                            path_imgs += [prev, cur]
                        L_ps = L_ps / len(legs)

                        if a.w_path_gan > 0:
                            for img in path_imgs:
                                p = D["mix"](img)
                                L_pg = L_pg + crit_gan(p, torch.ones_like(p))
                            L_pg = L_pg / len(legs)
                            fakes["path"] = (torch.cat(path_imgs, 0), "mix")

                L_gan = torch.zeros((), device=DEV)
                for _k, (img, dk) in fakes.items():
                    if dk == "mix":
                        continue                       # already counted in L_pg
                    p = D[dk](img)
                    L_gan = L_gan + crit_gan(p, torch.ones_like(p))

                loss_G = (a.w_self * L_self + a.w_cross * L_cross + a.w_gan * L_gan
                          + a.w_cyc * L_cyc + a.w_latcyc * L_lat
                          + a.w_path_gan * L_pg + a.w_path_smooth * L_ps)
                loss_G.backward()
                if a.clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt_G.param_groups for p in g["params"]
                         if p.requires_grad], a.clip)
                opt_G.step()

                # ---- discriminators ----
                req(list(D.values()), True); opt_D.zero_grad()
                real = {"T1": T, "FA": F, "mix": torch.cat([T, F], 0)}
                by_d = {}
                for _k, (img, dk) in fakes.items():
                    by_d.setdefault(dk, []).append(img.detach())
                d_tot = 0.0
                for dk, imgs in by_d.items():
                    pr = D[dk](real[dk]); lr_ = crit_gan(pr, torch.ones_like(pr))
                    pf = D[dk](pools[dk].query(torch.cat(imgs, 0)))
                    lf = crit_gan(pf, torch.zeros_like(pf))
                    ld = 0.5 * (lr_ + lf); ld.backward(); d_tot += ld.item()
                opt_D.step()

                for k, v in [("G", loss_G.item()), ("D", d_tot), ("self", float(L_self)),
                             ("cross", float(L_cross)), ("gan", float(L_gan)),
                             ("cyc", float(L_cyc)), ("latcyc", float(L_lat)),
                             ("path_gan", float(L_pg)), ("path_sm", float(L_ps))]:
                    acc[k] += v

            acc = {k: v / max(nb, 1) for k, v in acc.items()}
            vloss = val_monitor(m, vl, stage, a.w_latcyc)
            ep_global += 1
            row = [ep_global, stage] + [round(acc[k], 4) for k in
                                        ["G", "D", "self", "cross", "gan", "cyc",
                                         "latcyc", "path_gan", "path_sm"]] + [round(vloss, 5)]

            if (e + 1) % 10 == 0 or e == n_ep - 1:
                r = evaluate(m, el)
                print(f"[s{stage} ep{e+1}/{n_ep}] G={acc['G']:.3f} self={acc['self']:.3f} "
                      f"cross={acc['cross']:.3f} lat={acc['latcyc']:.3f} val={vloss:.4f} | "
                      f"selfT1 {r['self_T1']:.3f} selfFA {r['self_FA']:.3f} | "
                      f"T1→FA {r['T1toFA']:.3f} FA→T1 {r['FAtoT1']:.3f} | "
                      f"fw={r['flow_work']:.2f} gap={r['latent_gap']:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                row += [round(r[k], 4) for k in ["self_T1", "self_FA", "T1toFA", "FAtoT1",
                                                 "flow_work", "latent_gap"]]
            else:
                print(f"[s{stage} ep{e+1}/{n_ep}] G={acc['G']:.3f} self={acc['self']:.3f} "
                      f"cross={acc['cross']:.3f} lat={acc['latcyc']:.3f} "
                      f"path={acc['path_gan']:.3f}/{acc['path_sm']:.3f} val={vloss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                row += ["", "", "", "", "", ""]
            row.append(round(time.time() - t0, 1)); W.writerow(row); log.flush()

            if vloss < best - 1e-5:
                best, bad, best_sn = vloss, 0, snapshot()
            else:
                bad += 1
            if bad >= patience:
                print(f"  [stage{stage}] early stop @ep{e+1} (best val {best:.5f})", flush=True)
                break

        if best_sn is not None:
            restore(best_sn)
            print(f"  [stage{stage}] restored best (val {best:.5f})", flush=True)
        snap = dict(model=m.state_dict(), args=vars(a), stage=stage)
        snap.update({f"d_{k}": D[k].state_dict() for k in D})   # so a resume is exact
        torch.save(snap, os.path.join(RESULTS, f"stage{stage}.pth"))

    # --- final ------------------------------------------------------------
    torch.save(dict(model=m.state_dict(), args=vars(a)), os.path.join(RESULTS, "model.pth"))
    r = evaluate(m, el, shuffle_probe=True)
    txt = (f"variant={a.variant}\n"
           f"self_T1={r['self_T1']:.4f}\nself_FA={r['self_FA']:.4f}\n"
           f"ssim_T1toFA={r['T1toFA']:.4f}\nssim_FAtoT1={r['FAtoT1']:.4f}\n"
           f"shuffled_T1toFA={r['shuf_FA']:.4f}\nshuffled_FAtoT1={r['shuf_T1']:.4f}\n"
           f"shuffle_floor_FA={r['ref_FA']:.4f}\nshuffle_floor_T1={r['ref_T1']:.4f}\n"
           f"flow_work={r['flow_work']:.4f}\nlatent_gap={r['latent_gap']:.4f}\n")
    open(os.path.join(RESULTS, "final_eval.txt"), "w").write(txt)
    print(f"\n[MMCLAST-cg/{a.variant}] FINAL")
    print(f"  self   T1 {r['self_T1']:.4f}   FA {r['self_FA']:.4f}")
    print(f"  cross  T1→FA {r['T1toFA']:.4f}{'✓' if r['T1toFA']>CFM_T1FA else '✗'}   "
          f"FA→T1 {r['FAtoT1']:.4f}{'✓' if r['FAtoT1']>CFM_FAT1 else '✗'}  (vs CFM {CFM_T1FA}/{CFM_FAT1})")
    print(f"  u-shuffle  T1→FA {r['shuf_FA']:.4f} (Δ {r['T1toFA']-r['shuf_FA']:+.4f})   "
          f"FA→T1 {r['shuf_T1']:.4f} (Δ {r['FAtoT1']-r['shuf_T1']:+.4f})")
    print(f"  shuffle floor (two different real slices): FA {r['ref_FA']:.4f}  T1 {r['ref_T1']:.4f}"
          f"   -> latent is load-bearing iff shuffled ≈ floor")
    print(f"  flow_work {r['flow_work']:.4f}   latent_gap {r['latent_gap']:.4f}")
    print("Out:", RESULTS, flush=True)
    log.close()


if __name__ == "__main__":
    main()
