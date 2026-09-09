#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conditional DDPM for T1<->FA cross-modal translation (ADNI pilot).

Design
------
* Conditional epsilon-prediction DDPM. The UNet takes 2 input channels
  ``concat([x_noised_target, x_source], dim=1)`` and predicts the noise added to
  the target (1 output channel).  Conditioning on the source modality is done by
  plain channel concatenation -- the simplest robust translation scheme.
* TWO separate models are trained (one for T1->FA, one for FA->T1).  This is
  simpler and more robust than a single shared model with a direction flag and is
  what the train/eval scripts use.
* Standard DDPM forward process: T=1000 train steps, linear beta schedule
  (1e-4 -> 0.02), MSE loss on epsilon.
* Sampling: DDIM (deterministic) so we get a clean, reproducible denoising
  trajectory (t = T -> 0) for the transition visualization.

Images are kept in [0,1] from the dataset and mapped to [-1,1] inside the
diffusion process (and back to [0,1] for viz/metrics).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------
def sinusoidal_embedding(timesteps, dim):
    """Standard transformer / DDPM sinusoidal time embedding.

    timesteps: [B] (long or float).  Returns [B, dim].
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, with time embedding injection + residual."""

    def __init__(self, in_ch, out_ch, t_dim, groups=8):
        super().__init__()
        g_in = math.gcd(groups, in_ch)
        g_out = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention block as in DDPM (Ho et al. 2020) — applied at the low-res
    bottleneck. GroupNorm -> qkv -> softmax attention -> proj -> residual."""

    def __init__(self, ch, groups=8):
        super().__init__()
        g = math.gcd(groups, ch)
        self.norm = nn.GroupNorm(g, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        B, C, H, W = h.shape
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)   # B,N,C
        k = self.k(h).reshape(B, C, H * W)                    # B,C,N
        w = torch.bmm(q, k) * (C ** -0.5)
        w = torch.softmax(w, dim=-1)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)   # B,N,C
        out = torch.bmm(w, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CondUNet(nn.Module):
    """Modest UNet for conditional epsilon prediction.

    in_ch = 2  (noised target + source)
    out_ch = 1 (predicted noise on the target)
    3 resolutions (112 -> 56 -> 28), 2 res-blocks per stage.
    """

    def __init__(self, in_ch=2, out_ch=1, base=64, ch_mult=(1, 2, 4), t_dim=256, groups=8):
        super().__init__()
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        chs = [base * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # ---- encoder ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_chs = [chs[0]]
        prev = chs[0]
        for li, c in enumerate(chs):
            stage = nn.ModuleList([ResBlock(prev, c, t_dim, groups),
                                   ResBlock(c, c, t_dim, groups)])
            self.down_blocks.append(stage)
            skip_chs += [c, c]
            prev = c
            if li != len(chs) - 1:
                self.downsamples.append(Downsample(c))
                skip_chs.append(c)
            else:
                self.downsamples.append(None)

        # ---- middle (ResBlock -> Attn -> ResBlock, as in DDPM) ----
        self.mid1 = ResBlock(prev, prev, t_dim, groups)
        self.mid_attn = AttnBlock(prev, groups)
        self.mid2 = ResBlock(prev, prev, t_dim, groups)

        # ---- decoder ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for li, c in reversed(list(enumerate(chs))):
            stage = nn.ModuleList([
                ResBlock(prev + skip_chs.pop(), c, t_dim, groups),
                ResBlock(c + skip_chs.pop(), c, t_dim, groups),
                ResBlock(c + skip_chs.pop(), c, t_dim, groups),
            ])
            self.up_blocks.append(stage)
            prev = c
            if li != 0:
                self.upsamples.append(Upsample(c))
            else:
                self.upsamples.append(None)

        g_out = math.gcd(groups, prev)
        self.out_norm = nn.GroupNorm(g_out, prev)
        self.out_conv = nn.Conv2d(prev, out_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(sinusoidal_embedding(t, self.t_dim))
        h = self.in_conv(x)
        skips = [h]
        for stage, down in zip(self.down_blocks, self.downsamples):
            for blk in stage:
                h = blk(h, temb)
                skips.append(h)
            if down is not None:
                h = down(h)
                skips.append(h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for stage, up in zip(self.up_blocks, self.upsamples):
            for blk in stage:
                h = torch.cat([h, skips.pop()], dim=1)
                h = blk(h, temb)
            if up is not None:
                h = up(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Gaussian diffusion (epsilon-prediction, linear schedule, DDIM sampler)
# ---------------------------------------------------------------------------
class GaussianDiffusion:
    """Holds the noise schedule and provides q_sample / DDIM sampling.

    All buffers live on ``device``.  Works in [-1,1] space.
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64, device=device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = ac.float()
        self.sqrt_acp = torch.sqrt(ac).float()
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - ac).float()

    def to(self, device):
        self.device = device
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    # ---- forward (training) ----
    def q_sample(self, x0, t, noise):
        """x0:[B,1,H,W] in [-1,1]; t:[B] long; noise same shape as x0."""
        a = self.sqrt_acp[t][:, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None]
        return a * x0 + b * noise

    # ---- ancestral DDPM sampling (Ho et al. 2020, the faithful sampler) ----
    @torch.no_grad()
    def p_sample_loop(self, model, source):
        """Full T-step ancestral reverse process. x_{t-1} = 1/sqrt(a_t) (x_t -
        beta_t/sqrt(1-acp_t) eps) + sqrt(beta_t) z. Fixed variance sigma_t^2=beta_t."""
        B = source.shape[0]; dev = source.device
        x = torch.randn_like(source)
        for i in reversed(range(self.T)):
            t_b = torch.full((B,), i, device=dev, dtype=torch.long)
            eps = model(torch.cat([x, source], dim=1), t_b)
            alpha = self.alphas[i]; acp = self.alphas_cumprod[i]; beta = self.betas[i]
            mean = (x - beta / torch.sqrt(1 - acp) * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)

    # ---- DDIM sampling (deterministic, eta=0) ----
    @torch.no_grad()
    def ddim_sample(self, model, source, n_steps=50, eta=0.0, return_traj_at=None):
        """Generate the target conditioned on ``source`` (both in [-1,1]).

        model: CondUNet, input concat([x_t, source]).
        source: [B,1,H,W] in [-1,1].
        Returns (x0_final, traj) where traj is a dict {requested_t : x_predicted_x0}
        if return_traj_at is given (list of original-scale timesteps), else None.

        The trajectory stores the model's predicted *x0* at the given diffusion
        timesteps -- this is the clean "denoised estimate" at each point of the
        reverse trajectory and is what we visualize (noise -> clean target).
        """
        B = source.shape[0]
        dev = source.device
        # evenly spaced subsequence of timesteps, descending
        step_idx = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=dev)
        x = torch.randn_like(source)

        traj = {}
        want = set(int(v) for v in return_traj_at) if return_traj_at is not None else set()

        for i in range(n_steps):
            t = step_idx[i]
            t_b = torch.full((B,), int(t), device=dev, dtype=torch.long)
            inp = torch.cat([x, source], dim=1)
            eps = model(inp, t_b)

            acp_t = self.alphas_cumprod[t]
            sqrt_acp_t = torch.sqrt(acp_t)
            sqrt_omacp_t = torch.sqrt(1.0 - acp_t)
            x0_pred = (x - sqrt_omacp_t * eps) / sqrt_acp_t
            x0_pred = x0_pred.clamp(-1, 1)

            # record predicted x0 closest to requested timesteps
            if want:
                ti = int(t)
                # snap each requested timestep to the nearest sampled step once
                for w in list(want):
                    if abs(ti - w) <= (self.T // n_steps):
                        if w not in traj:
                            traj[w] = x0_pred.detach().cpu().clone()
                            want.discard(w)

            if i == n_steps - 1:
                x = x0_pred
                break

            t_next = step_idx[i + 1]
            acp_next = self.alphas_cumprod[t_next]
            sqrt_acp_next = torch.sqrt(acp_next)
            # deterministic DDIM (eta=0): sigma=0
            sigma = eta * torch.sqrt((1 - acp_next) / (1 - acp_t) * (1 - acp_t / acp_next))
            dir_xt = torch.sqrt((1 - acp_next - sigma ** 2).clamp(min=0.0)) * eps
            x = sqrt_acp_next * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        # make sure any unmatched requested timesteps still get the final x0
        for w in list(want):
            traj[w] = x.detach().cpu().clone()

        return x, (traj if return_traj_at is not None else None)


# ---------------------------------------------------------------------------
def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    # quick shape sanity check
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = CondUNet().to(dev)
    x = torch.randn(2, 2, 112, 112, device=dev)
    t = torch.randint(0, 1000, (2,), device=dev)
    print("out", net(x, t).shape, "params(M)", count_params(net) / 1e6)
    diff = GaussianDiffusion(device=dev)
    src = torch.randn(2, 1, 112, 112, device=dev)
    y, traj = diff.ddim_sample(net, src, n_steps=10, return_traj_at=[999, 500, 0])
    print("sample", y.shape, "traj keys", sorted(traj.keys()))
