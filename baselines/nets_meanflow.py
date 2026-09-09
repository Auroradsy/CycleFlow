#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-time-conditioned U-Net for MeanFlow (Geng et al. 2025).

MeanFlow learns the AVERAGE velocity u_theta(z_t, r, t) over the interval [r, t],
satisfying the identity
        u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t)
where v is the instantaneous (conditional) velocity x1 - x0 on the linear
interpolant z_t = (1-t) x0 + t x1, and d/dt u is the total derivative computed
by a JVP along (v, dr=0, dt=1).

One-step generation:  x1_hat = x0 + (1 - 0) * u_theta(x0, r=0, t=1).

This network is the CFM VelocityUNet extended to take a second time input r:
both r and t get sinusoidal embeddings, summed into the shared time-MLP, so the
architecture / param count stays comparable to the CFM baseline for fairness.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class MeanVelocityUNet(nn.Module):
    """(x [B,1,112,112], r [B], t [B]) -> average velocity u [B,1,112,112]."""

    def __init__(self, in_ch=1, base=64, ch_mult=(1, 2, 4), t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        chs = [base * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.enc = nn.ModuleList(); self.downs = nn.ModuleList()
        prev = chs[0]
        for i, c in enumerate(chs):
            self.enc.append(ResBlock(prev, c, t_dim)); prev = c
            self.downs.append(Down(c) if i != len(chs) - 1 else None)

        self.mid1 = ResBlock(prev, prev, t_dim)
        self.mid2 = ResBlock(prev, prev, t_dim)

        self.dec = nn.ModuleList(); self.ups = nn.ModuleList()
        for i in reversed(range(len(chs))):
            c = chs[i]
            self.ups.append(Up(prev) if i != len(chs) - 1 else None)
            self.dec.append(ResBlock(prev + c, c, t_dim)); prev = c

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, r, t):
        # combine the two time embeddings additively before the shared MLP
        emb = sinusoidal_embedding(t, self.t_dim) + sinusoidal_embedding(r, self.t_dim)
        temb = self.t_mlp(emb)
        h = self.in_conv(x)
        skips = []
        for blk, down in zip(self.enc, self.downs):
            h = blk(h, temb); skips.append(h)
            if down is not None: h = down(h)
        h = self.mid1(h, temb); h = self.mid2(h, temb)
        for blk, up in zip(self.dec, self.ups):
            if up is not None: h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = blk(h, temb)
        return self.out_conv(F.silu(self.out_norm(h)))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = MeanVelocityUNet()
    x = torch.randn(2, 1, 112, 112); r = torch.zeros(2); t = torch.rand(2)
    print("out", tuple(net(x, r, t).shape), "params", count_params(net))
