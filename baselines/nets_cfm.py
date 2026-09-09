#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time-conditioned U-Net velocity field for Conditional Flow Matching (CFM).

We learn v_theta(x_t, t) that transports the source modality image x0 to the
target modality image x1 along the rectified-flow / linear interpolant path

    x_t = (1 - t) * x0 + t * x1 ,   target velocity  u = x1 - x0 .

The net is a small U-Net operating on 1-channel 112x112 images. Time t in [0,1]
is turned into a sinusoidal embedding, projected by an MLP, and added to every
residual block (FiLM-style additive bias). Two separate nets are trained, one
per direction (T1->FA, FA->T1), which keeps the code and reasoning simple.

Param count for the default width (base=64) is ~16M, comfortably "tens of M".
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def sinusoidal_embedding(t, dim):
    """Standard transformer sinusoidal embedding for a (B,) tensor of times."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # zero-pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """GN -> SiLU -> conv, with additive time bias, plus residual skip."""

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
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class VelocityUNet(nn.Module):
    """Time-conditioned U-Net: (x_t [B,1,112,112], t [B]) -> v [B,1,112,112].

    112 -> 56 -> 28 -> 14 spatial, three resolution levels.
    """

    def __init__(self, in_ch=1, base=64, ch_mult=(1, 2, 4), t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        chs = [base * m for m in ch_mult]               # e.g. [64,128,256]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # encoder
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        self.enc_chs = []
        for i, c in enumerate(chs):
            self.enc.append(ResBlock(prev, c, t_dim))
            self.enc_chs.append(c)
            prev = c
            if i != len(chs) - 1:
                self.downs.append(Down(c))
            else:
                self.downs.append(None)

        # bottleneck
        self.mid1 = ResBlock(prev, prev, t_dim)
        self.mid2 = ResBlock(prev, prev, t_dim)

        # decoder (mirror); each level concatenates the matching skip
        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(chs))):
            c = chs[i]
            if i != len(chs) - 1:
                self.ups.append(Up(prev))
            else:
                self.ups.append(None)
            # input = upsampled prev + skip (c)
            self.dec.append(ResBlock(prev + c, c, t_dim))
            prev = c

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        temb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        h = self.in_conv(x)
        skips = []
        for i, (blk, down) in enumerate(zip(self.enc, self.downs)):
            h = blk(h, temb)
            skips.append(h)
            if down is not None:
                h = down(h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for blk, up in zip(self.dec, self.ups):
            if up is not None:
                h = up(h)
            skip = skips.pop()
            # guard against off-by-one spatial size from transpose conv
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = blk(h, temb)

        return self.out_conv(F.silu(self.out_norm(h)))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = VelocityUNet()
    x = torch.randn(2, 1, 112, 112)
    t = torch.rand(2)
    y = net(x, t)
    print("out", tuple(y.shape), "params", count_params(net))
