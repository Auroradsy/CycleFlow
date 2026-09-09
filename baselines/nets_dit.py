#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful DiT (Peebles & Xie 2023), adapted to latent cross-modal translation
with the *minimal* change: condition on the source latent by channel-concat at the
patch-embed input (the standard latent-diffusion image-conditioning), keeping every
DiT-defining choice intact:
  * operates on pretrained-VAE latents (MedVAE, 28x28x1)
  * patchify -> tokens, frozen 2D sin-cos positional embedding
  * adaLN-zero transformer blocks (timestep conditioning), zero-init gates
  * learn_sigma: predicts eps AND the variance-interp v (out_ch = 2 * latent_ch)
  * DiT-S config (depth 12, hidden 384, 6 heads, patch 2) by default

For classifier-free guidance the "null" condition is an all-zero source latent.
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ---- frozen 2D sin-cos positional embedding (DiT/MAE style) ----
def get_2d_sincos_pos_embed(dim, grid):
    g = np.arange(grid, dtype=np.float32)
    gx, gy = np.meshgrid(g, g)
    gx = gx.reshape(-1); gy = gy.reshape(-1)
    assert dim % 2 == 0
    emb_h = _1d_sincos(dim // 2, gy)
    emb_w = _1d_sincos(dim // 2, gx)
    return np.concatenate([emb_h, emb_w], axis=1)   # [grid*grid, dim]


def _1d_sincos(dim, pos):
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos[:, None] * omega[None]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sinusoidal_timestep(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq=256):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        return self.mlp(sinusoidal_timestep(t, self.freq))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        h = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, h), nn.GELU(approximate="tanh"),
                                 nn.Linear(h, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), sh1, sc1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None] * a
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), sh2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(hidden, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, c):
        sh, sc = self.ada(c).chunk(2, dim=1)
        return self.lin(modulate(self.norm(x), sh, sc))


class DiT(nn.Module):
    """DiT for latent translation. forward(x_t, t, src) where x_t and src are
    latents [B, C, H, W]; returns [B, 2C, H, W] (eps, v) when learn_sigma."""
    def __init__(self, latent_size=28, latent_ch=1, patch=2,
                 hidden=384, depth=12, heads=6, learn_sigma=True):
        super().__init__()
        self.latent_ch = latent_ch
        self.out_ch = latent_ch * (2 if learn_sigma else 1)
        self.patch = patch
        self.grid = latent_size // patch
        in_ch = latent_ch * 2                       # noised target + source (concat)
        self.x_embed = nn.Conv2d(in_ch, hidden, patch, patch)
        pos = get_2d_sincos_pos_embed(hidden, self.grid)
        self.register_buffer("pos", torch.from_numpy(pos).float()[None])
        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(depth)])
        self.final = FinalLayer(hidden, patch, self.out_ch)

    def unpatchify(self, x):
        B = x.shape[0]; p = self.patch; g = self.grid; c = self.out_ch
        x = x.reshape(B, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, c, g * p, g * p)

    def forward(self, x_t, t, src):
        h = self.x_embed(torch.cat([x_t, src], dim=1)).flatten(2).transpose(1, 2) + self.pos
        c = self.t_embed(t)
        for blk in self.blocks:
            h = blk(h, c)
        return self.unpatchify(self.final(h, c))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
