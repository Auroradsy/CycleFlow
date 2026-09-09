#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A small latent autoencoder, standing in for MedVAE on non-medical data.

The ADNI latent-DiT runs in MedVAE's frozen latent space (112x112x1 -> 28x28,
a 4x spatial compression).  MedVAE is a pretrained *medical* single-channel
model and does not transfer to 3-channel 64px digits, so the MNIST-PET/CT DiT
needs an equivalent of its own: same role, same 4x compression, frozen once
trained, and used through its posterior MEAN exactly as the ADNI code uses
`post.mode()`.

Trained on both domains at once so the two modalities share one latent space,
which is what the DiT then transports between.
"""
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _blk(i, o, down):
    return nn.Sequential(
        nn.Conv2d(i, o, 4, 2, 1) if down else nn.Conv2d(i, o, 3, 1, 1),
        nn.GroupNorm(8, o), nn.SiLU(),
        nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())


class LatentAE(nn.Module):
    """img_ch x S x S  <->  latent_ch x (S/4) x (S/4).  KL is tiny: the latent is
    used deterministically (the mean), so this is a regularised AE, not a
    generative VAE -- which is also how MedVAE is used downstream."""

    def __init__(self, img_ch=3, base=64, latent_ch=4):
        super().__init__()
        self.latent_ch = latent_ch
        self.enc = nn.Sequential(_blk(img_ch, base, True),
                                 _blk(base, base * 2, True))
        self.to_moments = nn.Conv2d(base * 2, 2 * latent_ch, 1)
        self.from_z = nn.Conv2d(latent_ch, base * 2, 1)
        self.dec = nn.Sequential(
            _blk(base * 2, base * 2, False), nn.Upsample(scale_factor=2, mode="nearest"),
            _blk(base * 2, base, False), nn.Upsample(scale_factor=2, mode="nearest"),
            _blk(base, base, False), nn.Conv2d(base, img_ch, 3, 1, 1), nn.Sigmoid())
        # per-modality normalisation, filled by fit_stats (as MedVAE does)
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("std", torch.ones(2))

    def moments(self, x):
        mu, logvar = self.to_moments(self.enc(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def decode(self, z):
        return self.dec(self.from_z(z))

    def forward(self, x):
        mu, logvar = self.moments(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar

    # ---- the MedVAELatent interface the DiT code expects -------------------
    @torch.no_grad()
    def _encode(self, x):
        return self.moments(x)[0]                      # the mean == post.mode()

    @torch.no_grad()
    def fit_stats(self, loader, n_batches=12, device="cuda"):
        za, zb = [], []
        for bi, (a, b, _) in enumerate(loader):
            if bi >= n_batches:
                break
            za.append(self._encode(a.to(device))); zb.append(self._encode(b.to(device)))
        za, zb = torch.cat(za), torch.cat(zb)
        self.mean = torch.tensor([za.mean(), zb.mean()], device=device)
        self.std = torch.tensor([za.std(), zb.std()], device=device)
        print(f"[latent-ae] A mean={self.mean[0]:.3f} std={self.std[0]:.3f} | "
              f"B mean={self.mean[1]:.3f} std={self.std[1]:.3f}", flush=True)

    @torch.no_grad()
    def encode_norm(self, x, m):
        return (self._encode(x) - self.mean[m]) / self.std[m]

    @torch.no_grad()
    def decode_norm(self, z, m):
        return self.decode(z * self.std[m] + self.mean[m]).clamp(0, 1)
