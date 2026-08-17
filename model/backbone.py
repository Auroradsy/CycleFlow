#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CycleGAN models for T1<->FA cross-modal translation (112x112, 1-channel).

Standard CycleGAN (Zhu et al. 2017):
  - ResNet-based generator: c7s1-64, downsample x2, 6 residual blocks,
    upsample x2, c7s1-1, tanh output.
  - 70x70 PatchGAN discriminator.

The generator exposes its encoder bottleneck (feature map right before the
residual blocks) via forward(..., return_feat=True) so it can be used for the
representation/clustering evaluation.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """1->1 channel ResNet generator. tanh output in [-1, 1].

    Layout for ngf=64:
      c7s1-64 -> d128 -> d256 -> 6 x R256 -> u128 -> u64 -> c7s1-1
    The "bottleneck" feature (return_feat) is the 256-channel map AFTER the two
    downsampling layers and BEFORE the residual blocks.
    """

    def __init__(self, in_ch=1, out_ch=1, ngf=64, n_blocks=6):
        super().__init__()
        # c7s1-64 (initial)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        # downsampling x2
        self.down = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )
        # residual blocks
        self.res = nn.Sequential(*[ResidualBlock(ngf * 4) for _ in range(n_blocks)])
        # upsampling x2
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        # c7s1-1 (output)
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7, bias=True),
            nn.Tanh(),
        )

    def encode(self, x):
        """Return the bottleneck feature map (B, ngf*4, H/4, W/4)."""
        h = self.head(x)
        h = self.down(h)
        return h

    def forward(self, x, return_feat=False):
        feat = self.encode(x)          # bottleneck before residual blocks
        h = self.res(feat)
        h = self.up(h)
        out = self.tail(h)
        if return_feat:
            return out, feat
        return out


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (LSGAN: no sigmoid, MSE loss applied outside)."""

    def __init__(self, in_ch=1, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=4,
                          stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=4,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Weight init (standard CycleGAN: normal, 0.02 std)
# ---------------------------------------------------------------------------
def init_weights(net, gain=0.02):
    def fn(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm" in cls and getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(fn)
    return net
