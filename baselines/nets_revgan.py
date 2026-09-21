#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RevGAN: a CycleGAN whose two generators are one reversible core.

van der Ouderaa & Worrall, "Reversible GANs for Memory-efficient Image-to-Image
Translation" (CVPR 2019).  The ResNet generator's residual stack is replaced by
additive-coupling (RevNet) blocks, and that stack is SHARED by the directions:

    G_{A->B} = Dec_B  o  R      o  Enc_A
    G_{B->A} = Dec_A  o  R^{-1} o  Enc_B

so the two translations are one set of core weights traversed in opposite
senses.  The core is exactly invertible; the composition is not, because the
four encoders and decoders are ordinary lossy convolutions trained
independently.  That gap -- reversibility in the core, approximate at the image
level -- is what this baseline exists to expose, and is why RevGAN still needs
the cycle penalty that a construction with a load-bearing shared bottleneck
does not.

A RevNet block splits the bottleneck in half along the channels and does

    y1 = x1 + F(x2)                 x2 = y2 - G(y1)
    y2 = x2 + G(y1)                 x1 = y1 - F(x2)

with F and G the two-convolution body of CycleGAN's residual block acting on
half the channels.  `core_hidden = ngf * 4` makes one reversible block cost the
same parameters as the residual block it replaces (1.18 M at ngf=64), so the
comparison is not confounded by capacity; the model as a whole is still about
half of CycleGAN's two generators, because the core is shared rather than
duplicated -- which is the paper's point.

The paper's memory-saving backward pass (recomputing each block's input from
its output rather than storing activations) is deliberately NOT implemented
here.  It changes peak memory, not the model or any number it reports, and at
112^2 / 64^2 memory is not what binds.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reversible core
# ---------------------------------------------------------------------------
class CouplingBody(nn.Module):
    """conv-IN-ReLU-conv-IN, i.e. model.backbone.ResidualBlock without its skip.

    The skip is what the coupling adds back (y1 = x1 + F(x2)), so the body
    itself is the residual branch alone.
    """

    def __init__(self, ch, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, hidden, kernel_size=3, bias=True),
            nn.InstanceNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden, ch, kernel_size=3, bias=True),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return self.net(x)


class RevBlock(nn.Module):
    """Additive coupling on a channel split; exactly invertible for any F, G."""

    def __init__(self, ch, hidden):
        super().__init__()
        if ch % 2:
            raise ValueError(f"reversible core needs an even channel count, got {ch}")
        self.half = ch // 2
        self.F = CouplingBody(self.half, hidden)
        self.G = CouplingBody(self.half, hidden)

    def forward(self, x):
        x1, x2 = x[:, :self.half], x[:, self.half:]
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = y[:, :self.half], y[:, self.half:]
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=1)


class ReversibleCore(nn.Module):
    """n_blocks additive couplings, alternating which half is transformed.

    The alternation is the channel swap between blocks; without it the first
    half would never be written by F alone across the whole stack.  It is
    implemented by flipping the two halves, which is its own inverse.
    """

    def __init__(self, ch, n_blocks=6, hidden=256):
        super().__init__()
        self.half = ch // 2
        self.blocks = nn.ModuleList([RevBlock(ch, hidden) for _ in range(n_blocks)])

    def _swap(self, x):
        return torch.cat([x[:, self.half:], x[:, :self.half]], dim=1)

    def forward(self, x):
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i != len(self.blocks) - 1:
                x = self._swap(x)
        return x

    def inverse(self, y):
        for i, b in enumerate(reversed(self.blocks)):
            if i != 0:
                y = self._swap(y)
            y = b.inverse(y)
        return y


# ---------------------------------------------------------------------------
# Per-domain encoder / decoder (CycleGAN's, with the residual stack removed)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """c7s1-ngf -> d2ngf -> d4ngf, i.e. ResnetGenerator.head + .down."""

    def __init__(self, in_ch=1, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """u2ngf -> ungf -> c7s1-out with a tanh, i.e. ResnetGenerator.up + .tail."""

    def __init__(self, out_ch=1, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
class RevGAN(nn.Module):
    """Both translation directions in one module; inputs and outputs in [-1,1]."""

    def __init__(self, img_ch=1, ngf=64, n_blocks=6, core_hidden=None):
        super().__init__()
        core_ch = ngf * 4
        self.enc_a = Encoder(img_ch, ngf)
        self.enc_b = Encoder(img_ch, ngf)
        self.dec_a = Decoder(img_ch, ngf)
        self.dec_b = Decoder(img_ch, ngf)
        self.core = ReversibleCore(core_ch, n_blocks, core_hidden or core_ch)

    def a2b(self, x):
        return self.dec_b(self.core(self.enc_a(x)))

    def b2a(self, y):
        return self.dec_a(self.core.inverse(self.enc_b(y)))

    def forward(self, x, direction="a2b"):
        return self.a2b(x) if direction == "a2b" else self.b2a(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    for img_ch, size in ((1, 112), (3, 64)):
        net = RevGAN(img_ch=img_ch).eval()
        x = torch.randn(2, img_ch, size, size)
        with torch.no_grad():
            z = net.enc_a(x)
            err = (net.core.inverse(net.core(z)) - z).abs().max().item()
            out = net.a2b(x)
        print(f"img_ch={img_ch} size={size}  out {tuple(out.shape)}  "
              f"core round-trip {err:.2e}  params {count_params(net) / 1e6:.2f}M "
              f"(core {count_params(net.core) / 1e6:.2f}M)")
