#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-space conventions, in one place.

The dataset stores slices in [0, 1]; the networks use tanh outputs in [-1, 1].
Every script needs the same two conversions and the same SSIM call, so they live
here rather than being re-declared five times with a chance of drifting apart.
"""
import numpy as np
import torch


def to_pm1(x):
    """[0, 1] -> [-1, 1]  (dataset -> network)."""
    return x * 2.0 - 1.0


def to_01(x):
    """[-1, 1] -> [0, 1], clamped  (network -> dataset / metric)."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def first_frame(t):
    """A (B,1,H,W) network output -> the first image as a [0,1] numpy array."""
    return to_01(t)[0, 0].detach().cpu().numpy()


def ssim(a, b):
    """SSIM between two [0,1] HxW arrays.  Always pass data_range explicitly:
    skimage infers it from the data otherwise, which silently changes the metric
    when an image happens not to span the full range."""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=1.0))


def ssim_batch(pred, gt):
    """Mean SSIM over a batch of (B,1,H,W) tensors already in [0,1]."""
    p = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
    g = gt.detach().cpu().numpy() if torch.is_tensor(gt) else gt
    return float(np.mean([ssim(g[i, 0], p[i, 0]) for i in range(p.shape[0])]))
