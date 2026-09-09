#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FID, for the unpaired runs where SSIM-against-ground-truth does not exist.

On ADNI every metric could be scored against the paired slice.  On an unpaired
dataset there is no target image, so quality has to be read distributionally.
Two things get scored:

  endpoint  FID(D_B(f(E_A(a))), real B)   — the usual horse->zebra number, and
            the one comparable to published CycleGAN results.
  path      FID(D_B(s_t), real A U real B) for the intermediate states of the
            walk.  This is the unpaired replacement for the hole metric: it asks
            whether a frame that is neither a horse nor a zebra is nonetheless a
            plausible image, without needing ground-truth mid-frames.

The Inception weights are the TF-ported ones from `pytorch-fid`, so the endpoint
numbers are on the same scale as the literature.  Real-side statistics are
cached — they depend only on the folder, and recomputing them every epoch would
cost more than the training step.
"""
import os

import numpy as np
import torch

_INCEPTION = None


def _inception(device, dims=2048):
    global _INCEPTION
    if _INCEPTION is None:
        from pytorch_fid.inception import InceptionV3
        idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        _INCEPTION = InceptionV3([idx], resize_input=True,
                                 normalize_input=True).to(device).eval()
    return _INCEPTION


@torch.no_grad()
def features(batches, device, dims=2048):
    """Pool3 features for an iterable of (B,C,H,W) tensors in [0, 1]."""
    net = _inception(device, dims)
    out = []
    for x in batches:
        x = x.to(device, non_blocking=True).float().clamp(0, 1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f = net(x)[0]
        out.append(f.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(out, 0)


def stats(feat):
    return feat.mean(0), np.cov(feat, rowvar=False)


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    """Frechet distance between two Gaussians (Dowson & Landau / Heusel et al.)."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        # Singular product: nudge the diagonals and retry, as pytorch-fid does.
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


@torch.no_grad()
def folder_stats(paths, device, cache, batch=32, crop=256, dims=2048):
    """Cached Inception statistics for a list of image files.

    The cache key is the file list itself, not just its length: a dataset that
    gained or lost images must not silently reuse the old reference.

    `crop` is part of the key too.  It was not, and that silently returned
    stats computed at one resolution to a caller asking for another: on a 64px
    dataset, reference stats written by a default-crop (256) call were reused
    by fid_against(crop=64), mixing 256-crop reals against 64-crop fakes.  The
    horse2zebra runs never tripped it because everything there was 256.
    """
    from PIL import Image
    key = str(hash(tuple(sorted(os.path.basename(p) for p in paths))))
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if (str(d.get("key", "")) == key and int(d["dims"]) == dims
                and int(d.get("crop", -1)) == crop):
            return d["mu"], d["sigma"]

    def gen():
        buf = []
        for p in paths:
            im = Image.open(p).convert("RGB").resize((crop, crop), Image.BICUBIC)
            buf.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
                       .permute(2, 0, 1))
            if len(buf) == batch:
                yield torch.stack(buf); buf = []
        if buf:
            yield torch.stack(buf)

    mu, sigma = stats(features(gen(), device, dims))
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez(cache, mu=mu, sigma=sigma, key=key, dims=dims, crop=crop)
    return mu, sigma


@torch.no_grad()
def translated_stats(paths, fn, device, batch=16, crop=256, img_ch=3, dims=2048):
    """Inception statistics of `fn` applied to every image in `paths`.

    `fn` takes and returns a tensor in [-1, 1]; images are fed once each, in
    order, so the sample is the whole split rather than a random view of it.
    """
    from PIL import Image
    mode = "RGB" if img_ch == 3 else "L"

    def gen():
        buf = []
        for p in paths:
            im = Image.open(p).convert(mode).resize((crop, crop), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0
            if a.ndim == 2:
                a = a[:, :, None]
            buf.append(torch.from_numpy(a).permute(2, 0, 1))
            if len(buf) == batch:
                yield torch.stack(buf); buf = []
        if buf:
            yield torch.stack(buf)

    out = []
    for x in gen():
        y = fn(x.to(device) * 2.0 - 1.0)
        out.append(((y + 1.0) * 0.5).clamp(0, 1).cpu())
    return stats(features(out, device, dims))


def fid_against(paths, fn, ref_paths, device, cache=None, **kw):
    """FID( fn(paths), ref_paths ).  The reference side is cached on disk."""
    mu_r, s_r = folder_stats(ref_paths, device, cache,
                             crop=kw.get("crop", 256), dims=kw.get("dims", 2048))
    mu_f, s_f = translated_stats(paths, fn, device, **kw)
    return frechet(mu_f, s_f, mu_r, s_r)
