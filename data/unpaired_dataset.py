#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unpaired image-folder dataset in the CycleGAN layout.

    <root>/trainA  <root>/trainB  <root>/testA  <root>/testB

Returns `(a, b, 0)` with both images in [0, 1] and shape (C, H, W), so it is a
drop-in for `PairedADNISliceDataset` — `train.py` calls `to_pm1` on both and
never looks at the third element outside the ADNI split logic.

The one thing that is NOT drop-in is the meaning of the pair: `a` and `b` are
two unrelated images.  Every loss that reads them as ground truth for each other
(`L_cross`, and the `T1toFA` / `FAtoT1` SSIM in `evaluate`) is meaningless here
and must be switched off by the caller — see `--data folder` in train.py.

Sampling follows the reference CycleGAN: the epoch length is max(|A|, |B|), A is
walked in order, and B is drawn at random so the two domains are not locked into
a fixed pairing across epochs.  `train=False` makes B deterministic (index
modulo) so evaluation is reproducible.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def list_images(d):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} is not a directory")
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
    if not fs:
        raise FileNotFoundError(f"{d} contains no images ({'/'.join(EXTS)})")
    return [os.path.join(d, f) for f in fs]


class UnpairedFolderDataset(Dataset):
    """CycleGAN-layout folders, with the reference augmentation.

    train: resize to `load_size`, random crop to `crop_size`, random h-flip.
    eval : resize straight to `crop_size` (no crop, no flip) so the test set is
           one fixed set of images rather than a random view of it.
    """

    def __init__(self, root, split="train", load_size=286, crop_size=256,
                 train=True, img_ch=3, seed=42, limit=None, skip=0, flip=True,
                 pair=False):
        self.A = list_images(os.path.join(root, f"{split}A"))[skip:]
        self.B = list_images(os.path.join(root, f"{split}B"))[skip:]
        # `pair=True` turns this into a PAIRED dataset: A[i] and B[i] are the
        # same underlying sample, matched by filename, and the same geometric
        # augmentation is applied to both.  Only meaningful for folders built
        # that way (e.g. data/make_mnist_petct.py --paired_train); the
        # alignment is asserted rather than assumed, because a silent mismatch
        # would make L_cross compare two unrelated images and look like a
        # training failure rather than a data bug.
        self.pair = pair
        if pair:
            na = [os.path.basename(p) for p in self.A]
            nb = [os.path.basename(p) for p in self.B]
            if na != nb:
                raise SystemExit(
                    f"pair=True but {split}A and {split}B filenames do not align "
                    f"({len(na)} vs {len(nb)}); rebuild with --paired_train")
        if limit:
            self.A, self.B = self.A[:limit], self.B[:limit]
        self.load_size, self.crop_size = load_size, crop_size
        self.train, self.img_ch = train, img_ch
        # Horizontal flip is standard for photographs and WRONG for anything
        # whose identity is chiral: a mirrored 2 is not a 2, and feeding one to
        # the FID reference poisons the very distribution we score against.
        # The reference CycleGAN carries --no_flip for the same reason.
        self.flip = flip
        # Augmentation draws from the `random` MODULE, not a private Random():
        # DataLoader re-seeds Python's global RNG per worker AND per epoch, so a
        # private generator would be cloned identically into every worker and
        # replay the same crop sequence each epoch.  `seed` is kept only so the
        # signature documents that the stream is under torch's control.
        self.seed = seed

    def __len__(self):
        return len(self.A) if self.pair else max(len(self.A), len(self.B))

    def _aug(self):
        """One geometric draw, so a paired item can reuse it for both images.

        Cropping and flipping A and B independently would destroy exactly the
        correspondence L_cross depends on."""
        r = self.load_size - self.crop_size
        return (random.randint(0, r), random.randint(0, r),
                self.flip and random.random() < 0.5)

    def _load(self, path, aug=None):
        mode = "RGB" if self.img_ch == 3 else "L"
        im = Image.open(path).convert(mode)
        if self.train:
            im = im.resize((self.load_size, self.load_size), Image.BICUBIC)
            x0, y0, fl = aug if aug is not None else self._aug()
            im = im.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))
            if fl:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            im = im.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = a[:, :, None]
        return torch.from_numpy(a).permute(2, 0, 1).contiguous()

    def __getitem__(self, i):
        if self.pair:
            g = self._aug() if self.train else None
            return self._load(self.A[i], g), self._load(self.B[i], g), 0
        a = self._load(self.A[i % len(self.A)])
        j = random.randrange(len(self.B)) if self.train else i % len(self.B)
        b = self._load(self.B[j])
        return a, b, 0


def folder_loaders(root, batch, workers, load_size=286, crop_size=256,
                   img_ch=3, seed=42, val_frac=0.1, eval_batch=8, flip=True,
                   pair=False):
    """train / val / test loaders.

    CycleGAN datasets ship no validation split, so one is carved off the FRONT
    of each training folder and excluded from training.  It is loaded with
    train=False: early stopping must not move because a random crop moved.
    """
    from torch.utils.data import DataLoader
    n_val = max(1, int(round(val_frac * len(list_images(os.path.join(root, "trainA"))))))
    common = dict(load_size=load_size, crop_size=crop_size, img_ch=img_ch,
                  seed=seed, flip=flip, pair=pair)

    tr = UnpairedFolderDataset(root, "train", train=True,  skip=n_val, **common)
    va = UnpairedFolderDataset(root, "train", train=False, limit=n_val, **common)
    te = UnpairedFolderDataset(root, "test",  train=False, **common)
    mk = lambda ds, bs, sh, dl: DataLoader(ds, batch_size=bs, shuffle=sh, drop_last=dl,
                                           num_workers=workers, pin_memory=True)
    return (mk(tr, batch, True, True), mk(va, eval_batch, False, False),
            mk(te, eval_batch, False, False), len(tr), len(va), len(te))
