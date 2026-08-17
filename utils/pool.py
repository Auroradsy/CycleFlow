#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CycleGAN's history buffer of generated images."""
import random

import torch


class ImagePool:
    """Show the discriminator a mix of current and past fakes (default 50).

    Straight from the CycleGAN reference implementation.  Without it the
    discriminator only ever sees the generator's latest output and the pair
    oscillates instead of converging.
    """

    def __init__(self, pool_size=50):
        self.pool_size, self.images = pool_size, []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img); out.append(img)
            elif random.random() > 0.5:
                i = random.randint(0, self.pool_size - 1)
                out.append(self.images[i].clone()); self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)
