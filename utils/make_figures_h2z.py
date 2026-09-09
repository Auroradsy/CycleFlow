#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backwards-compatible entry point — the figures now live in one general
script shared with the MNIST-PET/CT runs.  Equivalent to:

    python -m utils.make_figures_folder --dataset h2z
"""
import sys

from utils.make_figures_folder import main

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv += ["--dataset", "h2z"]
    main()
