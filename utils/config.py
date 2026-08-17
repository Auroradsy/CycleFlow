#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YAML configs layered under argparse.

Precedence, lowest to highest:

    argparse defaults  <  --config file  <  flags typed on the command line

The last step is the one that needs care.  Naively applying the YAML with
`ap.set_defaults()` would let it clobber a flag the user actually typed, so we
ask the parser which flags were *explicitly given* (by re-parsing against a
sentinel) and re-apply only those on top.  That way

    python train.py --config configs/morph.yaml --w_latcyc 5

does what it looks like it does.

A config may name a `variant`; the variant presets in train.py are applied after
this merge, so anything the preset pins (w_latcyc = 0 in `base`, say) wins over
the file.  That is deliberate: a variant is a definition, not a default.
"""
import argparse

import yaml


def load(path):
    """Read a YAML config to a flat dict.  Comments and `_`-prefixed keys are
    ignored, so a file can carry notes without them reaching argparse."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def parse_with_config(ap, argv=None):
    """`ap.parse_args()`, but with `--config FILE` layered in at the right place.

    The parser must already define `--config`.  Unknown keys in the file are a
    hard error: a silently-ignored typo in a hyperparameter file is the kind of
    bug that only shows up as a run that mysteriously fails to reproduce.
    """
    a = ap.parse_args(argv)
    if not getattr(a, "config", None):
        return a

    cfg = load(a.config)
    known = {act.dest for act in ap._actions}
    unknown = set(cfg) - known
    if unknown:
        raise SystemExit(f"{a.config}: unknown key(s) {sorted(unknown)}\n"
                         f"  known: {sorted(known - {'help'})}")

    # Which flags were typed?  Re-parse with every default replaced by a
    # sentinel; whatever comes back as non-sentinel came from the command line.
    sentinel = object()
    probe = argparse.ArgumentParser(add_help=False, parents=[ap], conflict_handler="resolve")
    probe.set_defaults(**{d: sentinel for d in known})
    typed = {k: v for k, v in vars(probe.parse_known_args(argv)[0]).items()
             if v is not sentinel}

    merged = {**vars(a), **cfg, **typed}
    return argparse.Namespace(**merged)
