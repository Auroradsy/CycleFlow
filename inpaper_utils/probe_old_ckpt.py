#!/usr/bin/env python3
"""Inspect the packed old-workstation baseline checkpoints.

Their SOURCE.txt says the ADNI cfm/meanflow/ddpm weights load with
__outdated_files/baseline/<method>/nets.py, which is NOT in this repo. This
prints each checkpoint's key structure so we can tell whether the current
baselines/train.py build() can take them, and whether the CycleGAN generator
keys match what make_paper_compare expects (G_T1toFA / G_FAtoT1).
"""
import os
import torch

B = '/ix/lzhan/siyuan/exps/CycleFlow/_baselines_unpacked/baselines'
for name in ['adni_ddpm/best.pth', 'adni_cfm/best.pth',
             'adni_meanflow/best.pth', 'adni_cyclegan/last.pth']:
    p = os.path.join(B, name)
    print('=' * 70)
    print(p)
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
    except Exception as e:
        print('  LOAD FAILED:', type(e).__name__, e)
        continue
    if not isinstance(ck, dict):
        print('  type:', type(ck))
        continue
    print('  top-level keys:', list(ck.keys())[:12])
    for k, v in ck.items():
        if isinstance(v, dict) and v and all(hasattr(x, 'shape') for x in list(v.values())[:3]):
            names = list(v.keys())
            print(f'  [{k}] {len(names)} tensors; first 6: {names[:6]}')
        elif not isinstance(v, dict):
            print(f'  [{k}] = {repr(v)[:300]}')
