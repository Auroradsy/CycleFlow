#!/usr/bin/env python3
"""What sampling budget did the DDPM runs actually use?

The migrated ADNI DDPM stores args['n_steps'] = None, so make_paper_compare has
to supply a fallback. MNIST's DDPM reproduces its published score with the
current code, so whatever ITS args say is the setting that matches the numbers
in the paper table. Print both.
"""
import torch
from pathlib import Path

E = Path('/ix/lzhan/siyuan/exps/CycleFlow')
for label, p in [('mnist ddpm (reproduces the table)', E / 'ddpm_mnist/last.pth'),
                 ('adni  ddpm (migrated)', E / 'ddpm_adni/last.pth'),
                 ('mnist cfm  (control)', E / 'cfm_mnist/last.pth'),
                 ('adni  cfm  (migrated, reproduced)', E / 'cfm_adni/last.pth')]:
    print('=' * 70)
    print(f'{label}\n  {p}')
    if not p.exists():
        print('  MISSING')
        continue
    ck = torch.load(p, map_location='cpu', weights_only=False)
    ar = ck.get('args', {})
    if not isinstance(ar, dict):
        ar = vars(ar)
    print('  epoch   :', ck.get('epoch'))
    print('  n_steps :', repr(ar.get('n_steps')))
    print('  all args:', {k: v for k, v in sorted(ar.items()) if not k.startswith('_')})
