#!/usr/bin/env python3
"""Can the CURRENT baselines/ nets load the OLD packed ADNI weights?

make_paper_compare's baseline branch does
    ma, mb, diff = build(method, img_ch, base, dev)
    m.load_state_dict(ck['net_ab'], strict=True)
so both the dict key and every parameter name must match. The packed runs use
net_T1toFA/net_FAtoT1 (cfm, meanflow) and model_T1toFA/model_FAtoT1 (ddpm).
This reports, per method, whether a strict load actually succeeds.
"""
import torch
from baselines.train import build

B = '/ix/lzhan/siyuan/exps/CycleFlow/_baselines_unpacked/baselines'
CASES = [('cfm', 'adni_cfm/best.pth', 'net_T1toFA'),
         ('meanflow', 'adni_meanflow/best.pth', 'net_T1toFA'),
         ('ddpm', 'adni_ddpm/best.pth', 'model_T1toFA')]

for method, rel, key in CASES:
    print('=' * 70)
    print(f'{method}  <-  {rel}  [{key}]')
    ck = torch.load(f'{B}/{rel}', map_location='cpu', weights_only=False)
    ar = ck.get('args', {})
    if not isinstance(ar, dict):
        ar = vars(ar)
    print('  packed args:', {k: ar.get(k) for k in ('base', 'img_ch', 'n_steps', 'epochs')})
    ma, _, _ = build(method, ar.get('img_ch', 1), ar.get('base', 64), torch.device('cpu'))
    own, sd = ma.state_dict(), ck[key]
    print(f'  current net : {len(own)} tensors; first 6: {list(own)[:6]}')
    print(f'  packed  net : {len(sd)} tensors; first 6: {list(sd)[:6]}')
    try:
        ma.load_state_dict(sd, strict=True)
        print('  STRICT LOAD: OK  -> this baseline can be re-run with current code')
    except Exception as e:
        print('  STRICT LOAD: FAILED ->', str(e)[:500])
