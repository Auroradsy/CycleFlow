#!/usr/bin/env python3
"""Inference time of every Table 2 method on one GPU, same inputs, same batch.

  python -m inpaper_utils.bench_infer --n 256 --batch 64      # GPU

Uses the generators of make_paper_compare, i.e. exactly the samplers behind the
comparison figures: CycleGAN one generator pass (native reverse model for B->A),
CFM 10 Euler steps, MeanFlow one step, DDPM 50 DDIM steps, DiT 1000 ancestral steps
without guidance plus the latent autoencoder, MMCLAST-cg encoder -> flow -> decoder.
Reported: milliseconds per image for each direction (first n held-out pairs, after
a warm-up batch), and their mean. JSON -> EXPS/bench_cost/infer.json.

`--only CFM,RectFlow` times just those methods and MERGES them into the existing
file, which is how a row is added without re-timing -- and so possibly moving --
the rows already in the paper. A method whose checkpoint is missing is skipped
with a note, as in make_paper_compare.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
OURS = {'adni': [('MMCLAST-cg, morph', 'adni_morph'), ('MMCLAST-cg, morph-smooth', 'adni_morph_smooth')],
        'mnist': [('MMCLAST-cg, morph', 'mnist_p_morph'), ('MMCLAST-cg, morph-smooth', 'mnist_p_morph_smooth')]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--only', help='comma-separated method names; merge into the existing JSON')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    sys.path.insert(0, str(ROOT))
    import torch
    from inpaper_utils.make_paper_compare import (load_test, baseline_generators, ours_generators,
                                                  resolve_tag, BASELINES)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    dest = EXPS / 'bench_cost'
    out = {'device': torch.cuda.get_device_name(0), 'n_images': a.n, 'batch': a.batch}
    if only and (dest / 'infer.json').exists():
        out = json.loads((dest / 'infer.json').read_text())
        out['device_of_added'] = torch.cuda.get_device_name(0)
    for ds in ('adni', 'mnist'):
        pair, _ = load_test(ds)
        x = [pair[s][:a.n].contiguous() for s in (0, 1)]
        img_ch, size = x[0].shape[1], x[0].shape[-1]
        out.setdefault(ds, {})
        for name, tag in [(n, t) for n, t in BASELINES[ds]] + OURS[ds]:
            if only and name not in only:
                continue
            tag = resolve_tag(tag, ds)
            if not (EXPS / tag).is_dir():
                print(f'SKIP {ds} {name}: no checkpoint directory {EXPS / tag}', flush=True)
                continue
            gens = ours_generators(tag, dev, {}) if name.startswith('MMCLAST') else \
                baseline_generators(name, tag, dev, img_ch, size, {}, 1.0, ds)
            ms = []
            for side, gen in enumerate(gens):
                with torch.inference_mode():
                    # Warm up at the batch size that is about to be TIMED. cudnn.benchmark
                    # keys its algorithm search on the shape, so a warm-up at another batch
                    # size leaves the search to be paid by the first timed direction -- which
                    # is invisible for a 250 ms sampler and a factor of four for a 0.2 ms one.
                    for _ in range(2):
                        gen(x[side][:a.batch].to(dev))
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for i in range(0, a.n, a.batch):
                        gen(x[side][i:i + a.batch].to(dev))
                    torch.cuda.synchronize()
                ms.append((time.perf_counter() - t0) / a.n * 1e3)
            out[ds][name] = {'ms_per_image_A2B': round(ms[0], 3), 'ms_per_image_B2A': round(ms[1], 3),
                             'ms_per_image_mean': round(sum(ms) / 2, 3)}
            print(ds, name, out[ds][name], flush=True)
            del gens; torch.cuda.empty_cache()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'infer.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1), '\n->', dest / 'infer.json', flush=True)


if __name__ == '__main__':
    main()
