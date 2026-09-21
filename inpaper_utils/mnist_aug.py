#!/usr/bin/env python3
"""E1c: are the decoded flow states useful training data, and is the *path* what makes them so?

  python -m inpaper_utils.mnist_aug --labels 10,25,50,100 --seeds 3

Task: digit classification on PET. Labelled PET is scarce; labelled MRI is not, and its
labels travel with a translation. The 12,000 training pairs are split once into two
disjoint halves, so the MRI corpus that is translated never contains the paired partner of
a labelled PET image -- a synthetic target generated from an image already in the training
set would add no information about its label, and that objection is designed out here.

Arms, all with the same labelled PET seed set:

  pet_only    n labelled PET per class, nothing else
  +translate  plus D_B(f(E_A(ct))) for the whole MRI corpus            (endpoint only, 1x)
  +path       plus the decoded states D_B(z^(l)), l = 2,3,4           (the flow's path, 3x)
  +interp     plus D_B((1-t) z^(0) + t f(z^(0))) at the same t        (same endpoints and
                                                                       the same 3x, straight
                                                                       line instead; t=1 is
                                                                       the endpoint itself)
  +mixup      the endpoint plus two pixel-space mixups of it          (a generic augmenter, 3x)
  oracle      plus the *real* PET of the MRI corpus                    (upper bound)

+path against +interp is the comparison the paper needs: identical endpoints, identical
number of extra images, and the only difference is whether the intermediate states come
from the bijection or from a straight line through the latent space.

Result JSON -> EXPS/mnist/aug/<run>/aug.json.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
DATA = Path('/ix/lzhan/siyuan/datasets/processed_datas')
# 'pet_only' is historical: it is the labelled *target* modality alone, whichever that is.
ARMS = ('pet_only', '+translate', '+path', '+interp', '+mixup', 'oracle')


def load_train(root):
    """trainA/trainB of the paired build, with the digit from the filename."""
    import torch
    from PIL import Image
    root = Path(root or os.environ.get('MNIST_PAIRED_ROOT') or DATA / 'MNIST_CycleFlow/mnist_petct_paired')
    names = sorted(p.name for p in (root / 'trainA').glob('*.png'))
    assert names == sorted(p.name for p in (root / 'trainB').glob('*.png')), 'trainA/trainB not aligned'
    read = lambda side: torch.from_numpy(np.stack(
        [np.asarray(Image.open(root / side / n).convert('RGB'), dtype=np.uint8) for n in names])).permute(0, 3, 1, 2)
    y = np.array([int(n.split('_d')[1][0]) for n in names])
    return read('trainA').contiguous(), read('trainB').contiguous(), y


def translate(m, src, dev, bs, ts, inverse=False):
    """Endpoint, intermediate and straight-line decodings of the source corpus, as uint8.

    `inverse` runs the same bijection backwards, PET -> MRI, which is the identical set of
    weights traversed in the other direction rather than a second model.
    """
    import torch
    enc, dec_to = (m.enc_B, m.dec_A) if inverse else (m.enc_A, m.dec_B)
    end, path, interp = [], [], []
    for i in range(0, len(src), bs):
        x = src[i:i + bs].to(dev).float() / 255
        z0 = enc(x * 2 - 1)
        st = m.walk(z0, inverse=inverse)
        L = len(st) - 1
        dec = lambda z: (((dec_to(z) + 1) / 2).clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        end.append(dec(st[-1]))
        # The blocks the intermediate states come from, and the matching points on the
        # straight line from z^(0) to f(z^(0)) -- same endpoints, same count.
        path.append(torch.stack([dec(st[k]) for k in ts], 1))
        interp.append(torch.stack([dec((1 - k / L) * z0 + (k / L) * st[-1]) for k in ts], 1))
    return torch.cat(end), torch.cat(path), torch.cat(interp)


def pick_labelled(y, n, seed, pool):
    """n indices per class, drawn from the labelled-PET half."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(10):
        idx = pool[y[pool] == c]
        out.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.concatenate(out)


def build_arm(arm, lab_idx, data, seed, c):
    """(x uint8, y) for one arm, using the first `c` images of the MRI corpus."""
    import torch
    y, cidx = data['y'], data['corpus_idx'][:c]
    end = data['end'][:c]
    xs, ys = [data['tgt'][lab_idx]], [y[lab_idx]]
    yc = y[cidx]
    if arm == '+translate':
        xs.append(end); ys.append(yc)
    elif arm == 'oracle':
        xs.append(data['tgt'][cidx]); ys.append(yc)
    elif arm in ('+path', '+interp'):
        t = data['path' if arm == '+path' else 'interp'][:c]       # (c, |ts|, 3, H, W)
        xs.append(t.flatten(0, 1)); ys.append(np.repeat(yc, t.shape[1]))
    elif arm == '+mixup':
        g = torch.Generator().manual_seed(seed)
        xs.append(end); ys.append(yc)                    # same endpoint as the other arms
        for _ in range(data['path'].shape[1] - 1):
            p = torch.randperm(len(end), generator=g)
            lam = 0.5 + 0.5 * torch.rand(len(p), 1, 1, 1, generator=g)   # keep the label of `end`
            xs.append((end.float() * lam + end[p].float() * (1 - lam)).to(torch.uint8))
            ys.append(yc)
    return torch.cat(xs), np.concatenate(ys)


def fit_eval(x, y, xte, yte, dev, steps, seed, batch=128):
    """A fixed number of gradient steps, not epochs.

    The arms differ in training-set size by two orders of magnitude (1,000 labelled PET
    images against 19,000 with augmentation). Counting epochs would hand the augmented
    arms 18x the optimisation budget and the comparison would measure that, not the data;
    batches are therefore drawn with replacement and every arm gets the same `steps`."""
    import torch
    import torch.nn.functional as F
    from inpaper_utils.adni_clf_gate import small_cnn
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = small_cnn(3, 10).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    yt = torch.from_numpy(y).long()
    net.train()
    for _ in range(steps):
        i = torch.randint(len(x), (min(batch, len(x)),), generator=g)
        xb = x[i].to(dev).float() / 255
        # Digits are not flipped (a mirrored digit is in neither domain); shift only.
        dx, dy = (int(torch.randint(-3, 4, (), generator=g)) for _ in range(2))
        xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
        loss = F.cross_entropy(net(xb), yt[i].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    ok = 0
    with torch.no_grad():
        for i in range(0, len(xte), 512):
            ok += (net(xte[i:i + 512].to(dev).float() / 255).argmax(1).cpu().numpy() == yte[i:i + 512]).sum()
    return float(ok / len(yte))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='morph_smooth')
    ap.add_argument('--labels', default='10,25,50,100', help='labelled PET per class')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=1500,
                    help='gradient steps per run, identical for every arm')
    ap.add_argument('--states', default='2,3,4', help='flow blocks used as extra samples; '
                                                       'the last one is the endpoint f(z)')
    ap.add_argument('--corpus', type=int, default=6000, help='MRI images translated')
    ap.add_argument('--corpus-sweep', help='comma-separated MRI corpus sizes to sweep instead; '
                                           'the labelled-PET count is then fixed at the first --labels value. '
                                           'With a corpus large enough to saturate the task every arm scores '
                                           'the same and the path/interp comparison cannot separate.')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--root')
    ap.add_argument('--direction', default='a2b', choices=['a2b', 'b2a'],
                    help='a2b: scarce labelled PET, translate a labelled MRI corpus into PET. '
                         'b2a: the mirror -- scarce labelled MRI, translate labelled PET into MRI. '
                         'The same bijection serves both, traversed in opposite senses.')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    from inpaper_utils.make_flow_viz import load_model
    from inpaper_utils.make_paper_compare import load_test

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m, ck = load_model('mnist', a.variant)
    m = m.to(dev).eval()

    ct, pet, y = load_train(a.root)
    inverse = a.direction == 'b2a'
    # The scarce, labelled modality is the one being classified; the abundant one is
    # translated into it. b2a just swaps which is which.
    src_all, tgt_all = (pet, ct) if inverse else (ct, pet)
    sname, tname = ('PET', 'MRI') if inverse else ('MRI', 'PET')
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    half = len(y) // 2
    tgt_pool, corpus_idx = perm[:half], perm[half:half + a.corpus]  # disjoint by construction
    ts = [int(s) for s in a.states.split(',')]
    print(f'{len(y)} training pairs, {sname}->{tname} -> {len(tgt_pool)} labelled-{tname} pool, '
          f'{len(corpus_idx)} {sname} corpus, states {ts}, on {dev}', flush=True)
    with torch.no_grad():
        end, path, interp = translate(m, src_all[corpus_idx], dev, a.batch, ts, inverse)
    (test_ct, test_pet), meta = load_test('mnist')
    test_tgt = test_ct if inverse else test_pet
    xte = (test_tgt * 255).round().to(torch.uint8)                 # test images of the target
    yte = np.array([mm['digit'] for mm in meta])
    data = {'tgt': tgt_all, 'y': y, 'corpus_idx': corpus_idx, 'end': end, 'path': path,
            'interp': interp}

    out = {'variant': a.variant, 'checkpoint': str(ck), 'states': ts, 'corpus': int(len(corpus_idx)),
           'direction': a.direction, 'source': sname, 'target': tname,
           'steps': a.steps, 'seeds': a.seeds, 'n_test': int(len(yte)), 'device': str(dev),
           'results': {}}
    labels = [int(v) for v in a.labels.split(',')]
    if a.corpus_sweep:
        cells = [(labels[0], int(c)) for c in a.corpus_sweep.split(',')]
    else:
        cells = [(n, len(corpus_idx)) for n in labels]
    out['cells'] = [{'labels_per_class': n, 'corpus': c} for n, c in cells]
    for n, c in cells:
        key = f'n{n}_c{c}'
        out['results'][key] = {}
        for arm in ARMS:
            accs, size = [], 0
            for s in range(a.seeds):
                lab_idx = pick_labelled(y, n, seed=1000 + s, pool=tgt_pool)
                x, yy = build_arm(arm, lab_idx, data, seed=s, c=c)
                size = len(x)
                accs.append(fit_eval(x, yy, xte, yte, dev, a.steps, seed=s))
            out['results'][key][arm] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
                                        'per_seed': accs, 'train_size': int(size),
                                        'labels_per_class': n, 'corpus': c}
            print(f'  [{sname}->{tname}] n={n:4d} corpus={c:5d} {arm:11s} '
                  f'{np.mean(accs):.4f} +- {np.std(accs):.4f}  '
                  f'({size} images)', flush=True)
    d = EXPS / 'mnist' / 'aug' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                                  + '-' + uuid.uuid4().hex[:8])
    d.mkdir(parents=True, exist_ok=True)
    (d / 'aug.json').write_text(json.dumps(out, indent=1))
    print('wrote', d / 'aug.json')


if __name__ == '__main__':
    main()
