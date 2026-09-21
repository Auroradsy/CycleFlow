#!/usr/bin/env python3
"""Paper-ready baseline comparison (cross-modal generation), one figure per dataset.

Two stages, so layout can be iterated without re-running the samplers:

  python -m inpaper_utils.make_paper_compare infer  --dataset adni    # GPU: full test set
  python -m inpaper_utils.make_paper_compare render --dataset adni    # CPU (htc): select + draw

`infer` translates every held-out pair with every available method, scores each
image with SSIM against its ground truth, and caches uint8 predictions under
EXPS/<dataset>/paper_compare/<run>/cache. `render` ranks test pairs by
margin = min(SSIM of our variants) - max(SSIM of the baselines), keeps the top
ones per direction with distinct subjects (ADNI) or digits (MNIST), and draws an
ICLR-style grid: method names on top, direction on the left, nothing else.
`--pick` overrides the automatic choice with explicit test indices.

DEC-FA preview (ADNI): V1 exists for a few TRAINING subjects only, so
  infer  --dataset adni --samples v1            # those slices, cache under paper_compare_v1/
  render --dataset adni --samples v1 --dec-fa   # z=44 of each, no ranking
writes 00_inpaper_fig_compare_decfa_preview to snapshot_results/adni/decfa_preview/.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
DATA = Path('/ix/lzhan/siyuan/datasets/processed_datas')

# Display order is fixed across datasets. A baseline whose checkpoint directory
# is missing is skipped with a note, so adding weights later needs no code change.
BASELINES = {
    'adni': [('CycleGAN', 'adni_host'), ('RevGAN', 'revgan_adni'), ('DDPM', 'ddpm_adni'),
             ('MeanFlow', 'meanflow_adni'), ('CFM', 'cfm_adni'), ('RectFlow', 'recflow_adni'),
             ('DiT', 'dit_adni_scratch')],
    'mnist': [('CycleGAN', 'mnist_host'), ('RevGAN', 'revgan_mnist'), ('DDPM', 'ddpm_mnist'),
              ('MeanFlow', 'meanflow_mnist'), ('CFM', 'cfm_mnist'), ('RectFlow', 'recflow_mnist'),
              ('DiT', 'dit_mnist')],
}
# The row label in the paper is not the method keyword baselines/train.py knows it by.
METHOD = {'CFM': 'cfm', 'RectFlow': 'recflow', 'MeanFlow': 'meanflow', 'DDPM': 'ddpm'}
# Every paper visualization shows the morph-smooth variant of the tables. On MNIST that is
# the validation-selected stage-3 config (w_gan 0.1, w_path_smooth 0.3; tune_mnist_collect).
OURS = {
    'adni': [('Ours', 'adni_morph_smooth')],
    'mnist': [('Ours', 'tune_mnist_smooth_g0.1_s0.3')],
}
DOMAINS = {'adni': ('T1', 'FA'), 'mnist': ('MRI', 'PET')}
SUB = {'adni': 'adni', 'mnist': 'mnist_petct'}


# ---------------------------------------------------------------- infer (GPU)
def resolve_tag(tag, dataset):
    """Checkpoints live either at EXPS/<tag> (migrated runs) or at
    EXPS/<dataset>/checkpoints/<tag> (runs trained here). Return whichever exists."""
    for candidate in (tag, f'{dataset}/checkpoints/{tag}'):
        if (EXPS / candidate).is_dir():
            return candidate
    return tag


def load_test(dataset, samples='test', root=None, split='test'):
    """Held-out pairs; with samples='v1' (ADNI) the V1-covered slices instead, which
    belong to TRAINING subjects and exist only for the DEC-FA preview. `root`
    points MNIST at another build of the dataset (sensitivity variants).
    split='val' (MNIST) returns the validation pairs train.py holds out for early
    stopping -- the first 10% of the training folders (data.unpaired_dataset.
    folder_loaders) -- at native resolution, for model selection without the test set."""
    import torch
    if dataset == 'adni':
        from data.paired_dataset import subject_level_split, PairedADNISliceDataset
        if samples == 'v1':
            from inpaper_utils.decfa import preview_indices
            ds = PairedADNISliceDataset(torch.tensor(preview_indices()), 'label_4')
        else:
            _, test = subject_level_split(42, .2, 'label_4', 40, 49)
            ds = PairedADNISliceDataset(test, 'label_4')
        items = [ds[i] for i in range(len(ds))]
        meta = [{'test_index': i, 'cache_index': int(ds.indices[i]),
                 'subject': ds.subjects[int(ds.subj_idx[ds.indices[i]])],
                 'z': int(ds.z_idx[ds.indices[i]])} for i in range(len(ds))]
        a = torch.stack([t[0] for t in items]); b = torch.stack([t[1] for t in items])
    else:
        if samples != 'test':
            raise SystemExit('--samples v1 is ADNI only')
        from PIL import Image
        # MNIST_PAIRED_ROOT: a node-local copy (jobs untar it to $SLURM_SCRATCH; 10k small PNGs over NFS are slow)
        root = Path(root or os.environ.get('MNIST_PAIRED_ROOT') or DATA / 'MNIST_CycleFlow/mnist_petct_paired')
        sa, sb = ('testA', 'testB') if split == 'test' else ('trainA', 'trainB')
        names = sorted(p.name for p in (root / sa).glob('*.png'))
        assert names == sorted(p.name for p in (root / sb).glob('*.png'))
        if split == 'val':
            names = names[:max(1, int(round(0.1 * len(names))))]     # folder_loaders' val_frac

        def read(side, name):
            with Image.open(root / side / name) as im:
                return np.asarray(im.convert('RGB'), dtype=np.float32) / 255

        a = torch.from_numpy(np.stack([read(sa, n) for n in names])).permute(0, 3, 1, 2)
        b = torch.from_numpy(np.stack([read(sb, n) for n in names])).permute(0, 3, 1, 2)
        meta = [{'test_index': i, 'filename': n, 'digit': int(n.split('_d')[1][0])}
                for i, n in enumerate(names)]
    return (a.contiguous(), b.contiguous()), meta


def batched(fn, x, dev, bs):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            out.append(fn(x[i:i + bs].to(dev)).float().clamp(0, 1).cpu())
    y = torch.cat(out)
    assert torch.isfinite(y).all(), 'non-finite prediction'
    return y


def dit_generators(tag, dev, img_ch, size, ckpts, cfg=1.0):
    import torch
    from baselines.latent_ae import LatentAE
    from baselines.nets_dit import DiT
    from baselines.diffusion_iddpm import IDDPM
    ae = LatentAE(img_ch=img_ch).to(dev).eval()
    ae.load_state_dict(torch.load(EXPS / tag / 'latent_ae.pth', map_location='cpu',
                                  weights_only=False)['ae'], strict=True)
    ck = torch.load(EXPS / tag / 'last.pth', map_location='cpu', weights_only=False)
    ar = ck['args']
    # The saved AE carries default buffers; the fitted statistics survive in the log.
    stats = re.findall(r'A mean=([-\d.]+) std=([-\d.]+) \| B mean=([-\d.]+) std=([-\d.]+)',
                       (EXPS / tag / 'train.log').read_text())
    if not stats:
        raise ValueError(f'missing DiT latent statistics in {EXPS / tag / "train.log"}')
    am, ast, bm, bst = map(float, stats[-1])
    ae.mean.copy_(torch.tensor([am, bm], device=dev)); ae.std.copy_(torch.tensor([ast, bst], device=dev))
    # cfg=1.0 is plain conditional sampling (one forward per step, no guidance).
    ckpts[tag] = {'T': ar['T'], 'cfg_used': cfg, 'cfg_train_default': ar['cfg'],
                  'latent_mean': [am, bm], 'latent_std': [ast, bst]}
    diff = IDDPM(T=ar['T'], device=dev)
    gens = []
    for side, key in enumerate(('net_ab', 'net_ba')):
        net = DiT(latent_size=size // 4, latent_ch=ae.latent_ch, patch=ar['patch'], hidden=ar['hidden'],
                  depth=ar['depth'], heads=ar['heads']).to(dev).eval()
        net.load_state_dict(ck[key], strict=True)

        def gen(x, net=net, side=side):
            z = ae.encode_norm(x, side)
            out = diff.p_sample_loop(net, z.shape, {'src': z}, cfg_scale=cfg,
                                     null_kwargs={'src': torch.zeros_like(z)})
            return ae.decode_norm(out, 1 - side)
        gens.append(gen)
    return gens


def baseline_generators(name, tag, dev, img_ch, size, ckpts, dit_cfg=1.0, dataset=None):
    import torch
    if name == 'DiT':
        return dit_generators(tag, dev, img_ch, size, ckpts, dit_cfg)
    ck = torch.load(EXPS / tag / 'last.pth', map_location='cpu', weights_only=False)
    ar = ck.get('args', {})
    ckpts[tag] = {k: ar[k] for k in ('method', 'n_steps', 'base', 'ngf', 'n_blocks', 'coupling',
                                     'reflow_sim_steps', 'core_hidden') if k in ar}
    if name == 'RevGAN':
        # One model, both directions: A->B runs the shared core forward, B->A inverts it.
        from baselines.nets_revgan import RevGAN
        g = RevGAN(img_ch, ar.get('ngf', 64), ar.get('n_blocks', 6),
                   ar.get('core_hidden') or None).to(dev).eval()
        g.load_state_dict(ck['revgan'], strict=True)
        return [lambda x: (g.a2b(x * 2 - 1) + 1) / 2, lambda y: (g.b2a(y * 2 - 1) + 1) / 2]
    if name == 'CycleGAN':
        from model.backbone import ResnetGenerator

        def generator(state, args):
            g = ResnetGenerator(img_ch, img_ch, args.get('ngf', 64), args.get('n_blocks', 6)).to(dev).eval()
            g.load_state_dict(state, strict=True)
            return lambda x, g=g: (g(x * 2 - 1) + 1) / 2

        gens = [generator(ck['G_T1toFA'], ar), generator(ck['G_FAtoT1'], ar)]
        # Prefer a CycleGAN trained natively in B->A (train_host.py --swap_domains,
        # which stores that generator as G_T1toFA) over this host's reverse generator.
        rev = EXPS / (dataset or '_') / 'checkpoints' / 'host_rev' / 'last.pth'
        if rev.exists():
            rck = torch.load(rev, map_location='cpu', weights_only=False)
            rar = rck.get('args', {})
            if not rar.get('swap_domains'):
                raise ValueError(f'{rev} was not trained with --swap_domains; its G_T1toFA is not B->A')
            gens[1] = generator(rck['G_T1toFA'], rar)
            ckpts[str(rev)] = {'role': 'native B->A CycleGAN', 'epoch': rck.get('epoch'),
                               'swap_domains': True, 'ngf': rar.get('ngf'), 'n_blocks': rar.get('n_blocks')}
            print(f'CycleGAN B->A: native reverse model {rev}', flush=True)
        return gens
    from baselines.train import build, cfm_generate, meanflow_generate, ddpm_generate
    method = METHOD[name]
    ma, mb, diff = build(method, img_ch, ar.get('base', 64), dev)
    # Runs trained here store the two directions as net_ab/net_ba. The ADNI runs
    # migrated from the old workstation name them net_<A>to<B> (cfm, meanflow) or
    # model_<A>to<B> (ddpm); the nets are the same modules, so accept either.
    KEYS = (('net_ab', 'net_T1toFA', 'model_T1toFA'),
            ('net_ba', 'net_FAtoT1', 'model_FAtoT1'))
    # Those checkpoints carry n_steps as an explicit None, so a dict default never
    # fires; CFM counts Euler steps and DDPM DDIM steps, hence separate fallbacks.
    n_steps = ar.get('n_steps') or (50 if method == 'ddpm' else 10)
    if method == 'recflow' and ck.get('stage') != 2:
        raise ValueError(f'{tag}: RecFlow checkpoint stopped at stage {ck.get("stage")}; '
                         'net_ab/net_ba must be the reflowed field')
    gens = []
    for m, keys in zip((ma, mb), KEYS):
        state = next((ck[k] for k in keys if k in ck), None)
        if state is None:
            raise KeyError(f'{tag}: none of {keys} present in checkpoint (has {list(ck)})')
        m.load_state_dict(state, strict=True); m.eval()
        if method in ('cfm', 'recflow'):
            gens.append(lambda x, m=m: cfm_generate(m, x, n_steps))
        elif method == 'meanflow':
            gens.append(lambda x, m=m: meanflow_generate(m, x))
        else:
            gens.append(lambda x, m=m: ddpm_generate(m, diff, x, n_steps))
    return gens


def ours_generators(tag, dev, ckpts):
    import torch
    from model import MMCLASTcg
    ck = torch.load(EXPS / tag / 'model.pth', map_location='cpu', weights_only=False)
    ar = ck['args']
    m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']),
                  img_ch=ar.get('img_ch', 1)).to(dev).eval()
    m.load_state_dict(ck['model'], strict=True)
    ckpts[tag] = {'variant': ar.get('variant'), 'n_flow': ar['n_flow']}
    return [lambda x: (m.cross_A2B(x * 2 - 1) + 1) / 2, lambda x: (m.cross_B2A(x * 2 - 1) + 1) / 2]


def ssim_all(pred, gt):
    from skimage.metrics import structural_similarity
    p = pred.permute(0, 2, 3, 1).numpy(); g = gt.permute(0, 2, 3, 1).numpy()
    if p.shape[-1] == 1:
        return np.array([structural_similarity(g[i, ..., 0], p[i, ..., 0], data_range=1)
                         for i in range(len(p))])
    return np.array([structural_similarity(g[i], p[i], data_range=1, channel_axis=-1)
                     for i in range(len(p))])


def psnr_all(pred, gt):
    """Per-image PSNR in dB over all pixels, as baselines/common.ssim_psnr."""
    mse = ((pred - gt) ** 2).flatten(1).mean(1).clamp_min(1e-12).numpy()
    return 10 * np.log10(1.0 / mse)


def to_u8(t):
    return (t.permute(0, 2, 3, 1).numpy() * 255 + .5).astype(np.uint8)


def infer(a):
    os.environ['CYCLEFLOW_PURPOSE'] = PURPOSE[a.samples]
    from server_paths import experiment_root
    run = Path(experiment_root())
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pair, meta = load_test(a.dataset, a.samples)
    if a.limit:
        pair, meta = tuple(x[:a.limit] for x in pair), meta[:a.limit]
    n, img_ch, size = pair[0].shape[0], pair[0].shape[1], pair[0].shape[-1]
    what = 'held-out pairs' if a.samples == 'test' else 'V1-covered pairs (training subjects)'
    print(f'{a.dataset}: {n} {what}, {img_ch}x{size}^2, device={dev}', flush=True)
    cache = run / 'cache'; cache.mkdir()
    np.save(cache / 'gt_A.npy', to_u8(pair[0])); np.save(cache / 'gt_B.npy', to_u8(pair[1]))
    methods, ckpts, skipped, ssim, psnr = [], {}, [], {}, {}
    todo = [(nm, tg, 'baseline') for nm, tg in BASELINES[a.dataset]] + \
           [(nm, a.ours_tag or tg, 'ours') for nm, tg in OURS[a.dataset]]
    old = None
    if a.reuse:
        # Baseline predictions are deterministic given (checkpoint, seed rule, test set); when only
        # our checkpoint changes, copy them from an earlier cache instead of re-sampling.
        old = json.loads((Path(a.reuse) / 'meta.json').read_text())
        assert old['n'] == n and old['samples'] == meta, '--reuse cache holds a different test set'
        old_ssim, old_psnr = dict(np.load(Path(a.reuse) / 'ssim.npz')), dict(np.load(Path(a.reuse) / 'psnr.npz'))
    for idx, (name, tag, role) in enumerate(todo):
        tag = resolve_tag(tag, a.dataset)
        if not (EXPS / tag).is_dir():
            print(f'SKIP {name}: no checkpoint directory {EXPS / tag}', flush=True)
            skipped.append({'name': name, 'tag': tag}); continue
        prev = [m for m in (old['methods'] if old else []) if m['tag'] == tag and m['role'] == role == 'baseline']
        if prev:
            k = prev[0]['key']
            methods.append({'name': name, 'tag': tag, 'role': role, 'key': f'm{idx}', 'reused_from': a.reuse})
            # CycleGAN also records its second (native B->A) run under a path key
            ckpts.update({c: v for c, v in old['checkpoints'].items()
                          if c == tag or (name == 'CycleGAN' and 'CycleGAN' in v.get('role', ''))})
            for side in (0, 1):
                shutil.copy2(Path(a.reuse) / f'{k}_{side}.npy', cache / f'm{idx}_{side}.npy')
                ssim[f'm{idx}_{side}'], psnr[f'm{idx}_{side}'] = old_ssim[f'{k}_{side}'], old_psnr[f'{k}_{side}']
                print(f'{name:>16s} dir {side}: mean SSIM {ssim[f"m{idx}_{side}"].mean():.4f}  '
                      f'PSNR {psnr[f"m{idx}_{side}"].mean():.2f}  (reused)', flush=True)
            continue
        gens = ours_generators(tag, dev, ckpts) if role == 'ours' else \
            baseline_generators(name, tag, dev, img_ch, size, ckpts, a.dit_cfg, a.dataset)
        methods.append({'name': name, 'tag': tag, 'role': role, 'key': f'm{idx}'})
        for side, gen in enumerate(gens):
            torch.manual_seed(42 + side)
            t0 = time.monotonic()
            pred = batched(gen, pair[side], dev, a.batch)
            s = ssim_all(pred, pair[1 - side])
            q = psnr_all(pred, pair[1 - side])
            ssim[f'm{idx}_{side}'], psnr[f'm{idx}_{side}'] = s, q
            np.save(cache / f'm{idx}_{side}.npy', to_u8(pred))
            print(f'{name:>16s} dir {side}: mean SSIM {s.mean():.4f}  PSNR {q.mean():.2f}  '
                  f'({time.monotonic() - t0:.0f}s)', flush=True)
        del gens
        torch.cuda.empty_cache()
    np.savez(cache / 'ssim.npz', **ssim)
    np.savez(cache / 'psnr.npz', **psnr)
    info = {'dataset': a.dataset, 'samples_mode': a.samples, 'domains': DOMAINS[a.dataset], 'n': n, 'methods': methods,
            'skipped': skipped, 'checkpoints': ckpts, 'samples': meta, 'seed_rule': 'torch.manual_seed(42+direction)',
            'reused_baselines_from': a.reuse,
            'mean_ssim': {m['name']: [float(ssim[f"{m['key']}_{s}"].mean()) for s in (0, 1)] for m in methods},
            'mean_psnr': {m['name']: [float(psnr[f"{m['key']}_{s}"].mean()) for s in (0, 1)] for m in methods}}
    (cache / 'meta.json').write_text(json.dumps(info, indent=2) + '\n')
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    print('Cache ->', cache, flush=True)


# --------------------------------------------------------------- render (CPU)
PURPOSE = {'test': 'paper_compare', 'v1': 'paper_compare_v1'}


def latest_cache(dataset, samples='test'):
    runs = sorted((EXPS / dataset / PURPOSE[samples]).glob('*/cache/meta.json'))
    if not runs:
        raise FileNotFoundError(f'no paper_compare cache for {dataset}; run `infer` first')
    return runs[-1].parent


def select(info, ssim, per_dir, pick):
    base = [m for m in info['methods'] if m['role'] == 'baseline']
    ours = [m for m in info['methods'] if m['role'] == 'ours']
    key = 'subject' if info['dataset'] == 'adni' else 'digit'
    chosen, used, report = [], set(), []
    for side in (0, 1):
        o = np.min([ssim[f"{m['key']}_{side}"] for m in ours], 0)
        b = np.max([ssim[f"{m['key']}_{side}"] for m in base], 0)
        margin = o - b
        if pick:
            idx = pick[side]
        else:
            idx = []
            for i in np.argsort(-margin):
                k = info['samples'][i][key]
                # Distinct subjects / digits, within and across the two directions.
                if k in used:
                    continue
                idx.append(int(i)); used.add(k)
                if len(idx) == per_dir:
                    break
        chosen.append(idx)
        report.append([{'test_index': int(i), **info['samples'][i], 'ours_min': float(o[i]),
                        'baseline_max': float(b[i]), 'margin': float(margin[i]),
                        'margin_rank': int((margin > margin[i]).sum()) + 1} for i in idx])
    return chosen, report


def render(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cache = Path(a.cache) if a.cache else latest_cache(a.dataset, a.samples)
    info = json.loads((cache / 'meta.json').read_text())
    ssim = dict(np.load(cache / 'ssim.npz'))
    pick, how = None, 'top per direction by min(ours) - max(baselines) SSIM, distinct subject/digit'
    if a.pick:
        pick, how = [[int(v) for v in grp.split(',')] for grp in a.pick.split(';')], 'manual --pick'
        assert len(pick) == 2, '--pick "i,j,k;l,m,n" (first direction; second direction)'
    elif a.samples == 'v1':
        # Only a handful of subjects have V1: show each one's z=44 slice, unranked.
        z44 = [i for i, s in enumerate(info['samples']) if s['z'] == 44]
        pick, how = [z44, z44], 'z=44 slice of every V1-covered subject (training split), not ranked'
    chosen, report = select(info, ssim, a.per_dir, pick)
    if a.dec_fa and info['dataset'] == 'adni' and any('cache_index' not in s for s in info['samples']):
        # Caches written before cache_index was recorded: recover it from (subject, z).
        import torch
        from data.paired_dataset import CACHE
        p = torch.load(CACHE, map_location='cpu', weights_only=False)
        s2i = {s: i for i, s in enumerate(p['subjects'])}
        where = {(int(si), int(zi)): k for k, (si, zi) in enumerate(zip(p['subj_idx'], p['z_idx']))}
        for s in info['samples']:
            s.setdefault('cache_index', where[(s2i[s['subject']], s['z'])])
    if a.dec_fa:
        from inpaper_utils import decfa
        miss = decfa.missing([info['samples'][i].get('cache_index') for idx in chosen for i in idx])
        if miss:
            raise SystemExit(f'{len(miss)} shown examples have no V1 (or the cache predates cache_index). '
                             'Use --samples v1 for a preview.')
    gt = [np.load(cache / 'gt_A.npy', mmap_mode='r'), np.load(cache / 'gt_B.npy', mmap_mode='r')]
    methods = info['methods']
    preds = {f"{m['key']}_{s}": np.load(cache / f"{m['key']}_{s}.npy", mmap_mode='r')
             for m in methods for s in (0, 1)}
    # The cache records the domain names as they were spelled when it was inferred;
    # the figure prints today's spelling, so that renaming a domain is a re-render.
    dom = DOMAINS[a.dataset]
    # Display order: source, ground truth, ours, then the baselines in their fixed order.
    shown = [m for m in methods if m['role'] == 'ours'] + [m for m in methods if m['role'] == 'baseline']
    cols = ['Input', 'Ground truth'] + [m['name'] for m in shown]
    ncol, nrow = len(cols), sum(len(c) for c in chosen)

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': a.fontsize,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    # Exact physical layout (inches) so the text is true size at \linewidth.
    W, left, top, gap, group_gap = a.width, .17, .30 if any('(' in c for c in cols) else .20, .025, .07
    s = (W - left - (ncol - 1) * gap) / ncol
    H = top + nrow * s + (nrow - 1) * gap + (group_gap - gap) + .01
    fig = plt.figure(figsize=(W, H))
    y = H - top
    for side, idx in enumerate(chosen):
        y_group_top = y
        for i in idx:
            y -= s
            ims = [gt[side][i], gt[1 - side][i]] + [preds[f"{m['key']}_{side}"][i] for m in shown]
            doms = [dom[side]] + [dom[1 - side]] * (len(ims) - 1)
            for c, (im, dn) in enumerate(zip(ims, doms)):
                ax = fig.add_axes([(left + c * (s + gap)) / W, y / H, s / W, s / H])
                if a.dec_fa and dn == 'FA':
                    ax.imshow(decfa.colour(im, info['samples'][i]['cache_index']), interpolation='none')
                else:
                    ax.imshow(im[..., 0] if im.shape[-1] == 1 else im, cmap='gray', vmin=0, vmax=255,
                              interpolation='none')
                ax.set_axis_off()
            y -= gap
        y += gap
        fig.text((left - .06) / W, (y + y_group_top) / 2 / H,
                 rf'{dom[side]}$\,\rightarrow\,${dom[1 - side]}', rotation=90, ha='center', va='center')
        y -= group_gap
    for c, name in enumerate(cols):
        label = name.replace(' (', '\n(')
        weight = 'bold' if name.startswith('Ours') else 'normal'
        fig.text((left + c * (s + gap) + s / 2) / W, (H - top + .04) / H, label,
                 ha='center', va='bottom', fontweight=weight, linespacing=1.0)

    out = ROOT / 'snapshot_results' / SUB[a.dataset] / ('decfa_preview' if a.samples == 'v1' else '')
    out.mkdir(parents=True, exist_ok=True)
    stem = '00_inpaper_fig_compare' + ('_decfa' if a.dec_fa else '') + ('_preview' if a.samples == 'v1' else '')
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=400, facecolor='white')
        shutil.copy2(out / f'{stem}.{ext}', cache.parent / f'{stem}.{ext}')
        print('Saved', out / f'{stem}.{ext}', flush=True)
    plt.close(fig)
    manifest = {'cache': str(cache), 'columns': cols, 'figure_width_in': W, 'font_size_pt': a.fontsize,
                'selection': how, 'samples_mode': a.samples, 'dec_fa': a.dec_fa,
                'skipped_baselines': info['skipped'], 'checkpoints': info['checkpoints'],
                'full_test_mean_ssim': info['mean_ssim'], 'full_test_mean_psnr': info.get('mean_psnr'),
                'n_test': info['n'],
                'rows': {f'{dom[s]}->{dom[1 - s]}': report[s] for s in (0, 1)}}
    # Only the figure is paper-facing; its provenance stays with the run under EXPS.
    (cache.parent / f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['infer', 'render'])
    ap.add_argument('--dataset', required=True, choices=['adni', 'mnist'])
    ap.add_argument('--batch', type=int, default=500, help='infer: images per forward batch')
    ap.add_argument('--limit', type=int, default=0, help='infer: first N test pairs only (smoke test)')
    ap.add_argument('--samples', default='test', choices=['test', 'v1'],
                    help='test: held-out pairs; v1: V1-covered TRAINING slices (ADNI DEC-FA preview)')
    ap.add_argument('--dec-fa', action='store_true', help='render: draw FA panels as DEC-FA')
    ap.add_argument('--dit-cfg', type=float, default=1.0,
                    help='infer: DiT classifier-free guidance scale; 1.0 = none (plain conditional sampling)')
    ap.add_argument('--reuse', help='infer: cache dir whose baseline predictions are copied instead of re-sampled')
    ap.add_argument('--ours-tag', help='infer: checkpoint tag drawn as "Ours" instead of OURS[dataset]')
    ap.add_argument('--cache', help='render: cache dir (default: latest infer run)')
    ap.add_argument('--per-dir', type=int, default=3, help='render: examples per direction')
    ap.add_argument('--pick', help='render: explicit test indices, "i,j,k;l,m,n"')
    ap.add_argument('--width', type=float, default=5.5, help='render: figure width in inches (ICLR \\linewidth=5.5)')
    ap.add_argument('--fontsize', type=float, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    infer(a) if a.stage == 'infer' else render(a)


if __name__ == '__main__':
    main()
