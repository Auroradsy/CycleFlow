#!/usr/bin/env python3
"""The two Params columns of Tables 1 and 2, recomputed from the constructors.

  python -m inpaper_utils.count_params            # both datasets

`infer` counts every module a translation runs, in BOTH directions; `train`
counts every parameter that had to be optimised to produce them, across all the
runs the row stands for -- which is why CycleGAN's train column is more than
twice its infer column (two hosts, forward and native reverse, each with two
generators and two discriminators) while CFM's two columns agree.

The rows that were already in the paper are recomputed here as a check: if one
of them no longer matches the table, the constructor changed under it.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path(os.environ.get('MMCLAST_EXPS', '/ix/lzhan/siyuan/exps/CycleFlow'))
# (dataset, image channels); the published table for each
SPEC = {'mnist': 3, 'adni': 1}
PUBLISHED = {'mnist': {'DDPM': (28.91, 28.91), 'CycleGAN': (15.68, 42.41), 'MeanFlow': (16.56, 16.56),
                       'DiT': (65.83, 65.83), 'CFM': (16.56, 16.56)},
             'adni': {'DDPM': (28.91, 28.91), 'CycleGAN': (15.65, 42.35), 'MeanFlow': (16.56, 16.56),
                      'DiT': (65.93, 65.93), 'CFM': (16.56, 16.56)}}


def n(m):
    return sum(p.numel() for p in m.parameters())


def rows(ch, size):
    from model.backbone import ResnetGenerator, PatchDiscriminator
    from baselines.nets_cfm import VelocityUNet
    from baselines.nets_meanflow import MeanVelocityUNet
    from baselines.nets_ddpm import CondUNet
    from baselines.nets_dit import DiT
    from baselines.latent_ae import LatentAE
    from baselines.nets_revgan import RevGAN
    g, d = n(ResnetGenerator(ch, ch, 64, 6)), n(PatchDiscriminator(ch, 64))
    cfm, mf = n(VelocityUNet(in_ch=ch)), n(MeanVelocityUNet(in_ch=ch))
    ddpm = n(CondUNet(in_ch=2 * ch, out_ch=ch, base=64))
    ae = LatentAE(img_ch=ch)
    dit = n(DiT(latent_size=size // 4, latent_ch=ae.latent_ch, patch=2, hidden=384, depth=12, heads=6))
    rev = RevGAN(ch)
    return {
        'DDPM': (2 * ddpm, 2 * ddpm),
        'CycleGAN': (2 * g, 2 * (2 * g + 2 * d)),          # forward host + native reverse host
        'RevGAN': (n(rev), n(rev) + 2 * d),                # ONE model for both directions
        'MeanFlow': (2 * mf, 2 * mf),
        'DiT': (2 * dit + n(ae), 2 * dit + n(ae)),
        'CFM': (2 * cfm, 2 * cfm),
        'RectFlow': (2 * cfm, 4 * cfm),                    # the reflow refits, it does not fine-tune
    }


def main():
    sys.path.insert(0, str(ROOT))
    for ds, ch in SPEC.items():
        size = 64 if ds == 'mnist' else 112
        print(f'\n{ds}  ({ch} channels, {size}^2)')
        print(f'{"method":10s} {"infer (M)":>10s} {"train (M)":>10s}   published')
        for name, (inf, tr) in rows(ch, size).items():
            pub = PUBLISHED[ds].get(name)
            mark = ''
            if pub:
                mark = 'ok' if (round(inf / 1e6, 2), round(tr / 1e6, 2)) == pub else f'MISMATCH {pub}'
            print(f'{name:10s} {inf / 1e6:10.2f} {tr / 1e6:10.2f}   {mark}')


if __name__ == '__main__':
    main()
