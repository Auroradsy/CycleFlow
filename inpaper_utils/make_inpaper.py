#!/usr/bin/env python3
"""Reproducible five-example paper panels from migrated checkpoints.

python -m inpaper_utils.make_inpaper --dataset adni --with-dit
python -m inpaper_utils.make_inpaper --dataset mnist --with-dit
Outputs and logs are archived under EXPS; figures are also copied to the
requested repository snapshot_results folder. No training is performed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import re
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path('/ix/lzhan/siyuan/exps/CycleFlow')
DATA = Path('/ix/lzhan/siyuan/datasets/processed_datas')


def load_ck(path, manifest):
    print('Loading', path, flush=True)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    manifest['checkpoints'].append({'path': str(path), 'bytes': path.stat().st_size,
                                    'mtime_ns': path.stat().st_mtime_ns,
                                    'args': ck.get('args', {})})
    return ck


def samples(dataset):
    if dataset == 'adni':
        from data.paired_dataset import subject_level_split, PairedADNISliceDataset
        _, test = subject_level_split(42, .2, 'label_4', 40, 49)
        ds = PairedADNISliceDataset(test, 'label_4')
        # Five evenly spaced held-out subjects, all at the same axial level.
        eligible = [i for i, k in enumerate(ds.indices) if int(ds.z_idx[k]) == 44]
        idx = [eligible[int(j)] for j in np.linspace(0, len(eligible)-1, 5)]
        ids = [{'test_index': i, 'cache_index': int(ds.indices[i]),
                'subject': ds.subjects[int(ds.subj_idx[ds.indices[i]])], 'z': 44} for i in idx]
        pairs = [ds[i][:2] for i in idx]
        dom = ('T1', 'FA')
    else:
        from PIL import Image
        root = DATA / 'MNIST_CycleFlow/mnist_petct_paired'
        names = sorted(p.name for p in (root/'testA').glob('*.png'))
        assert names == sorted(p.name for p in (root/'testB').glob('*.png'))
        # First five distinct digit classes in filename order, no score selection.
        idx, seen = [], set()
        for i, name in enumerate(names):
            digit = int(name.split('_d')[1][0])
            if digit not in seen:
                idx.append(i); seen.add(digit)
            if len(idx) == 5:
                break
        def read(side, name):
            with Image.open(root/side/name) as im:
                a = np.array(im.convert('RGB').resize((64,64), Image.Resampling.BICUBIC), dtype=np.float32)/255
            return torch.from_numpy(a).permute(2,0,1)
        pairs = [(read('testA',names[i]),read('testB',names[i])) for i in idx]
        ids = [{'test_index': i, 'filename': names[i], 'digit': int(names[i].split('_d')[1][0])} for i in idx]
        dom = ('MRI','PET')
    return tuple(torch.stack([p[d] for p in pairs]).to(DEV) for d in range(2)), ids, dom


def arr(t, pm1=False):
    t = (t+1)/2 if pm1 else t
    assert torch.isfinite(t).all(), 'Non-finite generated image'
    return t.clamp(0,1).detach().cpu().permute(0,2,3,1).numpy()


def score(a, b):
    from skimage.metrics import structural_similarity
    if a.shape[-1] == 1:
        return float(structural_similarity(a[...,0],b[...,0],data_range=1))
    return float(structural_similarity(a,b,data_range=1,channel_axis=-1))


def display(ax, im):
    ax.imshow(im[...,0] if im.shape[-1] == 1 else im, cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, out, name, manifest):
    for ext in ('png','pdf'):
        path = out/f'00_inpaper_{name}.{ext}'
        fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
        manifest['files'].append(path.name)
        print('Saved',path,flush=True)
    plt.close(fig)


def comparison(out, kind, gt, predictions, ids, dom, dataset, manifest):
    cols = ['Input','Target'] + list(predictions) if kind == 'cross_gen' else ['Input / GT'] + list(predictions)
    fig, axes = plt.subplots(10,len(cols),figsize=(1.55*len(cols),15),squeeze=False)
    metrics = {}
    for sample in range(5):
        for side in range(2):
            row = sample*2+side
            target = 1-side if kind == 'cross_gen' else side
            imgs = [gt[side][sample]]
            if kind == 'cross_gen': imgs.append(gt[target][sample])
            imgs += [predictions[k][side][sample] for k in predictions]
            for c, im in enumerate(imgs):
                display(axes[row,c],im)
                if c >= len(cols)-len(predictions):
                    sc = score(im,gt[target][sample])
                    axes[row,c].set_xlabel(f'{sc:.3f}',fontsize=8,labelpad=2)
                    metrics.setdefault(cols[c],[]).append({'sample':sample+1,'input_domain':dom[side], 'target_domain':dom[target],'ssim':sc})
            detail = f'z={ids[sample]["z"]}' if dataset=='adni' else f'digit {ids[sample]["digit"]}'
            direction = f'{dom[side]} → {dom[target]}'
            axes[row,0].set_ylabel(f'#{sample+1} · {detail}\n{direction}',rotation=0,ha='right',va='center',fontsize=9,labelpad=10)
    for c, name in enumerate(cols): axes[0,c].set_title(name,fontsize=10,pad=10)
    title = 'Self-reconstruction' if kind=='self_recon' else 'Cross-modal generation'
    note = '\n* DiT normalization recovered from training log (3 decimals).' if 'DiT*' in predictions else ''
    fig.suptitle(f'{dataset.upper()}  |  {title}\nFive fixed held-out pairs · both directions · numbers: per-image SSIM{note}',fontsize=13,y=.999)
    fig.tight_layout(rect=(0,0,1,.95),h_pad=.6,w_pad=.3)
    manifest[kind+'_metrics'] = metrics
    save(fig,out,kind+'_baseline',manifest)


def trajectory(out, model, pair, gt, ids, dom, dataset, manifest):
    for side in range(2):
        x = pair[side]*2-1
        enc = model.enc_A if side==0 else model.enc_B
        states = model.walk(enc(x),inverse=bool(side))
        endpoint = model.a_to_b(enc(x)) if side==0 else model.b_to_a(enc(x))
        torch.testing.assert_close(states[-1],endpoint)
        decs = (model.dec_A,model.dec_B) if side==0 else (model.dec_B,model.dec_A)
        decoded = [[arr(dec(s),True) for s in states] for dec in decs]
        expected = model.cross_A2B(x) if side==0 else model.cross_B2A(x)
        torch.testing.assert_close(decs[1](states[-1]),expected)
        labels = [f'{dom[side]} input'] + ['State 0']+[f'Block {k}' for k in range(1,len(states))]+[f'{dom[1-side]} GT']
        fig,axes = plt.subplots(10,len(labels),figsize=(1.5*len(labels),14),squeeze=False)
        for i in range(5):
            for view in range(2):
                row=2*i+view
                ims=[gt[side][i]]+[s[i] for s in decoded[view]]+[gt[1-side][i]]
                for c,im in enumerate(ims): display(axes[row,c],im)
                axes[row,0].set_ylabel(f'#{i+1}\n{dom[side if view==0 else 1-side]} decoder',rotation=0,ha='right',va='center',fontsize=9,labelpad=10)
        for c,l in enumerate(labels): axes[0,c].set_title(l,fontsize=10)
        fig.suptitle(f'{dataset.upper()}  |  {dom[side]} → {dom[1-side]} native flow progression (morph)\nFive fixed held-out pairs · every actual flow block · both decoder views',fontsize=13,y=.995)
        fig.tight_layout(rect=(0,0,1,.965),h_pad=.5,w_pad=.3)
        save(fig,out,'process_'+('a2b' if side==0 else 'b2a'),manifest)



def dit_cross(pair, ae_tag, ae, manifest):
    from baselines.nets_dit import DiT
    from baselines.diffusion_iddpm import IDDPM
    ck=load_ck(EXPS/ae_tag/'last.pth',manifest); ar=ck['args']
    log=EXPS/ae_tag/'train.log'
    matches=re.findall(r'A mean=([-\d.]+) std=([-\d.]+) \| B mean=([-\d.]+) std=([-\d.]+)',log.read_text())
    if not matches: raise ValueError(f'Missing DiT latent statistics: {log}')
    am,ast,bm,bst=map(float,matches[-1])
    ae.mean.copy_(torch.tensor([am,bm],device=DEV)); ae.std.copy_(torch.tensor([ast,bst],device=DEV))
    manifest['dit_sampling']={'T':ar['T'],'cfg':ar['cfg'],'normalization_source':str(log),
        'mean':[am,bm],'std':[ast,bst],'precision_note':'Normalization recovered from training log rounded to three decimals; not exact original evaluation.',
        'seeds':[42,43]}
    pred=[]
    diff=IDDPM(T=ar['T'],device=DEV)
    for side,key in enumerate(('net_ab','net_ba')):
        m=DiT(latent_size=pair[side].shape[-1]//4,latent_ch=ae.latent_ch,patch=ar['patch'],hidden=ar['hidden'],depth=ar['depth'],heads=ar['heads']).to(DEV).eval()
        m.load_state_dict(ck[key],strict=True)
        torch.manual_seed(42+side)
        with torch.inference_mode():
            z=ae.encode_norm(pair[side],side)
            # Same full ancestral sampler as training, with progress logging.
            calls=[0]; started=time.monotonic()
            def progress_model(*args,**kwargs):
                result=m(*args,**kwargs); calls[0]+=1
                if calls[0]%100==0:
                    print(f'DiT direction {side+1}: {calls[0]//2}/{ar["T"]} steps, {time.monotonic()-started:.0f}s',flush=True)
                return result
            generated=diff.p_sample_loop(progress_model,z.shape,{'src':z},cfg_scale=ar['cfg'],null_kwargs={'src':torch.zeros_like(z)})
            pred.append(arr(ae.decode_norm(generated,1-side)))
        del m
    return tuple(pred)


def main():
    global torch, np, plt, DEV
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,choices=['adni','mnist'])
    ap.add_argument('--with-dit',action='store_true',help='Include full 1000-step DiT cross generation with normalization recovered from the training log.')
    a=ap.parse_args()
    os.environ['CYCLEFLOW_PURPOSE']='inpaper_visualization'
    from server_paths import experiment_root
    run=Path(experiment_root())
    print('Importing numerical libraries...',flush=True)
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    torch.manual_seed(42); np.random.seed(42)
    DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    sub='adni' if a.dataset=='adni' else 'mnist_petct'
    out=run/'snapshot_results'/sub; out.mkdir(parents=True,exist_ok=True)
    manifest={'dataset':a.dataset,'seed':42,'device':str(DEV),'torch_version':torch.__version__,'checkpoints':[],'files':[],
              'notes':['Examples selected before model inference; shared across all panels.',
                       'CycleFlow base is an ablation baseline; latcyc and morph are variants.',
                       'DiT AE self reconstruction uses posterior mean, not stochastic sampling.',
                       'Self reconstructions are direct autoencoding, not CycleGAN cycle reconstructions.',
                       'DiT cross generation, if requested, uses log-recovered normalization; see dit_sampling precision note.']}
    if a.dataset=='adni': manifest['notes'].append('ADNI external translation baseline weights (CycleGAN, CFM, DDPM, MeanFlow) not present in migrated CycleFlow directory; the only external translation baseline in the cross panel is DiT.')
    print('Preparing fixed test examples on',DEV,flush=True)
    pair,ids,dom=samples(a.dataset); manifest['samples']=ids
    print('Selected samples:',ids,flush=True)
    gt=tuple(arr(x) for x in pair)
    selfs={}; crosses={}
    from baselines.latent_ae import LatentAE
    ae_tag='dit_adni_scratch' if a.dataset=='adni' else 'dit_mnist'
    ck=load_ck(EXPS/ae_tag/'latent_ae.pth',manifest)
    ae=LatentAE(img_ch=pair[0].shape[1]).to(DEV).eval()
    ae.load_state_dict(ck['ae'],strict=True)
    with torch.inference_mode(): selfs['DiT AE\n(baseline)']=tuple(arr(ae.decode(ae.moments(x)[0])) for x in pair)
    if a.with_dit:
        crosses['DiT*']=dit_cross(pair,ae_tag,ae,manifest)
    del ae,ck
    if a.dataset=='mnist':
        from model.backbone import ResnetGenerator
        ck=load_ck(EXPS/'mnist_host/last.pth',manifest); ar=ck.get('args',{})
        pred=[]
        for side,key in enumerate(('G_T1toFA','G_FAtoT1')):
            m=ResnetGenerator(3,3,ar.get('ngf',64),ar.get('n_blocks',6)).to(DEV).eval()
            m.load_state_dict(ck[key],strict=True)
            with torch.inference_mode(): pred.append(arr(m(pair[side]*2-1),True))
            del m
        crosses['CycleGAN']=tuple(pred); del ck
        # The training module's generators retain the original evaluation protocol.
        from baselines.train import build,cfm_generate,meanflow_generate,ddpm_generate
        for method,label in [('cfm','CFM'),('meanflow','MeanFlow'),('ddpm','DDPM')]:
            ck=load_ck(EXPS/f'{method}_mnist/last.pth',manifest); ar=ck['args']
            ma,mb,diff=build(method,3,ar.get('base',64),DEV)
            pred=[]
            for side,(m,key) in enumerate(((ma,'net_ab'),(mb,'net_ba'))):
                m.load_state_dict(ck[key],strict=True); m.eval(); torch.manual_seed(42+side)
                with torch.inference_mode():
                    if method=='cfm': y=cfm_generate(m,pair[side],ar.get('n_steps',10))
                    elif method=='meanflow': y=meanflow_generate(m,pair[side])
                    else: y=ddpm_generate(m,diff,pair[side],ar.get('n_steps',10))
                    pred.append(arr(y))
            crosses[label]=tuple(pred); del ma,mb,m,ck
            print('Finished',label,flush=True)
    from model import MMCLASTcg
    prefix='adni_' if a.dataset=='adni' else 'mnist_p_'
    for variant in ('base','latcyc','morph'):
        ck=load_ck(EXPS/(prefix+variant)/'model.pth',manifest); ar=ck['args']
        m=MMCLASTcg(ar['ngf'],ar['n_blocks'],ar['n_flow'],ar['flow_hidden'],bool(ar['pre_relu']),img_ch=ar.get('img_ch',1)).to(DEV).eval()
        m.load_state_dict(ck['model'],strict=True)
        label='CycleFlow\n'+variant
        with torch.inference_mode():
            selfs[label]=(arr(m.self_A(pair[0]*2-1),True),arr(m.self_B(pair[1]*2-1),True))
            crosses[label]=(arr(m.cross_A2B(pair[0]*2-1),True),arr(m.cross_B2A(pair[1]*2-1),True))
            if variant=='morph': trajectory(out,m,pair,gt,ids,dom,a.dataset,manifest)
        del m,ck
    comparison(out,'self_recon',gt,selfs,ids,dom,a.dataset,manifest)
    comparison(out,'cross_gen',gt,crosses,ids,dom,a.dataset,manifest)
    shutil.copy2(Path(__file__),run/'make_inpaper.py')
    (out/'00_inpaper_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dest=ROOT/'snapshot_results'/sub; dest.mkdir(parents=True,exist_ok=True)
    # Only figures are paper-facing; the manifest stays in the archived run under EXPS.
    for path in out.iterdir():
        if path.suffix in ('.pdf','.png'): shutil.copy2(path,dest/path.name)
    print('Archived:',out,'\nPublished copies:',dest,flush=True)


if __name__=='__main__':
    main()
