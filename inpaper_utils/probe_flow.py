import sys, torch, numpy as np
sys.path.insert(0, '/ihome/lzhan/sid51/projects/Brain/CycleFlow')
from data.paired_dataset import subject_level_split, PairedADNISliceDataset
from model import MMCLASTcg
from skimage.metrics import structural_similarity as ssim
torch.set_num_threads(8)
_, test = subject_level_split(42, .2, 'label_4', 40, 49)
ds = PairedADNISliceDataset(test, 'label_4')
A = torch.stack([ds[i][0] for i in range(len(ds))]); B = torch.stack([ds[i][1] for i in range(len(ds))])
for tag in ('adni_morph', 'adni_morph_bi'):
    ck = torch.load(f'/ix/lzhan/siyuan/exps/CycleFlow/{tag}/model.pth', map_location='cpu', weights_only=False); ar = ck['args']
    m = MMCLASTcg(ar['ngf'], ar['n_blocks'], ar['n_flow'], ar['flow_hidden'], bool(ar['pre_relu']), img_ch=1).eval()
    m.load_state_dict(ck['model'], strict=True)
    print('=====', tag)
    with torch.inference_mode():
        za, zb = m.enc_A(A * 2 - 1), m.enc_B(B * 2 - 1)
        print('latent', tuple(za.shape), 'za std %.3f zb std %.3f' % (za.std(), zb.std()))
        for L in m.flow.layers[::2]:
            print('actnorm log a: min %.3f max %.3f mean %.3f | b: min %.3f max %.3f' % (
                L.log_scale.min(), L.log_scale.max(), L.log_scale.mean(), L.bias.min(), L.bias.max()))
        for d, (z0, zt, inv) in enumerate(((za, zb, False), (zb, za, True))):
            x = z0; print('-- dir', d, 'rel dist to target enc: state0 %.3f' % ((x - zt).norm() / zt.norm()))
            order = reversed(range(4)) if inv else range(4)
            for k in order:
                an, cp = m.flow.layers[2 * k], m.flow.layers[2 * k + 1]
                if not inv:
                    x, _ = an(x)
                    x1, _ = cp._split(x); s, t = cp.net(x1).chunk(2, 1); s = torch.tanh(s)
                    x, _ = cp(x)
                else:
                    y1, _ = cp._split(x); s, t = cp.net(y1).chunk(2, 1); s = torch.tanh(s)
                    x, _ = cp.inverse(x); x, _ = an.inverse(x)
                cm_s, rms_s = s.mean(1).abs().mean(), s.pow(2).mean(1).sqrt().mean()
                cm_t, rms_t = t.mean(1).abs().mean(), t.pow(2).mean(1).sqrt().mean()
                print('block %d: s |chan-mean| %.3f vs RMS %.3f ; t |chan-mean| %.3f vs RMS %.3f ; s range [%.2f,%.2f] ; rel dist to target %.3f' % (
                    k + 1, cm_s, rms_s, cm_t, rms_t, s.min(), s.max(), (x - zt).norm() / zt.norm()))
        lat = (m.b_to_a(m.a_to_b(za)) - za).abs().max(), (m.a_to_b(m.b_to_a(zb)) - zb).abs().max()
        print('latent cycle max abs err A %.2e B %.2e' % lat)
        xb = (m.cross_A2B(A * 2 - 1) + 1) / 2; xa_c = (m.cross_B2A(xb.clamp(0, 1) * 2 - 1) + 1) / 2
        xa = (m.cross_B2A(B * 2 - 1) + 1) / 2; xb_c = (m.cross_A2B(xa.clamp(0, 1) * 2 - 1) + 1) / 2
        sr_a = (m.self_A(A * 2 - 1) + 1) / 2; sr_b = (m.self_B(B * 2 - 1) + 1) / 2
        f = lambda p, g: np.mean([ssim(g[i, 0].numpy(), p[i, 0].clamp(0, 1).numpy(), data_range=1) for i in range(len(g))])
        print('image cycle SSIM T1->FA->T1 %.4f  FA->T1->FA %.4f | self-recon T1 %.4f FA %.4f' % (f(xa_c, A), f(xb_c, B), f(sr_a, A), f(sr_b, B)))
        print('image cycle MAE T1 %.4f FA %.4f' % ((xa_c.clamp(0, 1) - A).abs().mean(), (xb_c.clamp(0, 1) - B).abs().mean()))
