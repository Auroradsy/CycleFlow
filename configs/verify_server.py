"""Read migrated data and run a small CPU backward pass; no training job."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from server_paths import experiment_root, DATA_ROOT
experiment_root()
from data.paired_dataset import CACHE, LABELS_CSV, subject_level_split, PairedADNISliceDataset
from data.unpaired_dataset import UnpairedFolderDataset
from model import MMCLASTcg

torch.set_num_threads(2)
train, test = subject_level_split(seed=42, test_frac=0.2, label_scheme='label_4', z_lo=40, z_hi=49)
a, b, label = PairedADNISliceDataset(train[:1], 'label_4')[0]
assert a.shape == b.shape == (1, 112, 112)
assert torch.isfinite(a).all() and torch.isfinite(b).all()
print(f'ADNI: cache={CACHE}, labels={LABELS_CSV}, train={len(train)}, test={len(test)}, shape={tuple(a.shape)}')
root = DATA_ROOT / 'MNIST_CycleFlow/mnist_petct_paired'
for split in ('train', 'test'):
    ds = UnpairedFolderDataset(str(root), split, train=False, img_ch=3, crop_size=64, pair=True, flip=False)
    x, y, _ = ds[0]
    assert x.shape == y.shape == (3, 64, 64)
    print(f'MNIST paired {split}: {len(ds)}, shape={tuple(x.shape)}')
# Exercise the migrated PyTorch installation without a costly full epoch.
from model.backbone import ResnetGenerator
net = ResnetGenerator(ngf=8, n_blocks=1)
x = torch.randn(1, 1, 32, 32)
y = net(x)
y.square().mean().backward()
assert any(p.grad is not None for p in net.parameters())
print('CPU forward/backward OK; torch=', torch.__version__, 'CUDA build=', torch.version.cuda, 'GPU visible=', torch.cuda.is_available())
