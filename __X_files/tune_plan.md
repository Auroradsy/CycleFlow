# ADNI bimodal tuning plan

A scratchpad to track which dial we touch next on the T1↔FA CLAST pilot
(`adni_pilot/`). Treat each row in §0 as an experiment slot,update **best
metric + notes** as you go. Detailed instructions for each tunable are below.

> **Baseline (frozen reference,from RESULTS_ADNI.md §1 / §6):**
> - subject-level acc (concat T1‖FA, K=4) = **0.442**
> - slice CCA = **0.945**,subject CCA = **0.023**
> - subject-level NMI = 0.099,ARI = 0.032
> - 5,778 paired axial slices,214 subjects (1 AD dropped → 213 used)
> - `image_size=112`, `latent_dim=8`, `K=4`, `n_flow_layers=4`, batch=128, 50 ep

---

## 0. Experiment log (fill as you run)

| # | Tuning item | Status | Best subj acc | Best NMI | sub-CCA | Notes |
|---|---|---|---|---|---|---|
| baseline | – |  done | 0.442 | 0.099 | 0.023 | RESULTS_ADNI §1 |
| 1 | Increase sample size | tbd| | | | |
| 2 | Align K with #classes | skip | – | – | – | K already aligned everywhere (MNIST 10, ADNI 4) |
| 3 | latent_dim sweep | tbd | | | | |
| 4 | n_flow_layers sweep | tbd | | | | |
| 5 | ResNet18 encoder | done(negative) | – | – | – | **MNIST eval**:pretrained=0.884 vs baseline=0.938. `viz_mnist_pretrained.py` + `figures/14_mnist_pretrained_vs_baseline_tsne.png` |
| 6 | MedVAE 2D pretrained | done(negative) | – | – | – | **ADNI T1 单模 eval**:medvae=0.280 vs baseline=0.297。MedVAE 2D 是 X-ray 域(胸/乳腺),不是脑 MRI。NMI≈0(无聚类信号)。`results_single_t1_medvae/best_vae.pth` |
| 7 | Pair loss alternative | ⬜ | | | | |

Standard eval per run(`adni_pilot/eval_subject_level.py`):
- subject-level acc (concat T1‖FA) — primary
- subject-level NMI / ARI
- slice-CCA + subject-CCA
- training trajectories (loss / cca / pair) — `figures/`

---

## 1. 增大样本量

**Why**: 5,778 paired slices 太少,单模/双模都没爬过 majority baseline,
flow 退化成恒等(参见 RESULTS §8)。

| 子方向 | 操作 | 增量 | 工程量 |
|---|---|---|---|
| **A. 扩切片范围** | `paired_dataset.py` 里 `Z_LO=20, Z_HI=70`(50 层) | ×1.85 → ~10.7k slices | 5 行 + 重 build cache |
| **B. 加 sagittal + coronal** | 三个正交方向各取 30 切片 | ×3 → ~17k slices | 重写 cache builder |
| **C. 数据增强** | random flip / 旋转 ±10° / 强度 jitter ±0.1 | 等效 ×3–5 | 加到 dataset `__getitem__` |
| **D. 跨多次扫描** | 同被试取多个 session(不只第一个) | ×2–5 | 改 `first_session` |
| **E. 加邻近队列预训练** | OASIS_T1 / AOMIC-ID1000 (~5k subj) 做单模 VAE 预训练再 fine-tune | 上限大 | 1–2 天,需新 dataset adapter |

**推荐序**: C (augmentation) → A (扩 z) → D (多 session) → E (跨队列)。
C 是几乎零代价的;A/D 不改模型;E 是真正的"扩样本"。

---

## 2. Cluster K 与类别数量对齐

**Why**: 当前 K=4 跟 label_4 对齐,但单模训练 `train_single.py` 里也是 K=4 —— 
encoder 没有 GMM 直接监督,可能 K=4 太死板。

| 试 | K(train) | K(eval) | 评 |
|---|---|---|---|
| 1 | 2 | 2 | 退回 label_2(CN vs impaired,107/107) |
| 2 | **4** ← baseline | 4 | 当前 |
| 3 | 6 | 6 | label_6(SMC 单独,AD=1 仍 drop) |
| 4 | 8 | **4** | overcluster + Hungarian map → 让 GMM 自由,再降回 4 评 |
| 5 | 16 | **4** | 更激进 overcluster |

代码改动: `train_single.py` / `train_unsup.py` 里 `CFG.num_clusters`。
对于 (4)(5)需要在 `eval_subject_level.py` 加一层 cluster→GT 映射 (`hungarian_accuracy` 已支持不等数,但要 K_pred > K_gt)。

**推荐**: 优先试 K=8 train + K=4 eval(过聚类 + 映射)。

---

## 3. `latent_dim` 参数分析

**Why**: 8 维可能太紧,但更宽不一定好(VAE 容易 posterior collapse 或失 KL)。
做一个 sweep,把 acc / CCA / 重建质量画成 latent_dim 的曲线。

| 试 | latent_dim | flow_hidden | 备注 |
|---|---|---|---|
| 1 | 4 | 64 | 极紧瓶颈,迫使 encoder 抽象 |
| 2 | **8** ← baseline | 128 | 当前 |
| 3 | 16 | 128 | |
| 4 | 32 | 256 | flow 容量配套加 |
| 5 | 64 | 512 | latent ≈ feature vector 级别 |

每跑一档,记录:
- 训练时间
- 重建 BCE(decoder 质量)
- u-space t-SNE 是否更结构化
- subject-level acc

**最关键的图**: latent_dim vs subject-acc + CCA 曲线,在最后 RESULTS_ADNI.md 加一节
"latent dim ablation"。

---

## 4. `n_flow_layers` tune

**Why**: 4 层 flow 在 MNIST-PET 上观察到真做了非平凡 morph(figures/12-13),
但在 ADNI 上几乎恒等。可能 4 层太浅 + ramp 太慢,或反之 flow 容量不需要那么大。

| 试 | n_flow_layers | flow_hidden | 期待 |
|---|---|---|---|
| 1 | 2 | 128 | 看 flow 是否被砍掉影响小 |
| 2 | **4** ← baseline | 128 | |
| 3 | 6 | 128 | 中等加深 |
| 4 | 8 | 128 | 较深 |
| 5 | 12 | 256 | 极深 |

`adni_pilot/flow_morph.py` 可视化每层的中间状态(已写好),跑完每个配置都出一张
step-by-step 图,看 flow 是否真的开始"动 z"。

---

## 5. Encoder backbone 换 ResNet / Swin  ❌ done(MNIST sanity = 负迁移)

**结果**(MNIST 单模,10k 样本,20 epochs):

| | 编码器 | 参数 | best test acc | NMI |
|---|---|---|---|---|
| baseline | 自定义 conv (image_size=28) | ~1M | **0.938** | ~0.85 |
| ResNet18 pretrained | ImageNet (image_size=64) | 28M | **0.884** | 0.83 |

ImageNet 自然图特征 → 1ch 灰度数字 = 负迁移。t-SNE
(`figures/14_mnist_pretrained_vs_baseline_tsne.png`)显示 baseline 簇更紧。

医学数据(ADNI T1/FA)上 ImageNet 大概率也是负迁移(域差 ImageNet→脑 MRI 更大)。
直接跳到 §6 MedVAE(同域预训练)。

文件:
- `models/pretrained_encoder.py` — ResNet18Encoder 类(可重用)
- `train_mnist_pretrained.py` — 单模 MNIST 训练
- `viz_mnist_pretrained.py` — t-SNE 对比图生成
- `results_mnist_pretrained_resnet18/{best_vae.pth, metrics.csv}` — 训练产物

---

## 5b. (skipped) ResNet sweep — 原来计划要做的

**Why**: 我们 encoder 是从头训的小 conv(103M 主要在 FC),迁移学习几乎没用上。
脑切片虽然不是 ImageNet 分布,但 ResNet 的浅层 edge/texture filters 普适。

| 候选 backbone | 来源 | latent 接口 | 工程量 |
|---|---|---|---|
| **ResNet-18 (ImageNet)** | `torchvision.models.resnet18(pretrained=True)` | 去掉最后 FC,512-D → 8-D 线性头 | 简单,1 小时 |
| ResNet-50 | `torchvision` | 2048-D → 8-D | 类似 |
| Swin-Tiny | `timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)` | 768-D | 需 resize 到 224² |
| ConvNeXt-Tiny | `timm` | 768-D | 类似 |
| EfficientNet-B0 | `timm` | 1280-D | 类似 |

decoder 这边没现成预训练,只能从头训(或参考 §6 用 MedVAE)。

**推荐**: ResNet-18 先打基础。改 `models/vae.py` 的 `Encoder` 类:

```python
import torchvision.models as tvm
class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1, ...):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)  # adapt to in_channels
        m.fc = nn.Identity()
        self.backbone = m
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
    def forward(self, x):
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)
```

---

## 6. MedVAE 2D 预训练 encoder  done(negative on ADNI T1)

**结果**(ADNI T1 单模,4590 训练 slices,30 epochs,frozen encoder + trainable decoder/head/GMM):

| | encoder | trainable params | best test acc | NMI |
|---|---|---|---|---|
| baseline(from-scratch) | 自定义 conv (image_size=112) | 103M(全部) | **0.297** | ~0 |
| MedVAE 2D 4x1c frozen | StanfordMIMI MedVAE 2D(X-ray 域) | 52M(decoder+head) | **0.280** | ≈0 |

**为什么没用**(进一步证实 §5 的判断):MedVAE 2D 训练数据是 chest X-ray + breast mammography
(README §"Model Description"),**不是脑 MRI**。X-ray 域→脑 MRI 域差距比 ImageNet→脑 MRI 小,
但仍然不够。重建 BCE 跟 baseline 相当(rec=2135 vs ~2155),但 latent 没学到 dx 信号
(NMI≈0,acc 跟 random 差不多)。

文件:
- `models/medvae_encoder.py` — MedVAEEncoder wrapper(可重用)
- `adni_pilot/train_single_medvae.py` — ADNI 单模 medvae 训练
- `adni_pilot/results_single_t1_medvae/best_vae.pth` — 权重
- `/data_new3/nfs_share/siyuan/data/medvae/` — 预训练权重缓存(下次复用)

**下次该试的(同域预训练)**:

| 模型 | 训练数据 | 输入 | 备注 |
|---|---|---|---|
| **MONAI BraTS bundle** ⭐ | 脑 MR(T1/T2/FLAIR) | 多种 | brain MRI 同域,该试 |
| **MAISI 3D**(NVIDIA) | whole-body MRI/CT | 256³ | 3D,但同域;升 3D 一起做 |
| **MedVAE 3D 4x1c** | whole-body MRI/CT | 128³ | 3D,同域 |
| **自己在 OASIS 1326 + ADNI 1144 上预训** | 脑 T1 | 112² | 数据量小但完全同域 |

⚠️ 教训:**ImageNet pretrained (§5) + MedVAE 2D X-ray pretrained (§6) 都对脑 MRI 无效**。
两次负迁移说明 encoder 选择必须**严格同域**(brain MRI 训的)才有意义,
domain transfer 不是 free lunch。

---

## 6b. (deprecated) 原 §6 计划

| 模型 | 路径 | 输入要求 | 输出 latent | 备注 |
|---|---|---|---|---|
| **MedVAE 2D** | `huggingface.co/stanfordmimi/MedVAE` | 256² 灰度 | 64-D latent | 直接 swap,但需把 112 → 256 |
| **MedVAE 3D** | 同上 | 128³ | 256-D | 如果上 3D 模型可用 |
| **MONAI BRATS autoencoder** | `monai bundle download brats_mri_segmentation`(找对应 autoencoder) | 192×192 | varies | 脑分割任务的 backbone |
| **MAISI 3D**(NVIDIA/MONAI) | huggingface `monai/maisi-3d-rflow` | 256³ | latent | 3D 强,适合升 3D 时用 |

工程步骤(以 MedVAE 2D 为例):
1. `pip install monai[all]` 或 `pip install medvae`(看 README)
2. 下载 checkpoint
3. 写 wrapper:把我们 `BiModalGMMVAE.vae_A` 和 `.vae_B` 替换成 MedVAE 模块
4. 冻结 encoder/decoder,只训 flow + GMM(快)→ 第二轮 unfreeze fine-tune
5. 评估对比 baseline

**注意**: 输入 size 不匹配(112 vs 256)是大坑,要 resize。也可以直接把我们
input pad 到 256 而不是 112。

---

## 7. 损失:改 paired loss

**Why**: 现在 pair loss = `‖u_A − u_B‖²`(L2 on slice-pair),只逼一对接近,
不管同被试不同 slice 之间的关系。这是 subject-CCA 崩塌的根因(0.945 → 0.023)。

| 选项 | 公式 | 改动量 | 效果预期 |
|---|---|---|---|
| (a) **subject-mean L2** | `‖mean(u_A in subject) − mean(u_B in subject)‖²` | `losses.py` 加 5 行 | 直接修 subject-CCA |
| (b) **InfoNCE / contrastive** | 同 batch 同对(u_A_i, u_B_i)为正,其它为负 | 重写 pair loss | 同被试 ID 拉近 + 推开他人 |
| (c) **subject-id triplet** | 锚 = u_A,正 = u_B 同被试,负 = u_B 不同被试 | 略复杂 | 强化 subject-level |
| (d) **CLIP-style symmetric** | softmax over batch in both directions | InfoNCE 变体 | 跨 batch 强对齐 |
| (e) **MMD** | maximum mean discrepancy between u_A and u_B | 易加 | 分布级对齐 |

代码位置: `synth_datas/multimodal/losses.py:paired_alignment_loss`。

**推荐先试**: (b) InfoNCE,batch 内 cosine similarity 矩阵,temp=0.1:
```python
def paired_infonce(u_A, u_B, temp=0.1):
    u_A = F.normalize(u_A, dim=-1); u_B = F.normalize(u_B, dim=-1)
    logits = u_A @ u_B.T / temp           # [B, B]
    labels = torch.arange(u_A.shape[0], device=u_A.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
```

---

## 推荐尝试顺序(从低代价到高代价)

| 顺序 | 实验 | 预计墙钟 | 期望收益 |
|---|---|---|---|
| 1 | **数据增强 + 扩 z 范围**(§1 A+C) | 30 min 重 build + 1 hr 训 | 中(基础打牢) |
| 2 | **K=8 overcluster + 映射**(§2) | 1 hr 训 + 30 min eval | 低-中 |
| 3 | **InfoNCE pair loss**(§7b) | 1 hr 训 | **高**(修 subject-CCA 根因) |
| 4 | **latent_dim 4/16/32 sweep**(§3) | 3 × 1 hr = 3 hr | 中(产 ablation 曲线) |
| 5 | **ResNet-18 encoder**(§5) | 1.5 hr | 中-高 |
| 6 | **MedVAE encoder**(§6) | 半天工程 + 1.5 hr 训 | **高** |
| 7 | **n_flow_layers sweep**(§4) | 4 × 1 hr | 低-中(诊断用) |
| 8 | **跨队列预训练**(§1E) | 1-2 天 | **高**(scale 升级) |

> 1+3 是 quick wins,跑完应该直接报到 RESULTS_ADNI.md 主表。

---

## 评估协议(每个实验跑完都要做)

```bash
# 1. 训完后保存 best_bimodal.pth
# 2. subject-level eval
CUDA_VISIBLE_DEVICES=1 python adni_pilot/eval_subject_level.py
# 3. 出图
CUDA_VISIBLE_DEVICES=1 python adni_pilot/viz_results.py
# 4. 填这个文档 §0 表
```

回到 §0 把 best subj acc / NMI / sub-CCA 填进对应行,顺手注释 "比 baseline 提升 +X.X pp"。

---

## Quick reference: 已存的 artifacts

```
adni_pilot/
├── results_single_t1/best_vae.pth         # T1 单模 warm-start
├── results_single_fa/best_vae.pth         # FA 单模 warm-start
├── results_bimodal_t1_fa/best_bimodal.pth # 双模 baseline
├── labels.csv                             # 214 行 × {label_2/3/4/6}
├── cache/paired_112.pt                    # 5778 paired slices, 112² (~580 MB)
└── figures/                               # baseline 全套图
```
