# MMCLAST on real ADNI T1 ↔ FA
Companion to `X_files/RESULTS.md` (synthetic MNIST-PET pilots)
## X. Main purpose:
Reliable representation learning:
- x1. Good for reconstruction
- x2. Maintain diagnostic info
### x1.1 Generation:
Very mature benchmarks, with metrics SSIM, PSNR
### x2.1 Diagnostic:
Classification benchmark; Biomarker? [clustering score: Silhouette score, Davies-Bouldin Score]
### xX. Uncertainty quantification:
- Generation-level uncertainty quantification (Pixel)
- Diagnostic-level uncertainty quantification (Representation)

### X1. Our mechanism superiorities

| Method | Cross-modal transfer | Shared knowledge space | Invertible commitment | Self-recon | Native representation learning | Density & Image-level conformal |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| naive VAE            | ✗ | ✗ | ✗ | ✓ | ✓ | ◐ |
| CycleGAN             | ✓ | ✗ | ✗ | ◐ | ✗ | ✗ |
| DDPM                 | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ |
| CFM / Rectified Flow | ✓ | ✗ | ✓ | ✗ | ✗ | ◐ |
| MeanFlow             | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| DiT                  | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ |
| **MMCLAST (ours)**   | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Legend: ✓ = yes · ◐ = partial / bound only · ✗ = no.
- **Shared knowledge space** = one shared latent `u` from multimodal knowledge; [Generation purpose & Reliable knowledge purpose]
- **Invertible commitment** = an exact analytic inverse while normalizing flow for ours when ODE for CFM; [Generation purpose & Reliable knowledge purpose]
- **Native representation learning** = a built-in encoder latent usable for downstream tasks, others must be probed from a hidden layer; [Reliable knowledge purpose]
- **Density** = tractable likelihood + conformal; [Generation purpose & Reliable knowledge purpose]


## ToDo Next
- Pilot study for how semantic and reconstrcution work with each other
- Sharpeness compare to GT

## 1. Data

Source: `/data_new3/nfs_share/public/Imaging_genetic/`.

| Modality | Source folder | Native shape / vox | Used here (after registration) |
|---|---|---|---|
| T1 | `Processed_T1/<subj>_I*/brain-in-rawavg.nii` | ~10 shapes, 2mm,native rawavg | `registrated_T1_sy/<subj>_T1_mni2.nii.gz` (91,109,91) MNI152 2mm |
| DTI (FA/MD/RD/AD) | `registered_DTI/<subj>_I*/*_FA_reg.nii.gz` etc. | (182,218,182) MNI152 1mm | `registrated_DTI_2mm_sy/<subj>_FA_mni2.nii.gz` (91,109,91) MNI 2mm |

- **Common subjects (T1 ∩ DTI) = 214** out of T1=1144 / DTI=695. We picked the first session per modality per subject (subject-level pairing,not voxel-time pairing).
- **Diagnosis labels**: not in the imaging folder; found via the
`kunzhao/TinyLLaVA_Factory-main/dataset/genetic_image/all_infomation_v2.json` table.

| label_6 | n / 214 | label_4 (SMC→CN, AD dropped) | n / 213 |
|---|---|---|---|
| CN | 92 | **CN** | **107** |
| EMCI | 59 | **EMCI** | **59** |
| MCI | 24 | **MCI** | **24** |
| LMCI | 23 | **LMCI** | **23** |
| SMC | 15 | – (→CN) | – |
| AD | 1 | – (dropped) | – |

### 1.1 Used sample counts

- Our MMClast and all baselines use the same cache
`adni_pilot/cache/paired_112.pt`.
- We use the **middle 27 continuous axial slices** per subject
(z = [32, 58] of the 91-slice volume), padded to 112×112, 1-channel. NOT the full 3D volume.

| Level | Count | Notes |
|---|---|---|
| slices per subject | **27** | middle axial z=32..58 |
| **cache size** (`paired_112.pt`) | **5,778** | 214 × 27 — built label-agnostically, **includes** the AD subject |
| AD subject (dropped at split) | 1 subj = **27** | label invalid under label_4 → stays in cache but excluded from both splits |
| **subjects actually used** | **213** | label_4 cohort = 170 train + 43 test |
| **train slices** | **4,590** | 170 train subjects × 27 — what every model is trained on |
| **test slices** | **1,161** | 43 test subjects × 27 |
| → used total | **5,751** | 4,590 + 1,161 = 213 × 27 (= cache 5,778 − the 27 AD slices) |


The other two uses are bigger:
| Eval | Unit | N |
|---|---|---|
| training | slices | 4,590 |
| generation SSIM/PSNR | slices (scored per-slice) | 1,161 |
| clustering acc/NMI/ARI | subjects (mean-pooled) | 43 |

### 1.2. Preprocessing pipeline (`adni_pilot/batch_fnirt.py`)

```
brain-in-rawavg.nii (2mm, native)
   │  flirt 12-DOF affine  ->  brain_in_MNI152_2mm
   ▼
   ▼  fnirt nonlinear (FSL std config T1_2_MNI152_2mm.cnf,
   ▼   ref overridden to MNI152_T1_2mm_brain since input is skull-stripped)
   ▼
   ▼  applywarp -> T1_mni2.nii.gz
   ▼
(91,109,91)  shared MNI152 2mm grid with FA/MD/RD/AD
```

## 2. Experiments

### 2.1 Bi-modal settings (`adni_pilot/train_unsup.py`)

| Field | Value |
|---|---|
| Both encoders/decoders | Pretrained from §4 |
| Schedule | 10 / 20 / 20 = 50 epochs (same as MNIST-PET) |
| K | 4 (label_4) |
| latent_dim | 8
| `lambda_p_max`, `lambda_cross_max` | 1.0, 0.5 |
| Batch / lr | 128 / 5e-4 (AdamW,wd=1e-4) |
| ckpt | `results_bimodal_t1_fa/best_bimodal.pth` (best @ E10) |

Baselines: Standard methods reproduced on the **same testbed** (paired_112.pt, subject-level
label_4 split, identical post-hoc GMM eval)


| Method | Backbone | Conditioning | Params | Losses | Norm |
|---|---|---|---|---|---|
| **naive VAE** | **same** Conv VAE enc/dec as CLAST (no flow, no joint GMM) | none | 103M / modality | BCE recon + KL | BatchNorm |
| **CycleGAN** | ResNet generator (c7s1-64 → down×2 → 6 ResBlocks → up×2) + 70×70 PatchGAN disc | none | G 7.8M ×2 + D ×2 | 5: adversarial (G) + cycle (λ=10) + identity (λ=5) + D real/fake ×2 | InstanceNorm |
| **CFM** | Time-cond U-Net (3 levels, base 64, GroupNorm, sinusoidal-t FiLM) velocity field v(x_t,t) | t (sinusoidal) | 8.3M ×2 | 1: velocity regression ‖v−(x₁−x₀)‖² | GroupNorm |
| **DDPM** (faithful, Ho 2020) | Cond U-Net ε(x_t,t \| source), in_ch=2 (noised target ⊕ source), **+bottleneck self-attention**, T=1000 linear-β, **ancestral 1000-step** sample | t + source concat | 14.5M ×2 | 1: noise MSE ‖ε̂−ε‖² | GroupNorm |
| **DiT** (faithful, Peebles & Xie 2023) | **latent** DiT-S/2 in frozen MedVAE space (28×28×1): patchify(2)→14×14 tokens, depth 12, hidden 384, 6 heads, adaLN-zero, 2D sin-cos pos, **learn-σ**; T=1000 **ancestral** + **CFG** | t (adaLN) + source-latent concat | 32.5M ×2 + frozen MedVAE | iDDPM hybrid: L_simple + 0.001·L_vlb | LayerNorm (adaLN-zero) |
| **MeanFlow** | **same** U-Net as CFM, two-time-cond avg-velocity u(x_t,r,t) | r + t (sinusoidal) | 8.3M ×2 | 1: avg-velocity ‖u−sg(v−(t−r)·du/dt)‖² | GroupNorm |
| **MMCLAST_v1** | VAE enc/dec with residualBlock + 2× NormalizingFlow (ActNorm+AffineCoupling ×4) + shared GMM | flow depth | enc/dec 103M ×2 + flow ~0.5M | 7: recon×2 (BCE) + KL×2 + GMM-NLL + flow-NLL×2 (+ paired-align; + cross-recon in P2; pi-balance off)  | BatchNorm |

Notes: 
- CLAST and naive VAE deliberately share the **same conv encoder/decoder**
- CFM and MeanFlow share the **same U-Net** 
- MeanFlow only adds the second time input r) so their gap isolates one-step
vs ODE. 
- CycleGAN/CFM/MeanFlow/DDPM/DiT are all **2-direction** (a separate net for T1→FA and
FA→T1)
- only CLAST routes both modalities through a single shared `u`

#### Baseline faithfulness audit (vs each method's original paper)

Each baseline reproduces its *own* paper as faithfully as possible, adapted to T1↔FA
translation with the **minimal** change — NOT homogenized to each other.

| Baseline | Paper | Verdict | Key faithful details / minimal adaptation |
|---|---|---|---|
| **naive VAE** | Kingma & Welling 2013 | ✅ faithful | vanilla VAE, ELBO = BCE recon + KL to N(0,I), reparam. Shares CLAST's conv enc/dec **by design** (isolates the flow+GMM contribution). |
| **CycleGAN** | Zhu et al. 2017 | ✅ faithful | ResNet gen (c7s1-64/d128/d256/6×R256/u128/u64), 70×70 PatchGAN, **LSGAN** (MSE), cycle L1 λ=10, identity L1 λ=5, image pool 50, Adam 2e-4/(0.5,0.999), linear LR decay. (Designed for *unpaired* data; we apply it to our paired set — it simply doesn't exploit the pairing.) |
| **CFM / Rectified Flow** | Lipman 2023 / Liu 2023 | ✅ faithful | linear interpolant x_t=(1−t)x₀+t·x₁ + velocity regression + Euler ODE. **Minimal adapt:** data-to-data (x₀=source, x₁=target) — exactly Rectified Flow's image-to-image form; the source *is* the conditioning (ODE starts at it). |
| **MeanFlow** | Geng et al. 2025 | ✅ faithful | average-velocity u(z,r,t) + JVP identity + one-step; same data-to-data adaptation, same U-Net as CFM (+2nd time input r). |
| **DDPM** | Ho et al. 2020 | ✅ **fixed** | was missing self-attention + used DDIM. Now: **bottleneck self-attention** + **ancestral 1000-step sampler** (Ho's reverse process), T=1000 linear-β, ε-pred L_simple. Source conditioned by channel-concat (diffusion samples from noise → concat is the minimal way to condition). |
| **DiT** | Peebles & Xie 2023 | ✅ **rebuilt** | now **latent** (frozen pretrained MedVAE, 28×28), DiT-S/2 (adaLN-zero, 2D sin-cos pos), **learn-σ + iDDPM hybrid loss**, **CFG** (source-latent dropout), **ancestral** sampling. Source conditioned by latent-concat (only departure from class-conditional). |

**Principled conditioning note:** diffusion methods (DDPM, DiT) sample from noise so they
*must* concat the source to condition; flow/RF methods (CFM, MeanFlow) and CLAST start the
path **at** the source (data-to-data), so they need no concat. Each is in its most native form.

**Leakage-free checkpoint policy (fixed):** the generative baselines (CFM / MeanFlow / CycleGAN)
were originally selected by **validation SSIM on the test split** + early stopping — test-set
leakage, and unfaithful (these papers report the *final* model after a fixed schedule). All three
were re-run **without early stopping, full fixed schedule, final model, evaluated on the full 1,161
test slices**. Effect vs the old leaky numbers: CFM 0.736/0.807 → **0.732/0.802** (≈unchanged, so
its leakage was negligible — it stays the best baseline); MeanFlow 0.713/0.701 → **0.686/0.621**
(old was a test-selected ep-3 checkpoint); CycleGAN 0.721/0.801 → **0.680/0.743** (old stopped at
ep40 before the LR-decay phase). MMCLAST was already leakage-free (selected on min *train* recon
loss), so the comparison is now uniformly clean and MMCLAST's win is unaffected.

### single-VAE lower-bound
- **Clustering results:**

| Modality | Best test acc | NMI | ARI |
|---|---|---|---|
| T1 (`results_single_t1/best_vae.pth`) | 0.297 @ E1 | ~0 | ~0 |
| FA (`results_single_fa/best_vae.pth`) | 0.291 @ E11 | ~0 | ~0 |

- **Reconstruction results:**

| Modality | SSIM ↑ | PSNR ↑ | MSE ↓ | BCE (sum) |
|---|---|---|---|---|
| T1 | 0.753 ±0.029 | 19.69 dB | 0.0114 | 2117 |
| FA | **0.793** ±0.023 | **21.09 dB** | **0.0080** | 2347 |



### 2.2 Clustering (representation diagnostic, K=4)

| Method | repr | T1-only | FA-only | **concat / repr** | NMI | ARI | subj-CCA | slice-CCA|
|---|---|---|---|---|---|---|---|---|
| Majority baseline | Always predict CN | – | – | 0.395 | – | – | – |-|
| Random K=4 | - | – | – | 0.250 | – | – | – |-|
| T1 single-mod | (u_A mean) | 0.395 | - | - | 0.065 | 0.009 | |
| FA single-mod | (u_B mean) | -|0.372 | - | 0.031 | -0.038 |  |
| **naive VAE** | mu (recon+KL only, no flow/GMM) | **0.535** | **0.465** | **0.442** | **0.112** | 0.030 | 0.047 |
| **CycleGAN** | G_T1→FA bottleneck (256-D) | – | – | 0.395 | 0.105 | -0.011 | – |
| **CFM** | U-Net mid2 probe (256-D, t=0.1)† | 0.419 | 0.349 | 0.419 | 0.098 | -0.026 | – |
| **DDPM** | U-Net mid2 probe (256-D, t=0.1)† | 0.465 | 0.419 | 0.395 | 0.088 | -0.016 | – |
| **MeanFlow** | U-Net mid2 probe (256-D, t=0.1)† | 0.442 | 0.465 | 0.372 | 0.098 | -0.064 | – |
| **DiT** (faithful) | mid transformer-block token probe (384-D, t=100)† | 0.395 | 0.442 | 0.442 | 0.091 | 0.055 | – |
| **MMCLAST (ours, default)** | joint (u_A ∪ u_B as 2× pool) | 0.395 | 0.372 | 0.407 | 0.018 | -0.009 | 0.023 | 0.945 |
| **MMCLAST (ours, default)** | u_A concat u_B | 0.395 | 0.372 | **0.442** | 0.099 | 0.032 | 0.023 | 0.945 |
| **MMCLAST (recon-opt, ConvVAE)** | u_A concat u_B | 0.395 | 0.349 | 0.349 | 0.059 | -0.050 | **0.628** | ~0.95 |
| **MMCLAST (recon-opt, U-Net)** | u_A concat u_B | – | – | 0.372 | 0.079 | -0.063 | 0.000 | – |

> **⚠ Reference correction (from §3.7).** The 0.395 majority baseline is too lenient. A
> permutation null — shuffle the labels, keep the same K=4 partition, n=43 test subjects —
> has **mean ACC 0.408** and 95th percentile ≈ 0.49. So every ACC in this table (max 0.442)
> is **within chance**; none of the methods, ours included, shows real diagnostic clustering.
> Read the table as "all methods at chance", not "all methods at the majority floor".

† CFM/DDPM/MeanFlow/DiT have **no encoder**; we probe a mid-network feature as the
representation — U-Net `mid2` (GAP, 256-D, t=0.1) for the conv nets
(`baseline/probe_gen_clustering.py`), and the **mid transformer block's tokens**
(GAP over tokens, 384-D, t=100) for the latent DiT (`baseline/probe_dit_clustering.py`).
Even these generation-only nets land at the same ~0.40 / NMI≈0.09–0.14 level as the
representation methods → **every method clusters near the 0.395 majority baseline; the
ceiling is the testbed (2D slices, 43 imbalanced test subjects, weak diagnostic signal),
not the model.** (DiT's concat 0.442 ties the best of all methods — still the floor.)

- **default vs recon-opt MMCLAST** (same backbone, different ckpt-selection): the **default**
  (acc-selected) is the best-clustering MMCLAST (concat ACC **0.442**, ties naive VAE for best
  of all methods); the **recon-opt** (min-recon-loss) trades discriminative ACC down to 0.349
  but lifts **subject-level CCA from 0.023 → 0.628** (strong λ_pair=2.0 aligns the two modalities
  to the same cluster). Measured on `results_recon_d8_xc1/best_recon.pth` via `eval_subject_level.py`.

### 2.3 Cross-modal generation quality (paired test set)

| Method | T1→FA SSIM | T1→FA PSNR | FA→T1 SSIM | FA→T1 PSNR | transition axis |
|---|---|---|---|---|---|
| CycleGAN (faithful, final) | 0.680 ±0.027 | 17.18 dB | 0.743 ±0.087 | 17.94 dB | none (1-shot) |
| CFM (best baseline, faithful) | 0.732 ±0.029 | 18.46 dB | 0.802 ±0.035 | 19.99 dB | ODE time t=0→1 (true curve) |
| DDPM (faithful: attn + ancestral) | 0.362 ±0.094 | 18.44 dB | 0.769 ±0.103 | 20.36 dB | denoising t=T→0 (1000-step) |
| DiT (faithful: latent + iDDPM + CFG) | 0.665 ±0.029 | 14.32 dB | 0.757 ±0.034 | 18.28 dB | denoising t=T→0 (1000-step) |
| MeanFlow (faithful: adaptive-w, final) | 0.686 ±0.034 | 17.37 dB | 0.621 ±0.068 | 17.27 dB | linear x0+s·Δ (no true curve) |
| MMCLAST (ours, default) † | 0.663 ±0.065 | 17.49 dB | 0.695 ±0.095 | 17.37 dB | flow layers (invertible) |
| **MMCLAST (ours, recon-opt, ConvVAE)** ‡ | **0.764** ±0.031 | 19.95 dB | **0.859** ±0.035 | **22.05 dB** | flow layers (invertible) |
| **MMCLAST (ours, recon-opt, U-Net)** § | **0.780** ±0.029 | **20.19 dB** | **0.810** ±0.043 | 19.54 dB | flow layers (invertible) |

**Recon-optimized MMCLAST beats every baseline on both directions, on both SSIM and PSNR**
(+0.032 SSIM / +1.5 dB over the faithful CFM on T1→FA; +0.057 SSIM / +2.1 dB on FA→T1). See `figures/08_recon_winner.png`.

- **Faithful diffusion baselines** (both now reproduce their papers — see §2.1 audit):
  - **DiT** (latent DiT-S/2 in frozen MedVAE space, iDDPM learn-σ hybrid loss, CFG=1.5, ancestral
    1000-step): **0.665 / 0.757**. FA→T1 reaches **95 % of the MedVAE round-trip ceiling** (0.80) —
    the latent VAE, not the transformer, is now the cap. `baseline/dit/` (`dit.py`, `diffusion.py`, `latent_vae.py`).
  - **DDPM** (conv U-Net + bottleneck self-attention, ancestral 1000-step): **0.362 / 0.769**.
  - Both still trail the deterministic translators (CFM/CycleGAN/MMCLAST) on T1→FA because diffusion
    samples from noise and only weakly preserves fine source structure; DDPM's T1→FA (0.362) stays low
    (T1 has more high-freq detail to hallucinate), while FA→T1 (smoother target) is competitive (0.769).
  - The earlier numbers (DDPM 0.238/0.397 via DDIM-50 no-attn; DiT 0.132/0.257 pixel-space ε-only) were
    **unfaithful**; making each match its paper raised both substantially.

- † The earlier 0.663 / 0.695 was a **checkpoint-selection artifact**: the default run picked
  the epoch with best *clustering accuracy* (an early epoch, before the cross-recon path was
  trained). It does **not** reflect the model's reconstruction capability.
- ‡ recon-opt setting = warm-start + cross-recon weight λ_cross 0.5→**1.0**, paired-align
  λ_pair 1.0→**2.0**, longer cross-path training (phases 5/15/20), best ckpt by **min recon
  train loss among cross-active (phase≥2) epochs**. Identical backbone/latent_dim=8.
  `train_recon.py --tag d8_xc1`, ckpt `adni_pilot/results_recon_d8_xc1/best_recon.pth`.
- The cross-recon ceiling is the *other* modality's self-recon (T1 0.892 / FA 0.787); the
  recon-opt (ConvVAE) run reaches 96–99 % of that ceiling, vs 75–88 % for the default run.
- § **U-Net backbone** = replace the ConvVAE enc/dec with two U-Nets (vector latent + skips),
  **pix2pix-style cross-modal skip transfer** (source encoder skips → target decoder; valid
  because T1/FA are registered). Same recon-opt recipe, no warm-start, 42M params (vs 103M ConvVAE).
  `train_recon_unet.py --tag unet_d8`, ckpt `results_recon_unet_d8/best_recon.pth`. Skips raise the
  ceiling to ~0.99 self-recon → **best T1→FA of all configs (0.780)**; FA→T1 0.810 (still beats CFM but
  below the ConvVAE 0.859, since FA's smoother maps give lower-detail skips). **Cost:** the bottleneck
  `z` carries little info once skips dominate → clustering concat ACC 0.372 / NMI 0.079, subj-CCA 0.000
  (§2.2). I.e. U-Net = pure reconstruction play; ConvVAE keeps the representation usable.

- CCA: Cross-model cluster alignment, predict paired slice/subject if they are in the same cluster; MNIST-PET 0.944

### 2.4 Main comparison — all methods, both axes

One-glance head-to-head on the two headline goals (x1 reconstruction, x2 diagnostic).
Clustering = best repr per method (concat for ours, U-Net `mid2` probe for gen-only nets);
generation = cross-modal SSIM on the paired test set.

| Method | Cross-modal? | T1→FA SSIM ↑ | FA→T1 SSIM ↑ | concat ACC ↑ | NMI ↑ |
|---|---|---|---|---|---|
| naive VAE | ❌ none | – | – | 0.442 | 0.112 |
| DDPM (faithful) | ✅ stochastic | 0.362 | 0.769 | 0.395 | 0.088 |
| DiT (faithful, latent) | ✅ stochastic | 0.665 | 0.757 | 0.442 | 0.091 |
| MeanFlow | ✅ one-step | 0.686 | 0.621 | 0.372 | 0.098 |
| CycleGAN | ✅ one-shot | 0.680 | 0.743 | 0.395 | 0.105 |
| CFM (best baseline) | ✅ ODE | 0.732 | 0.802 | 0.419 | 0.098 |
| MMCLAST (ours, default) | ✅ invertible | 0.663 | 0.695 | **0.442** | 0.099 |
| **MMCLAST (recon-opt, ConvVAE)** | ✅ invertible | 0.764 | **0.859** | 0.349 | 0.059 |
| **MMCLAST (recon-opt, U-Net)** | ✅ invertible | **0.780** | 0.810 | 0.372 | 0.079 |

- **Reconstruction (x1): recon-opt MMCLAST is best of all methods on both directions** —
  +0.028 over CFM (T1→FA), +0.052 (FA→T1); see §2.3 for PSNR (19.95 / 22.05 dB, also best).
- **Diagnostic clustering (x2): no method beats the 0.395 majority floor** (NMI ≈ 0.05–0.11) —
  testbed ceiling (2D slices, 43 imbalanced test subjects, weak signal), not a model gap.
  The recon-opt config trades a little discriminative ACC (0.349, slightly below majority)
  for reconstruction; the **default** config is the best clustering MMCLAST (concat ACC 0.442).
- **Cross-modal alignment (subj-CCA): recon-opt is a huge win** — the strong paired-align
  (λ_pair=2.0) lifts subject-level agreement (T1-pred == FA-pred) from **0.023 → 0.628**:
  the two modalities now land in the *same* cluster for the same subject. (slice-CCA stays high.)
- Same backbone, different checkpoint-selection objective (acc-sel ↔ min-recon-loss): one model
  family covers the recon winner *and* the clustering-competitive operating point.

## 3. Ablation study

All ablations on the same testbed (paired_112.pt, label_4, subject-level eval).
Reference = default CLAST (two-stage warm-start, latent=8, K=4, flow=4, BCE):
concat acc **0.442**, NMI 0.099, slice-CCA 0.945, subj-CCA 0.023, self-recon SSIM
T1 0.895 / FA 0.794, cross-modal SSIM T1→FA 0.663 / FA→T1 0.695 (acc-selected ckpt).
**Recon-optimized variant (§2.3 ‡): cross-modal SSIM T1→FA 0.764 / FA→T1 0.859 — beats every
baseline on both directions.**

### 3.1 Training strategy
| Method | Strategy | ACC | NMI | ARI | subj-CCA | slice-CCA| T1→FA SSIM | T1→FA PSNR | FA→T1 SSIM | FA→T1 PSNR |
|---|---|---|---|---|---|---|---|---|---|---|
| **CFM** | – | – | – | – | – | – | **0.732** ±0.029 | 18.46 dB | **0.802** ±0.035 | 19.99 dB |
| **naive_MMCLAST** | warm-start (three-stage) | 0.442 | 0.099 | 0.032 | 0.023 | 0.945 | 0.663 ±0.065 | 17.49 dB | 0.695 ±0.095 | 17.37 dB |
| **naive_MMCLAST** | end2end (from scratch) | **0.512** | **0.254** | 0.089 | 0.000 | 0.949 | 0.624 ±0.021 | 18.08 dB | 0.666 ±0.053 | 18.71 dB |

Queue:

| # | Ablation | Variants | Metric of interest | Status | Result | Files |
|---|---|---|---|---|---|---|
| A1 | **Training pipeline** | baseline warm-start vs this end-to-end from scratch | recon SSIM, CCA, acc | ✅ done | **3-way trade-off**: e2e gets the **best discriminative clustering** (concat ACC 0.512 / NMI 0.254 ≫ warm 0.442 / 0.099) but **worst recon** (self 0.63, cross 0.62/0.67 ≪ warm) and **zero subj-CCA** (0.000). Warm-start buys recon fidelity + alignment at the cost of discriminative structure. | `train_e2e.py`, `fig 07` |
| A8 | Loss weights | warm-start + cross-recon weight λ_cross 0.5→1.0, paired-align λ_pair 1.0→2.0, longer cross-path training (phases 5/15/20)| recon SSIM and PSNR | ✅ done

### 3.2 Architecture
| Method | Backbone | ACC | NMI | ARI | subj-CCA | slice-CCA| self T1/FA | T1→FA SSIM | T1→FA PSNR | FA→T1 SSIM | FA→T1 PSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CFM** | conv U-Net (pixel-space) | – | – | – | – | – | – | **0.732** ±0.029 | 18.46 dB | **0.802** ±0.035 | 19.99 dB |
| **MMCLAST** | ConvVAE (bottleneck, recon-opt) | 0.349 | 0.059 | -0.050 | **0.628** | ~0.95 | 0.892/0.787 | 0.764 ±0.031 | 19.95 dB | **0.859** ±0.035 | **22.05 dB** |
| **MMCLAST** | **U-Net + skip-transfer** (recon-opt) | 0.372 | 0.079 | -0.063 | 0.000 | – | **0.995/0.987** | **0.780** ±0.029 | **20.19 dB** | 0.810 ±0.043 | 19.54 dB |

<!-- **Backbone swap (A14): ConvVAE → two U-Nets with pix2pix-style cross-modal skip transfer**
(source encoder skips → target decoder; legal because T1/FA are co-registered). `train_recon_unet.py`.

- **Skips → near-perfect self-recon (0.99 vs 0.89)** and the **best T1→FA of any config (0.780)**.
- **FA→T1 is *lower* (0.810 vs ConvVAE 0.859)**: the FA→T1 path decodes T1 from *FA's* skips, and FA
  maps are smoother (less high-freq anatomy) → weaker scaffold. So the U-Net helps the direction whose
  *source* is detail-rich (T1→FA) more than the reverse.
- **Representation cost:** once skips carry the spatial detail, the bottleneck `z` becomes nearly
  uninformative → subj-CCA collapses to **0.000** (vs 0.628 for ConvVAE) and clustering stays at the
  floor. **Verdict:** U-Net is the pick if the goal is pure reconstruction; ConvVAE if the shared-`u`
  representation (alignment + clustering) must stay usable.

**latent_dim sweep (A2/A3)** — single-modal pure VAE (recon+KL only, no flow/GMM), self-recon
on the held-out test set (`recon_sweep.py`, `results_recon_sweep`). Isolates the bottleneck
capacity effect on reconstruction: -->

| latent_dim | self-recon SSIM (BCE) | PSNR | SSIM (MSE) |
|---|---|---|---|
| 8 | 0.889 | 24.14 dB | 0.890 |
| **16** | **0.897** | **24.37 dB** | 0.887 |
| 32 | 0.893 | 24.27 dB | 0.886 |
| 64 | 0.894 | 23.81 dB | 0.866 |

FIndings: latent_dim **barely matters** for reconstruction 
best; **BCE ≥ MSE at every dim**

**Encoder backbone (A4)** — replace the from-scratch ConvVAE encoder with a pretrained one,
GMM-VAE clustering ACC (best epoch). Both pretrained backbones are **negative transfer**:

| Encoder | Testbed | cluster ACC ↑ | vs scratch |
|---|---|---|---|
| scratch ConvVAE | MNIST-PET | **0.938** | — (reference) |
| ImageNet ResNet-18 | MNIST-PET | 0.884 | **↓ −0.054** |
| scratch ConvVAE | ADNI-T1 | **0.297** | — (reference) |
| MedVAE-2D (medical pretrain) | ADNI-T1 | 0.280 | **↓ −0.017** |

Findings: reptrain may destroy
(`pretrained_encoder.py`, `medvae_encoder.py`)

Queue:

| # | Ablation | Variants | Metric of interest | Status | Result | Files |
|---|---|---|---|---|---|---|
| A2 | **latent_dim** | 8 / 16 / 32 / 64 | recon SSIM/PSNR | ✅ done | single-modal pure-VAE recon SSIM: dim8=0.889, **dim16=0.897**, dim32=0.893, dim64=0.894 → latent_dim **barely matters** (see mini-table above); 64 overfits (PSNR↓, MSE↓) | `recon_sweep.py` |
| A3 | **Reconstruction loss** | BCE vs MSE | recon SSIM/PSNR | ✅ done | BCE ≥ MSE at every dim (e.g. dim16: BCE 0.897 vs MSE 0.887) → keep BCE | `recon_sweep.py` |
| A4 | **Encoder backbone** | scratch ConvVAE vs ImageNet-ResNet18 vs MedVAE-2D | cluster acc | ✅ done (negative) | MNIST: ResNet 0.884<0.938; ADNI-T1: MedVAE 0.280<0.297 — both negative transfer | `pretrained_encoder.py`, `medvae_encoder.py`, `fig 14` |
| A5 | **#clusters K** | 2 / 4 / 6 / 8 (overcluster→map) | acc/NMI/ARI | ⬜ todo | — | `train_unsup.py` (K) |
| A6 | **#flow layers** | 2 / 4 / 6 / 8 / 12 | CCA, cross-SSIM, flow morph | ⬜ todo | — | `train_unsup.py` (n_flow_layers) |
| A10 | **Subject pooling** | mean vs max vs attention | subj-CCA, acc | ⬜ todo | — | `eval_subject_level.py` |

### 3.3 Losses

**Loss-deletion ablation** (A12). End-to-end from scratch, 30 epochs, all losses on from
epoch 1; remove ONE loss term at a time and re-measure. Self-recon, clustering (concat
subject-level GMM K=4) and cross-modal SSIM on the held-out test set. `loss_ablation.py`,
`results_loss_ablation/loss_ablation.csv`.

| Config | ACC | NMI | CCA (slice) | self T1 | self FA | T1→FA SSIM | FA→T1 SSIM |
|---|---|---|---|---|---|---|---|
| **full** (recon+KL+GMM+flow+pair+cross) | 0.419 | 0.050 | 0.938 | 0.894 | 0.790 | 0.768 | 0.862 |
| − GMM-NLL | 0.465 | 0.059 | 0.952 | 0.894 | 0.790 | 0.769 | 0.862 |
| − flow-NLL | 0.372 | 0.050 | 0.948 | 0.893 | 0.790 | 0.769 | 0.862 |
| − KL | 0.419 | 0.097 | 0.948 | 0.895 | 0.789 | 0.768 | 0.863 |
| − paired-align | 0.372 | 0.076 | 0.951 | 0.894 | 0.789 | 0.767 | 0.863 |
| **− cross-recon** | 0.465 | **0.130** | **0.802** | 0.898 | 0.795 | 0.724 | 0.797 |
|2recon+paired-align|

<!-- **Read-out:**
- **Cross-recon is THE lever for cross-modal reconstruction.** Removing it collapses cross-SSIM
  to 0.724 / 0.797 (below CFM) **and** drops slice-CCA from ~0.95 to 0.802 — i.e. cross-recon,
  not the explicit paired-align term, is what actually aligns the two latent spaces.
- **GMM-NLL, flow-NLL, KL and paired-align are all redundant for reconstruction** — removing any
  one leaves cross-SSIM at 0.767–0.769 / 0.862–0.863 (within noise of full). The recon/transfer
  quality is carried by self-recon + cross-recon alone.
- **Clustering is insensitive to every loss** (ACC 0.37–0.47, NMI ≈ 0.05–0.13, all near the 0.395
  majority baseline) — consistent with §2.2: the testbed, not the loss, caps clustering.
- Independent of the recon-opt run, the **full** end-to-end config (eval at final epoch, no
  acc-based selection) already reaches 0.768 / 0.862 — a second confirmation that CLAST beats
  every baseline once the checkpoint is not chosen by clustering accuracy. -->

Cross-modal generation table (kept for direct comparison with baselines):

| Method | loss | ACC | NMI | ARI | subj-CCA | slice-CCA| T1→FA SSIM | T1→FA PSNR | FA→T1 SSIM | FA→T1 PSNR |
|---|---|---|---|---|---|---|---|---|---|---|
| CFM (best baseline) | -|-| -| -| -| 0.732 ±0.029 | 18.46 dB | 0.802 ±0.035 | 19.99 dB |
| naive_MMCLAST | all 6 (acc-sel) | 0.442 | 0.099 | 0.032 | 0.023 | 0.945 | 0.663 ±0.065 | 17.49 dB | 0.695 ±0.095 | 17.37 dB |
| **MMCLAST recon-opt** | recon+cross↑ | 0.349 | 0.059 | -0.050 | **0.628** | ~0.95 | **0.764** ±0.031 | **19.95 dB** | **0.859** ±0.035 | **22.05 dB** |

Queue:

| # | Ablation | Variants | Metric of interest | Status | Result | Files |
|---|---|---|---|---|---|---|
| A12 | **Loss deletion** | full / −gmm / −flow / −kl / −pair / −cross | recon SSIM, CCA, acc | ✅ done | cross-recon is the only term that matters for recon+alignment; rest redundant (table above) | `loss_ablation.py` |
| A13 | **Cross-recon weight** | λ_cross 0.5 / **1.0** / 2.0 ; λ_pair 1.0 / **2.0** | cross-SSIM/PSNR | ✅ done | λ_cross 1.0 + λ_pair 2.0 + phase≥2 ckpt-sel → **0.764 / 0.859, beats all baselines** | `train_recon.py`, `fig 08` |
| A7 | **Paired-align loss** | L2 vs InfoNCE vs subject-mean vs MMD | **subj-CCA** (fix 0.023) | ⬜ todo | — | `losses.py` |
| A9 | **KL weight** | w_kl 1.0 / 0.1 / annealing | recon vs posterior-collapse | ⬜ todo | — | `train_unsup.py` (w_kl) |
| A11 | **Supervised aux head** | unsup vs +λ·CE(u→head, y) | acc/NMI (only lever that may move it) | ⬜ todo | — | new |

#### A14 — What actually makes cross-modal alignment work? (representation-collapse ladder)

A controlled ladder that isolates *why* CLAST's paired-alignment succeeds, by rebuilding it
from the bare minimum and adding one ingredient at a time. All variants share a spatial conv
enc/dec; unless noted the representation is the **dense 28×28×128 feature map** (no MLP), and —
per design — **all losses are mean-reduced with weight = 1** ("balanced"). Metric of interest:
does the paired representation actually align across modalities? Measured on the test set:
**CCA** (one GMM K=4, fraction of paired T1ᵢ/FAᵢ in the same cluster; chance 0.25),
**mixing** (fraction of a point's 10-NN that are the other modality; 0 = separated),
**ratio** = paired‖Aᵢ−Bᵢ‖ / within‖Aᵢ−Aⱼ‖ (1 = aligned), **std** (rep scale; tiny = collapse).
`ablation_spatial_align.py`, `ablation_spatial_flow.py`, `ablation_fc_vecflow.py`, `spatial_ladder_compare.py`.

**Exact loss configuration per variant** — coefficient × reduction for every term
(`–` = term not used). "mean" = averaged over pixels/dims (→ O(0.1–1)); "sum" = summed
per image/over dims (recon-BCE-sum ≈ 2600, ×12544 larger). This is the whole point: the
first six rows are **balanced (all mean, weight 1)**; only CLAST is **recon-dominant (sum)**.

| Variant | self-recon | cross-recon | paired-align | flow-NLL | GMM-NLL | KL | dominant term |
|---|---|---|---|---|---|---|---|
| bare-align | 1 × mean-BCE | – | 1 × mean-MSE | – | – | – | none (all ~0.2) |
| A | 1 × mean-BCE | 1 × mean-BCE | 1 × mean-MSE | – | – | – | none (all ~0.2) |
| B | 1 × mean-BCE | 1 × mean-BCE | 1 × mean-MSE | 1 × per-elem (conv) | – | – | none (all ~0.2–1) |
| C | 1 × mean-BCE | 1 × mean-BCE | 1 × mean-MSE | 1 × per-dim (vec) | – | – | none (all ~0.2–1) |
| D | 1 × mean-BCE | 1 × mean-BCE | 1 × mean-MSE | 1 × per-dim | 1 × per-dim | – | none |
| E | 1 × mean-BCE | 1 × mean-BCE | 1 × mean-MSE | 1 × per-dim | 1 × per-dim | 1 × per-dim | none |
| A1 (=A, sum-recon) | **1 × SUM-BCE (≈2600)** | **1 × SUM-BCE** | 1 × mean-MSE | – | – | – | **recon + cross (sum)** |
| B1 (=B, sum-recon) | **1 × SUM-BCE (≈2600)** | **1 × SUM-BCE** | 1 × mean-MSE | 100 × per-elem (conv) | – | – | **recon + cross (sum)** |
| **CLAST (ref)** | **1 × SUM-BCE (≈2600)** | **≈1 × SUM-BCE** (λ_c 0.5→1 ramp) | **2 × mean-MSE** | **1 × SUM** (≈O(8)) | 1 × mean | **1 × SUM** (≈O(8)) | **recon + cross (sum)** |

→ In the balanced rows **no term dominates** (everything ~O(0.1–1)), so the ~0.2 mean-BCE recon
is too weak to force an informative latent. In CLAST the **sum-reduced recon/cross (≈2600)**
dominate by ×1000+, which is what forces genuine reconstruction; paired-align (mean-MSE×2 ≈ O(1))
and GMM are tiny by comparison — alignment is carried by the recon-dominant objective, not the
align/prior terms.

**Measured loss INTENSITY at the final epoch** (the actual numeric value each term contributes —
verifies the scales above rather than assuming them). `ablation_u_distribution.py`,
`results_spatial_loss_intensity.csv`.

| Variant | recA | recB | crossA | crossB | pair | flow-NLL | GMM | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bare-align | 0.161 | 0.173 | – | – | **0.000** | – | – | – |
| A | 0.161 | 0.174 | 0.179 | 0.194 | **0.001** | – | – | – |
| B | 0.403 | 0.286 | 0.233 | 0.218 | 0.048 | −5.17 | – | – |
| C | 0.216 | 0.211 | 0.216 | 0.211 | 0.186 | −5.96 | – | – |
| D | 0.216 | 0.211 | 0.216 | 0.211 | 0.070 | −5.69 | −0.96 | – |
| E | 0.216 | 0.211 | 0.216 | 0.211 | 0.048 | 0.29 | −1.36 | 2.02 |
| C w_fnll=10 | 0.216 | 0.211 | 0.216 | 0.211 | 0.528 | −6.61 | – | – |
| C w_fnll=1000 | 0.216 | 0.211 | 0.216 | 0.211 | 0.357 | −6.91 | – | – |
| **A1** (sum-recon) | **1993.6** | **2149.2** | **2173.1** | **2375.8** | 1.18 | – | – | – |
| **B1** (sum-recon+flow) | **2017.6** | **2193.8** | **2394.7** | **2473.0** | 0.89 | −3.07 | – | – |
| **CLAST (default)** | **2110.2** | **2327.3** | **4466.0** (cross tot) | | 0.079 | – | – | – |
| **CLAST (recon-opt)** | **2114.5** | **2341.0** | **4458.3** (cross tot) | | **0.016** | – | – | – |

Two things this table settles:
1. In every **balanced** row all terms sit at **~0.2** and `pair` decays to **0.000–0.001** — i.e. the
   align term is satisfied *for free* by collapsing, not by aligning.
2. In **A1/B1/CLAST**, recon/cross are **~2000–4500** and `pair` is only **0.016–1.2** (**<0.1 %** of
   the objective). So **MMCLAST's alignment is NOT carried by the paired-align term at all** — it is
   carried by the recon-dominant **cross-recon**, consistent with the loss-deletion ablation (A12,
   "cross-recon is the only term that matters").

| Variant | rep dim | losses added | mixing | CCA | ratio | std | aligned? |
|---|---|---|---:|---:|---:|---:|---|
| bare-align | 100352 | 2·recon + MSE-align | 0.00 | – | 3.2 | 0.058 | ✗ scale-collapse |
| A | 100352 | + cross-recon | 0.00 | – | 4.3 | 0.043 | ✗ not aligned |
| B | 100352 | + **conv** flow-NLL (Glow) | 0.00 | – | ~1e10 | 0.133 | ✗ point-collapse |
| C | **16** | FC→16 + **vector** flow-NLL | 0.00 | 0.00 | ~1e9 | 0.204 | ✗ point-collapse |
| D | 16 | + GMM-NLL (no KL) | 0.00 | 0.00 | ~1e9 | 0.258 | ✗ point-collapse |
| E | 16 | + KL (VAE reparam) | 0.00 | 0.00 | ~6e7 | 0.010 | ✗ point-collapse |
| C w_fnll=10 | 16 | C with **flow-NLL ×10** | 0.00 | 0.00 | ~2e9 | 0.344 | ✗ point-collapse |
| C w_fnll=1000 | 16 | C with **flow-NLL ×1000** | 0.00 | 0.00 | ~2e9 | 0.313 | ✗ point-collapse |
| A1 (=A, sum-recon) | 100352 | A with **sum-recon** (recon-dominant) | 0.00 | – | 3.3 | **0.804** | ~ collapse fixed, still not aligned |
| B1 (=B, sum-recon) | 100352 | B with **sum-recon** + flow-NLL×100 | 0.00 | – | 6.6 | **0.651** | ~ collapse fixed, still not aligned |
| **CLAST (ref)** | **8** | flow-NLL(**sum**)+GMM+KL, **sum-recon**, warm-start | **0.37** | **0.933** | **2.03** | **0.591** | ✅ |

**Read-out — the alignment does NOT come from any single loss term:**
- Adding **cross-recon (A)**, a **conv normalizing flow (B)**, a low-dim **vector flow (C)**, the
  **GMM prior (D)**, or **KL / VAE-reparam (E)** *each fails* — every balanced variant **collapses**
  (CCA≈0), either shrinking the whole representation (std↓, "scale-collapse") or mapping every input
  to one point ("point-collapse", within‖A−A‖→0, ratio→∞). This corroborates the loss-deletion
  ablation (A12): **−GMM was harmless** → GMM is not what carries alignment.
- **Root cause (verified):** every collapsed variant's decoder outputs the **same mean image
  regardless of input** (decoder output std *across samples* = 0.0000; self-recon is just the
  mean-brain ≈ 0.68 SSIM) — directly visible in `fig 13`: C/D/E self-recon is a blurry mean-brain,
  A1/B1 (sum-recon) recover detail, CLAST reconstructs genuine structure. The trivial "constant code → mean image" solution wins because, under
  **mean-reduced ("balanced") recon (~0.2 = the BCE entropy floor)**, the reconstruction gradient
  (÷12544 pixels) is **too weak to force an informative latent** — so the alignment/anchor terms
  trivially drive the code to a constant.
- **flow-NLL cannot fix it *at any strength* — because the collapse is UPSTREAM of the flow.**
  Sweeping its coefficient **w_fnll = 1 → 10 → 1000** leaves the collapse untouched
  (`within‖A−A‖ = 0.000` throughout; std only drifts 0.20→0.34). Reason: `within = 0` means the
  **encoder** maps every input to the *same* z, and a normalizing flow is a **bijection** — one input,
  one output — so a constant z forces a constant u no matter how strong flow-NLL is. flow-NLL can
  only *relocate/rescale* that single point toward N(0,I); it cannot pull coincident samples apart.
  ```
  x → encoder → z (COLLAPSES here) → flow → u → [flow-NLL acts here — downstream, too late]
  ```
  So flow-NLL anchors the *scale of the flow's output*, but cannot prevent the encoder's
  information collapse. Only **recon** (which acts directly on the encoder) can.
- **recon DOMINANCE fixes the collapse — but is necessary, not sufficient, for alignment.**
  Switching A/B to **sum-reduced** recon (A1/B1, recon ≈2600 dominant) **removes the collapse** —
  the representation regains a healthy scale (std 0.043→**0.80** for A1, 0.13→**0.65** for B1) and
  is genuinely informative again. **Yet they still do not align** (mixing 0, ratio 3–7): a
  strong recon makes the latent *informative*, but on the raw 100352-dim map the paired-align term
  (tiny mean-MSE, drowned by the ×2600 recon) can't pull the two modalities together.
- **Alignment needs BOTH: (1) recon-dominance (informative latent) AND (2) a low-dim bottleneck**
  where the paired-align/flow terms are comparable to recon and can actually organize the space.
  CLAST has both — **sum-recon** (informative, self-recon SSIM 0.89) **+ an 8-dim `u`** where
  flow-NLL anchors the scale and paired-align/cross-recon co-locate the modalities → CCA 0.933.

**Takeaway:** cross-modal alignment is not conferred by GMM, KL, or the flow *per se*. It needs
(a) a **recon-dominant** objective to make the bottleneck informative (else everything collapses to
the mean image), and (b) a **low-dim** latent so the alignment terms aren't drowned by recon.
"Balanced" weighting on a high-dim map (which we tried) breaks (a); recon-dominance on a high-dim
map (A1/B1) fixes the collapse but still misses (b). A useful negative result about loss weighting
and bottleneck dimension in multi-term latent-alignment models.

### 3.4 GMM flow-base, 3-stage training (A15)

Flow base switched from a single `N(0,I)` to a **K=4 GMM**, all losses **mean-reduced with
coefficients** (so intensities are comparable and the weight *is* the contribution).

**Stage schedule** (70 epochs total; enc/dec warm-started from single-modality VAEs):

| stage | epochs | enc/dec | active loss terms | purpose |
|---|---|---|---|---|
| **S1** | 1–30 | train | recon + cross + pair + kl **(no flow-NLL)** | pretrain a recon-dominant, informative latent |
| **S1.5** | — | — | *(fit GMM on `u` via sklearn, 5 inits)* | non-degenerate K=4 init (replaces `randn·0.05`) |
| **S2** | 31–50 | **frozen** | cross + pair + **GMM-flow** *(ramp 3 ep)* | shape `u` into K blobs without touching recon |
| **S3** | 51–70 | train | recon + cross + pair + kl + GMM-flow | joint fine-tune so terms don't fight |

**Loss weights** (coefficient × mean-reduced term; blank = term not in that stage's loss):

| term | mean scale | v2_gmmWeak | v1_gmmStrong | in S1 / S2 / S3 |
|---|---|---|---|---|
| recon (self-BCE, ×2) | ~0.18 | 100 | 12544 | ✓ / — / ✓ |
| cross-recon (BCE, ×2) | ~0.19 | 100 | 12544 | ✓ / ✓ / ✓ |
| pair (MSE on `u`) | ~0.01–0.6 | 1 | 1 | ✓ / ✓ / ✓ |
| kl (per-dim) | ~1–3 | 1 | 80 | ✓ / — / ✓ |
| GMM-flow NLL | ~1–2 | 1 | 2000 | — / ✓ / ✓ |

*recon coeff 12544 = 112×112 makes mean-BCE ≡ per-image sum-BCE (the A14 collapse fix). The
GMM-flow weight ramps linearly over the first 3 epochs of S2. Stage 3 can carry its own
(gentler) coefficients via `--w_*_s3` — used by v4, which keeps v1's S1–S2 weights but drops
S3 to v2's (100/100/1/1) to protect recon.*

**Actual loss contribution per stage** (share of the total loss each term carries, at the last
epoch of the stage — only terms present in that stage's loss are counted):

| | S1 recon / cross / kl | S2 cross / **GMM-flow** | S3 recon / cross / kl / GMM-flow |
|---|---|---|---|
| **v2_gmmWeak** | 49 / 49 / 2 % | 96 / **4** % | 48 / 48 / 2 / **1** % |
| **v3_noPair** | 49 / 49 / 2 % | 96 / **4** % | 48 / 48 / 2 / **1** % |
| **v4_strongS12_weakS3** | 49 / 49 / 1 % | 63 / **37** % | 48 / 48 / 2 / **1** % |
| **v5_d16_v4_noPair** | 49 / 49 / 1 % | 63 / **37** % | 48 / 48 / 2 / **1** % |
| **v1_gmmStrong** | 49 / 49 / 1 % | 63 / **37** % | 44 / 44 / 3 / **10** % |

*(pair ≈ 0 % everywhere — confirms A12: paired-align is negligible; v3 (pair off) is bit-identical
to v2. In S2 enc/dec are frozen so recon/kl are dropped; the GMM-flow term is the only thing
reshaping `u`, and that is where the runs diverge — **37 % (v1, v4) vs 4 % (v2, v3)**. The design
of v4 is visible here: its S2 = v1's (strong, 37 %) but its S3 = v2's (weak, 1 %). By S3 recon
dominates again for every run except v1, which keeps GMM-flow at 10 %.)*

**Results.** Runs sweep the GMM-flow weight, its S3 schedule (S12/S3 = weights used in Stages 1–2 /
Stage 3), latent dim, and warm-start. `dim` = latent_dim; `warm` = enc/dec warm-started from
single-modality VAEs.

| run | dim / warm | S12 / S3 recon | S3 GMM share | self T1/FA | **cross T1→FA / FA→T1** | u_std | K | **sil(gmm)** ↑ | **DB(gmm)** ↓ | sil(dx) |
|---|---|---|---|---|---|---|---|---|---|---|
| **v5_d16_v4_noPair** | 16 / ✗ | 12544 / 100 | ~1% | **0.895 / 0.792** | 0.765 / 0.864 | 0.646 | 4 | 0.264 | 1.371 | −0.016 |
| **v2_gmmWeak** | 8 / ✓ | 100 / 100 | ~1% | 0.892 / 0.788 | **0.770** / 0.863 | 0.279 | 4 | 0.314 | 1.211 | −0.016 |
| **v3_noPair** | 8 / ✓ | 100 / 100 | ~1% | 0.892 / 0.789 | 0.769 / 0.864 | 0.824 | 4 | 0.269 | 1.344 | −0.017 |
| **v4_strongS12_weakS3** | 8 / ✓ | 12544 / 100 | ~1% | 0.891 / 0.789 | 0.768 / 0.863 | 0.709 | 4 | 0.305 | 1.137 | −0.012 |
| **v1_gmmStrong** | 8 / ✓ | 12544 / 12544 | ~10% | 0.877 / 0.772 | 0.763 / 0.854 | 0.698 | 4 | **0.585** | **0.579** | −0.013 |

*(CFM baseline 0.732 / 0.802 — all runs beat it on both directions. v3 = v2 with `w_pair=0`;
v5 = v4 schedule at dim 16, from scratch, no pair.)*

- **All avoid collapse** (u_std 0.28–0.82) and keep all 4 clusters alive (`K used = 4`, balanced π).
  The GMM base + Stage-1.5 init fixes the A14 collapse cause.
- **GMM strength trades recon for blob structure.** Strong GMM throughout (v1) nearly doubles
  silhouette (0.585 vs 0.314) but costs cross-recon (−0.007) and pushes `u_std` up, which
  **degrades off-manifold intermediate decoding** — fig 18/19 show v1's FA→T1 intermediates blow
  out to a saturated, fragmented brain.
- **Blob structure is not a heritable asset (v4).** v4 builds v1's strong structure in S1–S2 then
  goes gentle in S3. It recovers the best recon (0.768 / 0.863) and the cleanest generation
  (fig 18-1/19-1 have no blow-up), but silhouette falls back to 0.305 — once S3 unfreezes enc/dec
  with a weak GMM, **recon washes the S2 blobs back toward a single ball.** The 0.585 structure
  needs *continuous* strong GMM pressure to persist; it cannot be built once and frozen.
- **pair loss is redundant (v3).** Dropping it (`w_pair=0`) leaves recon/cross bit-for-bit
  identical to v2 (0.769 / 0.864) and sil essentially unchanged — confirms A12. Its only effect is
  a looser latent (u_std 0.82 vs 0.28), which slightly *worsens* off-manifold decoding, so pair
  offers no benefit here.
- **dim 16 + from-scratch works, and gives the best recon (v5).** No warm-start needed — a
  from-scratch dim-16 run reaches the **highest self-recon (0.895 / 0.792)** and matches v4 on
  cross, with the **smoothest generation process** (fig 19-v5: wide bottleneck → gentler
  per-coupling steps → intermediates stay on-manifold). Cost: silhouette drops to 0.264 — a wider
  bottleneck **dilutes** the GMM/flow shaping (A14's low-dim-bottleneck lesson). Confirms the whole
  pipeline needs no single-modality pretrain.
- **Structure is geometric, not diagnostic.** `sil(dx) < 0` for all — the blobs the GMM carves are
  real but **unrelated to disease label** (§2.2 ceiling, A12).
- **Choice:** **v5** (dim 16, from scratch) for best recon + cleanest generation and one fewer
  dependency; **v4** for best generation at dim 8; **v2** for the simplest schedule. v1 maximizes
  unsupervised structure (sil 0.585) but at a recon and generation-quality cost, with no diagnostic
  payoff. **Recon/generation and blob-separation trade off; neither reaches disease structure.**

Figures: `15_3stage_trajectory` (v1/v2) · `16_3stage_u3d_gaussianity` · `17_3stage_recon` ·
`18_3stage_generation_process` (these 3 show all 4 dim-8 runs). Per-run flow traces:
`19-v1` … `19-v4` (dim 8) and `19-v5` (dim 16, smoothest process).

### 3.5 Where does U-Net sharpness come from? (skip vs shared-u, A16)

The recon-opt **U-Net + skip** config (§2.3/§3.2) gets the highest SSIM (self 0.99, cross 0.78)
— but a decode-without-skips probe shows the sharpness bypasses the latent entirely. Decode the
same net **without skips** (only the 8-D `u`):

| config | self-recon (u path) | cross T1→FA | cross FA→T1 |
|---|---|---|---|
| U-Net **+skips** (reported) | 0.99 | 0.79 (src-skip) | 0.82 |
| U-Net **no skips** (u only) | **0.14–0.17** | **0.26** | **0.17** |
| U-Net cross, tgt-enc on src (OOD) | – | 0.57 | – |
| U-Net cross, **oracle** tgt-skip (cheats) | – | 0.98 | 0.99 |
| ConvVAE bottleneck (u path) | **0.88** | 0.76 | 0.86 |

- **Finding:** in the U-Net variant the shared `u` contributes ≈0 to both self- and cross-recon
  (u-only 0.14–0.26); all detail comes from the SOURCE encoder skips → it is **pix2pix** (copies
  source spatial structure; works only because T1/FA are registered), **not** the CLAST mechanism.
  Any source-derived skip (even via the target encoder) bypasses `u`; only oracle target skips
  (which need the answer) recover detail. **ConvVAE is the honest config** — its `u` genuinely
  carries the reconstruction (0.88) and does the cross-modal transfer (0.76/0.86).
- Figures: `20_unet_vs_bottleneck_selfrecon`, `21_unet_cross_skip_vs_u`.

### 3.6 Two-channel + FiLM: keep `u` usable AND generate (A17)

Motivated by A16 — resolve the "sharp OR usable-`u`" conflict by **splitting the representation**:
`z_low → flow → u` does GMM/clustering/alignment; a separate **dense** channel carries recon
detail; the decoder is **FiLM-modulated by a flow state** so walking the flow morphs the output.
`models/film_vae.py`, `adni_pilot/train_film.py`. Guard metric = **u-only decode SSIM** (does `u`
carry content, or is it bypassed like U-Net skips?).

| variant | u-only decode ↑ | cross T1→FA / FA→T1 | sil(gmm) | u_std |
|---|---|---|---|---|
| **FiLM two-channel (A17)** | **0.882** | **0.767 / 0.851** ✓ | 0.325 | 0.294 |
| U-Net skip (A16, ref) | 0.17 | 0.78 / 0.82 (skip-driven) | – | – |
| ConvVAE bottleneck (ref) | 0.88 | 0.764 / 0.859 | ~0.31 | – |

- **Finding (conflict resolved):** `u`-only decode = 0.882 (vs U-Net skip's 0.17) — the two-channel
  design keeps `u` fully informative **and** it clusters (sil 0.325) **and** cross beats CFM. Usable
  representation + generation, no longer either/or.
- **Finding (FiLM morph, `fig 22`):** feeding successive flow states (z_low→b1→b2→b3→u) as the FiLM
  code produces a **smooth, on-manifold morph** — white-matter tracts emerge gradually stage by
  stage — vs the blown-out off-manifold intermediates of direct decoding (fig 18/19).
- **Finding (dense is redundant here):** the dense-path self-recon collapses to 0.19 while u-only
  reaches 0.88 — trained jointly, the model routes everything through `u` (which gets 3× the loss
  signal via u-only + cross×2) and ignores the dense channel. On this data an 8-D `u` alone already
  generates above baseline; the dense channel is unnecessary.
- Figure: `22_film_morph`.

### 3.7 The mid10 testbed — is the clustering ceiling caused by slice position? (A18)

`fig 23` showed every method's unsupervised clusters track **axial slice position**, not
diagnosis. `fig 24` showed why: over the cached z=32..58 the anatomy changes far more across z
than between subjects. So we rebuilt the testbed on the **middle 10 slices (z=40–49)** — where
z-variation is smallest — and **retrained everything on it** (1,700 train / 430 test slices;
`ADNI_Z_LO/HI` in `subject_level_split`, `RESULTS_SUFFIX` for the baselines).

**Generation (mid10, retrained):**

| Method | T1→FA SSIM | PSNR | FA→T1 SSIM | PSNR |
|---|---|---|---|---|
| CycleGAN | 0.680 | 17.87 | 0.813 | 19.80 |
| MeanFlow | 0.691 | 18.60 | 0.468 | 18.56 |
| CFM | 0.729 | 18.87 | 0.790 | 19.92 |
| **FiLM-CLAST (ours)** | **0.770** | – | **0.864** | – |

**Clustering (mid10, subject-level ACC/NMI/ARI; slice-level geometry), with calibration refs:**

| Method | ACC | NMI | ARI | sil(gmm) ↑ | DB(gmm) ↓ | sil(dx) |
|---|---|---|---|---|---|---|
| **FiLM-CLAST (ours)** | **0.442** | 0.082 | −0.004 | **0.201** | **1.623** | −0.023 |
| CFM | 0.419 | 0.095 | 0.019 | 0.116 | 2.106 | −0.024 |
| DDPM | 0.372 | **0.106** | **0.029** | 0.109 | 2.259 | −0.037 |
| MeanFlow | 0.372 | 0.088 | −0.012 | 0.141 | 1.938 | −0.031 |
| *raw pixels (no learning)* | – | – | – | *0.103* | *2.329* | *−0.017* |
| *random gaussian* | – | – | – | *0.012* | *6.026* | *−0.006* |

- **Finding (ceiling is NOT slice position):** removing the dominant z-variation does **not** lift
  diagnosis — ACC/NMI are unchanged vs the full-z testbed (ours 0.442/0.099 → 0.442/0.082). The
  slice-level cluster enrichment seen in `fig 23-mid10` disappears once slices are pooled per
  subject. The bottleneck is sample size (43 imbalanced test subjects) and 2D-slice information,
  not the z confound.
- **Finding (silhouette needs a calibration ref):** raw pixels already give sil 0.103, so the
  baselines' 0.11–0.14 is **no better than not learning at all**; only ours (0.201, DB 1.623)
  clearly exceeds the pixel reference. But by Rousseeuw's rubric sil < 0.25 is still "no
  substantial structure" — and `sil(dx) < 0` for **every** method *including raw pixels*, i.e.
  disease labels do not form clusters in image space at all.
- **⚠ Correction — ACC needs a permutation null.** Shuffling the labels while keeping the same
  partition gives a null **mean ACC of 0.408** (n=43, K=4) — *above* the 0.395 majority baseline.
  So the ACC ≈ 0.44 reported throughout §2.2 (naive VAE, DiT, ours) is **not distinguishable from
  chance**; the majority baseline was too lenient a reference. Use the permutation null.
- Figures: `23-mid10trained_clusters_*` (per-method clusters + samples, mid10-trained),
  `24_middle10_slices`, `22-mid10_film_morph`.

### 3.8 Can our module plug into CycleGAN? — no (A19)

Route-B test: keep the faithful CycleGAN host (LSGAN + cycle L1×10 + identity L1×5) and attach our
head — bottleneck → GAP+FC → VAE `z` (KL-anchored) → **one shared flow** → `u`, with a GMM base;
`u` is re-injected by FiLM. A single bijection `f` encodes modality by direction:
`u = f(z_A)` and `u = f⁻¹(z_B)`, so `z_A → u → z_B` is one chain. Two stages: (1) warmup, no GMM;
(2) encoders frozen, GMM on, decoder+D still training, lr×0.1; early stop on the non-adversarial
loss (leakage-free). `nets_clast.py`, `train_clast.py`.

| run | T1→FA | FA→T1 | u_std | K used | paired ‖u_A−u_B‖ | ACC (perm p) | sil | DB | sil(dx) |
|---|---|---|---|---|---|---|---|---|---|
| chain open (`w_chain=0`) | **0.738** | **0.818** | 4.80 | 1 | 28.1 | 0.442 (p=0.27) | 0.172 | 1.703 | −0.049 |
| chain closed (`w_chain=1`) | 0.724 | 0.661 | 0.46 | 1 | **0.265** | 0.488 (**p=0.033**) | 0.197 | 1.466 | −0.063 |

**Three independent diagnostics all say `u` is decorative:**

| probe | result |
|---|---|
| **u-shuffle** (same source, another sample's `u`) | SSIM identical to 3 decimals, both runs |
| **flow-block morph** (code walks z_A→u_A) | mean pixel change **0.6 %** |
| **u_A→u_B interpolation** (T1 code → FA code) | mean pixel change **0.01 %** |

- **Finding (the plug-in fails, and why):** with a 256×28×28 spatial bottleneck reaching the
  decoder, the low-dim `u` receives no gradient pressure — FiLM stays at its identity init and the
  host generates entirely through the spatial path. This is the **third** independent confirmation
  of the same law (U-Net skips §3.5; CycleGAN FiLM here, both runs): *whenever a high-capacity
  spatial bypass exists, the shared low-dim latent degenerates to decoration.* Our own FiLM model
  (§3.6) shows visible morph with the same FiLM mechanism — the difference is that there `u` is the
  only route.
- **Finding (constraining a useless latent costs generation):** closing the chain does make the two
  modalities coincide (‖u_A−u_B‖ 28.1 → 0.265), but it welds together a quantity that does not
  affect the output, and FA→T1 drops 0.818 → 0.661.
- **Caveat on the one positive signal:** the chain-closed run's ACC 0.488 is the only value in this
  project that beats its permutation null (p=0.033, uncorrected; ≈0.066 for the 2 runs tested).
  With sil 0.197 (≈ no structure) and sil(dx) < 0 there is no geometry supporting it, and n=43 means
  a 4-subject difference — a lead worth re-testing across seeds, not a result.
- Figures: `25_mid10_{nopair,pair}_cyclegan_clast_recon` (incl. u-shuffle column),
  `26_mid10_{nopair,pair}_clast_clusters`, `27_mid10_pair_clast_morph`.

### 3.9 Notes / priorities

- **Recon — SOLVED.** A13 (cross-recon weight↑ + phase-aware ckpt selection) makes MMCLAST
  beat every baseline on cross-modal SSIM/PSNR (0.764 / 0.859). A12 shows the win is driven
  by the **cross-recon loss**; all other terms are removable without hurting recon.
- **Recon ceiling:** self-recon (T1 0.892 / FA 0.787) bounds cross-recon; A2 showed latent_dim
  barely moves self-recon (dim8≈dim16), so the 8-D bottleneck is *not* the limiter here.
- **Clustering-focused:** §2.2 + A1 + A12 show schedule/backbone/loss tuning won't move NMI/ARI
  (stuck ≈0). Only **A11 (supervised signal)** is expected to help diagnosis separability.
- **Alignment fix:** A7 (InfoNCE / subject-mean) targets the subj-CCA 0.023 collapse.

## 4. Visualizations

All under `adni_pilot/figures/`:

| File | What |
|---|---|
| `01_trajectories.png` | 3-panel loss / clustering acc / CCA-and-per-modality-acc curves |
| `02_tsne_slice.png` | Slice-level t-SNE colored by modality (good mix) and by label_4 (no separation) |
| `03_tsne_subject.png` | 43-subject t-SNE in concat u space — diagnosis colors scattered,no natural clusters |
| `04_confusion.png` | Subject-level confusion at acc=0.442 — model effectively uses only 2 of the 4 clusters |
| `05_cross_modal_epoch_{005,030,050}.png` | 6-row generation grid: orig T1 / self-T1 / FA→T1 / orig FA / self-FA / T1→FA |
| `06_generation_process_all.png` | Unified cross-modal generation process, same sample, all methods + CLAST |
| `07_e2e_loss_evolution.png` | End-to-end (no warm-start) per-loss curves over training |
| **`08_recon_winner.png`** | **Recon-optimized MMCLAST vs all baselines: qualitative T1↔FA grid (per-sample SSIM) + cross-modal SSIM bar chart — MMCLAST wins both directions** |
| `09_spatial_align_losses{,_lp0.1/0.3/1.0}.png` | A14 spatial-align loss curves (2·recon + paired-align), per λ_p |
| `10_spatial_align_tsne{,_lp*}.png` | Single spatial-align t-SNE (by modality / by disease) |
| `11_spatial_align_tsne_grid.png` | A14 λ_p sweep (mean-BCE): 4-run t-SNE grid, modality/disease + cross-modal mixing |
| `12_alignment_ladder_tsne.png` | **A14 collapse ladder t-SNE (7 variants): only CLAST (recon-dominant) interleaves T1/FA (CCA 0.93); all balanced variants stay separated / collapsed** |
| **`13_ablation_reconstruction.png`** | **A14 reconstruction grid (rows = variants, cols = T1 in / self-recon T1 / self-recon FA / cross T1→FA / GT FA), shared colorbar, all panels on the same 0–1 intensity scale: C/D/E collapse to a blurry mean-brain; A1/B1 (sum-recon) recover detail but cross still off; CLAST recon-opt reconstructs both** |
| `14_ablation_u_distribution.png` | A14 latent distribution per variant (histogram of `u` post-flow, or the raw representation): a delta-like spike = collapse. std: E (+KL) **0.010** (worst collapse) · bare/A 0.04–0.06 · A1/B1 **0.62–0.81** · CLAST **0.58–0.70** (healthy) |
| `14b_ablation_u_3d.png` | A14 ladder in **joint 3-D** (PCA of `u`, colour=refit-GMM, marker=modality) — a marginal histogram can't separate a 4-blob mixture from one Gaussian; this can. Shows CLAST's `u` is a **single mixed ball** (sil 0.10–0.18), collapsed variants are empty boxes |
| **§3.4 A15 — GMM flow-base, 3-stage** (v1/v2 unless noted): | |
| `15_3stage_trajectory.png` | 6-panel training trajectory: sil(gmm)/DB/sil(dx)/cross-recon/u_std over 70 epochs, S1/S2/S3 bands — strong GMM (v1) sharpens blobs only in S3, at ~no recon cost; sil(dx) stays <0 throughout |
| `16_3stage_u3d_gaussianity.png` | **All 4 runs**, 3-D `u` layout (colour=own GMM blob) + per-blob **Gaussianity** check (whitened Mahalanobis radius vs χ(8)): blobs are real but **sub-Gaussian** (too concentrated); v1 most so, weak-GMM runs sit closer to χ(8) |
| `17_3stage_recon.png` | **All 4 runs**, bidirectional cross-modal recon grid (T1↔FA), slice-major: 2 samples × 4 settings, shared colorbar |
| `18_3stage_generation_process.png` | **All 4 runs**, cross-modal generation **process** (target emerges block-by-block through flow⁻¹). v1's FA→T1 intermediates blow out (u_std 0.70); v2/v4 clean |
| `19-{v1,v2,v3,v4,v5}_3stage_flow_step_by_step.png` | Full block-by-block flow trace by disease case (T1 morph → shared `u` → FA emerge), **one file per run**. v1's larger `u_std` gives noisier off-manifold intermediates; v2/v4 smoother; **v5 (dim 16) smoothest** (wider bottleneck → gentler coupling steps) |
| **§3.5 A16 — U-Net skip vs shared-u:** | |
| `20_unet_vs_bottleneck_selfrecon.png` | Self-recon: U-Net+skip (0.99) vs the same net decoded **without skips** (u only, 0.14–0.17) vs ConvVAE bottleneck (0.88) — U-Net sharpness lives in the skips, not `u` |
| `21_unet_cross_skip_vs_u.png` | Cross-modal: src-skip (0.79) ≈ oracle tgt-skip (0.98) ≫ u-only (0.27); tgt-encoder-on-source (0.57, OOD) — any source-derived skip bypasses `u` → pix2pix |
| **§3.6 A17 — two-channel + FiLM:** | |
| `22_film_morph.png` / `22-mid10` | FiLM morph: the code walks the flow (z_low→b1→b2→b3→u); the target modality **emerges gradually, smoothly, on-manifold** (vs the blown-out direct-decode of fig 18/19). `-mid10` = trained on z=40–49 |
| **§3.7 A18 — the mid10 testbed:** | |
| `23_clusters_<method>.png` / `23-mid10` / `23-mid10trained` | Per-method unsupervised clusters (GMM K=4) + 6 random T1 samples per cluster + per-cluster disease composition. Plain = full z, `-mid10` = mid10 slices w/ full-z models, `-mid10trained` = models retrained on mid10. **Clusters track slice position / ventricle size, never diagnosis** |
| `24_middle10_slices.png` | The middle 10 axial slices (z=40–49) for 4 random test subjects, T1+FA — shows z-variation is mild here while inter-subject (ventricle size) variation is visible |
| **§3.8 A19 — CycleGAN + our head:** | |
| `25_mid10_{nopair,pair}_cyclegan_clast_recon.png` | CycleGAN+CLAST recon, both directions, **plus a `u`-shuffled column**: swapping in another sample's `u` changes SSIM by 0.000 → `u` is decorative |
| `26_mid10_{nopair,pair}_clast_clusters.png` | Clusters of the shared `u` + samples + disease composition; chain-closed run has one LMCI-enriched cluster (39 % vs 16 % base) but sil 0.197 ≈ no structure |
| `27_mid10_pair_clast_morph.png` | Walking the shared flow as the FiLM code (blocks, and u_A→u_B): output changes by 0.6 % / **0.01 %** — the morph is dead, unlike fig 22 where `u` is the only route |


## 9. Interpretation

### What works 
- **End-to-end pipeline works** on real ADNI data (registration → cache → 3-stage training → eval → viz)
- **Cross-modal slice alignment** is excellent (CCA 0.945) — matches MNIST-PET pilot
- **Cross-modal generation beats every baseline** (recon-opt: T1→FA 0.764 / FA→T1 0.859 SSIM,
  +0.03–0.05 over CFM on both directions, on SSIM *and* PSNR) — §2.3 ‡, `fig 08`. The single
  invertible flow + shared latent does cross-modal transfer better than CFM/CycleGAN/MeanFlow/DDPM.
- **Loss ablation (A12)** shows the recon win is driven by the cross-recon term alone; GMM/flow/
  KL/paired-align are removable without hurting recon — a clean, minimal objective.

### What does not work 
- **Slice-level acc not good** (clustering capped by the testbed, not the model — see §2.2/A12)
- **Subject-level CCA very bad** (0.023 — A7 target)

## 10. Next steps
- Clustering is the remaining gap: only a **supervised aux head (A11)** is expected to move NMI/ARI.
- Fix subject-level CCA via A7 (InfoNCE / subject-mean paired loss).
- Optional: latent_dim↑ to raise the recon ceiling further (current win already clears all baselines).

### 10.1 To explore — resolve the flow-NLL ↔ GMM-NLL tension (from A14)

A latent regularizer on top of recon/cross-recon is necessary (recon alone leaves the latent
unconstrained → collapse, A14). But the **two regularizers we use pull in opposite directions**:

| term | shapes `u` toward | job |
|---|---|---|
| **flow-NLL** | single-peak **N(0, I)** | scale-anchor (anti-collapse) + exact density/UQ |
| **GMM-NLL** | **K separated peaks** (μ₁..μ_K) | cluster structure |

flow-NLL pulls all `u` to one center (0); GMM-NLL pushes `u` apart into K modes — **directly
conflicting targets**. In the current MMCLAST the tension is resolved by *weighting*: flow-NLL is
**sum-reduced (strong)** and GMM-NLL **mean-reduced (weak)**, so **flow-NLL dominates → `u` ≈ N(0,I)
and the GMM is essentially inert**. This matches A12 (**−GMM harmless**) and §2.2 (GMM never lifts
clustering). I.e. today GMM is mostly decorative.

**Proposed fix — make the flow's target distribution the GMM itself** (VaDE-style), instead of a
separate N(0,I):
```
now (conflict) :  flow-NLL: u → N(0,I)        ⟂   GMM-NLL: u → K peaks
to explore     :  flow-NLL evaluated UNDER the GMM:  −(log Σ_k π_k N(u; μ_k,Σ_k) + log|det J|)
```
Then a **single** flow-NLL-under-GMM does both jobs at once: `log|det J|` still penalizes
volume-collapse (anti-collapse ✓) while the mixture base gives K-peak cluster structure (✓) — no
more single-peak-vs-K-peak tug-of-war. Expected upside: GMM stops being inert (may finally help
clustering) without losing the scale-anchor that prevents A14-style collapse.

- **Experiment to run:** replace `flow_nll_standard_normal(u)` with a `flow_nll_under_gmm(u)` that
  uses the shared GMM as the flow's base density; sweep whether it (a) still prevents collapse, and
  (b) moves subj-CCA / clustering vs the current N(0,I)+separate-GMM design. `losses.py`, `flow.py`.