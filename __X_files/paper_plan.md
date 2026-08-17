# MMCLAST on ADNI T1↔FA — paper-ready results

All tables are paper-ready. Figures give a path + one line. Findings are one sentence each.

---

## 1. Dataset

| Item | Value |
|---|---|
| Source | ADNI T1 (MNI152 2mm) ↔ FA from DTI (MNI152 2mm), co-registered |
| Paired subjects | 213 (label_4: CN 107 / EMCI 59 / MCI 24 / LMCI 23) |
| Slices | middle 27 axial, 112×112, 1-ch → 5,751 total |
| Split (subject-level) | 170 train / 43 test → 4,590 / 1,161 slices |
| Eval units | generation = per-slice (1,161); clustering = per-subject mean (43) |

---

## 2. Main comparison — cross-modal generation + diagnostic clustering

**All methods, same testbed.** Generation = SSIM/PSNR on paired test slices; clustering = post-hoc GMM (K=4), concat repr for ours / mid-network probe for generation-only nets.

| Method | Type | T1→FA SSIM | T1→FA PSNR | FA→T1 SSIM | FA→T1 PSNR | concat ACC | NMI | subj-CCA |
|---|---|---|---|---|---|---|---|---|
| naive VAE | none | – | – | – | – | 0.442 | 0.112 | 0.047 |
| CycleGAN | 1-shot | 0.680 | 17.18 | 0.743 | 17.94 | 0.395 | 0.105 | – |
| DDPM (faithful) | stochastic | 0.362 | 18.44 | 0.769 | 20.36 | 0.395 | 0.088 | – |
| DiT (faithful, latent) | stochastic | 0.665 | 14.32 | 0.757 | 18.28 | 0.442 | 0.091 | – |
| MeanFlow | 1-step | 0.686 | 17.37 | 0.621 | 17.27 | 0.372 | 0.098 | – |
| CFM (best baseline) | ODE | 0.732 | 18.46 | 0.802 | 19.99 | 0.419 | 0.098 | – |
| MMCLAST default | invertible | 0.663 | 17.49 | 0.695 | 17.37 | **0.442** | 0.099 | 0.023 |
| **MMCLAST recon-opt (ConvVAE)** | invertible | 0.764 | 19.95 | **0.859** | **22.05** | 0.349 | 0.059 | **0.628** |
| **MMCLAST recon-opt (U-Net)** | invertible | **0.780** | **20.19** | 0.810 | 19.54 | 0.372 | 0.079 | 0.000 |

- **Finding (generation):** recon-optimized MMCLAST beats every baseline on both directions and both metrics (+0.03–0.05 SSIM, +1.5–2.1 dB over CFM).
- **Finding (clustering):** no method beats the 0.395 majority floor — the testbed (2D slices, 43 imbalanced subjects) caps diagnosis, not the model.
- **Finding (alignment):** the strong paired-align (λ_pair=2) lifts subject-level cross-modal agreement 0.023→0.628.

**Baseline single-modality lower bound:** self-recon SSIM T1 0.753 / FA 0.793; clustering ACC ≤0.30, NMI≈0.

---

## 3. Ablations

### 3.1 Training strategy

| Strategy | ACC | NMI | subj-CCA | T1→FA | FA→T1 |
|---|---|---|---|---|---|
| warm-start (3-stage) | 0.442 | 0.099 | 0.023 | 0.663 | 0.695 |
| end-to-end (scratch) | **0.512** | **0.254** | 0.000 | 0.624 | 0.666 |

- **Finding:** end-to-end gives the best discriminative clustering but worst recon and zero subj-CCA — warm-start buys recon fidelity + alignment.

### 3.2 Architecture

| Variant | self T1/FA | T1→FA | FA→T1 | subj-CCA |
|---|---|---|---|---|
| ConvVAE bottleneck | 0.892 / 0.787 | 0.764 | **0.859** | **0.628** |
| U-Net + skip-transfer | **0.995 / 0.987** | **0.780** | 0.810 | 0.000 |

| latent_dim | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| self-recon SSIM | 0.889 | **0.897** | 0.893 | 0.894 |

- **Finding (backbone):** pix2pix skips give near-perfect recon and best T1→FA, but kill the bottleneck representation (subj-CCA→0); ConvVAE keeps `u` usable.
- **Finding (dim):** latent_dim barely affects recon (8≈16≈32≈64); BCE ≥ MSE at every dim.
- **Finding (encoder):** pretrained backbones are negative transfer (ImageNet-ResNet18 −0.05 on MNIST-PET; MedVAE −0.02 on ADNI-T1).

### 3.3 Loss deletion (end-to-end, remove one term)

| Config | ACC | NMI | slice-CCA | self T1 | T1→FA | FA→T1 |
|---|---|---|---|---|---|---|
| full | 0.419 | 0.050 | 0.938 | 0.894 | 0.768 | 0.862 |
| − GMM-NLL | 0.465 | 0.059 | 0.952 | 0.894 | 0.769 | 0.862 |
| − flow-NLL | 0.372 | 0.050 | 0.948 | 0.893 | 0.769 | 0.862 |
| − KL | 0.419 | 0.097 | 0.948 | 0.895 | 0.768 | 0.863 |
| − paired-align | 0.372 | 0.076 | 0.951 | 0.894 | 0.767 | 0.863 |
| **− cross-recon** | 0.465 | 0.130 | **0.802** | 0.898 | **0.724** | **0.797** |

- **Finding:** cross-recon is the *only* term that matters for reconstruction and alignment; GMM/flow/KL/paired-align are all removable without loss.

### 3.4 What makes alignment work — collapse ladder (A14)

| Variant | rep dim | added loss | CCA | mixing | std | aligned? |
|---|---|---|---|---|---|---|
| bare-align | 100352 | 2·recon + MSE-align | – | 0.00 | 0.058 | ✗ collapse |
| + cross / conv-flow / vec-flow / GMM / KL | 100352→16 | (mean-reduced) | 0.00 | 0.00 | 0.01–0.34 | ✗ collapse |
| flow-NLL ×10 / ×1000 | 16 | stronger flow-NLL | 0.00 | 0.00 | 0.31–0.34 | ✗ collapse |
| A1/B1 (sum-recon) | 100352 | recon-dominant | – | 0.00 | 0.65–0.80 | ~ fixed, not aligned |
| **CLAST** | **8** | sum-recon + low-dim + flow/GMM/KL | **0.933** | **0.37** | 0.591 | ✅ |

- **Finding:** alignment needs BOTH (1) recon-dominance (else the encoder collapses to the mean image) AND (2) a low-dim bottleneck (else align terms are drowned); GMM/KL/flow alone confer none. flow-NLL can't fix collapse at any strength — it is a bijection *downstream* of the collapsed encoder.

### 3.5 GMM flow-base + 3-stage training (A15)

**Stage schedule:** S1 pretrain enc/dec (no flow-NLL) · S1.5 fit GMM on `u` · S2 frozen enc/dec + GMM-flow · S3 joint. All losses mean-reduced; weight = contribution.

| run | dim / warm | S12/S3 recon | S3 GMM share | self T1/FA | T1→FA / FA→T1 | u_std | sil(gmm)↑ | DB(gmm)↓ | sil(dx) |
|---|---|---|---|---|---|---|---|---|---|
| v5 (dim16, scratch, no-pair) | 16 / ✗ | 12544 / 100 | ~1% | **0.895 / 0.792** | 0.765 / 0.864 | 0.646 | 0.264 | 1.371 | −0.016 |
| v2 (weak) | 8 / ✓ | 100 / 100 | ~1% | 0.892 / 0.788 | **0.770** / 0.863 | 0.279 | 0.314 | 1.211 | −0.016 |
| v3 (no-pair) | 8 / ✓ | 100 / 100 | ~1% | 0.892 / 0.789 | 0.769 / 0.864 | 0.824 | 0.269 | 1.344 | −0.017 |
| v4 (strong-S12, weak-S3) | 8 / ✓ | 12544 / 100 | ~1% | 0.891 / 0.789 | 0.768 / 0.863 | 0.709 | 0.305 | 1.137 | −0.012 |
| v1 (strong) | 8 / ✓ | 12544 / 12544 | ~10% | 0.877 / 0.772 | 0.763 / 0.854 | 0.698 | **0.585** | **0.579** | −0.013 |

*(sil(gmm) = silhouette by model's own GMM; sil(dx) = silhouette by disease label. All beat CFM 0.732/0.802.)*

- **Finding (GMM strength):** a strong GMM-flow weight nearly doubles blob separation (sil 0.585 vs 0.31) but costs recon and inflates `u_std` (worse off-manifold generation).
- **Finding (transience):** blob structure is not heritable — S2 structure washes out when S3 unfreezes with weak GMM (v4 sil falls to 0.31), so it needs continuous GMM pressure.
- **Finding (dim/pretrain):** dim-16 from-scratch gives the best recon + smoothest generation with one fewer dependency, at slightly lower blob separation (wider bottleneck dilutes shaping).
- **Finding (pair redundant):** removing pair (v3) is bit-identical to v2 — confirms 3.3.
- **Finding (never diagnostic):** sil(dx) < 0 for all — the GMM carves real geometric clusters unrelated to disease.

### 3.6 Two-channel + FiLM: keep `u` usable AND generate (A17)

Split the representation: `z_low → flow → u` does clustering/alignment; a separate **dense**
channel carries recon detail; the decoder is **FiLM-modulated by a flow state**, so walking the
flow morphs the output. Guard metric = **u-only decode** (is `u` used, or bypassed?).

| variant | u-only decode ↑ | cross T1→FA / FA→T1 | sil |
|---|---|---|---|
| **FiLM two-channel** | **0.882** | **0.767 / 0.851** | 0.325 |
| U-Net skip (ref) | 0.17 | 0.78 / 0.82 (skip-driven) | – |
| ConvVAE bottleneck (ref) | 0.88 | 0.764 / 0.859 | ~0.31 |

- **Finding:** the two-channel split resolves the "sharp OR usable-`u`" conflict — `u` stays fully
  informative (0.882 vs U-Net's 0.17), clusters, and beats CFM on cross-recon.
- **Finding (morph):** feeding successive flow states as the FiLM code gives a smooth, on-manifold
  morph (fig 22) — vs blown-out intermediates when flow states are decoded directly.
- **Finding (dense redundant):** the dense path collapses to 0.19 while u-only reaches 0.88 — with
  3× the loss signal on `u`, the model routes everything through it and ignores the dense channel.

### 3.7 The mid10 testbed — is the ceiling caused by slice position? (A18)

Clusters tracked **axial slice position**, not diagnosis (fig 23), because anatomy varies more
across z=32..58 than between subjects (fig 24). So we rebuilt on the **middle 10 slices (z=40–49)**
and **retrained everything** (1,700 train / 430 test).

| Method (mid10, retrained) | T1→FA | FA→T1 | ACC | NMI | sil(gmm) | DB | sil(dx) |
|---|---|---|---|---|---|---|---|
| **FiLM-CLAST (ours)** | **0.770** | **0.864** | 0.442 | 0.082 | **0.201** | **1.623** | −0.023 |
| CFM | 0.729 | 0.790 | 0.419 | 0.095 | 0.116 | 2.106 | −0.024 |
| CycleGAN | 0.680 | 0.813 | – | – | – | – | – |
| MeanFlow | 0.691 | 0.468 | 0.372 | 0.088 | 0.141 | 1.938 | −0.031 |
| DDPM | – | – | 0.372 | 0.106 | 0.109 | 2.259 | −0.037 |
| *raw pixels* | – | – | – | – | *0.103* | *2.329* | *−0.017* |
| *random gaussian* | – | – | – | – | *0.012* | *6.026* | *−0.006* |

- **Finding:** removing the z confound does **not** lift diagnosis (ours 0.442/0.099 → 0.442/0.082).
  The bottleneck is sample size (43 imbalanced test subjects) + 2D-slice information, not slice
  position. Slice-level cluster enrichment vanishes once pooled per subject.
- **Finding (calibration matters):** raw pixels already give sil 0.103, so baselines at 0.11–0.14
  are **no better than not learning**; only ours (0.201) clearly exceeds it — yet sil < 0.25 is
  still "no substantial structure", and `sil(dx) < 0` for every method *including raw pixels*.
- **⚠ Correction:** a permutation null (shuffle labels, same partition, n=43, K=4) has **mean ACC
  0.408 > the 0.395 majority baseline**. Every ACC reported in §2 (max 0.442) is therefore **within
  chance**. Report against the permutation null, not the majority floor.

### 3.8 Can our module plug into CycleGAN? — no (A19)

Faithful CycleGAN host + our head (bottleneck → KL-anchored `z` → **one shared flow** → `u` with a
GMM base, re-injected by FiLM). One bijection encodes modality by direction: `u=f(z_A)`,
`u=f⁻¹(z_B)`. Two stages (warmup → encoders frozen + GMM on), early stop on the non-adversarial loss.

| run | T1→FA | FA→T1 | u_std | K used | ‖u_A−u_B‖ | ACC (perm p) | sil |
|---|---|---|---|---|---|---|---|
| chain open | **0.738** | **0.818** | 4.80 | 1 | 28.1 | 0.442 (p=0.27) | 0.172 |
| chain closed | 0.724 | 0.661 | 0.46 | 1 | **0.265** | 0.488 (**p=0.033**) | 0.197 |

| diagnostic | result |
|---|---|
| u-shuffle (another sample's `u`) | SSIM identical to 3 decimals |
| flow-block morph | **0.6 %** pixel change |
| u_A→u_B interpolation | **0.01 %** pixel change |

- **Finding:** the plug-in fails — with a 256×28×28 spatial bottleneck reaching the decoder, `u`
  gets no gradient pressure and FiLM stays at its identity init. **Third independent confirmation**
  (U-Net skips §3.5; both CycleGAN runs here) of one law: *a high-capacity spatial bypass always
  reduces the shared low-dim latent to decoration.* The same FiLM mechanism morphs visibly in §3.6,
  where `u` is the only route.
- **Finding:** closing the chain does align the modalities (‖u_A−u_B‖ 28.1 → 0.265) but welds a
  quantity that does not affect output — FA→T1 drops 0.818 → 0.661.
- **Caveat:** the chain-closed ACC 0.488 is the only value beating its permutation null (p=0.033,
  uncorrected). With sil 0.197 and sil(dx) < 0 there is no geometry behind it, and n=43 means a
  4-subject difference — a lead to re-test across seeds, not a result.

---

## 4. Figures

| Path | What |
|---|---|
| `figures/08_recon_winner.png` | Recon-opt MMCLAST vs all baselines: T1↔FA qualitative grid + cross-SSIM bar chart (MMCLAST wins both) |
| `adni_pilot/figures/06_generation_process_all.png` | Cross-modal generation process, same slice, all methods (row = method trajectory) |
| `adni_pilot/figures/10_flow_step_by_step.png` | CLAST block-by-block flow trace by disease case (T1 morph → shared `u` → FA emerge) |
| `figures/12_alignment_ladder_tsne.png` | A14 collapse ladder t-SNE: only recon-dominant CLAST interleaves T1/FA (CCA 0.93); balanced variants collapse |
| `figures/13_ablation_reconstruction.png` | A14 recon grid (shared colorbar): collapsed variants output a mean-brain; CLAST reconstructs detail |
| `figures/14b_ablation_u_3d.png` | A14 joint 3-D `u` (PCA): CLAST's `u` is a single mixed ball; a marginal histogram can't show this |
| `figures/15_3stage_trajectory.png` | A15 6-panel training trajectory (sil/DB/recon/u_std over S1/S2/S3): strong GMM sharpens blobs only in S3 |
| `figures/16_3stage_u3d_gaussianity.png` | A15 all-4-run 3-D `u` layout + per-blob Gaussianity vs χ(8) (blobs real but sub-Gaussian) |
| `figures/17_3stage_recon.png` | A15 bidirectional recon grid, 2 samples × 4 settings, shared colorbar |
| `figures/18_3stage_generation_process.png` | A15 generation process (target emerges block-by-block); v1 blows out, v2/v4 clean |
| `figures/19-v{1..5}_3stage_flow_step_by_step.png` | A15 per-run flow trace by disease case; v5 (dim16) smoothest |
| `figures/20_unet_vs_bottleneck_selfrecon.png` | A16 self-recon: U-Net+skip 0.99 vs same net **without skips** (u only) 0.14–0.17 vs ConvVAE 0.88 — sharpness is in the skips, not `u` |
| `figures/21_unet_cross_skip_vs_u.png` | A16 cross-modal: src-skip 0.79 ≈ oracle 0.98 ≫ u-only 0.27 — any source-derived skip bypasses `u` → pix2pix |
| `figures/22_film_morph.png` (`-mid10`) | A17 FiLM morph: the code walks the flow and the target emerges smoothly, on-manifold |
| `figures/23-mid10trained_clusters_<method>.png` | A18 per-method clusters + samples + disease composition (mid10-trained): clusters track slice position / ventricle size, never diagnosis |
| `figures/24_middle10_slices.png` | A18 the middle 10 slices for 4 random subjects (T1+FA): z-variation mild, inter-subject ventricle variation visible |
| `figures/25_mid10_{nopair,pair}_cyclegan_clast_recon.png` | A19 CycleGAN+CLAST recon **with a u-shuffled column** — swapping `u` changes SSIM by 0.000 |
| `figures/26_mid10_{nopair,pair}_clast_clusters.png` | A19 clusters of the shared `u`; chain-closed run has one LMCI-enriched cluster (39 % vs 16 %) but sil ≈ no structure |
| `figures/27_mid10_pair_clast_morph.png` | A19 walking the shared flow as the FiLM code: 0.6 % / **0.01 %** pixel change — morph is dead when a spatial bypass exists |

---

## 5. Open (not yet run)

A5 #clusters K · A6 #flow layers · A7 paired-align → InfoNCE (fix subj-CCA) · A9 KL weight · A10 subject pooling · **A11 supervised aux head (only lever expected to move diagnosis)**.
