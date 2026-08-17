# Baseline reproduction plan

Reproduce standard generative / cross-modal baselines on the **same ADNI T1↔FA
testbed** (and MNIST-PET where relevant) so CLAST can be compared apples-to-apples.
Two axes of comparison:

1. **Clustering / representation quality** — subject-level acc / NMI / ARI on label_4,
   plus *how the intermediate representation evolves during training* (new, §0).
2. **Cross-modal generation quality** — SSIM / PSNR / LPIPS on T1↔FA translation.

> Shared testbed (frozen):
> - data: `adni_pilot/cache/paired_112.pt` — 5,778 paired slices, 214 subj, 112²
> - split: subject-level 80/20 (`subject_level_split`, label_4)
> - eval: `adni_pilot/eval_subject_level.py` protocol
> - CLAST reference: subj acc **0.442**, NMI 0.099, slice-CCA 0.945, subj-CCA 0.023

---

## 0. Representation-evolution tracking (cross-cutting, do FIRST)

**Goal**: beyond final metrics, record how the *intermediate representation*
changes through training — for CLAST and every baseline, so the comparison is
about *learning dynamics*, not just endpoints.

What to log every N epochs (a reusable hook / callback):

| Metric | What it tells us | How |
|---|---|---|
| **linear-probe acc** | upper bound of dx info linearly decodable from repr | logistic reg on frozen repr → label_4, 5-fold |
| **silhouette score** | intrinsic cluster separation (no labels) | sklearn silhouette on repr |
| **GMM cluster-center drift** | are clusters still moving / merging? | ‖μ_k(t) − μ_k(t-1)‖ |
| **pair-dist distribution** | cross-modal alignment (not just mean) | histogram of ‖u_A−u_B‖, per epoch |
| **rank / effective-dim** | posterior collapse detector | participation ratio of repr covariance |
| **repr t-SNE snapshots** | qualitative drift | t-SNE every 10 ep → GIF |

Deliverable:
- `adni_pilot/repr_tracker.py` — callback that takes (model, loader, epoch) → dict
- Each training script calls it; appends to `repr_evolution.csv`
- `adni_pilot/viz_repr_evolution.py` — plots all the above vs epoch, multi-model overlay

**Why first**: once this exists, every baseline run produces the evolution data for
free, and the final comparison plot writes itself.

---

## 1. naive VAE (no GMM, no flow)

**Role**: ablate away CLAST's GMM prior + flow — pure recon+KL, post-hoc GMM on z.
We already did this for MNIST/SVHN/Printed (`train_purevae.py`); port to ADNI.

| Item | Spec |
|---|---|
| arch | Encoder + Decoder (our `models/vae.py`), latent_dim=8, no flow, no trainable GMM |
| loss | BCE recon + KL only |
| eval | fit GMM K=4 post-hoc on z (same as CLAST eval) |
| compare | does CLAST's flow+GMM beat post-hoc GMM on VAE? (MNIST showed ≤2pp) |
| files | `adni_pilot/baselines/train_naive_vae.py` |
| cost | ~30 min |

---

## 2. CycleGAN (unpaired cross-modal translation)

**Role**: classic GAN cross-modal baseline. T1↔FA via two generators + two
discriminators + cycle-consistency. No shared latent / clustering by design — but we
can probe the generator bottleneck as a representation.

| Item | Spec |
|---|---|
| arch | 2× ResNet/U-Net generator, 2× PatchGAN discriminator |
| loss | adversarial + cycle (λ=10) + identity (λ=0.5) |
| eval-gen | SSIM/PSNR of T1→FA, FA→T1 vs ground truth (we HAVE paired GT) |
| eval-repr | extract G_A2B encoder bottleneck → GMM K=4 → subj acc |
| ref | Zhu et al. 2017; adapt to 1ch 112² |
| files | `adni_pilot/baselines/cyclegan/` |
| cost | ~2-3 hr (GAN training slow) |

> Note: we have *paired* T1↔FA, so unlike vanilla CycleGAN we can also report
> supervised SSIM/PSNR — strong reference for generation quality.

---

## 3. DDPM (diffusion)

**Role**: diffusion generation quality upper bound. Unconditional + optionally
conditional (T1→FA via concat or classifier-free guidance).

| Item | Spec |
|---|---|
| arch | U-Net DDPM, T=1000 (or DDIM 50-step sampling) |
| variants | (a) uncond per-modality; (b) cond T1→FA |
| eval-gen | FID / SSIM / PSNR |
| eval-repr | U-Net mid-block features → GMM (weak,diffusion repr ≠ clustering) |
| ref | Ho et al. 2020 |
| files | `adni_pilot/baselines/ddpm/` |
| cost | **high** (~半天 train + slow sampling) — do later |

---

## 4. naive Flow Matching (CFM)

**Role**: the "flow" CLAST's proposal name evokes, done the modern way.
Conditional Flow Matching learns a velocity field; ODE-integrate to generate.

| Item | Spec |
|---|---|
| arch | velocity net v_θ(x_t, t); latent-space or pixel-space |
| target | T1↔FA transport (x0=source modality, x1=target) |
| loss | CFM regression: ‖v_θ(x_t,t) − (x1−x0)‖² (rectified/OT-CFM) |
| sampling | Euler/RK ODE, N steps |
| eval-gen | SSIM/PSNR + intermediate ODE-state viz (analog to our flow_morph) |
| eval-repr | if latent-space CFM: cluster the latent |
| ref | Lipman et al. 2023 (Flow Matching); Liu 2022 (Rectified Flow) |
| files | `adni_pilot/baselines/flowmatch/` |
| cost | ~1-2 hr |

> Direct conceptual rival to CLAST's normalizing flow. The step-by-step
> `flow_morph.py` viz we built transfers naturally to ODE integration steps.

---

## 5. Mean Flow

**Role**: one-step generation (2025). Learns *average* velocity over the interval
so you can jump t=0→1 in a single step.

| Item | Spec |
|---|---|
| arch | mean-velocity net u_θ(x_t, r, t) with the MeanFlow identity |
| target | same T1↔FA transport as §4 |
| sampling | 1-step (and few-step for comparison) |
| eval-gen | SSIM/PSNR @ 1-step vs CFM @ N-step |
| ref | MeanFlow (2025) |
| files | `adni_pilot/baselines/meanflow/` |
| cost | ~1-2 hr |

---

## 6. Unified comparison table (final)

| Method | subj acc | NMI | ARI | SSIM(T1→FA) | PSNR | 1-step? | repr-evolution |
|---|---|---|---|---|---|---|---|
| **CLAST (ours)** | 0.442 | 0.099 | 0.032 | – | – | – | TODO |
| naive VAE | | | | – | – | – | |
| CycleGAN | | | | | | ✗ | |
| DDPM | – | – | – | | | ✗ | |
| CFM | | | | | | ✗ | |
| MeanFlow | | | | | | ✓ | |

→ write into `X_files/RESULTS_ADNI.md` as a new "Baselines" section.

---

## Recommended order

| # | Task | Why this order | Cost |
|---|---|---|---|
| 0 | **repr tracker** | every later run reuses it | 1 hr |
| 1 | **naive VAE** | cheapest, direct CLAST ablation | 30 min |
| 4 | **CFM** | direct flow rival, reuses our viz | 1-2 hr |
| 5 | **MeanFlow** | extends CFM code | 1-2 hr |
| 2 | **CycleGAN** | different family, gen-quality ref | 2-3 hr |
| 3 | **DDPM** | most expensive, gen upper bound | 半天 |
| 6 | **unified table** | after all done | — |

> Env note: CFM/MeanFlow/DDPM may want newer torch; `brain` (torch 1.12) should
> still work for these from-scratch impls. CycleGAN fine on `brain`.

---

## Shared infra to build once

- `adni_pilot/baselines/common.py` — shared dataset loaders (reuse `paired_dataset.py`),
  SSIM/PSNR metrics, repr-probe eval, fixed viz batch
- all baselines write to `adni_pilot/baselines/<name>/results/`
- all call `repr_tracker` so the evolution comparison is uniform
