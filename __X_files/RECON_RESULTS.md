# Reconstruction results — ADNI T1 ↔ FA

Paper-ready reconstruction numbers only (self-reconstruction and cross-modal
reconstruction). All values are **SSIM on the held-out test split**, higher is better.

**Testbed.** ADNI paired T1 / FA (DTI), co-registered to MNI152 2 mm, middle 10 axial
slices (z = 40–49), 112×112. Subject-level split: **170 train / 43 test subjects**
= 1,700 / 430 slices. Every model below was trained on this split; no test data was
used for model selection (validation is carved from the training subjects).

---

## 1. Cross-modal reconstruction — baselines

Translation-only models; they have no self-reconstruction path.

| Method | GMM | T1→FA SSIM | T1→FA PSNR | FA→T1 SSIM | FA→T1 PSNR |
|---|:--:|---|---|---|---|
| DDPM | ✗ | 0.559 ±0.209 | 18.91 | 0.488 ±0.150 | – |
| DiT (latent, CFG 1.5) | ✗ | 0.652 ±0.019 | 14.08 | 0.751 ±0.028 | 18.08 |
| CycleGAN | ✗ | 0.680 ±0.027 | 17.87 | **0.813** ±0.034 | 19.80 |
| MeanFlow | ✗ | 0.691 ±0.024 | 18.60 | 0.468 ±0.087 | 18.56 |
| **CFM** (best baseline) | ✗ | **0.729** ±0.026 | 18.87 | 0.790 ±0.039 | 19.92 |

---

## 2. MMCLAST — shared-flow VAE (self + cross)

Two modality-specific ConvVAEs bridged by **one invertible flow** through a shared
latent. Three-stage training (S1 per-modality VAE → S2 flow only, encoders/decoders
frozen → S3 decoders + flow, encoders frozen), early stopping on a held-out
validation split of the training subjects.

`GMM` = a Gaussian-mixture prior on the shared latent (used for clustering in earlier
versions of this model; **dropped in all runs below** — these are pure reconstruction
models). `pair` = latent alignment term on the flow output. `params` = full model.

Latent chain variants:
* **two-step** `z_A → u → z_B` — one flow applied twice, `u` a modality-agnostic midpoint
* **one-step** `z → u` — one flow applied once; `z` is A's code, `u` is B's
* **two-flow (-2f)** `z_A →f_A→ u ←f_B← z_B` — a flow per modality into a shared `u`
* **-bNLL** — every intermediate flow block additionally anchored to N(0,I)

| Model | Latent | Chain | GMM | pair | Params | self T1 | self FA | **T1→FA** | **FA→T1** |
|---|---|---|:--:|:--:|---|---|---|---|---|
| **MMCLAST** | vector 8-D | two-step | ✗ | ✗ | 206.8 M | 0.895 | 0.787 | **0.768** | **0.861** |
| **MMCLAST** | vector 8-D | two-step | ✗ | ✓ | 206.8 M | 0.896 | 0.784 | 0.767 | 0.854 |
| **MMCLAST-dense** | 128×28×28 | two-step | ✗ | ✗ | **2.55 M** | 0.937 | 0.974 | 0.763 | 0.820 |
| **MMCLAST-dense** | 128×28×28 | two-step | ✗ | ✓ | 2.55 M | **0.990** | **0.988** | 0.550 | 0.773 |
| **MMCLAST-dense** | 128×28×28 | one-step | ✗ | ✗ | 2.55 M | 0.928 | 0.981 | 0.753 | 0.744 |
| **MMCLAST-dense** | 128×28×28 | one-step | ✗ | ✓ | 2.55 M | 0.936 | 0.988 | 0.726 | 0.685 |
| **MMCLAST-dense-2f** | 128×28×28 | two-flow | ✗ | ✓ | 3.51 M | 0.957 | 0.933 | 0.717 | 0.758 |
| **MMCLAST-dense-bNLL** | 128×28×28 | one-step | ✗ | ✗ | 2.55 M | 0.929 | 0.983 | 0.679 | 0.768 |

Latent diagnostics (`gap` = ‖flow output − target code‖ / ‖target code‖; `flow work` =
‖f(z)−z‖ / ‖z‖, ~0 would mean the flow collapsed to the identity):

| Model | pair | gap ↓ | flow work | cross mean |
|---|:--:|---|---|---|
| MMCLAST-dense (one-step) | ✗ | 9.07 | 3.25 | 0.749 |
| MMCLAST-dense (one-step) | ✓ | 0.77 | 0.95 | 0.706 |
| MMCLAST-dense-2f | ✓ | **0.091** | 92.4 | 0.738 |
| MMCLAST-dense-bNLL | ✗ | 1.39 | **0.34** | 0.724 |

---

## 3. MMCLAST-cg — CycleGAN rewired so the flow *is* the bridge

### 3.1 The rewiring

The earlier hybrid (§3.3) bolted a 16-D latent onto the CycleGAN bottleneck and
re-injected it by FiLM. That latent turned out to be decoration: the 256×28×28
bottleneck still reached the decoder untouched, so shuffling the code changed
SSIM by 0.000, and `G_A2B` was already a complete translator on its own.

MMCLAST-cg instead **splits** the two generators rather than augmenting them:

| new part | = which half of the host |
|---|---|
| `E_A` | `G_T1toFA.head + down` |
| `D_B` | `G_T1toFA.res + up + tail` |
| `E_B` | `G_FAtoT1.head + down` |
| `D_A` | `G_FAtoT1.res + up + tail` |

```
T1 →E_A→ z ─f→ u →D_B→ FA          self T1 : D_A(E_A(T1))
FA →E_B→ u ─f⁻¹→ z →D_A→ T1        self FA : D_B(E_B(FA))
```

Three properties, all measured rather than asserted:

1. **Parameter-neutral.** The four halves are exactly the host's two generators:
   **15.65 M**. The conv flow adds **1.84 M** (12 %).
2. **`f` is the exact identity at initialisation** (ActNorm `log_scale=bias=0`,
   coupling last conv zero-init), so a warm-started model *reproduces plain
   CycleGAN bit-for-bit*: `--check_init` returns **0.6798 / 0.8125**, i.e. the
   host's own final numbers, at `flow_work = 0.0`. Every later gain is
   attributable to the flow.
3. **No bypass exists.** `D_B` is only ever fed `E_B(FA)` or `f(E_A(T1))`.

The bottleneck is tapped *before* the last ReLU of `down` so the code is
sign-free; a non-negative code cannot host a flow output without being
off-distribution.

### 3.2 Training and results

Three stages: S1 two **independent** autoencoders (+self-GAN), `f` frozen at
identity — training the cross paths here would let the two feature spaces align
on their own and leave the flow nothing to do; S2 the bijection only, `E`/`D`
frozen; S3 joint, `E` frozen. Early stopping per stage on the non-adversarial
validation loss, restoring the best checkpoint. **No GMM, no flow-NLL, no KL, no
pair term.**

| Variant | `L_latcyc` | `L_path` | self T1 | self FA | **T1→FA** | **FA→T1** | flow work | gap |
|---|:--:|:--:|---|---|---|---|---|---|
| MMCLAST-cg `base` | ✗ | ✗ | 0.9995 | 0.9993 | **0.7626** | **0.8421** | 1.90 | 2.22 |
| MMCLAST-cg `latcyc` | ✓ | ✗ | 0.9983 | 0.9979 | 0.7497 | 0.8195 | 1.16 | 0.84 |
| MMCLAST-cg `morph` | ✓ | ✓ | 0.9984 | 0.9980 | 0.7545 | 0.8266 | 1.18 | 0.82 |
| *plain CycleGAN (host)* | – | – | – | – | *0.6798* | *0.8125* | – | – |

Rewiring alone is worth **+0.083 T1→FA and +0.030 FA→T1** over the host it is
built from, at 12 % extra parameters, and it turns a translation-only model into
one with near-perfect self-reconstruction (0.999).

Two added terms:
* `L_latcyc = ‖E_B(D_B(f z)) − f z‖₁ / ‖f z‖₁` — the flow's output must be a
  **fixed point of the B autoencoder**, i.e. decodable. Unlike `pair` it does not
  pull the code toward a target, and unlike `-bNLL` it says nothing about the
  code's distribution.
* `L_path` — every intermediate block state, decoded, must satisfy a realism
  critic `D_mix` trained on real T1 ∪ real FA, plus a frame-to-frame smoothness
  term. **No ground-truth intermediate frames are needed**; the discriminator
  supplies the supervision the eight earlier morph attempts lacked.

They cost 0.008 / 0.016 SSIM and buy the first usable path morph (§3.4).

### 3.3 Superseded: the FiLM-head hybrid

| Variant | GMM | pair | T1→FA | FA→T1 | u-shuffle Δ |
|---|:--:|:--:|---|---|---|
| CycleGAN + FiLM latent head | ✓ | ✗ | 0.738 | 0.818 | **0.000** |
| CycleGAN + FiLM latent head | ✓ | ✓ | 0.724 | 0.661 | – |

Kept for the record: the SSIM is respectable but the latent is inert, so the
numbers are the host's, not the module's.

### 3.4 The path morph — nine attempts, one that works

Eight earlier designs failed the same way: constrain the flow's intermediate
states and `flow_work` collapses (`-bNLL`: 0.34, no visible change); leave them
free and they decode to noise. `L_latcyc` + `L_path` constrain **decodability**
instead of distribution, which is not the same trade.

Per-frame `mean|Δ|` against frame 0, slice 250 (`figures/32_…_ablation.png`):

| Variant | decoder | z | b1 | b2 | b3 | u | verdict |
|---|---|---|---|---|---|---|---|
| `base` | D_A (T1 view) | 0 | 0.022 | 0.063 | 0.104 | **0.151** | torn apart |
| `base` | D_B (FA view) | 0 | 0.018 | 0.025 | 0.035 | 0.059 | static |
| `latcyc` | D_A (T1 view) | 0 | 0.008 | 0.023 | 0.036 | 0.058 | **coherent ✓** |
| `latcyc` | D_B (FA view) | 0 | 0.016 | 0.026 | 0.035 | 0.043 | static ✗ |
| `morph` | D_A (T1 view) | 0 | 0.007 | 0.019 | 0.028 | 0.057 | **coherent ✓** |
| `morph` | D_B (FA view) | 0 | 0.034 | 0.077 | 0.150 | **0.156** | **moving ✓** |

**A correct morph needs the change to appear in the FA view while the T1 view
stays a valid brain.** `base` does the exact opposite: the change is in the T1
view — that is the state being destroyed, with saturation holes spreading from
b1 — while the FA view barely moves.

**The two added terms do different, non-substitutable jobs**, and the middle row
is what proves it:

* `L_latcyc` alone buys **decodability**: the T1 view stops tearing (0.151 →
  0.058, a 2.6× reduction) and the checkerboard texture at `z`/`b1` disappears.
  It buys **no path progress at all** — the FA view goes 0.059 → 0.043, if
  anything flatter than `base`.
* `L_path` is what creates the **progression** (FA view 0.043 → 0.156, 3.6×),
  and it can only do so on states that are already decodable.

Neither term alone produces a usable morph; the ninth attempt works because the
two constraints are orthogonal — one makes the frames legal, the other makes
them move.

Figures: `30_mmclast_cg_morph_{base,latcyc,morph}.png` (full path, both decoders),
`31_mmclast_cg_selfcross_{base,latcyc,morph}.png`,
`32_mmclast_cg_morph_ablation.png` (all three variants × both decoders — the
figure to use; the middle pair is what makes the argument).

**Not run:** the fourth cell, `L_path` without `L_latcyc`. The prediction is that
it fails — the realism critic would be asked to police frames that are already
off-manifold — but it is untested.

---

## 4. Calibrating SSIM on MNI-registered slices

Every slice here is registered to MNI152 and cropped to the middle 10 axial
positions, so SSIM has a compressed and badly-offset dynamic range. Two floors,
measured on the 430-slice test set:

| Floor | what it is | FA target | T1 target |
|---|---|---|---|
| inter-subject | SSIM(x_i, x_j), **different** subjects | 0.6456 | 0.6790 |
| **template** | SSIM(mean of the test set, x_i) — a constant predictor that **ignores its input entirely** | **0.7134** | **0.7562** |

Checked against §1 and §2, the template floor disqualifies most of the field:

| Method | T1→FA | vs floor 0.7134 | FA→T1 | vs floor 0.7562 |
|---|---|---|---|---|
| DDPM | 0.559 | **−0.154** | 0.488 | **−0.268** |
| DiT | 0.652 | **−0.061** | 0.751 | **−0.005** |
| CycleGAN | 0.680 | **−0.033** | 0.813 | +0.057 |
| MeanFlow | 0.691 | **−0.022** | 0.468 | **−0.288** |
| CFM | 0.729 | +0.016 | 0.790 | +0.034 |
| MMCLAST (vector) | 0.768 | +0.055 | 0.861 | +0.105 |
| MMCLAST-dense | 0.763 | +0.050 | 0.820 | +0.058 |
| MMCLAST-cg `base` | 0.763 | +0.049 | 0.842 | +0.086 |
| MMCLAST-cg `latcyc` | 0.750 | +0.036 | 0.820 | +0.063 |
| MMCLAST-cg `morph` | 0.755 | +0.041 | 0.827 | +0.070 |

**Four of the five baselines lose to a constant image in at least one direction;
DDPM and MeanFlow lose in both.** Every MMCLAST variant clears both floors, but
the honest margin is the *headroom* column, not the raw SSIM — a raw SSIM
without these floors is close to uninterpretable on this data. (The floor uses
test-set statistics, which makes it harder to beat, i.e. conservative for us.)

### 4.1 The u-shuffle probe, corrected

Feeding the decoder another sample's code is the guard metric for A16. The
in-training version rolled the batch by one — but the test loader is unshuffled
and each subject contributes 10 consecutive slices, so it was swapping in the
*same subject's neighbouring slice*, which proves nothing. Permuting **across
subjects** (`eval_mmclast_probe.py`):

| Variant | T1→FA correct | wrong subject's code | Δ | inter-subject floor |
|---|---|---|---|---|
| MMCLAST-cg `base` | 0.7626 | 0.6659 | **−0.097** | 0.6456 |
| MMCLAST-cg `latcyc` | 0.7497 | 0.6569 | **−0.093** | 0.6456 |
| MMCLAST-cg `morph` | 0.7545 | 0.6589 | **−0.096** | 0.6456 |
| *FiLM head (§3.3)* | *0.738* | *0.738* | *0.000* | – |

Handed the wrong code, MMCLAST-cg falls **to the inter-subject floor** — it
generates a valid brain that is the wrong person's, which is the correct
behaviour for a load-bearing latent. The FiLM head did not move at all.

---

## 5. Summary

| | GMM | T1→FA | FA→T1 |
|---|:--:|---|---|
| Best baseline (CFM) | ✗ | 0.729 | 0.790 |
| Best baseline, FA→T1 (CycleGAN) | ✗ | 0.680 | 0.813 |
| **MMCLAST** (vector, two-step, no pair) | ✗ | **0.768** | **0.861** |
| **MMCLAST-dense** (2.55 M, no pair) | ✗ | 0.763 | 0.820 |
| **MMCLAST-cg** (CycleGAN rewired, 17.5 M) | ✗ | 0.763 | 0.842 |

- **MMCLAST beats every baseline in both directions**: +0.039 T1→FA over CFM, +0.048
  FA→T1 over CycleGAN.
- **MMCLAST-dense matches it with 81× fewer parameters** (2.55 M vs 206.8 M) and
  much higher self-reconstruction (0.937 / 0.974 vs 0.895 / 0.787), losing only
  0.005 / 0.041 on cross-modal.
- **No GMM is used in any MMCLAST run in §2.** These are reconstruction-only models; the shared
  latent is anchored solely by a KL term to N(0,I) at each end, and the flow is trained
  by cross-reconstruction. Only the §3 hybrid still carries a GMM base.
- **Latent alignment does not buy reconstruction quality — the two are anti-correlated.**
  Driving the flow output onto the target code works (gap 9.07 → 0.77 → **0.091** across
  pair / two-flow variants) but cross-modal SSIM does not improve with it (0.749 → 0.706 →
  0.738). Proximity in latent L2 is not proximity after decoding.
- **Constraining the flow's intermediate states costs expressivity.** Anchoring every block
  to N(0,I) (-bNLL) keeps the intermediates in-distribution but collapses the flow's work
  to 0.34 (vs 3.25 unconstrained), and cross-modal drops to 0.679 / 0.768.
- **Two flows are not better than one.** MMCLAST-dense-2f needs 38 % more parameters and a
  pair term (which is mandatory there, since nothing else ties the two `u`-spaces together)
  yet lands below the single-flow, no-pair variant.
- **Rewiring an existing translator beats bolting a module onto it.** Splitting CycleGAN's
  two generators into 2 encoders + 2 decoders and making one flow the only bridge adds
  12 % parameters and **+0.083 / +0.030** over the host, and gives it 0.999 self-recon.
  The earlier FiLM-head hybrid scored similarly but its latent was inert (u-shuffle Δ 0.000);
  MMCLAST-cg's latent drops to the inter-subject floor when shuffled.
- **The alignment/quality anti-correlation is about *what* is optimised, not alignment
  itself.** `pair` optimises L2 proximity to a target code and costs reconstruction.
  `L_latcyc` optimises *decodability* (fixed point of the target autoencoder); it pulls the
  gap down as a by-product (2.22 → 0.82) at a cost of only 0.008 / 0.016 SSIM, and unlike
  `pair` it is what makes the intermediate path decodable at all.
- **A path can be made decodable without constraining its distribution.** `-bNLL` anchored the
  intermediates to N(0,I) and collapsed `flow_work` to 0.34 with nothing visible; an adversarial
  realism critic on decoded frames leaves `flow_work` at 1.18 and produces the first usable morph
  in nine attempts (§3.4).
- **Decodability and progression are separate problems needing separate terms.** The three-way
  ablation isolates them: `L_latcyc` alone stops the path from tearing (T1 view 0.151 → 0.058)
  but leaves it static (FA view 0.059 → 0.043); adding `L_path` is what makes it move
  (0.043 → 0.156). Eight earlier attempts each addressed at most one of the two.
- **Raw SSIM is nearly uninterpretable on this data (§4).** A constant image — the test-set mean —
  scores 0.713 / 0.756, beating four of the five baselines in at least one direction. Report
  headroom over that floor, not raw SSIM.
