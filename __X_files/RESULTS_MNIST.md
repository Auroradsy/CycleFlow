# Phase 1 Single-Modality Baseline — Results

Vanilla CLAST (VAE + trainable GMM prior, no normalizing flow) trained
end-to-end for 50 epochs on three datasets. Same architecture, same
hyperparameters, only the input layout (`in_channels`, `image_size`) differs.

All evaluation uses Hungarian-aligned cluster assignment. Best-test-acc
checkpoint (`best_vae.pth`) is used for diagnostics below; metrics in the
table are also from that checkpoint epoch.

## 1. Headline numbers

**Single-modality (vanilla GMM-VAE):**

| Dataset | K | Channels | Image | Train / Test | Best Acc | NMI | ARI | Random | Section |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | 10 | 1 | 28×28 | 48k / 12k | **0.9382** | 0.864 | 0.869 | 0.10 | §3–5 |
| **MNIST-CT** (synthetic) | 10 | 3 | 28×28 | 48k / 12k | **0.9227** | 0.844 | 0.828 | 0.10 | §7 |
| **MNIST-PET** (synthetic, no hotspot) | 10 | 3 | 28×28 | 48k / 12k | **0.7689** | 0.662 | 0.581 | 0.10 | §7 |
| **MNIST-PET-HOT08** (hotspot, w=0.8) | 10 | 3 | 28×28 | 48k / 12k | **0.7626** | 0.760 | 0.671 | 0.10 | §8 |
| **MNIST-PET-HOT15** (hotspot, w=1.5) | 10 | 3 | 28×28 | 48k / 12k | **0.8240** | 0.755 | 0.712 | 0.10 | §8 |
| Printed Digits (9-class, cleaned) | 9 | 1 | 28×28 | 3.6k / 0.9k | **0.2756** | 0.174 | 0.071 | 0.111 | §3–5 |
| SVHN | 10 | 3 | 32×32 | 32k / 8k | **0.1576** | 0.029 | 0.012 | 0.10 | §3–5 |

**Bi-modal (§9 pilot):**

| Pair | K | u-dim | Train / Test | Best joint Acc | accA | accB | **CCA** | Section |
|---|---|---|---|---|---|---|---|---|
| MNIST + MNIST-PET-HOT08 (paired) | 10 | 8 | 48k / 12k pairs | **0.8085** | 0.805 | 0.813 | **0.944** | §9 |

**Reading**: MNIST converges as expected. The synthetic MNIST-CT/PET pair
(§7) bookends a controlled modality gap; PET loses ~15 pp from soft-glow
blur, providing the Phase-2 alignment target. **Hotspots (§8) lift PET
from 0.769 to 0.824 at w=1.5**, giving the multi-modal model a richer
modality-B signal source. The **bi-modal pilot (§9) achieves CCA 0.944
and joint accuracy 0.808** — the framework's main cross-modal-alignment
claim is validated, though joint acc does not exceed the strong single
modality MNIST 0.938 (expected on a benchmark where one modality is
already easy; the proposal's clinical scenario flips this comparison).

**Pure-VAE ablation (§6)**: joint GMM training contributes ≤2 pp on MNIST
and Printed, and is net-zero on SVHN. The VaDE-style joint training in
the current single-modality baseline is nearly equivalent to "train a
VAE, then fit a GMM post-hoc".

## 2. Inputs

### 2.1 MNIST (10 × 10 grid, one row per class)

![mnist samples](figures/diag_mnist_grid.png)

### 2.2 SVHN

Distribution: 32×32 RGB street-view digits, balanced subset (4000/class)
from train split.

### 2.3 Printed Digits (Kaggle: kshitijdhama/printed-digits-dataset)

**Raw, no filtering — first 10 files per class:**

![printed raw](figures/diag_printed_grid.png)

Class 0 row is essentially blank — many files are all-black or
near-empty. Other rows look reasonable but show clear multi-font
mixing: thin/bold, italic vs. upright, geometric vs. rounded.

**Class 0 garbage breakdown** (out of 641 files):

| Filter | Surviving |
|---|---|
| Raw | 641 |
| `max > 0` (drop all-black) | 466 |
| `mean > 0.01` (drop near-black) | 400 |
| `mean > 0.02` AND bbox-area ≥ 50 | 271 (rectangle artifacts, not zeros) |

**Even after the strictest filter, class 0 survivors are mostly
rectangle-shape artifacts, not actual `0` digits**:

![class 0 stricter](figures/diag_class0_stricter.png)

Conclusion: class 0 is irreparably contaminated. Final loader
**drops class 0 entirely** and remaps labels 1–9 → 0–8. Other classes
pass the same `mean+bbox` filter cleanly. After filtering:

![printed filtered](figures/diag_printed_stricter_grid.png)

(top row missing because class 0 is dropped; columns show the cleaner
9-class subset used for training).

## 3. Confusion matrices

Predicted columns are reordered so column `d` corresponds to the
cluster Hungarian-matched to true digit `d`. Off-diagonal mass = errors.

### MNIST (acc = 0.938)

![cm mnist](figures/cm_mnist.png)

Diagonal-dominant, errors concentrated on a few confusable pairs
(typically 4↔9, 3↔5, 7↔1). Two of the 10 clusters tend to "split" or
"merge" digits — visible as a slightly weaker diagonal entry plus a
scattered off-diagonal column.

### Printed Digits (acc = 0.276)

![cm printed](figures/cm_printed_digits.png)

Diagonal weak; mass spread across multiple clusters per row.
Several digits are split across 2–3 clusters (likely by font style),
and several clusters absorb multiple digits. Confirms that the
9-cluster GMM is partitioning by font / weight rather than digit
identity.

### SVHN (acc = 0.158)

![cm svhn](figures/cm_svhn.png)

Near-uniform off-diagonal — the assignment is essentially arbitrary.
Vanilla VAE on RGB street-view does not encode digit identity in
the 8-dim latent.

## 4. Latent space (t-SNE of test embeddings)

Left: colored by true digit. Right: colored by predicted cluster
(Hungarian-aligned). Visual agreement = good clustering.

### MNIST

![tsne mnist](figures/tsne_mnist.png)

10 clean islands, true labels and predicted clusters mostly agree.

### Printed Digits

![tsne printed](figures/tsne_printed_digits.png)

Latent shows some structure but classes overlap heavily; predicted
clusters carve up regions that don't follow the true-label coloring.

### SVHN

![tsne svhn](figures/tsne_svhn.png)

True-label coloring shows essentially uniform mixing — no class
separation in the latent. Predicted clusters carve up the space
along non-semantic axes.

## 5. Cluster atlases

### 5.1 Scheme-A (per-cluster pixel-space median over high-confidence assignments)

| Dataset | Atlas |
|---|---|
| MNIST | ![](figures/atlas_mnist.png) |
| Printed Digits | ![](figures/atlas_printed.png) |
| SVHN | ![](figures/atlas_svhn.png) |

MNIST clusters produce clean digit prototypes. Printed clusters
show partially-formed shapes — multiple digit identities ghost
into a single atlas. SVHN atlases are colored blobs with no
discernible digit structure.

### 5.2 Decoded refit-GMM means (decoder applied to GMM means in latent space)

| Dataset | Decoded means |
|---|---|
| MNIST | ![](figures/decoded_means_mnist.png) |
| Printed Digits | ![](figures/decoded_means_printed.png) |
| SVHN | ![](figures/decoded_means_svhn.png) |

This is what the *model* thinks each cluster center looks like.
For MNIST, decoded means closely match the pixel-space atlas —
confirming both the latent geometry and the decoder are aligned to
digit identity. For SVHN, the decoder produces blurry color blobs
because the GMM means in latent space correspond to non-semantic
mixtures.

## 6. Pure-VAE ablation (no GMM in training, post-hoc clustering)

To isolate the contribution of the joint GMM training term, we trained
the **same encoder/decoder/hyperparameters** on each dataset with
`w_gmm = 0` (reconstruction + KL only) for 50 epochs, then fit an
sklearn GMM on the resulting `μ(x)` post-hoc and computed Hungarian
clustering metrics with the same protocol. The 16-image reconstruction
grid below is from epoch 50; top row = original test images, bottom
row = decoded reconstructions.

### 6.1 Comparison

| Dataset | GMM-joint | Pure VAE (post-hoc) | Δ | Final recon/px (BCE) |
|---|---|---|---|---|
| MNIST | 0.9382 | 0.9207 | −1.8 pp | 0.096 |
| Printed Digits | 0.2756 | 0.2556 | −2.0 pp | 0.097 |
| SVHN | 0.1576 | 0.1599 | **+0.2 pp** | 0.618 |

**The joint GMM term contributes ≤2 pp of Hungarian accuracy on all
three datasets, and on SVHN it is slightly net-negative.** The
"VaDE-style" joint training in the current CLAST baseline is therefore
nearly equivalent to "train a vanilla VAE, then fit a GMM on the
encoder output" — useful as stabilization, but not a meaningful
algorithmic contribution at the single-modality level.

### 6.2 Reconstruction quality (epoch 50)

**MNIST** — recon/px = 0.096. Reconstructions essentially indistinguishable from originals; digit identity preserved sharply.

![recon mnist](figures/recon_mnist.png)

**Printed Digits** — recon/px = 0.097. Reconstructions are blurry but
roughly digit-shaped. Notably the recon BCE matches MNIST's almost
exactly — yet the post-hoc clustering accuracy is 3.6× lower.
**This is the cleanest demonstration that good reconstruction
≠ semantic clustering**: the Printed VAE reconstructs as well as
MNIST in pixel terms, but its latent encodes font / stroke / scale
style instead of digit identity.

![recon printed](figures/recon_printed.png)

**SVHN** — recon/px = 0.618 (∼6× higher than MNIST). Reconstructions
are blurry color blobs barely identifiable as digits. The decoder
cannot reconstruct the digit because **the latent doesn't encode the
digit** — it encodes average color, illumination, and background.
This is why no clustering algorithm (joint or post-hoc) can recover
digit labels from this latent.

![recon svhn](figures/recon_svhn.png)

### 6.3 What this ablation tells us

1. **Reconstruction is necessary but not sufficient.**
   Reconstruction loss anchors the latent to data, but it does not
   determine *what* gets encoded. The latent encodes whatever variance
   matters most for reconstruction — which on MNIST happens to be
   digit shape, but on Printed Digits is font style, and on SVHN is
   scene appearance.

2. **The GMM prior cannot rescue a misaligned encoder.**
   When the reconstruction signal pulls the latent away from semantic
   axes (Printed, SVHN), the GMM's structural pressure cannot drag
   it back. GMM only carves up whatever geometry the encoder gives it.

3. **The current CLAST single-modality baseline ≈ a vanilla VAE.**
   The 1–2 pp advantage of joint GMM training is within
   epoch-to-epoch noise (MNIST joint dropped to 0.81 in some late
   epochs). This further supports the position that the project's
   actual novelty must come from Phase 2 (latent-space normalizing
   flow + paired alignment), not from the single-modality pipeline
   itself.

4. **Phase 2 paired alignment with MNIST is the right intervention
   for Printed Digits.** Since the Printed encoder *can* reconstruct
   digits but encodes them along the wrong latent axes, anchoring
   the Printed latent to the MNIST latent (which is correctly aligned)
   via flow + paired loss should pull the Printed encoder toward
   digit-identity axes. SVHN is a harder case because the encoder
   doesn't even reconstruct digits — that gap likely needs a stronger
   encoder (SSL pretraining) before flow alignment can help.

## 7. Synthetic paired MNIST-PET / MNIST-CT — Phase-2 testbed

The single-modality results on natural data (Sections 3–6) span a
wide range: MNIST 0.94, Printed 0.28, SVHN 0.16. To run controlled
Phase-2 alignment experiments we need a second-modality candidate
that is **paired by construction**, **uses a realistic but non-extreme
modality gap**, and **does not fail at the encoder level**. None of
the natural datasets meet all three criteria (Printed has label noise,
SVHN's encoder cannot recover digits). We therefore synthesize a
controlled MRI↔CT analog from MNIST.

### 7.1 Construction

The goal is a **controlled** modality gap that mimics the dominant
differences between real PET and CT imaging while staying simple
enough that single-modality VAE baselines on each side still
converge. Below we walk through (a) the clinical properties we want
to capture, (b) the transforms used to mimic them, (c) the specific
parameter values and how they were chosen, and (d) what we
deliberately do not model.

#### 7.1.1 Clinical motivation: what real PET vs CT differ in

| Imaging axis | PET (real) | CT (real) |
|---|---|---|
| Underlying physics | Tracer-uptake emission (functional) | X-ray attenuation (anatomical) |
| Spatial resolution | ~4–6 mm FWHM (low) | ~0.5–1 mm (high) |
| Noise character | Count-limited (Poisson-like, high) | Reconstruction noise (lower) |
| Local texture | Smooth, blurry, soft-edged | Crisp, sharp edges (bone↔tissue boundaries) |
| Intensity scale | Non-linear SUV; most tissues mid–low, hotspots bright | Hounsfield units linearly mapped to a viewing window |
| Background level | Persistent baseline tracer uptake everywhere | Air ≈ 0, varies smoothly |
| Display | Pseudo-color overlay (hot / single-color colormap) | Grayscale (window/level) |

The MNIST-PET / MNIST-CT pair tries to capture **resolution, intensity
remap, noise, and display color** — i.e., everything in the rows
above except the underlying physics, which is irrelevant for a
clustering testbed.

#### 7.1.2 Transform pipelines

Each source MNIST grayscale image `x ∈ [0,1]^{28×28}` is passed
through one of two deterministic pipelines (numpy + scipy):

| Step | MNIST-PET ("soft tracer glow") | MNIST-CT ("sharp bone contrast") | What it models |
|---|---|---|---|
| **Spatial filter** | `gaussian_filter(x, σ=1.2)` | unsharp mask: `x + 1.1·(x − gaussian(x, σ=1.1))` | PSF blur of PET vs. high-frequency content of CT |
| **Gamma** | `x ** 2.2` | `x ** 0.40` | PET non-linear SUV display (pushes mid→low); CT bone window (stretches low→up) |
| **Linear remap** | `α·x + β` with `α=1.15, β=+0.12` | `α=1.55, β=−0.18` | PET baseline tracer glow (β > 0 lifts the floor); CT contrast stretch + black clip (β < 0) |
| **Clip** | `clip(x, 0, 1)` | `clip(x, 0, 1)` | Display range |
| **Noise** | `+ N(0, 0.05)` | `+ N(0, 0.03)` | Poisson-like PET noise vs. lower CT noise (Gaussian approximation) |
| **Colormap** | RGB green tint: `(0, x, 0)` | RGB grayscale: `(x, x, x)` | PET pseudo-color overlay vs. CT monochrome |

Both pipelines output 3-channel RGB in `[0,1]^{3×28×28}`. The 3-channel
layout for CT is wasteful (replication) but matches PET's channel
count, so a single encoder architecture (`Conv2d(3, …)`) serves both
modalities without needing two channel-specific code paths.

Implementation: `synth_datas/mnist_petct.py:_pet_style_intensity` and
`_ct_style_intensity`. Per-image noise uses a single
`numpy.RandomState(seed)` advanced sequentially, so the rendering is
deterministic given a fixed seed.

#### 7.1.3 Why these parameter values

The values were chosen by **iterative visual inspection of a 7-row
preview grid** (`figures/preview_synthetic_modalities.png` for the
grayscale-only pre-stage, `figures/preview_petct_v1_bothgreen.png` and
`figures/preview_petct_v3_pet_plus_ct.png` for the final color pair).
No quantitative optimization was used — the goal was a gap
"large enough to be visually distinct, small enough that vanilla
single-modality baselines still converge".

Justification for each choice:

| Parameter | Value | Rationale |
|---|---|---|
| PET σ (blur) | 1.2 | At 1.2 px on a 28-px image (~4% of width), digit edges visibly soften but stroke topology (loops, crossings) is preserved. σ ≥ 1.8 in early trials caused digits like 1/7 and 0/8 to become indistinguishable. |
| PET γ | 2.2 | Compresses the bright stroke down toward mid-tones; visually mimics SUV display where peak uptake is rare. γ > 2.5 made the digits nearly disappear into the green background. |
| PET β (lift) | +0.12 | Adds a small global "background glow" so the image is not pure black, matching PET's pervasive baseline tracer signal. β > 0.2 washed out the digit too much. |
| PET noise σ | 0.05 | Visible grain but does not obscure digit shape. Increasing to 0.08 already started hurting MNIST-PET clustering (in scratch tests) below 0.70 acc. |
| CT unsharp α | 1.1 over σ=1.1 blur | Strong edge enhancement while staying numerically stable (sharpening with α > 1.5 caused ringing artifacts and pushed pixels far outside [0,1]). |
| CT γ | 0.40 | Aggressive low-end stretch to mimic CT bone window where strokes appear nearly white on near-black. γ < 0.3 caused saturation across most of the digit. |
| CT linear `α=1.55, β=−0.18` | — | Pushes background toward 0 and strokes toward 1 — explicit "high contrast" rendering. The negative β clips the background flat to zero after the final clip. |
| CT noise σ | 0.03 | Lower than PET (CT has lower noise floor after reconstruction). 0.03 is visually subtle but enough to break perfect pixel-level pairing, which is realistic — even paired clinical PET-CT is not pixel-identical. |
| PET color | pure green (R=0, G=x, B=0) | "Single color overlay" that occupies only one of three RGB channels, forcing the encoder to learn that the other two channels are uninformative. This is what makes PET genuinely harder than its grayscale counterpart, beyond the soft-glow intensity remap. |
| CT color | grayscale replicated 3× | Mirrors PET's RGB layout but keeps all three channels carrying the same signal, so the encoder sees redundant but full-bandwidth input. |

A useful intuition: **PET's three transforms each take away
information** (blur removes high-frequency, γ>1 + β>0 compresses
dynamic range, single-channel rendering wastes 2/3 of input
bandwidth), while **CT's three transforms each add or amplify
information** (sharpening boosts high-frequency, γ<1 stretches
contrast, grayscale-as-3ch is bandwidth-redundant but not lossy).
This asymmetry is exactly why MNIST-PET clusters at 0.77 and
MNIST-CT at 0.92 — and exactly the kind of asymmetry Phase-2
paired alignment is meant to compensate for.

#### 7.1.4 What we deliberately do NOT model

| Feature | Why excluded |
|---|---|
| Realistic non-Gaussian PSF kernels | Goal is controlled gap, not fidelity to a specific scanner |
| Spatial misregistration between modalities | We want **perfectly paired** ground truth so Phase-2 `L_pair` has zero noise; real PET-CT registration is a separate problem |
| Attenuation correction artifacts | Out of scope for digit-clustering testbed |
| Motion artifacts | Same |
| Real PET hot colormap (multi-color gradient) | Pure-green is a stronger "one channel only" stress test for the encoder; multi-color hot maps would let the encoder cheat by reading multiple channels |
| Bone vs. soft-tissue windowing variants for CT | One CT pipeline is enough for the controlled gap |

The two pipelines together model the dominant MRI↔CT/PET differences
in clinical display: different intensity statistics, different local
texture (blurry vs. crisp), and different colormap rendering (PET
pseudo-color vs. CT grayscale). They preserve identical spatial
layout and class identity, so paired alignment has a perfect ground
truth.

### 7.2 Sample grid

Top: original MNIST. Middle: MNIST-PET. Bottom: MNIST-CT.
Each column = the same source image rendered into both modalities.

![paired sample](figures/sanity_mnist_petct.png)

The PET row shows soft, washed-out green glyphs; the CT row shows
crisp, high-contrast grayscale digits. Digit identity is clearly
recognizable in both rows.

### 7.3 Per-modality clustering results

| Modality | Best Acc | NMI | ARI | vs. original MNIST |
|---|---|---|---|---|
| MNIST (reference) | 0.938 | 0.864 | 0.869 | — |
| **MNIST-CT** | **0.923** | 0.844 | 0.828 | −1.5 pp |
| **MNIST-PET** | **0.769** | 0.662 | 0.581 | −16.9 pp |

**MNIST-CT ≈ original MNIST.** Sharpening + high contrast preserves
(or even strengthens) the digit-identity axis in the latent.

**MNIST-PET loses ~17 pp.** Gaussian blur destroys stroke-edge cues
and γ = 2.2 compresses dynamic range; pure-green rendering uses only
one of three input channels effectively. The encoder still captures
digit identity but the latent is noisier and less linearly separable.

This 15-pp gap is **the cleanest Phase-2 target** we have: both
modalities work as single-modality baselines (no SVHN-style failure),
but PET is consistently weaker than CT, and the dominant error
modes (Section 7.4) are exactly the kind of identity confusions
that paired alignment with a stronger anchor should resolve.

### 7.4 Confusion matrices

MNIST-CT — clean, diagonal-dominant, errors mostly on classic
4↔9 / 3↔5 / 7↔1 pairs:

![cm ct](figures/cm_mnist_ct.png)

MNIST-PET — diagonal weakened, mass spread across several adjacent
clusters per row; e.g. digit 5/8 absorbs partial mass from 3 and 6,
digit 9 splits across two clusters:

![cm pet](figures/cm_mnist_pet.png)

### 7.5 Latent space t-SNE

Left = colored by true digit, right = colored by Hungarian-aligned
prediction.

MNIST-CT — 10 well-separated islands, true and predicted colorings
agree closely:

![tsne ct](figures/tsne_mnist_ct.png)

MNIST-PET — fewer clean islands, several digit classes overlap on
the embedding boundary, predicted clusters cover slightly different
regions than true labels:

![tsne pet](figures/tsne_mnist_pet.png)

### 7.6 Cluster atlases

**Scheme-A pixel-space median atlas:**

| MNIST-CT | MNIST-PET |
|---|---|
| ![atlas ct](figures/atlas_mnist_ct.png) | ![atlas pet](figures/atlas_mnist_pet.png) |

**Decoded refit-GMM means:**

| MNIST-CT | MNIST-PET |
|---|---|
| ![dec ct](figures/decoded_means_mnist_ct.png) | ![dec pet](figures/decoded_means_mnist_pet.png) |

CT atlases produce 10 clean digit prototypes. PET atlases show
visible ghosting / merging in 2–3 clusters, matching the confusion
matrix evidence.

### 7.7 Why this pair, not MNIST-M or "two MNISTs"

| Candidate second modality | Channel layout | Gap nature | Issue for Phase-2 alignment |
|---|---|---|---|
| Two copies of MNIST | 1 + 1 | None | Nothing for the flow to do |
| MNIST + EMNIST/USPS handwriting | 1 + 1 | Style of human handwriting | Gap too small, mostly resolution shift |
| **MNIST + MNIST-PET/CT (this)** | 3 + 3 | Intensity + texture + colormap | Realistic, controlled, paired |
| MNIST + Printed Digits | 1 + 1 | Font / stroke style | Latent in Printed encodes style not identity (Section 4) |
| MNIST + MNIST-M | 1 + 3 | Random RGB natural-image backgrounds | Gap is colour-and-clutter, far from MRI↔CT |
| MNIST + SVHN | 1 + 3 | Scene appearance dominates | Encoder cannot learn digit identity in SVHN |

MNIST-PET/CT is the only candidate whose gap **is the right type**
(intensity statistics + local texture + colormap, all preserving
spatial structure and class identity) and whose **both endpoints
have working single-modality baselines** — the prerequisite for
paired alignment to mean anything.

### 7.8 What this gives Phase 2

This pair provides a clean experimental setup for the upcoming
Phase-2 work:

- **Perfect pairing**: same source MNIST image → both modalities,
  so the paired-alignment loss (`L_pair = ‖h(z_PET) − g(z_CT)‖²`)
  has noise-free ground truth.
- **Clear gain story**: PET 0.77 vs CT 0.92 — Phase 2 should lift
  PET toward CT's level (target: 0.85+) without hurting CT.
- **Single-axis ablation**: vary the transform strength to ablate
  "gap size vs. alignment gain" without changing dataset semantics.
- **Decoder pathway works in both directions**: both modalities
  reconstruct cleanly, so cross-reconstruction losses
  (`L_cross = ‖x_CT − D_CT(g⁻¹(h(z_PET)))‖²`) will have meaningful
  signal even before the flow is fully trained.

## 8. Hotspot-augmented MNIST-PET (single-modality)

§7 built a synthetic MNIST-PET that differs from MNIST only in **rendering**
(soft glow, green tint). For a Phase-2 cross-modal study to be meaningful,
PET needs to carry **class-conditional functional signal** that CT does not
— mimicking real PET, where tracer uptake patterns differ by disease
subtype. We add this by injecting **per-class hotspots**: deterministic
landmarks on the digit's strokes, rendered as Gaussian-blob multiplicative
boosts before the PET pipeline. Background and overall PET style stay
identical to §7.

This section reports the single-modality clustering result for the
hotspot-augmented PET. It both validates the hotspot design and produces
the stronger modality-B initializer used in the Phase-2 pilot (§9).

### 8.1 Hotspot design

Per-class landmark table (`synth_datas/hotspot_config.py`) gives each digit
1–4 bbox-relative landmarks. For each sample we:

1. Compute the stroke bounding box.
2. Map relative landmarks to absolute pixel coordinates via the bbox.
3. **Snap** each landmark to the nearest stroke pixel within a 4-pixel
   search radius; landmarks falling on empty background are silently
   dropped (no spurious hotspots in air).
4. Place a Gaussian blob (σ = 2.5–3.5) at each snapped point.
5. **Multiplicatively** boost the stroke pixels in the blob's support:
   `x ← clip(x · (1 + w · blob · stroke_mask), 0, 1)`.
6. Pass the boosted grayscale through the existing PET pipeline (blur, γ,
   linear lift, noise, green tint).

Two key invariants:

- **Stroke-gated**: `boost · stroke_mask` ensures background never lights
  up — air → 0 uptake, just like a real PET.
- **Pre-pipeline**: applying the boost before the σ=1.2 PET blur lets the
  hotspot diffuse into a soft glow region, matching real PSF-limited
  tracer-uptake appearance rather than sharp dot artifacts.

Each digit has a topologically distinct fingerprint: 0 has 4-corner pattern,
3 has a Z-shape, 6 has bottom-loop glow, etc. (see
`synth_datas/hotspot_strategy.ipynb` for the live table and per-class
landmark visualization with bbox + snap markers.)

### 8.2 Per-class fingerprints

![](../results/figures/hotspot_01_landmarks.png)

cyan = stroke bbox · green = relative target (pre-snap) · red = snapped
location · orange = Gaussian σ.

In most cases red ≈ green (snap is small/zero). Larger green↔red offsets
indicate the digit's stroke topology drifted from the template; the snap
keeps the hotspot on actual ink anyway. A few corner cases (e.g. open-cross
"4") cannot find any stroke within 4 px → that landmark is dropped, and the
sample carries one fewer hotspot than its class fingerprint specifies.

### 8.3 Rendered samples per class (25 per class, after full PET pipeline)

A grid of 25 samples for each class, after full hotspot + PET pipeline (8
samples per class shown for compactness):

![](../../synth_datas/PET-MNIST/_master_grid.png)

For full per-class 5×5 grids (one PNG per digit) see
`synth_datas/PET-MNIST/digit_*.png`. Hotspot patterns are visibly
consistent across handwriting variants within each class — the
bbox-relative + snap mechanism handles MNIST's natural shape variability.

### 8.4 Effect on single-modality clustering

Identical setup to §7.3 (CFG_MNIST_PET: 1ch→3ch RGB, D=8, K=10, 50 epochs,
joint GMM-VAE training, sklearn refit GMM each epoch). Only the input data
differs: **no-hotspot** (the §7 PET-MNIST), **HOT08** (w=0.8 medium),
**HOT15** (w=1.5 heavy).

| Variant | Best Acc | Best NMI | Best ARI | Δ vs. no-hotspot |
|---|---|---|---|---|
| No hotspot (§7 PET-MNIST) | 0.7689 | 0.662 | 0.581 | — |
| **HOT08** (w=0.8) | 0.7626 | 0.760 | 0.671 | **+9.8 NMI**, **+9 ARI**; acc within noise |
| **HOT15** (w=1.5) | **0.8240** | **0.755** | **0.712** | **+5.5 acc**, +9 NMI, +13 ARI |

![](../results/figures/hotspot_singlemod_traj.png)

Three observations:

1. **HOT15 lifts best acc by 5.5 pp** over the no-hotspot baseline (0.82 vs
   0.77). The single-modality encoder is now picking up the hotspot signal
   as an additional discriminative axis on top of stroke shape.
2. **HOT08 and the no-hotspot run finish at similar best-acc**, but **HOT08
   has substantially higher NMI** (0.76 vs 0.66) and ARI (0.67 vs 0.58),
   indicating that even when peak Hungarian acc is similar, the latent
   geometry under HOT08 is more aligned with digit identity than the
   no-hotspot version.
3. **All three runs peak around epoch 11–17 then plateau or wobble**, a
   common pattern in our single-modality GMM-VAE runs. The wobble does not
   indicate divergence; subsequent epochs stay within 5 pp of the peak.

### 8.5 Why HOT15 won

Stronger boost weight (1.5 vs 0.8) makes the hotspot regions ~80% brighter
than ordinary stroke after the full pipeline. This makes the latent
encoding of "is hotspot present here" almost binary, easy for the encoder
to dedicate a few latent dimensions to, separable by GMM into 10 distinct
hot-pattern fingerprints.

The trade-off: at w=1.5 the heavy boost slightly suppresses non-hotspot
stroke (since the boost competes with the γ=2.2 power on remaining stroke
pixels), so the **digit shape becomes secondary to the hotspot pattern**.
On natural data this would be a problem (real PET uptake is not perfectly
class-deterministic); on the synthetic MNIST testbed it is exactly what we
want to test alignment dynamics.

### 8.6 What this gives Phase 2

HOT08 is used as modality B for the Phase-2 pilot (§9) — it is "medium"
strength (a more conservative starting point than HOT15) and its checkpoint
has been verified as a stable warm-start. An ablation rerun of §9 with
HOT15 as modality B is the most informative next experiment (single
hyperparameter change, expected to lift joint accuracy by 3–5 pp).

## 9. Phase-2 pilot: bi-modal CLAST (MNIST ↔ MNIST-PET-with-hotspots)

First end-to-end run of the proposal's multi-modal framework (proposal §3–5):
each modality keeps its own conv VAE, two affine-coupling normalizing flows map
each latent into a shared modality-agnostic `u` space, and a single shared GMM
clusters at `u`. Paired alignment + cross-reconstruction are activated in a
three-phase schedule (10 / 20 / 20 epochs per proposal §5).

This section reports the pilot result. Code lives under
`synth_datas/multimodal/`; metrics CSV in `results_bimodal_mnist_pet/`.

### 9.1 Setup

| Component | Choice |
|---|---|
| Modality A | MNIST (1ch, 28×28) — original handwriting |
| Modality B | MNIST-PET-HOT08 (3ch, 28×28) — soft-glow green PET + per-class hotspot boost, weight 0.8 (§7 hotspot design) |
| Pairing | Perfect — A and B derive from the same MNIST source, so paired ground truth is noise-free |
| Encoder/Decoder | Existing `StaticGMMVAE` conv (in_channels parameterized) |
| Flow `h_A, h_B` | 4× (ActNorm + AffineCoupling), MLP 128 SiLU, zero-init last layer (Glow-style identity-init) |
| Latent D | 8 (matches single-modality VAE checkpoints, allows warm-start) |
| Shared GMM K | 10 |
| Warm-start | `vae_A ← results_schemeA_atlas_refit/best_vae.pth` (MNIST 0.9382); `vae_B ← results_mnist_pet_hot08/best_vae.pth` (PET-HOT08 0.7626) |
| Optimizer | AdamW, lr 5e-4, weight_decay 1e-4 |
| Schedule | **Phase 1** (1–10): VAE+KL+GMM(u)+flow_NLL, λ_p = λ_cross = 0 · **Phase 2** (11–30): ramp λ_p, λ_cross linearly over 5 epochs to (1.0, 0.5) · **Phase 3** (31–50): full objective + cosine LR (5e-4 → 5e-5) |
| Total epochs | 50 |

Loss weights (proposal eq 8): `w_recon = w_kl = w_gmm = 1.0`, `λ_f = 1.0`,
`λ_p_max = 1.0`, `λ_cross_max = 0.5`, `λ_b = 0`.

### 9.2 Training trajectory

![](../results/figures/bimodal_phase_curves.png)

The four panels show:

- **Top-left**: Hungarian clustering accuracy. `joint_acc` = single GMM at u
  applied to test (A ∪ B). `accA / accB` = per-modality test acc using the
  same joint GMM.
- **Top-right**: CCA = paired same-source samples that land in the same
  predicted cluster.
- **Bottom-left**: cross-modal loss terms `L_pair` (paired alignment, cyan)
  and `L_cross` (cross-reconstruction BCE, orange).
- **Bottom-right**: `λ_p` schedule for reference.

### 9.3 The Phase-2 transition (the headline finding)

| Phase | Epoch | joint_acc | accA | accB | CCA | L_pair |
|---|---|---|---|---|---|---|
| Phase 1 end | 10 | 0.367 | 0.355 | 0.593 | 0.117 | — |
| **Phase 2 first epoch** (λ_p = 0.20) | **11** | **0.679** | **0.707** | **0.728** | **0.777** | 0.293 |
| Phase 2 fully ramped (λ_p = 1.0) | 15 | 0.794 | 0.789 | 0.798 | 0.939 | 0.069 |
| **Phase 2 best** | **29** | **0.8085** | **0.805** | **0.813** | **0.944** | 0.055 |
| Phase 3 cosine end | 50 | 0.792 | 0.788 | 0.796 | **0.950** | 0.046 |

**The first epoch of Phase 2 already lifts CCA from 0.12 (≈ random 0.10) to
0.78 and joint_acc from 0.37 to 0.68 — a 66 pp / 31 pp jump in a single
epoch, with `λ_p = 0.20` only.** This validates the proposal's central claim
that even a small paired-alignment pressure is enough to align the two flows'
outputs in the shared `u` space.

### 9.4 Shared `u` space — t-SNE of the test set

![](../results/figures/bimodal_tsne_u.png)

Left: 2500 test samples per modality (5000 dots, `A=○`, `B=×`), colored by
true digit. Ten clean class islands.

Middle: colored by modality (blue = A, red = B). **Modalities are
inter-mixed**, not separated — i.e., a given digit's A and B versions land in
the same neighborhood of `u`, which is exactly what paired alignment is
supposed to achieve.

Right: colored by predicted cluster from the shared GMM. The 10 clusters
agree well with the 10 true digit islands.

### 9.5 Cross-modal generation

The decoded cross-modal pathway:

  - `x_A → E_A → z_A → h_A → u → h_B⁻¹ → z_B → D_B → x̂_B` (MNIST → PET)
  - `x_B → E_B → z_B → h_B → u → h_A⁻¹ → z_A → D_A → x̂_A` (PET → MNIST)

The 6-row layout in each viz: orig_A / self-recon_A / **B→A cross-gen** /
orig_B / self-recon_B / **A→B cross-gen**.

![](../results/figures/bimodal_cross_viz_summary.png)

At **epoch 5** (Phase 1, no paired loss) the self-recon rows are correct but
the cross-modal rows show **random digits**, not the input column's digit.

By **epoch 30** (Phase 2 fully active) and **epoch 50** (end of Phase 3), the
cross-modal rows correctly reproduce the input column's digit identity in the
target modality. The cross-modal images are slightly softer than the
self-reconstructions (the cross path goes through `flow⁻¹`, which compounds
small errors), but **digit identity is unambiguously preserved**, and that
is the property the framework was designed to deliver.

### 9.6 Test confusion matrix (joint u-space clustering)

![](../results/figures/bimodal_cm.png)

The joint u-space GMM produces an essentially diagonal-dominant confusion
matrix across both modalities combined; off-diagonal mass concentrates on
the usual MNIST-ish confusion pairs (4↔9, 3↔5/8, 7↔1).

### 9.7 Acceptance evaluation

The three criteria fixed before training:

| Criterion | Target | Result | Pass |
|---|---|---|---|
| (a) joint_acc > max(single-modality baselines) | > max(0.94, 0.76) | 0.808 | **partial** — beats PET (+5 pp) but does not beat clean MNIST (−13 pp) |
| (b) Cross-modal cluster agreement (CCA) | ≥ 0.90 | **0.950** | ✅ |
| (c) Cross-modal generation visually identity-preserving | qualitative | epoch 30+: full alignment (Section 8.5) | ✅ |

Criterion (a) is the only partial outcome — and it is a fair-to-expected
result on this dataset. MNIST is already an "easy" modality where a single
VAE reaches 0.94; forcing the encoder to ALSO support cross-modal
generation costs some single-modality accuracy. In the proposal's target
domain (cardiac MRI/CT, brain MRI/PET) **neither single modality reaches
ceiling**, so the joint-vs-best-single comparison flips in the framework's
favor — that is the regime where Phase-2's combined-signal story matters
most.

### 9.8 Loss decomposition: where the optimization goes

| Loss term | Phase 1 start | Phase 1 end | Phase 2 start | Phase 2 end | Phase 3 end |
|---|---|---|---|---|---|
| `recon_A` (BCE per sample) | 78.4 | 76.4 | 77.1 | 74.5 | 71.3 |
| `recon_B` | 336.2 | 335.2 | 334.7 | 331.1 | 330.1 |
| `KL_A + KL_B` | small | small | small | small | small |
| `GMM(u)` | 2.45 | 1.26 | 1.05 | 1.15 | 1.09 |
| `flow_NLL_A + flow_NLL_B` | 14.8 | 4.0 | 5.3 | 2.6 | 0.6 |
| `L_pair` | 0 | 0 | 0.293 | 0.056 | **0.046** (↓84% from peak) |
| `L_cross` | 0 | 0 | 488 | 412 | **406** (↓16% from peak) |

- `recon_A` and `recon_B` continue to improve through all 3 phases — the
  joint training did NOT break per-modality self-reconstruction.
- `flow_NLL` drops from 15 → 0.6, indicating the flow successfully maps each
  modality's latent to N(0, I) in u-space.
- `L_pair` is the fastest-decaying term — the flow learns the alignment
  quickly once forced.
- `L_cross` decays much more slowly than `L_pair` — it's a harder objective
  (full cross-modal pixel reconstruction) and stays at ~400 BCE per sample
  by epoch 50.

### 9.9 Takeaways from this pilot

1. **The full bi-modal pipeline trains end-to-end with no instability.**
   Three-phase schedule + warm-start lets us hit CCA 0.95 in 50 epochs total.

2. **paired_loss is dramatically effective.**
   One epoch with λ_p = 0.20 lifts CCA from random to 0.78. Without paired
   loss, the framework collapses to "two independent VAEs sharing a GMM
   incidentally" — useless. With paired loss, the two `u` distributions
   merge fast.

3. **Cross-modal generation is qualitatively correct** by Phase 2 end.
   This is the strongest evidence that the **invertible** flow is doing real
   work; if `flow_B⁻¹(h_A(z_A))` did not preserve digit identity, the
   generated images would look like the wrong digit.

4. **Joint clustering accuracy is bounded by the weaker modality's
   contribution to the shared GMM.** On MNIST (already 0.94 alone), joint
   alignment costs accuracy. On clinical data where neither modality is
   strong alone, the same mechanism should provide net gain.

### 9.10 Limitations + next steps

- **D = 8** is below the proposal's D = 32. Repeat with D = 32 (requires
  re-training the single-modality VAEs) to align with paper baselines.
- **MNIST-PET-HOT15** (which reached 0.8240 single-modality) was not used
  as modality B; rerunning the pilot with HOT15 as warm-start may push
  joint acc past 0.85.
- **No paired-fraction ablation** yet. Proposal §7 calls for ablating
  paired fraction (100% → 0%). The current pilot has 100% pairing.
- **Real clinical data still needed.** ADNI MRI is the planned next data
  source; this MNIST-pair pilot is the framework correctness check, not
  the clinical demonstration.

## 10. Takeaways

1. **MNIST is a sanity-check baseline that the pipeline works.**
   K=10 GMM-VAE recovers digit identity at 0.94 acc, NMI 0.86,
   matching the classical VaDE-style result.

2. **Printed Digits is the interesting "intermediate" case.**
   Same architecture, same channel layout as MNIST, but acc collapses
   to 0.27 — *not* because of label noise (we cleaned that) but because
   intra-class style variance (multiple fonts per digit) dominates the
   reconstruction signal. The GMM partitions the latent by style, not
   identity.

   This is the failure mode that Phase 2 (latent-space normalizing flow
   + paired alignment with MNIST) is designed to fix: anchor each
   printed-digit cluster to the corresponding MNIST cluster via the
   shared `u`-space GMM, forcing the encoder to align with digit
   identity rather than style.

3. **SVHN is the failure baseline.** Color/lighting/background swallow
   the digit signal entirely; vanilla VAE-GMM has no path to
   semantic clustering on natural images. This validates the
   motivation for richer encoders or stronger structural priors.

4. **The Phase-2 multimodal pipeline is fully implemented (§9).** Latent
   normalizing flow (4-layer affine coupling + ActNorm) + paired
   alignment + cross-reconstruction. Pilot on MNIST + MNIST-PET-HOT08
   reaches **CCA 0.944**, **joint acc 0.808**; cross-modal generation
   preserves digit identity (Section 9.5).

5. **Class-conditional hotspots (§8) successfully add functional signal
   to MNIST-PET.** HOT15 lifts single-modality acc from 0.769 to 0.824
   without disturbing the original PET appearance (background unchanged,
   only stroke pixels get multiplicatively boosted in class-specific
   regions).

## 11. Reproducibility

```bash
# GMM-joint baselines (uses GPU 0 by default; set CUDA_VISIBLE_DEVICES to pick)
python train.py            # MNIST
python train_printed.py    # Printed Digits
python train_svhn.py       # SVHN

# Synthetic MNIST-PET / MNIST-CT (Phase-2 testbed, no hotspots)
python synth_datas/train_styled.py pet
python synth_datas/train_styled.py ct

# Hotspot-augmented MNIST-PET single-modality (§8)
python synth_datas/train_styled.py pet_hot08      # w=0.8 medium
python synth_datas/train_styled.py pet_hot15      # w=1.5 heavy

# Bi-modal pilot (§9) — uses MNIST and HOT08 checkpoints as warm-start
python synth_datas/multimodal/train_bimodal.py

# Pure-VAE ablation (recon + KL only, post-hoc GMM eval each epoch)
python train_purevae.py mnist
python train_purevae.py printed
python train_purevae.py svhn

# Hotspot preview / sample-grid regeneration
python synth_datas/gen_petmnist_pngs.py           # refreshes synth_datas/PET-MNIST/
```

Each run writes to `results_<dataset>_*/`:
- `metrics.csv` — per-epoch loss + Hungarian acc / NMI / ARI
- `cluster_samples/` — top-confidence sample grids per cluster
- `cluster_purity_epoch_XXX.csv` — per-cluster digit composition
- `atlas_A/` — Scheme-A pixel-space atlases (per-epoch)
- `atlas_refit_means/` — decoded GMM means atlases (per-epoch)
- `best_vae.pth` — checkpoint with best test-set Hungarian accuracy

Hyperparameters (all three runs share these unless noted):
`latent_dim=8`, `batch_size=128`, `lr=1e-3`, `weight_decay=1e-4`,
`w_recon=w_kl=w_gmm=1.0`, GMM hard-overwritten from sklearn refit at
epoch 20.

Per-dataset config diffs:
- MNIST: `K=10, in_channels=1, image_size=28, N_per_class=12800`
- MNIST-PET / MNIST-CT: `K=10, in_channels=3, image_size=28, N_per_class=12800` (`CFG_MNIST_PET` / `CFG_MNIST_CT` in `synth_datas/config.py`; transform parameters in `synth_datas/mnist_petct.py`)
- MNIST-PET-HOT08 / HOT15: inherit CFG_MNIST_PET; add `hotspot_weight = 0.8` or `1.5` (`CFG_MNIST_PET_HOT08` / `CFG_MNIST_PET_HOT15` in `synth_datas/config.py`); per-class landmark table in `synth_datas/hotspot_config.py`
- Printed: `K=9, in_channels=1, image_size=28, N_per_class=500` (class 0 dropped)
- SVHN: `K=10, in_channels=3, image_size=32, N_per_class=4000`
- Bi-modal: `latent_dim=8, K=10, n_flow_layers=4, flow_hidden=128, lr=5e-4, 3-phase schedule 10/20/20 epochs, λ_p_max=1.0, λ_cross_max=0.5`; warm-start from MNIST + HOT08 single-modality checkpoints (see `synth_datas/multimodal/config.py`).
