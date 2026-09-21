# MMCLAST-cg on MNIST-PET/CT

```bash
python -m data.make_mnist_petct --aux_nohot      # build the dataset (~30 s)
python -m utils.make_figures_folder --dataset mnist
python -m utils.eval_mnist                       # the paired table
```

**Status: no arm has finished yet.** The host is training; the numbers tables
below are empty on purpose. Nothing here should be cited until they are filled.

## Why this dataset, after horse2zebra

horse2zebra established that the split-plus-flow survives without pairing, but
it could not say whether the output was any *good*. Unpaired data admits no
SSIM against ground truth, so quality rested on FID alone, and the only SSIM
available — cycle SSIM — is won outright by an identity map (`h2z_base`, which
did not translate at all, had the *highest* cycle SSIM of any arm). The
endpoint claim stayed unresolved for want of a metric.

MNIST-PET/CT removes that limitation without giving up the unpaired training
setting, because the PET modality is *derived* from the CT one:

| | ADNI (T1↔FA) | horse2zebra | **MNIST-PET/CT** |
|---|---|---|---|
| training | paired | unpaired | **unpaired** |
| evaluation | paired | unpaired | **paired** |
| `L_cross` | weight 10 | must be 0 | **must be 0** |
| quality metric | SSIM vs GT | FID only | **SSIM/PSNR vs GT + FID** |
| size | 1 × 112² | 3 × 256² | 3 × 64² |

The split is what buys this, and it is deliberate:

```
train/   DISJOINT MNIST indices between A and B.  No image's own PET
         counterpart is on the other side, so training never sees a pair.
test/    The SAME indices in both, written in matching filename order.
```

Unpaired training, paired evaluation. `data/make_mnist_petct.py` prints the
index overlap (must be 0) and the filename alignment (must be True) on every
build, and `utils/eval_mnist.py` re-asserts the alignment before scoring — it
is the one property the whole evaluation rests on.

## The two modalities

* **A = CT** — the original MNIST digit, grayscale replicated to RGB. Unstyled.
* **B = PET** — per-class hotspot boost → PSF blur (σ=1.2) → SUV-style display
  curve (γ=2.2, ×1.15, +0.12 lift) → acquisition noise (σ=0.05) → green tint.

### Provenance of the PET rendering

The pipeline and the per-class hotspot table come from
`__outdated_files/synth_datas/` (`hotspot_strategy.ipynb` cell 4/6/14,
`hotspot_config.py`, `mnist_petct.py`), copied rather than imported because
that tree is staged for deletion. Those sources disagree about which is
authoritative — `gen_petmnist_pngs.py` reads the notebook and calls it "source
of truth", while `hotspot_config.py` calls itself "the canonical, production
version" — so all three were checked against this implementation:

| comparison | result |
|---|---|
| notebook `REL_REGIONS` vs `hotspot_config.py` vs here | identical |
| notebook `boost_hotspots` vs `_boost_hotspots` | bit-identical (0.0 over 500 digits) |
| notebook `pet_pipeline` vs `_pet_intensity` | 5.96e-08 = float32 eps |
| rendered uint8 output | **0 differing pixels in 1,568,000** |

The 5.96e-08 is one ulp: the notebook casts the noise to float32 before adding,
this code casts after. It is invisible at uint8 (255 × 6e-8 ≈ 1.5e-5, far
inside one quantisation step), so the PET images here are byte-for-byte what
the notebook's scheme produces.

The whole pipeline runs at native 28² — the hotspot sigmas and the PSF blur are
tuned for it — and the result is upscaled to 64².

### One deliberate departure: CT is not styled

The original scheme rendered CT through its own pipeline too
(`_ct_style_intensity`: unsharp mask σ=1.1 at +1.1×, γ=0.40, ×1.55−0.18, noise
σ=0.03), simulating a CT bone window. That is **not** used here: A is the
original MNIST digit, grayscale replicated to RGB, untouched.

This makes the setup cleaner rather than merely simpler. With both modalities
styled, every domain difference is an artefact of two hand-written intensity
curves; with A left raw, the entire gap is produced by the PET side's physical
simulation, and the hotspots — the one signal that is genuinely absent from the
source domain — are what the model has to invent.

**The asymmetry is the point.** The hotspots are a class-conditional signal
that is simply *not present* in CT. So A→B must **invent** it and can only ever
be right distributionally, while B→A merely has to remove it and is
well-posed. On horse2zebra the two directions behaved differently and nothing
in the data explained why; here the asymmetry is designed in and measurable.

`--no_flip` is required and set in every config: a mirrored digit is not a
member of either domain, and augmenting with one would poison the same
distribution the FID reference is computed against.

## Metrics

Every number is reported against a floor and a ceiling, because a raw value
cannot be over-read — the same discipline the horse2zebra FID calibration
needed and the ADNI hole metric lacked.

| metric | floor | ceiling | what it answers |
|---|---|---|---|
| SSIM/PSNR vs GT | `_copy` (emit the input) | 1.0 | did it translate, and is the result right? |
| FID | real vs real | do-nothing | is the output *distribution* right? |
| cycle SSIM | — | 1.0 (identity wins) | **not** a quality metric; kept for continuity with h2z |
| hotspot contrast | `_nohot` PET | real PET | was class-conditional signal absent from CT invented? |

The hotspot column is the one this dataset exists for. `testB_nohot` is the
same PET rendering with the boost off and the *same noise draws*, so
`testB − testB_nohot` isolates the class-conditional signal exactly. Then

```
M        = pixels the boost materially raised
contrast = mean(X[M]) / mean(X[stroke \ M])
```

with the masks taken from the real pair — always, including for both
references, so every model is scored on the same pixels and only the
intensities inside them are its own. A model that ignores class identity lands
on the `_nohot` floor no matter how good its SSIM is.

Measured on all 5000 test pairs (`paired_table.json`):

| reference | SSIM | PSNR | hotspot contrast |
|---|---|---|---|
| `_copy` — emit the input | 0.4643 | 11.46 | — |
| `_nohot` PET — boost off | — | — | **1.216** (floor) |
| real PET | 1.0 | — | **1.378** (ceiling) |

So the hotspot window is 1.216 → 1.378. It is not a wide window, and a
difference of a few hundredths inside it is not a result; what it cleanly
separates is an arm sitting *on* the floor from one that moved off it.

## Configuration

`configs/mnist_{base,latcyc,morph}.yaml`, plus `configs/run_mnist_s3.sh`.

The schedule is **not** the ADNI/h2z one, and that is a deliberate departure:
12000 images per domain at batch 32 is 375 iterations per epoch against
horse2zebra's 133, so an epoch here carries ~3× the gradient steps and the
epoch caps come down to match. Batch is 32 rather than 8 because 64² with 6
resnet blocks leaves the card nearly empty at 8. `n_blocks: 6` follows the
reference CycleGAN, which uses `resnet_6blocks` below 256px.

The host is trained here from scratch (`exps/checkpoints/mnist_host`) rather
than warm-started from a third-party checkpoint as on horse2zebra, so the whole
chain — host, split, flow — is ours end to end.

### Where the host's speed actually comes from

The host is the only serialised step — every arm waits on it — so it was worth
measuring rather than guessing. One epoch, same data, same everything else:

| config | iters/epoch | s/epoch |
|---|---|---|
| 1 GPU, batch 32 | 375 | 253 |
| 1 GPU, batch 64 | 187 | 238 |
| **2 GPUs (`--dp`), batch 64** | 187 | **144** |

**Raising the batch on one card buys almost nothing** (−6%): the card is already
at 100% SM occupancy at batch 32, so halving the iteration count just makes each
iteration twice as long. This reproduces the horse2zebra measurement, where a
batch increase gave +0.4%.

The speedup is the second GPU, not the batch. `--batch 64 --dp` splits into 32
per card — the same per-GPU work as the measured single-GPU batch-32 run — and
halves the iterations: **1.76×**, 7 h → 4 h. The batch increase is a
*prerequisite* for feeding two cards, not itself the optimisation.

`lr` stays at 2e-4 rather than being scaled with the batch, so that the
already-recorded batch-32 curve stays a valid control. It tracks:

```
          ep1    ep2    ep3    ep4      (loss_G)
b32 1gpu  2.745  1.518  1.414  1.381
b64 2gpu  3.573  1.757  1.551  1.474    <- ~1 epoch behind, same shape, D stable
```

Half the gradient updates per epoch puts it about one epoch behind; the shape,
the D balance (~0.55 vs ~0.52) and the absence of any collapse are unchanged,
and there are 100 epochs with 50 of LR decay to absorb it.

**The arms do not get this treatment.** Two reasons. `nn.DataParallel` only
parallelises `forward()`, but `train.py` drives the model through
`m.cross_A2B`, `m.self_A`, `m.walk`, `m.enc_A` — those would silently bypass
the wrapper and run on one card. Routing them through a mode-dispatching
`forward()` means restructuring `MMCLASTcg`, which is precisely the code the
exact-invertibility and `--check_init` guarantees rest on. And it is
unnecessary: round 1 is two arms on two GPUs, already fully parallel with no
scatter/gather overhead. Their batch also stays at 32 — a single-card increase
is worth ~6%, while halving the gradient steps per epoch works against early
stopping, which counts epochs.

## Fixed before this dataset was run

The unpaired S2 early-stopping criterion was blind to `f`. With E/D frozen and
`f` an exact bijection, `f⁻¹(f(z))` cancels analytically inside the pixel cycle
term, so the criterion depended on `f` only to second order; with
`w_latcyc = 0` nothing else in it touched `f` at all. On horse2zebra that cost
a whole arm: `h2z_base` was killed at S2 epoch 20 and scored FID 226.2, sitting
on the do-nothing ceiling of 228.5, while the identical run with the stop
removed reached **109.6** — the best of any arm. The objective was fine; the
criterion was not.

`val_monitor` in `train.py` now always includes the latent-cycle residual in
the unpaired branch, even when it carries no weight in the objective: it is the
only S2-available quantity that depends on `f` to first order, and it measures
the right thing, since `back_u − u` is the autoencoder error evaluated at
`f(z_A)` and is small exactly when `f` lands inside the range `D_B` was trained
to decode. This is a deliberate break from "criterion = objective minus GAN".

## Pipeline validation

Before committing GPU time, one throwaway run (1 epoch per stage, no warm
start, `MMCLAST_EXPS` pointed at a scratch dir) exercised the whole path at
64². **These are not results** — a 1-epoch model is not a method — but they
confirm the plumbing and, incidentally, that the hotspot metric discriminates:

```
S1  selfA 0.978  selfB 0.977   flow_work 0.00   <- exact identity at init, as required
S2  val 1.9964                 flow_work 1.67   <- criterion now moves with f
S3  selfA 0.986  selfB 0.979   flow_work 1.42
paired:  SSIM A→B 0.864 (copy floor 0.464)      <- it translates
         hotspot  1.201 (floor 1.194, ceil 1.356 on that 512-image subset)
```

The last line is the one worth noting: after a single epoch the model already
produces something that *looks* like PET — SSIM far above the copy floor — while
sitting essentially **on** the hotspot floor, i.e. inventing none of the
class-conditional signal. Appearance and functional content come apart, and
this dataset can see the difference. horse2zebra could not.

## Results

_All arms complete, paired and unpaired._

**The dataset now exists in two builds.** `--paired_train` gives trainB the
same MNIST indices as trainA, so `L_cross` is available and the ADNI protocol
runs unchanged; the default gives them disjoint indices, so training is
genuinely unpaired. Test sets are identical and both settings warm-start from
the same host, so the pair of runs isolates the cross-path supervision and
nothing else.

### 0. What supervision recovers, and what SSIM cannot see

| training | model | SSIM CT→PET | PSNR | hotspot (% of gap) |
|---|---|---|---|---|
| — | host | 0.8280 | 28.47 | −4.5% |
| unpaired | base | 0.8323 | 28.75 | −1.7% |
| unpaired | latcyc | 0.8327 | 29.11 | +1.9% |
| **paired** | **base** | 0.8657 | 30.55 | **+20.6%** |
| **paired** | **latcyc** | **0.8680** | **30.74** | **−2.2%** |
| **paired** | **morph** | 0.8387 | 29.55 | **+11.6%** |

Two findings, and the second is why this dataset was built.

**Recovery tracks whether the objective asks for it.** Adversarial and cycle
losses never require a hotspot to land on the correct digit-specific location,
so the unpaired arms sit on the floor and the host sits *below* it. Adding
`L_cross`, which compares against the true PET pixel for pixel, lifts recovery
to 20.6% (t=52.2). Even direct supervision leaves four fifths unrecovered.

**SSIM and PSNR rank the two paired arms in the opposite order to the hotspot
measure.** `L_latcyc` improves SSIM (0.8657→0.8680) and PSNR (30.55→30.74)
while taking hotspot recovery from +20.6% to −2.2% — 22.8 points of the gap,
t=61.6, p≈0. The mechanism matches what the term asks for: it requires `f(z_A)`
to be a fixed point of the target autoencoder, i.e. to lie in the *typical*
region of `E_B`'s range, and the hotspots are exactly the atypical
class-specific detail such a constraint smooths away. It buys global fidelity
at the cost of rare local structure, and no reconstruction metric reports the
trade.

`morph` makes the same point from the other side. It carries `L_latcyc` too,
yet recovers +11.6% against `latcyc`'s −2.2% (t=44.0): supervising the
trajectory partly undoes the smoothing, because an intermediate frame has to
look like a real image and a real PET image has hotspots. It pays at the
endpoint, scoring worst of the three on CT→PET SSIM. Across the three paired
arms the two measures are close to anti-correlated:

```
by SSIM             latcyc (0.8680) > base (0.8657) > morph (0.8387)
by signal recovered base (+20.6%)   > morph (+11.6%) > latcyc (-2.2%)
```

The ordering is not an artifact of each arm training its own stage 1, at least
in the direction that matters: `morph` reached the *best* stage-1 validation of
the three (0.00918 vs 0.00970 and 0.00960) and still finished worst at the
endpoint, so the confound favoured it and it lost anyway.

This supersedes the flat negative result recorded below under conclusion 2,
which was measured on unpaired models only.

### 1. MMCLAST-cg beats its host on CT→PET — the first time on unpaired data

FID has no natural scale, so every value is read against a floor (real test vs
real train of the target domain) and a ceiling (the score for changing
nothing), both measured under the protocol the runs themselves use:

| | CT→PET | PET→CT |
|---|---|---|
| floor (real vs real) | 1.08 | 4.14 |
| **host** | **15.88** | **28.84** |
| ceiling (do nothing) | 253.61 | 258.86 |

| | FID A→B | FID B→A | SSIM A→B | PSNR | SSIM B→A | PSNR | cyc A | cyc B |
|---|---|---|---|---|---|---|---|---|
| `_copy` (do nothing) | — | — | 0.4643 | 11.46 | 0.4643 | 11.46 | | |
| **host** (plain CycleGAN) | 15.89 | **28.85** | 0.8280 | 28.47 | **0.9427** | 25.20 | 0.9891 | 0.9711 |
| **`mnist_latcyc`** | **12.98** | 76.83 | **0.8327** | **29.11** | 0.9287 | **25.26** | 0.9884 | 0.9776 |
| `mnist_base` | 17.85 | 105.54 | 0.8323 | 28.75 | 0.9246 | 24.58 | **0.9954** | **0.9847** |
| `mnist_base_longS2` † | 23.35 | 100.03 | 0.8278 | 28.38 | 0.8563 | 24.68 | 0.9957 | 0.9846 |
| `mnist_base_s3cut` * | 24.95 | 161.21 | 0.8324 | 28.88 | 0.8158 | 24.12 | 0.9955 | 0.9777 |

\* superseded: its S3 was cut at ep16 by an over-aggressive criterion, see below.
† control with S2 early stopping disabled, see below.

`latcyc` takes A→B outright — FID 12.98 vs 15.89, SSIM 0.8327 vs 0.8280, PSNR
29.11 vs 28.47. On horse2zebra the best arm was 109.6 against a host of 70.6,
so the endpoint claim did not transfer there; here it does, in the direction
that has to synthesise the harder modality.

**B→A is the direction that breaks**, and it is the opposite of what this
README predicted. The prediction was that A→B would be hard because it must
*invent* the hotspots. In fact CT is the awkward target: raw MNIST has an
exactly-zero background and near-binary strokes, so any residual generator
noise lands far away in Inception feature space, while PET's lifted noisy
background is forgiving. The prediction was about information content; FID is
dominated by background statistics.

### 2. Nothing meaningfully recovers the hotspots — and optimisation does not help

Scored on all 4997 measurable test pairs, against the floor (`_nohot`) and
ceiling (real PET), gap +0.1614:

| model | hotspot | vs floor | % of gap | t | p |
|---|---|---|---|---|---|
| host | 1.2090 | −0.0073 | **−4.5%** | −8.4 | 7e-17 |
| `mnist_base` | 1.2136 | −0.0027 | −1.7% | −3.1 | 2e-03 |
| `mnist_latcyc` | 1.2194 | +0.0031 | **+1.9%** | +4.0 | 6e-05 |

The host and `base` sit *significantly below* the no-hotspot floor: they
produce less hotspot contrast than a PET image rendered with the boost
switched off, i.e. they actively smooth the functional signal away.
`latcyc` is significantly above the floor, but recovers **1.9%** of the
available gap — real at n=4997, negligible in substance.

This is what the dataset exists to show. Over the course of these runs FID A→B
improved from 24.95 to 12.98 — a 48% gain, and a genuine one by every metric
normally reported — while the hotspot column moved from −1.7% to +1.9% of the
gap. **Distributional and pixel metrics improved substantially; recovery of
information absent from the source modality did not.**

It is the expected answer rather than a surprise: hotspot position is a
deterministic function of digit class, which *is* recoverable from CT, but
nothing in the objective rewards recovering it. An adversarial loss against the
PET distribution has no reason to place blobs correctly per-sample, and
averaging them away is safer for both the critic and the cycle term. No
unpaired benchmark reporting only FID can see this.

### 3. `L_latcyc` is load-bearing here — the opposite of horse2zebra

| | `w_latcyc` | S2 | S3 | FID A→B | latent gap |
|---|---|---|---|---|---|
| `mnist_latcyc` | 2 | **60, no early stop** | 61 | **12.98** | **1.011** |
| `mnist_base` | 0 | 20, early stop | 26 | 17.85 | 1.339 |

On horse2zebra the same comparison came out the other way (`base_longS2` best
at 109.6), but that was an artifact of a blind criterion. Here the criterion
works, and the two arms separate cleanly and mechanistically: `latcyc`'s latent
residual falls 0.892 → 0.465 and its latent gap ends at 1.011, while `base`'s
residual sits at ~1.98 throughout S2 and *rises* through S3, to 2.11.

The mechanism is visible in that rise. At S2 the decoders are frozen, so `f`
must itself land inside the range `D_B` can decode — `L_latcyc` is the only
term asking for that, and without it nothing does. At S3 the decoders unfreeze
and `D_B` retrains to decode whatever `f` produces, so latent alignment can be
routed around; the residual is free to drift upward. `L_latcyc` is load-bearing
at S2 and optional at S3.

### 4. A silent FID caching bug, found by a number that did not reconcile

The figure script reported the host at 20.12 / 23.21 where `train_host.py`
reported 15.89 / 28.85 — same protocol, same checkpoint, same reference sets.
Two bugs, one of them dangerous:

* `fid_reference` called `folder_stats` without `crop`, so it used the default
  256 on a 64px dataset;
* **`folder_stats` keyed its disk cache on the file list and `dims` but not on
  `crop`**, so the 256-crop reference stats those calls wrote were silently
  handed back to `fid_against(crop=64)` — 256-crop reals scored against
  64-crop fakes.

The second is the one that matters: it returns a plausible wrong number with no
error. horse2zebra never tripped it because every call there was 256. `crop` is
now part of the cache key, and the stale caches were deleted. The host now
reconciles at 15.88 / 28.84 across both scripts.

The lesson is the same one this project keeps relearning: the discrepancy was
only visible because the same quantity was computed twice by two paths. A
single path would have reported 20.12 and nothing would have looked wrong.

### 5. Two early-stopping errors, and what they cost

Both were mine, both are fixed, and the second was found only because the first
fix over-reached.

**S2 blindness (the original bug).** With E/D frozen and `f` an exact
bijection, `f⁻¹(f(z))` cancels analytically inside the pixel cycle, so the
unpaired criterion depended on `f` only to second order. With `w_latcyc = 0`
nothing else in it touched `f` at all. On horse2zebra this cost 117 FID points
(`h2z_base` 226.2 vs `h2z_base_longS2` 109.6). Fixed by always including the
latent-cycle residual at S2.

**S3 over-reach (introduced by that fix).** The fallback was applied at S3 too,
where the decoders train and `self` already tracks them — so the criterion was
not blind, and forcing in a term `base` does not optimise made it 96% of the
value (S3 val 2.073 = 1.983 latcyc + 0.011 self ×2 + cycle). It stopped
`mnist_base` at ep16 while its self loss was still falling, 0.011 → 0.006.
Restricting the fallback to S2 and re-running S3 from the *same* `stage2.pth`:

| | S3 epochs | FID A→B | FID B→A |
|---|---|---|---|
| over-aggressive criterion | 16 | 24.95 | 161.21 |
| corrected | 26 | **17.85** | **105.54** |

Cost of that error: 7 FID points A→B, 56 B→A.

**Was `base`'s S2 stop at ep20 real? Yes.** `mnist_base_longS2` resumes the
same `stage1.pth` with S2 early stopping disabled, so the stop is the only
difference between the two runs. It answers on both the criterion and the
endpoint:

| | S2 | S2 best val | FID A→B | FID B→A | SSIM A→B |
|---|---|---|---|---|---|
| `mnist_base` | ep20, early stop | **2.01207** | **17.85** | 105.54 | **0.8323** |
| `mnist_base_longS2` | 60, no early stop | 2.01468 | 23.35 | **100.03** | 0.8278 |

Forty extra S2 epochs found no better checkpoint by the criterion *and* made
the endpoint worse — 5.5 FID points on A→B. The stop was correct.

This is the cleanest evidence that the S2 fix works, precisely because it is
the opposite of the horse2zebra outcome. There the criterion was blind, its
"best" was near-random, and removing the stop gained 117 FID points. Here the
criterion is sighted, and its report of "no further progress" is true enough
that overriding it *costs* quality. Same experiment, opposite result, and the
difference is whether the criterion could see `f`.

`flow_work` kept rising all the way (1.72 at ep60 vs 1.49 at ep20) while the
endpoint got worse — the flow moves more without moving usefully, the same
drift seen in `base`'s rising S3 residual. `flow_work` says the flow is doing
something; it does not say the something is good.
