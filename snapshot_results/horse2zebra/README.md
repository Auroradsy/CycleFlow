# MMCLAST-cg on horse2zebra — figures

Regenerate everything with:

```bash
python -m utils.make_figures_h2z          # picks up every finished h2z_* arm
```

The script skips arms that have no `model.pth` yet, so it is safe to re-run
while the ablation is still going.

## Why this dataset

ADNI gave the method three conveniences that are not available in general, and
horse2zebra removes all of them at once:

| | ADNI (T1↔FA) | horse2zebra |
|---|---|---|
| pairing | paired, MNI-registered | **unpaired** — no target image exists |
| `L_cross` | weight 10, the spine of S2 | **must be 0** |
| channels / size | 1 × 112² | 3 × 256² |
| quality metric | SSIM against ground truth | **FID**, distributional |

The port is gated behind `--data folder`; the ADNI path is byte-identical and
was re-verified (`--config configs/morph.yaml --check_init` still reproduces the
host at 0.6798 / 0.8125).

## Reading the figures

**`00_host_reference.png`** — the plain CycleGAN that MMCLAST-cg is split from.
Everything else is a comparison against this. Warm-started from
`Ziyobek/cyclegan-horse2zebra` (official `resnet_9blocks` layout, verified by a
`strict=True` load); `--check_init` gives `max|split − host| = 0.000e+00`, i.e.
the rewiring reproduces it exactly before training starts.

**`01_fid_calibration.png`** — the important one. FID has no natural scale, so
every bar is plotted against two references measured under the *same* protocol:

```
floor    real test images vs real train images of the target domain
ceiling  real SOURCE images vs real target — the score for changing nothing
```

A bar sitting on the dashed line did not translate, no matter how the raw number
looks. This is the same discipline the ADNI hole metric needed (real T1 = 0.001,
real FA = 0.142): a detector without its baselines reads pathology into normal
data and vice versa.

**`02_fid_path.png`** — FID of each intermediate flow state against real
A ∪ B. This is the unpaired replacement for the hole metric, and a better one:
it is a standard distributional score rather than a hand-rolled proxy, and it
needs no ground-truth mid-frames. A flat curve means the walk goes nowhere.

**`10_cross_vs_host_<tag>.png`** — input / host / the arm's cross path / the
arm's self path. The self row is the control: a clean self row next to a failed
cross row says the autoencoders are fine and the flow is where it broke.

**`40_morph_{a2b,b2a}_<tag>.png`** — the native morph. Frames are the flow's own
block outputs, not an interpolation between two codes, and each is decoded by
*both* decoders so one row shows the source dissolving and the next the target
emerging.

## Reference values (`fid_reference.json`)

| | horse → zebra | zebra → horse |
|---|---|---|
| floor (real vs real) | 26.21 | 79.75 |
| **host (plain CycleGAN)** | **70.65** | **124.67** |
| ceiling (do nothing) | 228.52 | 217.27 |

The host's 70.65 lands inside the 74–85 range published for CycleGAN on
horse2zebra, which is the evidence that this FID protocol is sound and that a
number like 226 means what it appears to mean.

## Status

Endpoint FID, path FID (the flow's own block states scored against real A ∪ B),
and cycle SSIM.

**Read the table in two blocks.**  The five arms in the middle fork from the
SAME `h2z_latcyc/stage2.pth`, so S3 is the only thing that differs between them
and they are mutually controlled.  `h2z_base_longS2` is *not* one of them — it
has its own S1 and its own S2 — so it is comparable to `h2z_base` (identical
config but for the S2 early stop) and to the host, but not to the S3 ablation.

| arm | S3 terms | FID A→B | FID B→A | path FID f⁰…f⁴ | path mean | cyc SSIM |
|---|---|---|---|---|---|---|
| *floor* | — | 26.2 | 79.7 | | | |
| **host** | plain CycleGAN | **70.6** | **124.7** | — no intermediates exist — | | 0.801 |
| **`h2z_base_longS2`** | none, no `L_latcyc`; *own S1+S2* | **109.6** | **149.2** | 150.7 146.9 134.7 116.1 96.7 | 129.0 | **0.889** |
| `h2z_latcyc` | none | 132.2 | 162.1 | 156.6 147.5 141.9 109.8 **95.8** | 130.3 | 0.900 |
| `h2z_morph_ra` | D_mix (relativistic) + smooth | 158.4 | 170.2 | **127.2** 122.5 122.6 107.0 107.7 | **117.4** | 0.895 |
| `h2z_morph_abs` | D_mix (absolute) + smooth | 164.9 | 163.9 | 132.7 125.3 114.5 **105.0** 110.4 | 117.6 | 0.900 |
| `h2z_morph_smooth` | smooth only, no D_mix | 179.1 | 159.7 | 146.3 147.4 146.7 132.2 120.6 | 138.6 | 0.907 |
| `h2z_base` | none, no `L_latcyc`; S2 cut short | 226.2 | 185.3 | 155.4 154.1 155.7 156.3 156.6 | 155.6 | 0.911 |
| *ceiling* (do nothing) | — | 228.5 | 217.3 | | | |

### 1. `L_latcyc` is NOT required — the early-stopping criterion was broken

An earlier version of this file claimed the opposite, on the strength of
`latcyc` 132.2 against `base` 226.2.  That comparison was an artefact.

`h2z_base_longS2` is `h2z_base` with one change — the S2 early stop removed —
and it reaches **109.6**, better than `latcyc`'s 132.2 and the best endpoint of
any arm.  Its path FID falls monotonically to 96.7, statistically level with
`latcyc`'s 95.8, so the morph is intact too.  Same objective, `w_latcyc = 0`,
117 FID points of difference, and the only variable is how long S2 ran:

| | `w_latcyc` | S2 epochs | FID A→B |
|---|---|---|---|
| `h2z_base` | 0 | early stop @ **20** | 226.2 |
| `h2z_base_longS2` | 0 | full **80** | **109.6** |
| `h2z_latcyc` | 2 | early stop @ 65 | 132.2 |

The cause is diagnosed in full under "The `h2z_base` result" below and is fixed
in `val_monitor`.  On this dataset `L_latcyc` is not merely unnecessary, it
appears to *cost* endpoint quality — though `latcyc` also carries the shorter
S2, so the two are not perfectly separated and the honest statement is that
`w_latcyc = 0` with a properly trained S2 is the best configuration measured.

The `latcyc`-derived arms below remain valid as an ablation of S3, because they
all fork from the same checkpoint.  But that checkpoint is now known not to be
the best S2 available, so their *absolute* numbers understate what a
correctly-forked ablation would produce.

### 2. Path supervision buys path quality with endpoint quality

This is the clearest new result, and `02_fid_path.png` shows it as a crossing:

* `latcyc`, with no path terms at all, has the **best endpoint** (95.8 at f⁴,
  FID A→B 132.2) and the **worst early frames** (156.6 at f⁰).
* `ra` / `abs`, with `L_path`, have far better early frames (127.2 / 132.7) and
  a better path mean (117.4 / 117.6 against 130.3) but pay for it at the
  endpoint (158.4 / 164.9).

The curves cross between f² and f³.  Supervising the trajectory does exactly
what it is supposed to do — and it is not free.  ADNI could not see this
trade-off: `mean|Δ|` measures how much moved, not whether the frame is a
plausible image, and the hole detector only fired in one direction.

### 3. `smooth`-only is the weakest arm, on a second dataset

`w_path_smooth` alone is a one-sided penalty whose optimum is "all frames
identical" — nothing in it says an intermediate should look like an image.  On
ADNI that arm did not move (`mean|Δ|` 0.027 at the last frame against 0.153 for
`abs`).  Here it is worst among the three path arms on both axes: path mean
138.6 and endpoint 179.1, with a curve that is flat from f⁰ to f² before
dropping.  Two datasets, same conclusion, measured two different ways.

### 4. The `ra`-over-`abs` advantage does NOT transfer, as predicted

On ADNI the relativistic critic was the fix for D_mix's unreachable target
(`fake → 1` cannot be satisfied by a frame that belongs to neither real set).
Here `ra` and `abs` are effectively tied: path means 117.4 vs 117.6, and they
split the two directions (`ra` better A→B, `abs` better B→A).

That is the predicted outcome.  Horses and zebras share the entire background
distribution and differ only in the texture on the animal, so "real A ∪ B" is a
coherent image distribution and a half-striped horse sits near it.  T1 and FA
differ in global intensity statistics, so their union is bimodal and its
midpoint is nobody's real image — which is what made the absolute critic
misbehave there and what `ra` was introduced to repair.  Remove the pathology
and the fix stops mattering.

### What does NOT transfer from the ADNI results

**MMCLAST-cg does not beat its host on translation quality here.**  On ADNI the
rewiring alone gained +0.083 SSIM over the CycleGAN it was cut from; here the
best arm (109.6) is still well behind the host (70.6).  Dropping `L_cross`
costs real endpoint quality and no arm recovers it.  The path results above stand on their
own — they are about the trajectory — but the endpoint claim does not transfer
as stated.

**Cycle consistency is a genuine win, and structural.**  Every arm reaches
0.889–0.911 cycle SSIM against the host's 0.801, because `f` is an exact
bijection and `f⁻¹(f(z))` cancels analytically, whereas CycleGAN has to push two
independent generators together with `λ_cyc`.

But it is not a translation-quality claim, and the arms now rank in almost
exactly the wrong order to be one.  `base`, which does not translate at all,
has the *highest* cycle SSIM of any arm (0.911); `base_longS2`, which
translates best, has the *lowest* (0.889) and also the lowest SSIM(out,in)
(0.823); the host, which translates most, is lowest of all (0.801).  Cycle SSIM
here is close to a measure of how little the model changed, and an identity map
would score 1.0.

**SSIM against the input is the wrong metric here and inverts the ranking.**
See `ssim_table.json`: the host scores *worst* (0.751) precisely because it
translates most, while `base` and the morph arms score 0.85–0.88.  Anyone
reaching for "SSIM" on an unpaired dataset gets this number and draws the
opposite of the correct conclusion.  It is recorded as a warning, not a result.

### The `h2z_base` result, resolved

`h2z_base` scored 226.21 against a do-nothing ceiling of 228.52 — it did not
translate, and its morph figure is flat to match.  Two hypotheses were on the
table:

- **H1** the objective is at fault.  With `w_latcyc = 0` nothing points `f` at
  the range of `E_B`, so the frozen `D_B` cannot decode `f(z)` into a zebra.
- **H2** the early-stopping criterion is at fault.  At S2 the encoders and
  decoders are frozen and `f` is an *exact* bijection, so `f⁻¹(f(z))` cancels
  analytically inside the cycle term and the unpaired validation loss is blind
  to `f` to first order.  With `w_latcyc = 0` there is nothing else in it, so
  `base` "converged" at ep5 and patience killed S2 at ep20 with `flow_work`
  only 0.865 — while `latcyc`, whose criterion carries a first-order latcyc
  term, ran 65 epochs and reached 1.04.

**H2 is confirmed and H1 is refuted.**  `h2z_base_longS2` — same `stage1.pth`,
same objective, S2 early stop removed and nothing else changed — ran the full
80 epochs and scored **109.59**, the best endpoint of any arm.  The objective
was never the problem.

The fix is in `val_monitor` in `train.py`: the unpaired branch now always
includes the latent-cycle residual, even when it carries no weight in the
objective.  It is the only quantity available at S2 that depends on `f` to
first order, and it measures the right thing — `back_u − u` is the autoencoder
error evaluated at `f(z_A)`, which is small exactly when `f` lands inside the
range `D_B` was trained to decode.  This is a deliberate break from the
"criterion = objective minus GAN" rule that governs the paired path: a
criterion that cannot see the parameters being trained cannot select among
them.  The paired/ADNI branch is untouched and still reproduces the host at
0.6798 / 0.8125.

The asymmetry noted earlier survives: `base`'s A→B sat exactly on the ceiling
while B→A (185.3) was meaningfully below its own ceiling of 217.3.  Whatever
S2 did manage to learn in 20 epochs, it did not learn equally in both
directions.

### What is still outstanding

The five-arm S3 ablation forks from `h2z_latcyc/stage2.pth`, produced under the
broken criterion.  The comparison *among* those arms is controlled and stands,
but a re-fork from a properly-trained S2 would be needed before their absolute
numbers mean anything.  That re-run has not been done.
