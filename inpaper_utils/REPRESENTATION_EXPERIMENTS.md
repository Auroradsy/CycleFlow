# Representation experiments — plan and status

Two additions to the paper, both arguing that the learned representation (and the
trajectory through it) carries real semantic content rather than modality style.

Status legend: `[ ]` not started · `[~]` running · `[x]` done · `[-]` dropped, with why.

## E2 — latent geometry: where the intermediate states sit

No t-SNE/UMAP: they are non-metric, their inter-cluster distances are not interpretable,
and the picture is decided by `perplexity`/`n_neighbors`. Everything below is measured in
the latent space itself, with no embedding and no hyperparameter that changes the sign of
the answer.

- [ ] **E2.1 dual distance curve.** Per block $\ell$, relative distance from $z^{(\ell)}$
  to the source code $z^{(0)}$ and to the paired target code $E_B(y)$, mean ± std over the
  full test split, both directions. Overlay the linear-interpolation reference
  $\hat z(t)=(1-t)z^{(0)}+t\,E_B(y)$ at $t=\ell/L$, which is two straight lines by
  construction. Raw and whitened (per-channel std of the target endpoint codes), plus
  cosine; if raw and cosine disagree, that is itself reportable.
- [ ] **E2.2 path length / chord ratio.** $\sum_\ell\|z^{(\ell)}-z^{(\ell-1)}\|$ over
  $\|z^{(L)}-z^{(0)}\|$, per sample. $\approx 1$ would mean the flow is linear
  interpolation and the trajectory claim is empty. Per-block step norms alongside, to
  cross-check the division of labour seen in the affine figure.
- [ ] **E2.3 domain probe.** Logistic regression separating $E_A$ from $E_B$ codes, fitted
  on spatially pooled features with grouped k-fold (by subject on ADNI), scored on the
  held-out fold's intermediate states: $P(\text{B})$ as a function of $\ell$. Plus the
  non-parametric version — the fraction of each state's $k$ nearest neighbours that come
  from domain B.
- [ ] **E2.4 paired vs random target.** Distance to the paired $E_B(y)$ against distance
  to a random other subject's $E_B(y')$. If they coincide, the flow only does a
  modality-level style change; the gap is the subject-specific work.
- [ ] **E2.5 figure.** PCA fitted on the endpoint codes only (linear, so straightness and
  relative distance survive the projection), trajectories drawn in it.

## E1 — does the representation carry class information

A reviewer's first objection to "translate and augment" is that a synthetic target
generated from an image you already hold adds no information about its label. The three
framings below survive that objection; plain self-augmentation does not, and is not used.

- [ ] **E1a ADNI feasibility gate.** Before anything else: is the label predictable from
  these slices at all? Real T1 / real FA / both, `label_2` (107 healthy vs 107 impaired),
  subject-level 5-fold × 3 repeats over all 214 subjects, slice predictions averaged to a
  subject-level decision, balanced accuracy and AUC as mean ± std, plus a label-permutation
  control. The single 43-subject test split cannot resolve anything below ~15 points, so
  it is not used here. **If real data sits at chance, E1b/E1c are MNIST-only on ADNI's
  behalf and this is reported as a negative result, not quietly dropped.**
- [ ] **E1b cross-modal probe.** Fit a classifier on real $E_B(x_B)$ codes; without
  retraining, apply it to $f(E_A(x_A))$ of held-out data. Controls: real $E_B$ (upper
  bound), $E_A(x_A)$ with no flow (shows $f$ is what makes the codes readable), and the
  baselines' translations re-encoded by $E_B$.
- [ ] **E1c MNIST low-label augmentation.** Task: digit classification on PET. Labelled
  PET is scarce ($n\in\{10,25,50,100\}$ per class); a large labelled CT corpus is not, and
  its labels travel with the translation. Arms: PET only · + endpoint translation ·
  **+ decoded intermediate states** · + decoded linear-interpolation states · + pixel
  mixup. The comparison that matters is the flow's states against the interpolation's:
  it turns "the trajectory is principled rather than arbitrary" into a number.

## Results (2026-09-17)

### E2 — latent geometry: done, both datasets

Relative distance to the paired target code, per block, against the straight line between
the same two endpoints, and against a random other target:

```
ADNI  T1->FA   flow 1.290 1.202 1.100 0.986 0.831   line 1.290 0.968 0.645 0.323 0.000
      FA->T1   flow 1.290 1.155 1.070 0.961 0.778   random target ends at 0.927
MNIST CT->PET  flow 1.217 1.129 1.021 1.023 1.006   random target ends at 1.336
      PET->CT  flow 1.217 1.089 1.028 0.861 0.607   random target ends at 1.343
```

**Path length over chord**: ADNI 1.947 +- 0.017 and 2.020 +- 0.029; MNIST 2.151 +- 0.024
and 2.274 +- 0.036. About twice the straight line in all four directions, std under 0.04.
A ratio near 1 would have meant the flow is linear interpolation and the trajectory claim
is empty; it is 2.

**CT->PET is the direction that stalls**: its distance to the paired target code stops
falling after block 2 (1.021, 1.023, 1.006) while its distance to a *random* target stays
at 1.34. It lands in the right neighbourhood without approaching the specific code, which
is what a direction that has to invent the target's acquisition noise should do.

**Domain probe (E2.3)**, margin across the A/B discriminant, 0 = source side, 1 = target
side, endpoint probe accuracy 1.000 in every case:

```
ADNI  T1->FA   0.001 0.241 0.422 0.701 1.082      MNIST CT->PET  0.000 0.222 0.405 0.735 1.069
      FA->T1   0.999 0.757 0.589 0.281 -0.064           PET->CT  1.000 0.755 0.473 0.226 -0.034
```

Steady, monotone, and it overshoots slightly past the target side. `P(B)` itself saturates
to 0/1 once the endpoints are perfectly separable, which they are, so the margin is the
measure to report and P(B) only fixes the crossing point (between blocks 2 and 3).

Two things caught by their own controls, both of which would have produced a confident
wrong answer:

* **the pooled features were identically zero.** A global spatial mean of the latent is
  exactly 0 for every sample, because the encoder ends in `InstanceNorm2d(affine=False)`.
  The probe was reading noise (endpoint accuracy 0.485) and still produced a plausible
  rising curve. Pooling to 4x4 fixes it; `domain_probe.py` now refuses to report a curve
  when the endpoint accuracy is below 0.9.
* **whitening the distances was a no-op** for the same reason -- `channel_std` is
  1.000..1.000 in both domains. The column was removed rather than left as a mystery.

### E1a — the ADNI gate FAILS, and the failure is trustworthy

| input | balanced acc | AUC |
|---|---|---|
| real T1 | 0.540 +- 0.008 | 0.550 +- 0.014 |
| real FA | 0.508 +- 0.022 | 0.517 +- 0.006 |
| real T1+FA | 0.525 +- 0.037 | 0.531 +- 0.038 |
| permuted labels | 0.472 | 0.428 |
| **positive control: axial slice index, 10-way** | **0.980 accuracy** (chance 0.100) | |

214 subjects, 5-fold x 3 repeats, grouped by subject. The standard error of a balanced
accuracy over 214 subjects is about 0.034, so none of the three is distinguishable from
chance. The positive control settles the interpretation: the same network on the same
folds reads the slice index at 0.980, so this is a fact about the diagnosis label, not
about the classifier. `label_2` is the easiest form of the task, so the four-class version
cannot do better.

Consequence: E1c is MNIST-only and ADNI's contribution is E1b's label-free retrieval plus
E2. The gate is reported, not buried.

### E1b — cross-modal probe: the strongest result here

MNIST, a digit classifier fitted on real PET codes only and applied without retraining
(5,000 held-out pairs, chance 0.100):

| code | accuracy |
|---|---|
| `E_PET(y)` real target | 0.913 +- 0.008 |
| **`f(E_CT(x))` our translation** | **0.886 +- 0.008** |
| `E_CT(x)` no flow | **0.106 +- 0.002** |
| decoded and re-encoded | 0.907 +- 0.009 |
| `E_CT(x)` under its *own* probe | 0.929 +- 0.005 |

Three lines close the argument. The source code carries the digit perfectly well -- its own
probe reads it at 0.929, better than the PET probe reads real PET. The PET probe reads that
same source code at chance. And `f` makes it readable at 0.886, 97% of the upper bound. So
what the source code lacks is not the class, it is alignment with the target space, and the
flow is what supplies it.

On ADNI the own probe scores 0.588 +- 0.121 against a chance of 0.605, which is the gate's
verdict again from a third direction: there is no diagnosis signal to align.

ADNI retrieval, label-free, 430 real FA codes in the gallery (chance top-1 0.002):
`f(E_T1(x))` top-1 0.174 / top-5 0.393 / median rank 14, against 0.077 / 0.188 / 35 for
the un-flowed source code, and 0.256 / 0.444 / 8 after decoding and re-encoding.

Each dataset separates on a different measure, and both are reported: on MNIST retrieval
does *not* separate (`E_CT(x)` scores 0.908, higher than the flow output, because CT and
PET share spatial layout and retrieval rides on layout), while the label probe separates
cleanly; on ADNI the label probe is uninformative and retrieval separates.

### E1c — augmentation works; the *path* does not beat a straight line

MNIST PET digit accuracy, 3 seeds, every arm given the same 1,500 gradient steps:

| arm | n=10 | n=25 | n=50 | n=100 |
|---|---|---|---|---|
| PET only | 0.7651 | 0.8895 | 0.9259 | 0.9504 |
| + translation (endpoint) | 0.9542 | 0.9575 | 0.9625 | 0.9645 |
| + flow path | 0.9550 | 0.9569 | 0.9611 | 0.9635 |
| + straight line | 0.9564 | 0.9591 | 0.9634 | 0.9651 |
| + mixup | 0.9325 | 0.9383 | 0.9459 | 0.9541 |
| real PET (oracle) | 0.9718 | 0.9705 | 0.9706 | 0.9722 |

Translating a labelled CT corpus into PET recovers 91% of the gap between 100 labelled
images and the oracle at n=10 (84% at n=25), and the CT corpus is a disjoint half of the
training set, so no synthetic image is derived from a labelled PET image.

Corpus-size sweep at 10 labels per class, which removes the saturation confound -- with
6,000 translated images every augmented arm is already near the oracle:

| arm | 100 | 300 | 1000 | 3000 | 6000 |
|---|---|---|---|---|---|
| PET only | 0.7651 | 0.7651 | 0.7651 | 0.7651 | 0.7651 |
| + translation | 0.8507 | 0.9021 | 0.9301 | 0.9486 | 0.9542 |
| + flow path | 0.8533 | 0.9023 | 0.9323 | 0.9484 | 0.9550 |
| + straight line | 0.8523 | 0.9062 | 0.9339 | 0.9511 | 0.9564 |
| + mixup | 0.8390 | 0.8793 | 0.9085 | 0.9346 | 0.9325 |
| real PET (oracle) | 0.8633 | 0.9120 | 0.9452 | 0.9654 | 0.9718 |

**The flow's intermediate states are not better training data than a straight line through
the latent space.** `+path` and `+interp` are within seed noise in all nine settings --
four labelled counts and five corpus sizes -- with the straight line marginally ahead in
seven of them. The tie is not an artifact of saturation: it holds at corpus 100, where the
augmented arms are 10 points below the oracle and there is ample room to separate.

In hindsight this is what should have been expected: digit identity is preserved along any
path between the same two endpoints, so a classifier has nothing to distinguish. The
finding stands as a negative, and the trajectory's value stays where it is defensible --
auditability and the geometry of E2, not downstream gain. Translation-as-augmentation is
the positive result here and it does not depend on the path at all.

A step-budget confound was found and fixed before any of this was believed: counting
epochs gave `pet_only` 25 gradient steps against 1,200 for the augmented arms, which alone
put it at chance (0.106 rather than 0.765). All arms now get the same number of steps,
sampled with replacement.

## Order

E2 first (forward passes on existing checkpoints, no training). Then E1a, which gates the
ADNI half of E1. Then E1b, then E1c.
