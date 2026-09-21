# ADNI paper visualizations

Each panel uses the same five held-out paired examples. Each example has two rows (both modalities or decoder views). PNG is for preview; PDF is for the paper. Sample IDs, checkpoint paths, per-image SSIM and sampling details are in `00_inpaper_manifest.json`.

- `00_inpaper_self_recon_baseline`: direct self reconstruction, comparing the DiT latent autoencoder (posterior mean) and CycleFlow base / latcyc / morph. A translation model's cycle reconstruction is not treated as direct self reconstruction.
- `00_inpaper_cross_gen_baseline`: both translation directions on the same pairs. CycleFlow base is an ablation reference, not an independent external baseline. DiT uses its full ancestral sampler and CFG 1.5. Its latent normalization is recovered from the training log rounded to three decimals, because the saved AE contains default normalization; this precision limitation is marked with an asterisk.
- `00_inpaper_process_a2b` and `00_inpaper_process_b2a`: the morph variant's actual flow block outputs, decoded through both decoders. No image blending or latent interpolation. The last state is checked against the model's normal translation path.

Examples are chosen before inference, without ranking by reconstruction quality. Display range is fixed to [0,1] for all methods; no per-image contrast adjustment. SSIM is computed on individual images, not a whole-test-set metric.

ADNI uses five distinct test subjects, all at axial z=44, from the original seed-42 subject split (z=40–49, label_4). The migrated directory currently has no ADNI CycleGAN, CFM, DDPM or MeanFlow checkpoints; the available external translation comparison is DiT.

Regenerate from the repository root:

```bash
/ix/lzhan/siyuan/envs/cycleflow/bin/python -m inpaper_utils.make_inpaper --dataset adni --with-dit
```

Full run logs and archived figures: `/ix/lzhan/siyuan/exps/CycleFlow/adni/inpaper_visualization/<run>/`.

## Paper figures (2026-09-17 naming)

Everything the ICLR submission does not include was moved to `archived/` (138 files, plus
`decfa_preview/` and the caption files of the removed figures). What remains is named
`adni_<what>`, flat, because the paper's `figures/` directory holds both datasets and the
old `00_inpaper_*` stems collided with the MNIST ones. `<name>.pdf` goes in the paper,
`<name>.png` is the preview, `<name>_caption.tex` is the caption as it stands in the paper.

| name | paper figure | paper label | rendered from |
|---|---|---|---|
| `adni_compare` | Figure 2 | `fig:adni-ablation` | `00_inpaper_fig_compare_decfa` |
| `adni_path_a2b` / `adni_path_b2a` | Figure 4 (a)/(b) | `fig:adni-path-ours` | `00_inpaper_fig_path_ablation_{a2b,b2a}_ab_decfa` |
| `adni_features` | Figure 5 | `fig:adni-flow-features` | `00_inpaper_flow_features_sym_morph-smooth` |
| `adni_affine_a2b` / `adni_affine_b2a` | Figure 6 (a)/(b) | `fig:adni-flow-affine` | `00_inpaper_flow_affine_{a2b,b2a}_ab_morph-smooth_decfa` |
| `adni_cycle` | Figure 7 | `fig:adni-cycle-b2a` | `00_inpaper_cycle_ab_morph-smooth_decfa` |

`adni_pca_path` and `adni_domain_margin` are the ADNI halves of the representation
experiments. The paper shows the MNIST ones instead (Figure 8), so these two stay here
unreferenced; their caption file is `adni_pca_path_caption.tex`.

`adni_compare` shows CycleFlow against the external baselines, not the variant ablation,
although the caption in the paper currently calls it an ablation. The variant ablation is
`archived/00_inpaper_fig_ablation_decfa.pdf`.

The paper reads its own copies under `__X_files/iclr2027/figures/`. After re-rendering
anything here, restage them:

```bash
python -m inpaper_utils.sync_figures          # or --check to see what is stale
python -m inpaper_utils.collect_fig_sources   # refresh __X_files/iclr2027/figs_scripts/
```

`__X_files/iclr2027/figs_scripts/fig<N>/` holds, per figure, the code that drew it, the
run's manifests and metrics, and the pinned run directory it came from.
