# MNIST paper visualizations

Each panel uses the same five held-out paired examples. Each example has two rows (both modalities or decoder views). PNG is for preview; PDF is for the paper. Sample IDs, checkpoint paths, per-image SSIM and sampling details are in `00_inpaper_manifest.json`.

- `00_inpaper_self_recon_baseline`: direct self reconstruction, comparing the DiT latent autoencoder (posterior mean) and CycleFlow base / latcyc / morph. A translation model's cycle reconstruction is not treated as direct self reconstruction.
- `00_inpaper_cross_gen_baseline`: both translation directions on the same pairs. CycleFlow base is an ablation reference, not an independent external baseline. DiT uses its full ancestral sampler and CFG 1.5. Its latent normalization is recovered from the training log rounded to three decimals, because the saved AE contains default normalization; this precision limitation is marked with an asterisk.
- `00_inpaper_process_a2b` and `00_inpaper_process_b2a`: the morph variant's actual flow block outputs, decoded through both decoders. No image blending or latent interpolation. The last state is checked against the model's normal translation path.

Examples are chosen before inference, without ranking by reconstruction quality. Display range is fixed to [0,1] for all methods; no per-image contrast adjustment. SSIM is computed on individual images, not a whole-test-set metric.

MNIST uses the first five distinct digit classes in test filename order (7,2,1,0,4), aligned between testA and testB. External translation baselines: CycleGAN, CFM (10 Euler steps), MeanFlow (one step), DDPM (50 DDIM steps), and DiT. CycleGAN was trained unpaired; the chosen CycleFlow variants and other translation baselines were trained paired, as recorded in checkpoint args.

Regenerate from the repository root:

```bash
/ix/lzhan/siyuan/envs/cycleflow/bin/python -m inpaper_utils.make_inpaper --dataset mnist --with-dit
```

Full run logs and archived figures: `/ix/lzhan/siyuan/exps/CycleFlow/mnist/inpaper_visualization/<run>/`.

## Paper figures (2026-09-17 naming)

Everything the ICLR submission does not include was moved to `archived/` (31 files, plus the
caption files of the removed ablation and sensitivity figures). What remains is named
`mnist_<what>`, flat, because the paper's `figures/` directory holds both datasets and the
old `00_inpaper_*` stems collided with the ADNI ones. `<name>.pdf` goes in the paper,
`<name>.png` is the preview, `<name>_caption.tex` is the caption as it stands in the paper.

| name | paper figure | paper label | rendered from |
|---|---|---|---|
| `mnist_compare` | Figure 1 | `fig:mnist-compare` | `00_inpaper_fig_compare` |
| `mnist_path_a2b` / `mnist_path_b2a` | Figure 3 (a)/(b) | `fig:mnist-path-ours` | `00_inpaper_fig_path_ablation_{a2b,b2a}_ab` |
| `mnist_pca_path` | Figure 8 | `fig:mnist-pca` | `domain_probe.py` (E2.5) |
| `mnist_missing_modality` | Figure 9 | `fig:mnist-missing-modality` | `repr_collect.py` (E1b + E1c) |

`mnist_aug`, `mnist_aug_table.tex`, `mnist_domain_margin` and `mnist_latent_distance` are
the rest of the representation experiments; the paper does not include them.

The paper reads its own copies under `__X_files/iclr2027/figures/`. After re-rendering
anything here, restage them:

```bash
python -m inpaper_utils.sync_figures          # or --check to see what is stale
python -m inpaper_utils.collect_fig_sources   # refresh __X_files/iclr2027/figs_scripts/
```

`__X_files/iclr2027/figs_scripts/fig<N>/` holds, per figure, the code that drew it and the
data it was drawn from; Figures 8 and 9 redraw there from their bundled data alone.
