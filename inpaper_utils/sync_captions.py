#!/usr/bin/env python3
"""Copy figure blocks from the paper into their snapshot caption files.

  python -m inpaper_utils.sync_captions

For each (label, caption file, header) below, the \\begin{figure}...\\end{figure} block
that carries \\label{<label>} in __X_files/iclr2027/iclr2027_conference.tex replaces the
body of snapshot_results/<caption file>; the header comment lines are rewritten too.
The caption file sits next to the figure it describes and carries the provenance the
caption itself no longer states.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / '__X_files/iclr2027/iclr2027_conference.tex'
SNAP = ROOT / 'snapshot_results'
FIGURES = [
    ('fig:mnist-compare', 'mnist_petct/mnist_compare_caption.tex', """\
% fig:mnist-compare. Rendered 5.5in wide with 8pt labels, i.e. \\linewidth in the
% ICLR template. Column order: Input, Ground truth, Ours, then the baselines.
% "Ours" = morph-smooth with stage-3 weights 1 (mnist/checkpoints/mnist_p_morph_smooth),
% chosen because it reproduces the PET grain (inpaper_utils/grain_probe.py); Table 1's
% morph-smooth row is the validation-selected tune_mnist_smooth_g0.1_s0.3. Examples:
% automatic selection, top-3 per direction by min(ours) - max(baselines) SSIM with
% distinct digits; margins and picks in the run's _manifest.json.
% Rendered by inpaper_utils/fig34_mnist.sbatch (FIG_TAG=mnist/checkpoints/mnist_p_morph_smooth).
"""),
    ('fig:mnist-path-ours', 'mnist_petct/mnist_path_caption.tex', """\
% fig:mnist-path-ours: mnist_path_a2b.pdf over mnist_path_b2a.pdf, one PDF per
% direction, two samples each -- the first two pairs per direction of fig:mnist-compare.
% Rendered 5.5in wide, 8pt labels. Model: mnist/checkpoints/mnist_p_morph_smooth
% (stage-3 weights 1), the same checkpoint as fig:mnist-compare.
% Rendered by inpaper_utils/fig34_mnist.sbatch; grain numbers from inpaper_utils/grain_probe.py.
"""),
    ('fig:adni-ablation', 'adni/adni_compare_caption.tex', """\
% fig:adni-ablation. Rendered 5.5in wide with 8pt labels, i.e. \\linewidth; FA panels
% as DEC-FA (inpaper_utils/decfa.py). NOTE: this figure compares against the external
% baselines; the caption below calls it an ablation. The variant ablation is
% adni/archived/00_inpaper_fig_ablation_decfa.pdf.
% "Ours" = MMCLAST-cg w/ Morph-smooth (adni_morph_smooth). Examples: automatic selection,
% top-3 per direction by min(ours) - max(baselines) SSIM with distinct subjects; margins
% and picks in the run's _manifest.json.
"""),
    ('fig:adni-path-ours', 'adni/adni_path_caption.tex', """\
% fig:adni-path-ours: adni_path_a2b.pdf over adni_path_b2a.pdf, one PDF per direction,
% two held-out slices each, drawn from fig:adni-compare. Rendered 5.5in wide, 8pt labels; DEC-FA.
"""),
    ('fig:adni-cycle-b2a', 'adni/adni_cycle_caption.tex', """\
% fig:adni-cycle-b2a: both cycle directions in one figure, (a) T1->FA->T1 and
% (b) FA->T1->FA, two held-out slices each -- the slices of fig:adni-path-ours. DEC-FA; one shared
% error scale.
% Rendered by inpaper_utils/cycle_ab.sbatch (make_flow_viz --figures cycle_ab --ab-pick).
"""),
    ('fig:adni-flow-features', 'adni/adni_features_caption.tex', """\
% fig:adni-flow-features: the symmetric feature-map figure for morph-smooth, included at
% 0.92\\linewidth so the caption fits on the same float page. Numbers quoted "over the test
% set" are means over the 430 held-out ADNI slices.
% Rendered by make_flow_viz --figures features_sym.
"""),
    ('fig:mnist-missing-modality', 'mnist_petct/mnist_missing_modality_caption.tex', """\
% fig:mnist-missing-modality: E1b in (a), E1c in (b) and (c).
% (a) a linear probe fitted on real PET codes and applied unchanged to the other code
% sets, with the source modality's own probe as the fourth bar -- that bar is what makes
% the chance result in the third bar readable as a coordinate mismatch rather than as
% missing information.
% (b, c) the augmentation curves, one per translation direction, reduced to the three
% series that bear on the missing-modality claim: nothing added, translated, real. The
% path / straight-line / mixup arms answer a different question and live in mnist_aug.pdf.
% Drawn by inpaper_utils/repr_collect.py from cross_modal_probe.py and mnist_aug.py.
"""),
    ('fig:mnist-pca', 'mnist_petct/mnist_pca_path_caption.tex', """\
% fig:mnist-pca: both directions' flow states in a PCA fitted on the endpoint codes only.
% Linear projection on purpose -- relative distance and straightness survive it, which is
% not true of t-SNE or UMAP. Black = mean trajectory numbered by block, grey = ~25
% individual samples, clouds = the endpoint codes. Drawn by inpaper_utils/domain_probe.py
% from the pooled features of inpaper_utils/latent_geometry.py; the quantitative claims in
% the text (path/chord, probe margin) are measured in the latent space, not in this
% projection.
"""),
    ('fig:adni-flow-affine', 'adni/adni_affine_caption.tex', """\
% fig:adni-flow-affine: the affine terms of both directions, (a) T1->FA and (b) FA->T1,
% two held-out slices each -- the slices of fig:adni-path-ours. One PDF per direction, stacked; the
% two colour scales are shared and drawn once, under (b).
% Rendered by inpaper_utils/cycle_ab.sbatch (make_flow_viz --figures affine_ab --ab-pick).
"""),
]


def block(tex, label):
    k = tex.index(f'\\label{{{label}}}')
    i = tex.rindex('\\begin{figure}', 0, k)
    return tex[i:tex.index('\\end{figure}', k) + len('\\end{figure}')]


def main():
    from inpaper_utils.paper_tex import flat
    tex = flat()     # the figures now live in sections/*.tex
    for label, name, header in FIGURES:
        (SNAP / name).write_text(header + block(tex, label) + '\n')
        print('synced', label, '->', SNAP / name)


if __name__ == '__main__':
    main()
