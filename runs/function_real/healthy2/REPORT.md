# Barrier proxy against measured TER at adequate power: the Healthy-2 iRPE maturation series

Run 2026-09-21/22. Script `scripts/pimorph_function_healthy2.py` (subcommands in the order they were run:
`export-train`, `eval-segmenter`, `inventory`, `select`, `download`, `segment`, `stats`, `qc`); fine-tuning
with `runs/cellpose_ft/train_cpsam.py`. Heavy steps ran on the xAI devbox (one H100 each), statistics locally.

## Question and data

Does the structural part of PiMorph's barrier proxy (the resistor network of `pimorph.function.transport`)
carry information about measured transepithelial resistance (TER) once the sample is large enough to tell?
The earlier AMD test (`../REPORT.md`) had 10 wells over a 780 to 1030 Ohm cm^2 range (Spearman -0.52, p 0.13).

Data: NIST / NEI `Healthy2_Data` (Schaub, Hotaling, Bharti et al. 2020 JCI; DOI 10.18434/T4/1503229;
inventory in `data/rpe_nist/healthy2/INVENTORY.md`). Live QBAM bright-field absorbance (blue 488 filter,
1040 x 1388 float32 tiles, 4 x 3 grid per well) of iPSC-RPE from one healthy donor in 12-well Transwells,
imaged weekly; TER read the same day. 36 wells with TER (12 each: Control, Aphidicolin, HPI4), weeks 3 to 8,
216 TER values from 96 to 1146 Ohm. There is no junction stain in this series, so only the structural part
of the proxy can be tested: cell number, geometry and gaps with a CONSTANT conductance per interface. Junction
coverage, the quantity the barrier model was written for, is not observable here.

Subset analysed (`data/rpe_nist/healthy2/selection.csv`): 84 well-timepoints covering all 36 wells, every
week 3 to 8 (11 to 17 wells per week) and TER 100 to 1115 Ohm (Aphidicolin 30, Control 29, HPI4 25), two
adjacent centre tiles (`r001_c001`, `r002_c001`) per well-timepoint, 168 tiles, all downloaded
(`data/rpe_nist/healthy2/download.log`).

## 1. QBAM cell segmenter

Training pairs: the 1,032 registered 256 px AMD tiles (blue absorbance uint8 / 255 + hand-corrected border
masks) converted to instances by `pimorph.io.rpe_nist.qbam_instances` (borders closed by watershed, as
`skeleton_roi_to_instance`). Almost every tile (1,031 of 1,032) has a region the annotators left without
borders (median 21% of the tile); these are detected (`qbam_unannotated`), set to the tile median in the image
and to 0 in the labels, and excluded from scoring. Held out by well: `AMD1_A_D75` and `AMD2_B_D73` (175 tiles,
17%, two donors, clones absent from training); train 857 tiles, 126,639 cells.

Cellpose-SAM fine-tuned with the repository recipe (AdamW, lr 1e-5, weight decay 0.1, batch 8, 256 px, 100
epochs, one crop per tile per epoch): 45 min on one H100, weights
`runs/cellpose_ft/cpsam_qbam/models/cpsam_qbam` (devbox; 1.2 GB, not pulled), `train_info.json` pulled.

Held-out AMD wells, Cellpose proposer + constrained decoder (`CellposeProposer` -> `ConstrainedDecoder`,
vertex_weight 0.3), scored with `pimorph.metrics.structural.structural_metrics` against the curated
instances (unannotated regions ignored):

| model | tiles | pred / true cells | AP50 | AP75 | PQ | adjacency pair F1 | vertex F1 (P / R) | boundary F1 |
|---|---|---|---|---|---|---|---|---|
| cpsam_qbam (fine-tuned) | 175 | 23,295 / 24,769 | **0.741** | 0.535 | 0.692 | 0.703 | 0.657 (0.747 / 0.612) | 0.939 |
| cpsam zero-shot | 175 | 17,690 / 24,769 | 0.459 | 0.258 | 0.444 | 0.385 | 0.343 (0.750 / 0.320) | 0.701 |

Per well (fine-tuned): `AMD2_B_D73` AP50 0.801, PQ 0.743, adjacency 0.778, vertex F1 0.836; `AMD1_A_D75`
AP50 0.706, PQ 0.663, adjacency 0.659, vertex F1 0.554 (the darker, less regular well). Cellpose's raw
masks score within 0.01 of the decoded complexes (raw AP50 0.731, vertex F1 0.648), so the decoder neither
adds nor removes accuracy here. Vertex localisation error 1.3 px median. Per-tile values in
`segmenter_holdout_per_tile.csv` and `zeroshot/`.

The segmenter was trained on mature, well pigmented AMD cultures. On Healthy-2 it finds cells only where
pigment gives contrast: median tissue fraction (area of decoded cells / tile) is 0.54 in Aphidicolin wells,
0.47 in Control wells and 0.03 in HPI4 wells (HPI4 blocks maturation and pigmentation; the crops in `qc/`
show no cell borders visible to the eye either). 39 of 168 tiles (all HPI4) contain fewer than three cells
and yield no complex; 18 of the 25 HPI4 well-timepoints drop out of the main analysis for that reason. The
decoded Healthy-2 cells have median area 585 px^2 (radius 13.4 px) against 310 px^2 for the AMD truth; the
pixel size is recorded in neither deposit, so whether this is culture density or optics is not verifiable.

## 2. Per-tile quantities

Per decoded complex (`per_tile.csv`, 129 tiles with a complex, 150,527 cells): cell count and density
(cells per 1000 px^2 of tile), mean cell area and area CV, gap area fraction (1 - cell area / tile area),
number of gap faces, cell-cell edges, edges per cell, mean edge arclength; from `pimorph.function.transport`
with g_e = 1 per cell-cell interface and g_gap = 10 per enclosed gap face: parallel-model G_eff, permeability
index (G_eff per tissue area) and gap share of G_eff; in-plane effective conductance between the cells on the
left and right image borders (`solve_in_plane`); the same with g_e proportional to interface length (`_len`
columns). Aggregation: mean over the two tiles per well-timepoint (`per_well_timepoint.csv`).

Posterior (`posterior_tiles.csv`): 20 tiles stratified over TER, 512 px centre crops, `cellpose_hypotheses`
(Cellpose flow / cell-probability threshold grid x decoder perturbations and merge/split moves, 56
hypotheses per tile) weighted by the complex energy (`PosteriorEnsemble`, ESS 3.9). Median relative 90%
credible width: cell count 0.009, permeability index 0.013, in-plane conductance 0.005, area CV 0.057. The
fine-tuned network is confident, so the within-family uncertainty is one to two orders of magnitude below
the between-well spread; it does not capture the dominant error, which is cells the network cannot see at
all. Seven of the 20 tiles (six HPI4, one week-4 Aphidicolin) have no hypothesis with three cells.

## 3. Correlation with TER (n = 66 well-timepoints with a complex, 29 wells)

Spearman across well-timepoints with bootstrap 95% CI; Pearson against log TER; rank partial correlations
given condition dummies, given condition + week dummies, and given cell density + area CV; leave-one-well-out
(LOWO) range of Spearman with each well removed and pooled R^2 of a LOWO linear prediction. Full table
`ter_correlations.csv`, per-condition values there and in `tables.md`.

| statistic | kind | Spearman | p | 95% CI | partial rho given condition (p) | given condition + week (p) | given density + area CV (p) | LOWO Spearman range | LOWO R^2 |
|---|---|---|---|---|---|---|---|---|---|
| permeability index (g const) | proxy | **+0.524** | 6e-06 | [0.30, 0.70] | 0.375 (0.002) | 0.008 (0.95) | -0.017 (0.89) | [0.48, 0.59] | 0.198 |
| G_eff (parallel) | proxy | +0.531 | 5e-06 | [0.31, 0.70] | 0.335 (0.007) | -0.014 (0.92) | 0.048 (0.71) | [0.49, 0.59] | 0.123 |
| in-plane conductance | proxy | +0.464 | 9e-05 | [0.21, 0.65] | 0.252 (0.045) | -0.026 (0.84) | -0.035 (0.78) | [0.41, 0.53] | 0.062 |
| gap share of G_eff | proxy | +0.349 | 0.004 | [0.09, 0.57] | 0.112 (0.38) | -0.083 (0.53) | -0.254 (0.043) | [0.29, 0.45] | 0.078 |
| permeability index (g ~ length) | proxy | +0.529 | 5e-06 | [0.31, 0.70] | 0.372 (0.002) | 0.015 (0.91) | 0.020 (0.88) | [0.48, 0.59] | 0.199 |
| cell density | control | +0.524 | 6e-06 | [0.29, 0.70] | 0.342 (0.006) | -0.013 (0.92) | | [0.48, 0.59] | 0.163 |
| cell area CV | control | -0.193 | 0.12 | [-0.42, 0.08] | -0.224 (0.075) | -0.024 (0.86) | | [-0.26, -0.12] | -0.009 |
| gap area fraction | control | -0.532 | 4e-06 | [-0.70, -0.32] | -0.323 (0.009) | 0.018 (0.89) | -0.077 (0.55) | [-0.59, -0.49] | 0.175 |
| edges per cell | control | +0.570 | 6e-07 | [0.36, 0.73] | 0.353 (0.004) | -0.004 (0.98) | 0.309 (0.013) | [0.53, 0.62] | 0.238 |
| mean edge length | control | -0.393 | 0.001 | [-0.60, -0.14] | -0.369 (0.003) | -0.008 (0.95) | 0.031 (0.81) | [-0.44, -0.33] | 0.106 |

Per condition (Spearman): permeability index Control +0.10 (n 29, p 0.62), Aphidicolin +0.65 (n 30,
p 9e-05), HPI4 +0.54 (n 7, p 0.22); cell density +0.08 / +0.63 / +0.14. Within Aphidicolin wells both
numbers rise with week as the culture matures and pigments; within Control wells, where TER varies 117 to
774 Ohm, neither tracks it.

Sensitivity with all 84 well-timepoints, counting tiles without a complex as empty tissue (density and
G_eff 0, gap fraction 1; `ter_correlations_all_filled.csv`): permeability index Spearman +0.735 (95% CI
0.61 to 0.82), given condition 0.325 (p 0.003), given condition + week 0.060 (p 0.60); cell density
+0.735, given condition 0.297, given condition + week 0.040. The larger number is the HPI4 wells: lowest TER
and no visible cells.

Leave-one-well-out linear prediction of TER (`model_comparison_lowo.csv`, n 66):

| model | LOWO R^2 | LOWO R^2 with condition dummies | Spearman(prediction, TER) |
|---|---|---|---|
| cell density alone | 0.163 | 0.673 | 0.466 |
| density + area CV | 0.152 | 0.667 | 0.453 |
| density + area CV + gap area fraction | 0.095 | 0.666 | 0.399 |
| permeability index alone | 0.198 | 0.680 | 0.465 |
| in-plane conductance alone | 0.062 | 0.678 | 0.213 |
| density + area CV + permeability index | 0.142 | 0.664 | 0.417 |
| density + area CV + in-plane conductance | 0.139 | 0.667 | 0.441 |
| density + area CV + permeability index + in-plane conductance | 0.129 | 0.662 | 0.387 |

Condition dummies alone carry most of the predictable variance (R^2 about 0.66); no structural quantity adds
more than 0.02 on top of them, and adding any transport number to density + area CV lowers the out-of-well
R^2.

## 4. What this says about the proxy

1. The transport proxy has the wrong sign. The permeability index is a predicted leak (higher = leakier),
   yet it rises with TER: Spearman +0.52 (p 6e-06), stable under leave-one-well-out (0.48 to 0.59). With a
   constant conductance per interface, G_eff per area is the interface density, a monotone function of
   cell density, and denser, smaller cells are what a maturing RPE monolayer shows. The proxy therefore
   tracks maturation, and maturation raises TER.
2. It adds nothing over density and area CV. Partial correlation of the permeability index with TER given
   cell density and area CV: -0.02 (p 0.89); in-plane conductance -0.04 (p 0.78). The LOWO R^2 of density +
   area CV falls from 0.152 to 0.142 when the permeability index is added and to 0.129 when both transport
   numbers are added. The one structural quantity with residual information is edges per cell (partial
   +0.31, p 0.013), a plain connectivity statistic that the transport model does not use differently from
   density.
3. Within condition it survives (partial +0.375, p 0.002) only because weeks are pooled; within condition
   and week the partial correlation is +0.008 (p 0.95). Every column of the table behaves the same way:
   the well-to-well differences at a given week and treatment, which is what a barrier assay would have to
   resolve, are not seen by any of these quantities.
4. The largest single effect in the data, the difference between HPI4 wells (about 110 Ohm) and the rest,
   is captured only through the failure of the segmenter to find cells in unpigmented tissue. That is an
   absorbance signal, the one the paper's DNN-F exploits, not a structural one.

Conclusion: at n = 66 to 84 well-timepoints over a tenfold TER range the structural part of the barrier
proxy is not validated. It is anti-correlated with what it claims to predict, and it is redundant with cell
density. `barrier_report` keeps `validated: False`; the flag is not changed because the condition for
changing it (a significant within-condition association that survives leave-one-well-out) is met only in
the direction opposite to the model's meaning, and disappears within week. What remains untested is the
junction-coverage term, which needs a series with a ZO-1 or claudin stain and TER on the same wells.

## 5. Visual QC pack

`qc/`: 12 PNGs (900 x 480), one per well-timepoint stratified over TER (103 to 1115 Ohm), each a centre crop
sized for about 25 cells, QBAM left, decoded outlines (yellow) and multicellular vertices (cyan) right, TER
in the title. `qc/README.md` explains how to record counts, merges, splits and a verdict in
`qc/qc_judgments.csv`. The five lowest-TER crops (four HPI4 wells at 100 to 122 Ohm and a week-6 Control
dip at 139 Ohm) contain 0 to 2 predicted cells and no borders visible to the eye, and the week-4 Control
crop at 503 Ohm has 3; the seven others have 13 to 34 (`qc/qc_index.csv`). The reviewer should say whether
they see cells where the segmenter finds none. Not yet reviewed by a human.

## 6. Not verified

- Pixel size of either deposit (all geometry in px); the 585 vs 310 px^2 cell-area difference between
  Healthy-2 decodes and AMD truth is unexplained.
- Segmenter accuracy on Healthy-2 itself: the held-out numbers are for mature AMD wells; on immature and
  HPI4 wells the segmenter demonstrably misses most cells (tissue fraction 0.03 to 0.5) and no truth exists
  to score it. The visual QC pack is the only check.
- The `_len` variant and gap conductance 10 are model choices; other constants were not explored because
  no constant changes the sign argument above.
- TER is a single reading per well and date in Ohm (12-well Transwell, about 1.12 cm^2) from EVOM.csv; the
  paper reports no replicate SD for this series.
- Only two of the twelve tiles per well were used; well-centre tiles may not represent the whole well.

## Files

- `per_tile.csv`, `per_well_timepoint.csv` (with a complex), `per_well_timepoint_all.csv` (empty tiles
  filled), `ter_correlations.csv`, `ter_correlations_all_filled.csv`, `model_comparison_lowo.csv`,
  `tables.md` (auto-generated tables), `proxy_vs_ter.png`.
- `segmenter_holdout_per_tile.csv`, `segmenter_holdout_summary.json`, `zeroshot/` (same for zero-shot cpsam),
  `eval_segmenter.log`, `segment.log`; training log `../../cellpose_ft/cpsam_qbam.log`, `cpsam_qbam/train_info.json`.
- `posterior_tiles.csv`; `qc/`; decoded labels stay on the devbox (`runs/function_real/healthy2/labels/`).
- Data side: `data/rpe_nist/healthy2/INVENTORY.md`, `ter_long.csv`, `tile_listing.csv`, `selection.csv`,
  `download.log`, `meta/` (DataSummary, EVOM, VEGF, filters), tiles on the devbox
  (`data/rpe_nist/healthy2/tiles/`, 168 x 5.8 MB).
