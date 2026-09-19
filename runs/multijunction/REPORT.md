# Multi-junction fields and predicted barrier conductance on S-BIAD1169

Dataset: BioImage Archive S-BIAD1169 (Bromberger and Schossleitner, CC BY 4.0). Confluent primary dermal microvascular endothelial cells, 1 h DMSO or BRAF inhibitor (dabrafenib, vemurafenib, encorafenib, PLX8394 at 1, 10, 100 uM), confocal LSM-980 63x. One field per treatment folder.

## Channel roles

Source: biostudies file-list Channel attribute. Mapping of file suffix to marker: `_c1` = Prox1, `_c2` = VE-Cadherin, `_c3` = F-actin, `_c4` = Claudin-5, `_c5` = DAPI.
 The acquisition text lists DAPI (405), Claudin-5 (488), F-actin (514), VE-Cadherin (594), Prox1 (639). Exports are 8-bit RGB pseudocolor (VE-cadherin grey, claudin-5 cyan, F-actin yellow, DAPI blue), reduced to max over RGB. Calibration: burned-in 50 um scale bar (379 px), 0.1319 um/px; the TIFF resolution tag is a print dpi and was ignored. Roles used here: AJ = VE-Cadherin, TJ = Claudin-5, actin = F-actin; geometry from VE-Cadherin, nuclei from DAPI.

## Pipeline

ClassicalProposer on VE-cadherin with DAPI seeds, ConstrainedDecoder with cell_radius_px from the proposer, DAPI smoothed with sigma 0.5 um before seeding, multichannel_profiles with bin_px=2.0, half_width_px=3.0, Otsu thresholds on pooled strip samples per channel, a bin counts as occupied at >= 30% of strip samples above threshold. Field values are arclength-weighted means over cell-cell edges. Barrier: g_e = 0.05 + 1.0 (1 - TJ coverage) L_e + 0.5 (1 - AJ coverage) L_e with L_e in um, explicit gaps 10.0 each, G_eff = sum (parallel model), permeability index = G_eff per um^2 of cell area.

## WARNING: permeability numbers are untested predictions

`validated: False`. No TEER, tracer flux or any functional barrier measurement exists for these fields. G_eff and the permeability index are what the resistor model implies for the reconstructed junction coverage; they are prediction targets for a future functional experiment, not measurements.

## Per condition (n fields: {'BRAFi': np.int64(14), 'DMSO': np.int64(3)})

| statistic | BRAFi (mean +- sd) | DMSO (mean +- sd) | Mann-Whitney p |
|---|---|---|---|
| AJ_coverage | 0.394 +- 0.058 | 0.414 +- 0.007 | 0.676 |
| TJ_coverage | 0.358 +- 0.081 | 0.376 +- 0.036 | 1.000 |
| co_occupancy_AJ_TJ | 0.235 +- 0.064 | 0.248 +- 0.019 | 0.676 |
| jaccard_AJ_TJ | 0.389 +- 0.083 | 0.376 +- 0.041 | 0.859 |
| exclusive_AJ | 0.092 +- 0.019 | 0.095 +- 0.009 | 0.953 |
| exclusive_TJ | 0.085 +- 0.025 | 0.071 +- 0.028 | 0.432 |
| n_gaps | 1.143 +- 4.276 | 0.000 +- 0.000 | 0.758 |
| gap_fraction (untested prediction) | 0.011 +- 0.043 | 0.000 +- 0.000 | 0.758 |
| G_eff (untested prediction) | 1039.609 +- 146.994 | 980.519 +- 46.414 | 0.432 |
| permeability_index (untested prediction) | 0.057 +- 0.008 | 0.054 +- 0.002 | 0.362 |

### Vector-state fractions of interface arclength (mean over fields)

| pattern | BRAFi | DMSO | p |
|---|---|---|---|
| none | 0.395 | 0.359 | 0.156 |
| actin only | 0.088 | 0.099 | 0.432 |
| TJ only | 0.085 | 0.071 | 0.432 |
| TJ+actin | 0.038 | 0.057 | 0.021 |
| AJ only | 0.092 | 0.095 | 0.953 |
| AJ+actin | 0.068 | 0.070 | 0.859 |
| AJ+TJ | 0.096 | 0.101 | 0.676 |
| AJ+TJ+actin | 0.139 | 0.147 | 0.676 |

## Joint junction states (GMM with BIC over the multichannel edge vector, all fields pooled)

k = 4, features = ['AJ_coverage', 'AJ_continuity', 'AJ_width_mean_px', 'AJ_mean_intensity', 'TJ_coverage', 'TJ_continuity', 'TJ_width_mean_px', 'TJ_mean_intensity', 'actin_coverage', 'actin_continuity', 'actin_width_mean_px', 'actin_mean_intensity', 'co_occupancy_AJ_TJ', 'co_occupancy_AJ_actin', 'co_occupancy_TJ_actin'], bootstrap stability ARI = 0.544 +- 0.114 over 20 resamples.

| condition | state_0 | state_1 | state_2 | state_3 |
|---|---|---|---|---|
| BRAFi | 0.192 | 0.299 | 0.400 | 0.109 |
| DMSO | 0.182 | 0.216 | 0.432 | 0.170 |

State means (original units):

- state 0: AJ_coverage=0.436, AJ_continuity=0.346, AJ_width_mean_px=3.162, AJ_mean_intensity=73.011, TJ_coverage=0.227, TJ_continuity=0.184, TJ_width_mean_px=2.174, TJ_mean_intensity=36.097, actin_coverage=0.572, actin_continuity=0.459, actin_width_mean_px=3.824, actin_mean_intensity=85.683, co_occupancy_AJ_TJ=0.170, co_occupancy_AJ_actin=0.316, co_occupancy_TJ_actin=0.177
- state 1: AJ_coverage=0.266, AJ_continuity=0.157, AJ_width_mean_px=2.584, AJ_mean_intensity=54.690, TJ_coverage=0.117, TJ_continuity=0.093, TJ_width_mean_px=1.891, TJ_mean_intensity=26.579, actin_coverage=0.213, actin_continuity=0.157, actin_width_mean_px=2.757, actin_mean_intensity=52.257, co_occupancy_AJ_TJ=0.076, co_occupancy_AJ_actin=0.124, co_occupancy_TJ_actin=0.064
- state 2: AJ_coverage=0.467, AJ_continuity=0.331, AJ_width_mean_px=3.217, AJ_mean_intensity=85.166, TJ_coverage=0.580, TJ_continuity=0.434, TJ_width_mean_px=3.529, TJ_mean_intensity=66.104, actin_coverage=0.246, actin_continuity=0.182, actin_width_mean_px=2.903, actin_mean_intensity=58.626, co_occupancy_AJ_TJ=0.368, co_occupancy_AJ_actin=0.173, co_occupancy_TJ_actin=0.205
- state 3: AJ_coverage=0.755, AJ_continuity=0.691, AJ_width_mean_px=4.439, AJ_mean_intensity=118.907, TJ_coverage=0.834, TJ_continuity=0.773, TJ_width_mean_px=4.809, TJ_mean_intensity=89.515, actin_coverage=0.776, actin_continuity=0.714, actin_width_mean_px=4.761, actin_mean_intensity=116.230, co_occupancy_AJ_TJ=0.684, co_occupancy_AJ_actin=0.612, co_occupancy_TJ_actin=0.685

## Per field

| field_id | condition | treatment | n_cells | n_gaps | n_edges | AJ_coverage | TJ_coverage | co_occupancy_AJ_TJ | jaccard_AJ_TJ | exclusive_AJ | exclusive_TJ | n_gaps | gap_fraction | G_eff | permeability_index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Figure_5A__Figure_5A_D10__Snap-212-Bild-Export-34 | BRAFi | D10 | 14 | 0 | 28 | 0.462 | 0.459 | 0.334 | 0.533 | 0.097 | 0.077 | 0 | 0.000 | 846.488 | 0.046 |
| Figure_5A__Figure_5A_DMSO__Snap-184-Bild-Export-06 | DMSO | DMSO | 18 | 0 | 40 | 0.410 | 0.415 | 0.267 | 0.420 | 0.091 | 0.103 | 0 | 0.000 | 1027.448 | 0.056 |
| Figure_5A__Figure_5A_V10__Snap-190-Bild-Export-12 | BRAFi | V10 | 22 | 0 | 48 | 0.443 | 0.433 | 0.334 | 0.542 | 0.077 | 0.071 | 0 | 0.000 | 1121.516 | 0.061 |
| Figure_5A__Figure_5A_V100__Snap-204-Bild-Export-26 | BRAFi | V100 | 16 | 0 | 31 | 0.377 | 0.418 | 0.253 | 0.425 | 0.075 | 0.121 | 0 | 0.000 | 1075.618 | 0.059 |
| Supplementary_Figures_S6-S7__S6_D1__Snap-206-Bild-Export-28 | BRAFi | D1 | 18 | 0 | 38 | 0.376 | 0.315 | 0.204 | 0.374 | 0.120 | 0.073 | 0 | 0.000 | 1215.278 | 0.067 |
| Supplementary_Figures_S6-S7__S6_D100__Snap-218-Bild-Export-40 | BRAFi | D100 | 10 | 0 | 18 | 0.346 | 0.379 | 0.231 | 0.420 | 0.060 | 0.111 | 0 | 0.000 | 776.256 | 0.043 |
| Supplementary_Figures_S6-S7__S6_DMSO__Snap-180-Bild-Export-02 | DMSO | DMSO | 13 | 0 | 23 | 0.423 | 0.371 | 0.250 | 0.340 | 0.106 | 0.054 | 0 | 0.000 | 934.638 | 0.051 |
| Supplementary_Figures_S6-S7__S6_V1__Snap-186-Bild-Export-08 | BRAFi | V1 | 18 | 0 | 40 | 0.387 | 0.348 | 0.222 | 0.376 | 0.091 | 0.084 | 0 | 0.000 | 1118.947 | 0.061 |
| Supplementary_Figures_S6-S7__S6_V10__Snap-191-Bild-Export-13 | BRAFi | V10 | 15 | 0 | 31 | 0.434 | 0.456 | 0.268 | 0.358 | 0.084 | 0.131 | 0 | 0.000 | 868.206 | 0.048 |
| Supplementary_Figures_S6-S7__S6_V100__Snap-195-Bild-Export-17 | BRAFi | V100 | 13 | 0 | 27 | 0.255 | 0.200 | 0.109 | 0.266 | 0.075 | 0.071 | 0 | 0.000 | 1187.883 | 0.065 |
| Supplementary_Figures_S6-S7__S7_DMSO__Snap-231-Bild-Export-43 | DMSO | DMSO | 14 | 0 | 25 | 0.410 | 0.342 | 0.229 | 0.367 | 0.089 | 0.055 | 0 | 0.000 | 979.471 | 0.054 |
| Supplementary_Figures_S6-S7__S7_E1__Snap-239-Bild-Export-51 | BRAFi | E1 | 20 | 0 | 43 | 0.440 | 0.381 | 0.262 | 0.391 | 0.094 | 0.089 | 0 | 0.000 | 1147.476 | 0.063 |
| Supplementary_Figures_S6-S7__S7_E10__Snap-245-Bild-Export-57 | BRAFi | E10 | 13 | 16 | 26 | 0.415 | 0.302 | 0.203 | 0.363 | 0.120 | 0.066 | 16 | 0.159 | 1005.883 | 0.056 |
| Supplementary_Figures_S6-S7__S7_E100__Snap-248-Bild-Export-60 | BRAFi | E100 | 20 | 0 | 45 | 0.356 | 0.386 | 0.215 | 0.354 | 0.075 | 0.113 | 0 | 0.000 | 1225.130 | 0.067 |
| Supplementary_Figures_S6-S7__S7_P1__Snap-252-Bild-Export-64 | BRAFi | P1 | 12 | 0 | 23 | 0.388 | 0.290 | 0.206 | 0.319 | 0.107 | 0.054 | 0 | 0.000 | 879.731 | 0.048 |
| Supplementary_Figures_S6-S7__S7_P10__Snap-257-Bild-Export-69 | BRAFi | P10 | 18 | 0 | 34 | 0.480 | 0.411 | 0.298 | 0.461 | 0.113 | 0.078 | 0 | 0.000 | 993.328 | 0.054 |
| Supplementary_Figures_S6-S7__S7_P100__Snap-268-Bild-Export-80 | BRAFi | P100 | 13 | 0 | 26 | 0.364 | 0.229 | 0.147 | 0.269 | 0.100 | 0.049 | 0 | 0.000 | 1092.786 | 0.060 |

## Limitations

- One field per treatment (3 DMSO, 15 BRAFi); the Mann-Whitney test compares 3 against 15 fields and a significant p here is at most suggestive.
- Images are 8-bit RGB pseudocolor exports with a burned-in scale bar, not raw acquisitions; intensities are display-scaled per image, so thresholds are per-field Otsu and intensities are not comparable across fields.
- Geometry comes from VE-cadherin alone; where VE-cadherin is lost after BRAFi the decoder can miss or merge cells, which biases TJ coverage on the surviving edges.
- Explicit gaps depend on the decoder's gap map; their count is a reconstruction output, not an annotation.
- The barrier proxy is a passive resistor network with hand-set weights; G_eff and the permeability index are untested predictions (validated: False).

## Measured-function follow-up (2026-09-19)

See `runs/function_real/REPORT.md`: against the ECIS resistance and tracer permeability of the
S-BIAD1169 paper (Bromberger et al. 2024 Fig 4) the predicted G_eff does not correlate
(Spearman -0.03 and 0.20, n = 17 fields); on 10 iPSC-RPE wells with measured TER the
permeability index has the right sign (-0.52) but is not significant (p 0.13). `validated`
stays False.
