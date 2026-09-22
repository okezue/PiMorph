## Data

66 well-timepoints, 29 wells, weeks 3 to 8, TER 103 to 1115 Ohm, 129 tiles.

## Segmenter on held-out AMD wells

wells AMD1_A_D75, AMD2_B_D73: 175 tiles, 24769 true cells, 23295 predicted

| ap50 | ap75 | pq | adjacency_pair_f1 | vertex_f1 | vertex_precision | vertex_recall | boundary_f1 |
|---|---|---|---|---|---|---|---|
| 0.741 | 0.535 | 0.692 | 0.703 | 0.657 | 0.747 | 0.612 | 0.939 |
| AMD1_A_D75: 0.706 | 0.472 | 0.663 | 0.659 | 0.554 | 0.708 | 0.468 | 0.921 |
| AMD2_B_D73: 0.801 | 0.643 | 0.743 | 0.778 | 0.836 | 0.815 | 0.862 | 0.970 |

## Correlation with TER across well-timepoints

| statistic | kind | n | Spearman | p | 95% CI | Pearson (log TER) | partial rho, condition | p | partial rho, condition + week | p | partial rho given density + area CV | p | LOWO Spearman range | LOWO pred. R^2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n_cells | other | 66 | 0.524 | 6.2e-06 | [0.29, 0.70] | 0.538 | 0.342 | 0.0057 | -0.013 | 0.92 | -0.033 | 0.79 | [0.48, 0.59] | 0.163 |
| cell_density_per_kpx | control | 66 | 0.524 | 6.2e-06 | [0.29, 0.70] | 0.538 | 0.342 | 0.0057 | -0.013 | 0.92 | nan | nan | [0.48, 0.59] | 0.163 |
| cell_area_px | other | 66 | -0.218 | 0.078 | [-0.47, 0.06] | -0.502 | -0.241 | 0.055 | -0.023 | 0.86 | 0.069 | 0.59 | [-0.27, -0.14] | 0.079 |
| cell_area_cv | control | 66 | -0.193 | 0.12 | [-0.42, 0.08] | -0.327 | -0.224 | 0.075 | -0.024 | 0.86 | nan | nan | [-0.26, -0.12] | -0.009 |
| gap_area_frac | control | 66 | -0.532 | 4.4e-06 | [-0.70, -0.32] | -0.546 | -0.323 | 0.0091 | 0.018 | 0.89 | -0.077 | 0.55 | [-0.59, -0.49] | 0.175 |
| n_gaps | other | 66 | 0.475 | 5.6e-05 | [0.24, 0.66] | 0.405 | 0.279 | 0.026 | -0.036 | 0.79 | -0.224 | 0.075 | [0.43, 0.55] | 0.041 |
| n_edges | other | 66 | 0.547 | 2e-06 | [0.32, 0.72] | 0.504 | 0.341 | 0.0059 | -0.018 | 0.89 | 0.171 | 0.18 | [0.50, 0.61] | 0.138 |
| edges_per_cell | control | 66 | 0.570 | 5.9e-07 | [0.36, 0.73] | 0.617 | 0.353 | 0.0042 | -0.004 | 0.98 | 0.309 | 0.013 | [0.53, 0.62] | 0.238 |
| edge_length_px | control | 66 | -0.393 | 0.0011 | [-0.60, -0.14] | -0.529 | -0.369 | 0.0027 | -0.008 | 0.95 | 0.031 | 0.81 | [-0.44, -0.33] | 0.106 |
| G_eff | proxy | 66 | 0.531 | 4.5e-06 | [0.31, 0.70] | 0.489 | 0.335 | 0.0068 | -0.014 | 0.92 | 0.048 | 0.71 | [0.49, 0.59] | 0.123 |
| permeability_index | proxy | 66 | 0.524 | 6.4e-06 | [0.30, 0.70] | 0.587 | 0.375 | 0.0023 | 0.008 | 0.95 | -0.017 | 0.89 | [0.48, 0.59] | 0.198 |
| gap_fraction | proxy | 66 | 0.349 | 0.0041 | [0.09, 0.57] | 0.493 | 0.112 | 0.38 | -0.083 | 0.53 | -0.254 | 0.043 | [0.29, 0.45] | 0.078 |
| G_eff_len | other | 66 | 0.530 | 4.7e-06 | [0.31, 0.70] | 0.490 | 0.333 | 0.0071 | -0.017 | 0.9 | 0.042 | 0.74 | [0.49, 0.59] | 0.124 |
| permeability_index_len | proxy | 66 | 0.529 | 4.9e-06 | [0.31, 0.70] | 0.587 | 0.372 | 0.0024 | 0.015 | 0.91 | 0.020 | 0.88 | [0.48, 0.59] | 0.199 |
| in_plane_G | proxy | 66 | 0.464 | 8.6e-05 | [0.21, 0.65] | 0.364 | 0.252 | 0.045 | -0.026 | 0.84 | -0.035 | 0.78 | [0.41, 0.53] | 0.062 |
| in_plane_G_len | proxy | 66 | 0.459 | 0.00011 | [0.21, 0.65] | 0.361 | 0.243 | 0.053 | -0.033 | 0.81 | -0.047 | 0.71 | [0.41, 0.53] | 0.061 |
| tissue_frac | other | 66 | 0.532 | 4.4e-06 | [0.32, 0.70] | 0.546 | 0.323 | 0.0091 | -0.018 | 0.89 | 0.077 | 0.55 | [0.49, 0.59] | 0.175 |

## Per-condition Spearman

| statistic | Control | Aphidicolin | HPI4 |
|---|---|---|---|
| n_cells | 0.08 (p 0.68) | 0.63 (p 0.00019) | 0.14 (p 0.76) |
| cell_density_per_kpx | 0.08 (p 0.68) | 0.63 (p 0.00019) | 0.14 (p 0.76) |
| cell_area_px | -0.08 (p 0.68) | -0.32 (p 0.09) | 0.29 (p 0.53) |
| cell_area_cv | 0.17 (p 0.37) | -0.46 (p 0.011) | -0.54 (p 0.22) |
| gap_area_frac | -0.07 (p 0.73) | -0.58 (p 0.00071) | -0.36 (p 0.43) |
| n_gaps | 0.09 (p 0.66) | 0.53 (p 0.0025) | nan (p nan) |
| n_edges | 0.06 (p 0.74) | 0.62 (p 0.00026) | 0.39 (p 0.38) |
| edges_per_cell | 0.08 (p 0.69) | 0.62 (p 0.00026) | 0.36 (p 0.43) |
| edge_length_px | -0.14 (p 0.46) | -0.57 (p 0.0011) | -0.25 (p 0.59) |
| G_eff | 0.07 (p 0.7) | 0.61 (p 0.00034) | 0.39 (p 0.38) |
| permeability_index | 0.10 (p 0.62) | 0.65 (p 8.8e-05) | 0.54 (p 0.22) |
| gap_fraction | -0.01 (p 0.94) | 0.34 (p 0.062) | nan (p nan) |
| G_eff_len | 0.07 (p 0.7) | 0.61 (p 0.00031) | 0.39 (p 0.38) |
| permeability_index_len | 0.10 (p 0.6) | 0.66 (p 8.5e-05) | 0.43 (p 0.34) |
| in_plane_G | 0.01 (p 0.96) | 0.52 (p 0.0033) | 0.80 (p 0.03) |
| in_plane_G_len | 0.01 (p 0.98) | 0.50 (p 0.0046) | 0.80 (p 0.03) |
| tissue_frac | 0.07 (p 0.73) | 0.58 (p 0.00071) | 0.36 (p 0.43) |

## Leave-one-well-out linear prediction of TER

| model | n | LOWO R^2 | LOWO R^2 with condition dummies | Spearman(pred, TER) |
|---|---|---|---|---|
| density | 66 | 0.163 | 0.673 | 0.466 |
| density+areaCV | 66 | 0.152 | 0.667 | 0.453 |
| density+areaCV+gapfrac | 66 | 0.095 | 0.666 | 0.399 |
| permeability_index | 66 | 0.198 | 0.680 | 0.465 |
| in_plane_G | 66 | 0.062 | 0.678 | 0.213 |
| density+areaCV+permeability | 66 | 0.142 | 0.664 | 0.417 |
| density+areaCV+in_plane_G | 66 | 0.139 | 0.667 | 0.441 |
| density+areaCV+permeability+in_plane_G | 66 | 0.129 | 0.662 | 0.387 |

## Posterior credible intervals

20 tiles, 56 hypotheses per tile, ESS 3.9.
Median relative 90% credible width: cell count 0.009, permeability index 0.013, in-plane G 0.005, area CV 0.057.
Spearman of the posterior-mean permeability index with TER over these tiles: 0.44; of the MAP value: 0.44 (tiles with a complex: 13).
