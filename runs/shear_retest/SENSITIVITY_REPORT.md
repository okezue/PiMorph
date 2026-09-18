# Reconstruction-uncertainty intervals for EGM2 network statistics

Fields: 102 ({'6dyne': 30, 'high_shear': 38, 'static': 34}). Hypotheses per field from the neural:best.pt proposer +
constrained decoder perturbation grid and merge/split moves; every statistic is evaluated per
hypothesis and summarized by its posterior mean and 90% credible interval. Geometry channel is
the VE-cadherin channel (geometry_source = junction_channel), which is circular for junction
scoring and is recorded as such. Morphology labels are the heuristic feature-derived classes.

| Statistic | Condition | Posterior mean over fields (sd) | Median 90% CI half-width | Diff vs first | MWU p |
|---|---|---|---|---|---|
| reticular_fraction | static (n=34) | 0.112 (0.049) | 0.001 | +nan | nan |
| reticular_fraction | 6dyne (n=30) | 0.207 (0.049) | 0.001 | +0.095 | 2.63e-09 |
| reticular_fraction | high_shear (n=38) | 0.127 (0.083) | 0.001 | +0.014 | 0.407 |
| all_reticular_3clique_fraction | static (n=34) | 0.002 (0.004) | 0.000 | +nan | nan |
| all_reticular_3clique_fraction | 6dyne (n=30) | 0.011 (0.008) | 0.000 | +0.009 | 2.96e-07 |
| all_reticular_3clique_fraction | high_shear (n=38) | 0.007 (0.010) | 0.000 | +0.005 | 0.0175 |
| all_reticular_3clique_enrichment_z | static (n=34) | 0.003 (1.115) | 0.042 | +nan | nan |
| all_reticular_3clique_enrichment_z | 6dyne (n=30) | 0.025 (1.078) | 0.042 | +0.022 | 0.952 |
| all_reticular_3clique_enrichment_z | high_shear (n=38) | 0.381 (0.856) | 0.042 | +0.379 | 0.0105 |
| tricellular_realized_3clique_fraction | static (n=34) | 0.945 (0.029) | 0.000 | +nan | nan |
| tricellular_realized_3clique_fraction | 6dyne (n=30) | 0.966 (0.014) | 0.000 | +0.020 | 0.000617 |
| tricellular_realized_3clique_fraction | high_shear (n=38) | 0.938 (0.027) | 0.000 | -0.007 | 0.221 |
| area_degree_spearman | static (n=34) | 0.722 (0.102) | 0.006 | +nan | nan |
| area_degree_spearman | 6dyne (n=30) | 0.746 (0.065) | 0.006 | +0.024 | 0.431 |
| area_degree_spearman | high_shear (n=38) | 0.753 (0.100) | 0.006 | +0.031 | 0.101 |
| n_cells | static (n=34) | 172.483 (139.080) | 1.000 | +nan | nan |
| n_cells | 6dyne (n=30) | 141.310 (19.064) | 1.000 | -31.173 | 0.229 |
| n_cells | high_shear (n=38) | 154.009 (51.595) | 1.000 | -18.474 | 0.987 |
| mean_degree | static (n=34) | 5.119 (0.264) | 0.012 | +nan | nan |
| mean_degree | 6dyne (n=30) | 5.168 (0.121) | 0.012 | +0.049 | 0.835 |
| mean_degree | high_shear (n=38) | 5.009 (0.356) | 0.012 | -0.110 | 0.135 |

Reading: when the median credible half-width is comparable to or larger than the between-condition
difference, reconstruction ambiguity alone can account for the effect on these fields. The
enrichment z tests whether reticular contacts concentrate on 3-cliques beyond what the reticular
fraction predicts (z near 0: no concentration). The realized fraction states how many graph
3-cliques are actual tricellular vertices of the complex.