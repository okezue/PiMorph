# Reconstruction-uncertainty intervals for EGM2 network statistics

Fields: 4 (static, 6dyne), hypotheses per field from the classical
proposer + constrained decoder perturbation grid and merge/split moves. Geometry channel is
the VE-cadherin channel (geometry_source = junction_channel), which is circular for junction
scoring and is recorded as such.

| Statistic | Condition | Posterior mean (fields) | Median 90% CI half-width | Between-condition diff | MWU p (means) |
|---|---|---|---|---|---|
| reticular_fraction | static | 0.334 | 0.008 | -0.027 | 0.667 |
| reticular_fraction | 6dyne | 0.307 | 0.008 | -0.027 | 0.667 |
| all_reticular_3clique_fraction | static | 0.024 | 0.000 | +0.013 | 1 |
| all_reticular_3clique_fraction | 6dyne | 0.036 | 0.000 | +0.013 | 1 |
| area_degree_spearman | static | 0.510 | 0.010 | +0.185 | 0.333 |
| area_degree_spearman | 6dyne | 0.694 | 0.010 | +0.185 | 0.333 |

Reading: when the median credible half-width is comparable to or larger than the between-condition
difference, the reconstruction ambiguity alone can account for the effect on these fields, and the
finding needs the full 30+30 field set with the posterior propagated before it is claimed.