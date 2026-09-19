# Force inference validation

Forward model: vertex model on Voronoi sheets (60 cells, 256 x 256), per-cell reference area and perimeter, K_A=1.0, K_P=0.1, log-normal edge tensions, 3.0 px short-range core, relaxed at fixed topology to max residual force < 1e-6 (tension units). Edges are straight chords; ``edge_smooth`` carries the circular arcs implied by Laplace's law (kappa = delta p / effective tension).

Truth: the effective tension of an edge (bare gamma plus perimeter-elastic and core terms) is what the geometry encodes; bare gamma correlations are reported for reference. All inferred values are RELATIVE (mean tension = 1, pressure gauge as reported); the global scale is not identifiable.

Seeds: 9, relaxations converged: 18 / 18, all complexes valid: True. Skipped: seed 5 dispersion 0.3 (half-edge loop mixes faces; rotation system inconsistent); seed 5 dispersion 0.6 (half-edge loop mixes faces; rotation system inconsistent)

## Synthetic recovery (mean over seeds, minimum in parentheses)

| dispersion | noise px | pressures | curvature | ridge | Pearson eff | Spearman eff | Spearman bare | Pearson pressure |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.0 | False | False | 0.001 | 0.748 (0.693) | 0.743 (0.687) | 0.432 (0.336) | n/a |
| 0.3 | 0.0 | False | False | 0.1 | 0.778 (0.737) | 0.780 (0.713) | 0.502 (0.373) | n/a |
| 0.3 | 0.0 | True | False | 0.001 | 0.936 (0.901) | 0.932 (0.895) | 0.707 (0.595) | 0.859 (0.726) |
| 0.3 | 0.0 | True | False | 0.1 | 0.877 (0.836) | 0.874 (0.792) | 0.651 (0.475) | 0.780 (0.632) |
| 0.3 | 0.0 | True | True | 0.001 | 1.000 (1.000) | 1.000 (1.000) | 0.792 (0.744) | 1.000 (1.000) |
| 0.3 | 0.0 | True | True | 0.1 | 0.967 (0.940) | 0.970 (0.955) | 0.786 (0.731) | 0.979 (0.940) |
| 0.3 | 0.5 | False | False | 0.001 | 0.684 (0.631) | 0.669 (0.601) | 0.373 (0.259) | n/a |
| 0.3 | 0.5 | False | False | 0.1 | 0.734 (0.693) | 0.729 (0.657) | 0.446 (0.306) | n/a |
| 0.3 | 0.5 | True | False | 0.001 | 0.520 (0.373) | 0.485 (0.340) | 0.286 (0.116) | 0.150 (-0.241) |
| 0.3 | 0.5 | True | False | 0.1 | 0.763 (0.702) | 0.757 (0.665) | 0.516 (0.339) | 0.545 (0.419) |
| 0.3 | 0.5 | True | True | 0.001 | 0.775 (0.593) | 0.751 (0.592) | 0.535 (0.478) | 0.842 (0.749) |
| 0.3 | 0.5 | True | True | 0.1 | 0.883 (0.846) | 0.881 (0.848) | 0.672 (0.620) | 0.936 (0.891) |
| 0.3 | 1.0 | False | False | 0.001 | 0.538 (0.299) | 0.533 (0.327) | 0.303 (0.082) | n/a |
| 0.3 | 1.0 | False | False | 0.1 | 0.599 (0.384) | 0.594 (0.364) | 0.352 (0.109) | n/a |
| 0.3 | 1.0 | True | False | 0.001 | 0.316 (0.211) | 0.312 (0.194) | 0.153 (0.036) | 0.145 (-0.156) |
| 0.3 | 1.0 | True | False | 0.1 | 0.572 (0.367) | 0.575 (0.362) | 0.362 (0.130) | 0.322 (0.135) |
| 0.3 | 1.0 | True | True | 0.001 | 0.533 (0.323) | 0.538 (0.367) | 0.367 (0.208) | 0.643 (0.574) |
| 0.3 | 1.0 | True | True | 0.1 | 0.658 (0.473) | 0.666 (0.496) | 0.485 (0.303) | 0.831 (0.741) |
| 0.6 | 0.0 | False | False | 0.001 | 0.730 (0.623) | 0.721 (0.637) | 0.493 (0.413) | n/a |
| 0.6 | 0.0 | False | False | 0.1 | 0.754 (0.699) | 0.769 (0.715) | 0.573 (0.482) | n/a |
| 0.6 | 0.0 | True | False | 0.001 | 0.904 (0.848) | 0.916 (0.881) | 0.748 (0.647) | 0.779 (0.687) |
| 0.6 | 0.0 | True | False | 0.1 | 0.837 (0.748) | 0.858 (0.805) | 0.714 (0.565) | 0.689 (0.549) |
| 0.6 | 0.0 | True | True | 0.001 | 0.999 (0.998) | 0.999 (0.998) | 0.817 (0.774) | 1.000 (0.999) |
| 0.6 | 0.0 | True | True | 0.1 | 0.928 (0.844) | 0.948 (0.901) | 0.810 (0.755) | 0.949 (0.875) |
| 0.6 | 0.5 | False | False | 0.001 | 0.627 (0.569) | 0.612 (0.557) | 0.404 (0.296) | n/a |
| 0.6 | 0.5 | False | False | 0.1 | 0.674 (0.623) | 0.676 (0.611) | 0.474 (0.346) | n/a |
| 0.6 | 0.5 | True | False | 0.001 | 0.492 (0.393) | 0.475 (0.390) | 0.337 (0.275) | 0.220 (0.059) |
| 0.6 | 0.5 | True | False | 0.1 | 0.683 (0.601) | 0.701 (0.617) | 0.536 (0.412) | 0.441 (0.308) |
| 0.6 | 0.5 | True | True | 0.001 | 0.683 (0.425) | 0.687 (0.556) | 0.512 (0.451) | 0.679 (0.473) |
| 0.6 | 0.5 | True | True | 0.1 | 0.803 (0.766) | 0.816 (0.773) | 0.652 (0.595) | 0.859 (0.690) |
| 0.6 | 1.0 | False | False | 0.001 | 0.397 (-0.001) | 0.385 (0.118) | 0.232 (-0.028) | n/a |
| 0.6 | 1.0 | False | False | 0.1 | 0.473 (0.182) | 0.463 (0.182) | 0.303 (0.044) | n/a |
| 0.6 | 1.0 | True | False | 0.001 | 0.294 (0.181) | 0.285 (0.194) | 0.183 (0.085) | 0.146 (-0.037) |
| 0.6 | 1.0 | True | False | 0.1 | 0.455 (0.220) | 0.462 (0.236) | 0.322 (0.086) | 0.224 (0.055) |
| 0.6 | 1.0 | True | True | 0.001 | 0.425 (0.140) | 0.413 (0.214) | 0.298 (0.096) | 0.467 (0.277) |
| 0.6 | 1.0 | True | True | 0.1 | 0.545 (0.353) | 0.535 (0.327) | 0.411 (0.206) | 0.703 (0.516) |

Noise is added to the free vertices only; arc traces follow their endpoints. With curvature the tension-plus-pressure system is overdetermined and recovery on noiseless data is exact up to scale. Without curvature the pressures are constrained only through the chord pressure terms and the system is underdetermined (ridge prior fills the null space); without pressures the tension-only balance absorbs pressure forces.

## Real field

`runs/pimorph_ve_strat/Control_s1/complex.json`: 541 cells, 1622 edges, 1082 vertices. Inferred 1528 relative tensions and 541 relative pressures (pressure gauge: mean_zero). Tension CV 0.445, residual RMS 0.377 (tension units), fraction of interior vertices balanced within 5% 0.02 (within 20%: 0.20), condition number 3.16e+03, 3413 equations for 2160 unknowns (rank 2065), ridge 0.1.

These are RELATIVE tensions with no validation: no force measurement (laser ablation, micropipette, FRET sensor) exists for this field, and absolute tension, absolute pressure and the elastic moduli are not identifiable from geometry. The residual force imbalance is large compared with the synthetic equilibria (RMS below 0.05 there), so the vertex-balance model describes this reconstruction poorly: reticular VE-cadherin junctions and crack-corner vertex positions do not deliver tension-balanced angles. Treat the table as a diagnostic, not a measurement.

## Not validated

- absolute tension and pressure scale (gauge);
- pressures without edge curvature (underdetermined without the Laplace law);
- any mechanics on real data (only self-consistency: residual force balance);
- topology changes (T1/T2); the forward model runs at fixed topology.

## Real-data follow-up (2026-09-19)

See `runs/mechanics_real/REPORT.md`: laser-ablation recoil (Lang et al. 2019, 15 cuts), DLITE
ZO-1 colonies and TissueMiner consistency. The cut boundary cable is inferred above the field
mean in 14 / 15 fields and the cut-edge tension ranks recoil velocity with Spearman 0.64
(tension-only, ridge 0.1; 0.2 with pressures and curvature).
