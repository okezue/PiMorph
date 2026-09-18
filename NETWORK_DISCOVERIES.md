# Network-Level Findings in Endothelial Biology

## Re-test with PiMorph posteriors (2026-09-18)

The three findings below were re-tested on every EGM2 field (34 static, 30 at 6 dyn cm⁻², 38 at
18-20 dyn cm⁻²) with the PiMorph reconstruction instead of the legacy watershed: neural proposals
(`pimorph_proposals_v3_endo`, chosen by label-free self-consistency in
`runs/shear_retest/checkpoint_selection.csv`), the constrained decoder with open background
excluded from cells, and a posterior of 32 legal complexes per field. Every statistic was
evaluated per hypothesis and the condition test was run on posterior means
(`scripts/pimorph_sensitivity_egm2.py --proposer neural --per-condition 0 --n-perm 1000`;
outputs in `runs/shear_retest/`). Reconstruction ambiguity is negligible for all three
statistics (median 90% credible half-width 0.001 for the reticular fraction, 0.006 for the
area-degree correlation), so the verdicts below are about the segmentation model, not about
sampling noise within one model.

| Finding | Legacy (30 + 30) | PiMorph posterior (34 static, 30 at 6 dyn) | Verdict |
|---|---|---|---|
| Reticular fraction higher at 6 dyn cm⁻² | 51.0% vs 62.1%, +9.5 pp, p = 3e-6 | 0.112 vs 0.207, +0.095, MWU p = 2.6e-9; balanced 30 + 30 subset p = 1.6e-8; 6 dyn above static in all three biological replicates (0.261/0.136, 0.170/0.109, 0.190/0.135); 18-20 dyn 0.127 (p = 0.41 vs static) | **survives**, including the distinct high-shear regime. The absolute level is much lower under PiMorph's edge labeler (1-D strip profiles, Otsu threshold pooled over edges) than under the legacy band features, so the fraction itself is labeler-dependent; the condition difference is not. |
| All-reticular 3-cliques enriched at 6 dyn cm⁻² | 15.9% vs 25.4% raw, p = 1.5e-4; conditional null pending | raw 0.002 vs 0.011 (p = 3e-7) but enrichment z against the conditional null that fixes each field's reticular count: static 0.003, 6 dyn 0.025, MWU p = 0.95 | **withdrawn as an independent finding**: reticular edges do not concentrate on 3-cliques beyond what the higher reticular fraction predicts. The raw increase is a corollary of the finding above. 94.5% (static) and 96.6% (6 dyn) of graph 3-cliques are realized by a multicellular vertex of the complex, so "3-clique" and "tricellular junction" can be used interchangeably on these fields (p = 6e-4 for the small difference). Exploratory: 18-20 dyn shows z = 0.38 (p = 0.01 vs static). |
| Area-degree correlation stronger at 6 dyn cm⁻² | median r 0.455 vs 0.642, +0.187, p = 2.6e-7 | posterior-mean r 0.722 vs 0.746, +0.024, MWU p = 0.43; balanced 30 + 30 medians 0.746 vs 0.761, p = 0.96; no replicate shows a difference | **does not replicate**. With background excluded from cells the within-field correlation is 0.70 to 0.80 in every condition. The legacy watershed flooded background into cells (see the red outlines across empty regions in `runs/egm2_full/*/qc_cells.png`), which is most severe in sparse static fields and is the likely source of the weaker legacy static correlation. |

Two of the earlier caveats are now measured rather than assumed: reconstruction uncertainty
(negligible here) and the 3-clique / tricellular-vertex correspondence (about 95%). The remaining
caveats stand: morphology labels are heuristic, geometry is segmented from the VE-cadherin channel,
and no functional readout was tested. The four `rep1` fields from the 060721 batch (static) and
the eight high-shear `rep1` fields have reticular fractions near 0.002 and are excluded from the
balanced subset, as in the legacy analysis.

## Summary (legacy, 2026-09-17 and earlier)

Using graph/network analysis on the EndoPiGraph-AJmorph pipeline results, we identified **three robust findings** about how 6 dyn cm⁻² shear stress reorganizes endothelial cell contact networks relative to static. Two originally claimed effects (raw clustering coefficient, degree-occupancy correlation) were **withdrawn** after hardened per-image statistical testing. See the re-test above for their current status: one survives, one is reduced to a corollary, one does not replicate.

**Statistical validation:** All statistics use **per-image replicate testing** (n = number of images, not cells/edges) to avoid pseudo-replication. Effect sizes reported as rank-biserial correlation r.

**Regime caveat:** These findings compare static vs 6 dyn cm⁻². The 18-20 dyn cm⁻² regime is *not* a monotonic continuation of the 6 dyn cm⁻² phenotype; see "High-shear regime" below.

---

## Measured vs hypothesized

This document separates what the pipeline measured from how those measurements might be interpreted. Only the first list is supported by the data in this repository.

**Measured (per-image quantities and tests on them):**
- Per-image fraction of edges labelled reticular by the heuristic AJ classifier
- Per-image fraction of contact-graph 3-cliques whose three edges are all labelled reticular
- Per-image Spearman correlation between cell area and degree (number of neighbours)
- Per-image clustering coefficient and mean degree (used for the withdrawn Discovery 1)
- Per-image Spearman correlation between degree and AJ occupancy (used for the withdrawn degree-occupancy claim)
- Mann-Whitney U and Wilcoxon signed-rank tests, bootstrap CIs and rank-biserial effect sizes across images

**Hypothesized (interpretations, not tested here):**
- That reticular junctions represent "junction maturation"
- That reticular junctions indicate stronger cell-cell adhesion or barrier function
- That 3-cliques are tricellular junctions, or that tricellular junctions are permeability "hotspots" in these samples
- That a tighter area-degree correlation means the tissue is "more geometrically ordered"

Two further measurement caveats apply to every finding below. First, the AJ morphology labels are heuristic (see `CLOSEOUT_CHECKLIST.md`, Known Limitations). Second, cell geometry was segmented from the VE-cadherin channel, which is also the junction channel scored for morphology; the pipeline now records this as `geometry_source: junction_channel` in run provenance.

---

## ~~Discovery 1: Clustering Coefficient Increases Under Flow~~ (CONFOUNDED)

**Original finding:** Clustering coefficient appears to increase under flow.

| Condition | Median Clustering | Mean | Std | n |
|-----------|------------------|------|-----|---|
| Static    | 0.376            | 0.381| 0.073| 30 |
| 6 dyne    | 0.469            | 0.442| 0.055| 30 |
| high_shear| 0.331            | 0.309| 0.122| 30 |

**CRITICAL: Confounded by Graph Density**

Regression analysis controlling for mean degree:
```
clustering ~ condition + mean_degree + n_cells
R² = 0.937 (mean degree explains 94% of variance!)

is_6dyne coef: -0.039, p = 1.34e-04 (NEGATIVE after control)
is_high_shear coef: -0.037, p = 3.53e-05 (NEGATIVE after control)
mean_degree coef: 0.113, p = 7.69e-46
```

Normalized clustering (C/C_random): static vs 6dyne **p = 0.68** (NOT significant)

**Conclusion:** The raw clustering increase is driven by changes in graph density (more edges = higher clustering mechanically). After controlling for density, the effect disappears. This finding is **withdrawn**.

---

## Discovery 2: Reticular Junctions Increase Under Flow

**Finding:** The proportion of edges labelled reticular by the heuristic AJ classifier increases under flow.

| Condition | Median % Reticular | Mean | Std | n |
|-----------|-------------------|------|-----|---|
| Static    | 51.5%             | 51.0%| 5.2%| 30 |
| 6 dyne    | 61.1%             | 62.1%| 8.6%| 30 |

**Per-image replicate statistics:**
- Median difference: +9.5% [95% CI: 6.7%, 16.6%]
- Mann-Whitney U = 133.5, **p = 2.98e-06**
- Effect size r = 0.703 (large)

**Hypothesis (not tested here):** Flow promotes junction maturation, and reticular junctions indicate stronger cell-cell adhesion and barrier function. Neither claim is tested by this dataset; both require functional validation (for example permeability or adhesion assays).

---

## ~~Discovery 3: High-Degree Cells Have Stronger Junctions~~ (NOT CONFIRMED)

**Original claim:** Cells with more neighbors have higher AJ occupancy.

**Per-image replicate testing:**
- Median within-image Spearman r = 0.001
- Wilcoxon test: **p = 0.808** (not significant)

**Conclusion:** The original pooled analysis suffered from pseudo-replication. When properly tested at the image level, the degree-occupancy correlation does NOT hold. This finding is **withdrawn**.

---

## Discovery 3 (renumbered): All-Reticular 3-Cliques Increase Under Flow

**What was measured:** a 3-clique in the cell contact graph, i.e. three cells that are pairwise adjacent. The quantity below is the per-image fraction of 3-cliques whose three edges are all labelled reticular. A graph 3-clique is *not* automatically a tricellular junction: it corresponds to one only when the three cells meet at a common physical vertex. That correspondence is now measurable with the `pimorph` half-edge complex (`scripts/harden_network_stats.py --complex-dir`, which reports the fraction of 3-cliques realized by a common vertex per image); results are pending.

**Finding (raw):** The per-image fraction of all-reticular 3-cliques is higher at 6 dyn cm⁻² than static.

| Condition | Median % All-Reticular 3-Cliques | Mean | Std | n |
|-----------|----------------------------------|------|-----|---|
| Static    | 15.9%                            | 15.9%| 4.4%| 30 |
| 6 dyne    | 25.4%                            | 25.3%| 10.2%| 30 |

**Per-image replicate statistics (raw fraction):**
- Median difference: +9.6% [95% CI: 4.2%, 13.8%]
- Mann-Whitney U = 193.5, **p = 1.54e-04**
- Effect size r = 0.570 (large)

**Confound to be checked:** the raw fraction rises trivially when more edges are reticular (Discovery 2), because a 3-clique with three randomly placed reticular edges is more likely when the reticular fraction is higher. The claim that reticular edges *concentrate* on 3-cliques therefore has to be tested against a conditional null that preserves each image's reticular-edge count: labels are permuted across edges within the image (1000 permutations, seed 0), the all-reticular 3-clique fraction is recomputed, and each image gets `enrichment_z = (observed - null mean) / null sd` plus a two-sided permutation p-value. The condition comparison is then a Mann-Whitney U test on per-image `enrichment_z`. This check is implemented in `scripts/harden_network_stats.py` (`tests_enrichment_z` in the JSON output); its result will be reported here when the script is re-run on `runs/egm2_full`. Until then, the raw increase above should be read as consistent with, but not independent of, the reticular-fraction increase in Discovery 2.

**Hypothesis (not tested here):** Tricellular junctions (where 3 cells meet) are reported in the literature as permeability hotspots, and flow might drive junction maturation specifically at multi-cell vertices. This dataset does not test permeability, does not yet establish that the 3-cliques are tricellular junctions, and does not measure maturation.

---

## Discovery 4 (renumbered): Area-Degree Correlation Strengthens Under Flow

**Finding:** Cell area positively correlates with degree (number of neighbors), and this correlation strengthens under flow.

| Condition | Median r | Mean r | Range | n |
|-----------|----------|--------|-------|---|
| Static    | 0.455    | 0.445  | [0.17, 0.65] | 30 |
| 6 dyne    | 0.642    | 0.624  | [0.31, 0.76] | 30 |

**Per-image replicate statistics:**
- Both conditions: correlations differ from 0, **p = 1.86e-09**
- Condition comparison: Median diff = +0.187 [95% CI: 0.144, 0.246]
- Mann-Whitney U = 101.0, **p = 2.57e-07**
- Effect size r = 0.78 (large)

**Measured:** Larger cells have more neighbors, and the within-image correlation is stronger under flow.

**Hypothesis (not tested here):** The tighter area-degree relationship reflects a more geometrically ordered tissue. Geometric order was not measured directly; note also that cell areas come from VE-cadherin-based segmentation (see "Measured vs hypothesized").

---

## Overall Conclusion

**At 6 dyn cm⁻² relative to static, the contact network has a higher fraction of edges labelled reticular, a higher raw fraction of all-reticular 3-cliques, and a stronger area-degree correlation.**

Three findings survive hardened per-image statistical testing:
1. Reticular junction percentage increases
2. Raw all-reticular 3-clique fraction increases (enrichment beyond the reticular-fraction increase pending the conditional null; correspondence of 3-cliques to tricellular vertices pending the complex check)
3. Area-degree correlation strengthens

Two original claims were **withdrawn** after proper statistical validation:
- Raw clustering coefficient: 94% of variance is explained by mean degree; coefficient flips sign after density control.
- Degree-occupancy correlation: pooled analysis suffered from pseudo-replication; per-image Wilcoxon p = 0.81.

## High-shear regime is distinct, not a continuation

Per-image medians on the 18-20 dyn cm⁻² subset (n = 30) are closer to static than to 6 dyn cm⁻²:

| Metric | Static | 6 dyn cm⁻² | 18-20 dyn cm⁻² |
|---|---:|---:|---:|
| Reticular fraction (median) | 50.7% | 61.1% | 52.7% |
| All-reticular 3-cliques (median) | 15.9% | 25.4% | 15.5% |

Subject to batch and density caveats, this is consistent with **intermediate shear producing the highest reticular fraction and all-reticular 3-clique fraction** while higher shear shifts toward a distinct regime. This is reported as exploratory in the manuscript pending balanced-batch validation.

---

## Statistical Methods

- **Sampling unit:** Image (not individual cells/edges)
- **Between-condition test:** Mann-Whitney U (non-parametric)
- **Within-condition test:** Wilcoxon signed-rank (for correlations differing from 0)
- **Conditional null for all-reticular 3-cliques:** per-image label permutation across edges with the reticular-edge count fixed (1000 permutations, seed 0); reports `null_mean_pct`, `null_sd_pct`, `enrichment_z`, `perm_p`; conditions compared by Mann-Whitney U on `enrichment_z`
- **3-clique vs tricellular vertex:** optional `--complex-dir` check against the `pimorph` half-edge complex (fraction of 3-cliques with a common multicellular vertex)
- **Effect size:** Rank-biserial correlation r
  - |r| < 0.1: negligible
  - |r| 0.1-0.3: small
  - |r| 0.3-0.5: medium
  - |r| > 0.5: large
- **Confidence intervals:** Bootstrap (1000 resamples)
- **Dataset:** 95 EGM2-treated HUVEC images (32 static, 30 at 6 dyn cm⁻², 33 at 18-20 dyn cm⁻²; replicate testing on the balanced 30+30 static-vs-6 dyn cm⁻² subset)

---

## Key Files

- Hardened statistics script: `scripts/harden_network_stats.py`
- Per-image results: `runs/egm2_full/hardened_network_stats.json`
- Cell/edge data: `runs/egm2_full/*/cells.csv`, `edges.csv`
