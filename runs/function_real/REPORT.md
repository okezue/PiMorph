# Barrier proxy against measured barrier function (2026-09-19)

Question: does the resistor-network barrier proxy of `pimorph.function.transport` (G_eff,
permeability index from junction coverage along every reconstructed interface) track a
measured barrier readout on real monolayers? Until now it had only been run on synthetic
sheets and on S-BIAD1169 without any functional measurement (`runs/multijunction/`).

Verdict: **not validated**. On the one dataset that pairs junction images with a per-well TER
(10 iPSC-RPE wells) the prediction points the right way (Spearman -0.52 between predicted
permeability index and measured TER, curated complexes) but is not significant (p = 0.13,
bootstrap 95% CI -0.92 to 0.22), and it vanishes with PiMorph's own reconstruction
(-0.09). On S-BIAD1169 the predicted G_eff does not follow the published ECIS resistance
drop or tracer permeability (Spearman -0.03 and 0.20, n = 17 fields). `barrier_report`
therefore keeps `validated: False`; nothing in `function/barrier.py` was changed.

## Data obtained

### 1. NIST / NEI iPSC-RPE with TER (Schaub, Hotaling, Bharti et al. 2020 JCI)

Source: https://isg.nist.gov/deepzoomweb/data/RPEimplants (DOI 10.18434/T4/1503229), the
data record of "Deep learning predicts function of live retinal pigment epithelium from
quantitative microscopy". Downloaded to `data/rpe_nist/` (not in git; provenance in
`data/rpe_nist/README_provenance.md`, loader `src/pimorph/io/rpe_nist.py`):

- `AMD_Data/AnalysisOfSegmentedData/AMD_TER-Data_Mean-SD.csv`: TER (Ohm cm^2; mean and
  SD of three readings) for 20 AMD-iRPE wells (3 donors, 2 to 3 clones, culture days 73 to
  77). This is the measured barrier function.
- 1032 registered 256 x 256 tiles from 10 of those wells: ZO-1 immunofluorescence
  (`fluorescentZ01.zip`), hand-corrected border masks (`segmentation_mask.zip`) and QBAM
  absorbance. The masks are the paper's ground truth for DNN-Z, i.e. curated cell outlines.
- whole-well stitched ZO-1 images of the same 10 wells (`NN_Data/Unregistered_Images/ZO1`);
  the tiles are crops of these at the positions in their names (offset -3, -1 px, pixel
  correlation 1.000 on the checked tile).

Well identity: the image names use internal clone letters (`AMD1B`, `AMD1C`, `AMD2C`,
`AMD3B`, `AMD3C`), the TER table the paper's (AMD1 A, B; AMD2 and AMD3 A, B, C). The
mapping AMD1B -> 1A, AMD1C -> 1B, AMD2C -> 2B, AMD3B -> 3B, AMD3C -> 3C is the only one
consistent with which culture days exist per clone, and it is the naming used by the
`NN_Data/QBAM_DNNS` tiles of the same wells. Donor codes D2/D3/D4 = AMD1/2/3 come from
`AMD_Donor-Match_DNNI.csv`. TER range over the 10 wells: 780 to 1030 Ohm cm^2 (all mature
monolayers; between-well differences are of the order of the within-well SD, 17 to 71).

The pixel size is not recorded anywhere in the deposit (the paper gives no objective for
these images); cells are about 19 px across. All wells share the optics, so per-pixel
numbers are comparable across wells but not convertible to micrometres.

The Healthy-2 series (216 TER values over maturation, 291 GB) has only live QBAM
bright-field images and no junction stain, so it was not attempted; the RPE Mask R-CNN
training set already in `data/rpe_nei/` (figshare 28832501) contains no TER or well
metadata (checked: only VIA polygon annotations per z-frame; the 4.5 GB zip is truncated).

### 2. S-BIAD1169 with the published ECIS and permeability effects

The BioStudies record's publication is Bromberger et al. 2024, Life Science Alliance
7:e202402671 ("Off-targets of BRAF inhibitors disrupt endothelial signaling and vascular
barrier function"; PDF in `data/sbiad1169/paper/`). Fig 4A gives the ECIS resistance change
at 1 h per inhibitor and dose (mean of 5 to 10 replicates), Fig 4B the transwell tracer
permeability fold change at 1 h. The paper publishes no source-data table, so the values in
`scripts/pimorph_function_sbiad1169_vs_paper.py` were read off the figure (about +-50 ohm,
+-0.1 fold). Text of the paper: vemurafenib lowers resistance significantly at 10 to 100 uM,
PLX8394 at 50 to 100 uM, dabrafenib only at 100 uM, encorafenib not at all; permeability
rises for V100 and P100 only. The S-BIAD1169 fields are the 1 h treatments at 1, 10 and
100 uM of the same four inhibitors, so the comparison is field-level prediction against
condition-level measurement (17 fields, 13 conditions).

### 3. Searched, not obtained

Web searches for datasets pairing junction images with TEER, FITC-dextran, XPerT or Evans
blue at field or well level found none with per-well pairing. Closest: Zenodo 14969049
(TDP-43 BBB paper, ZO-1/VE-cadherin transwell stainings by condition, TEER only in
figures) and Zenodo 8377287/11185144 (TLNRD1, xCELLigence per condition); both would only
allow another condition-level direction test like S-BIAD1169 and were not processed.

## Pipeline (script `scripts/pimorph_function_rpe_ter.py`)

Per tile: ZO-1 and mask upsampled 2x (cells 19 px -> 38 px). Curated complex: cells =
4-connected interiors of the hand-corrected mask, grown 3 px so neighbours share a crack,
`extract_complex`. Neural complex: `NeuralProposer(models/pimorph_proposals_v6_pool.pt,
tta=True)` on the ZO-1 tile + `ConstrainedDecoder` with `vertex_weight 0.3` and the
proposer's cell radius (no nuclei). ZO-1 profiled as the TJ channel with
`multichannel_profiles(bin_px 2, half_width_px 3, per-tile Otsu)`; `barrier_report` with
weights `{TJ: 1.0}`, baseline 0.05, gap conductance 10. Field statistics are
arclength-weighted means; conductances are rescaled to native pixels. Per well: mean over
tiles. Correlation with TER: Spearman over wells with bootstrap CI, plus a tile-level
Spearman whose p-value comes from permuting TER across wells (2000 permutations), so the
clustering of tiles within wells is respected.

Reconstruction check on four tiles against the curated masks (2x upsampling): adjacency
pair F1 0.72 to 0.76, vertex F1 0.77 to 0.80, boundary F1 0.91 to 0.94 on three tiles,
0.46 / 0.60 / 0.78 on one tile whose mask is partial. The neural reconstruction has 236 +- 21
cells per tile against 148 +- 34 in the masks (the masks leave unannotated regions and the
decoder over-splits these small cells), and finds no gaps where the masks have 5 +- 4
unlabelled pockets per tile.

## Results: RPE TER (n = 10 wells, 1032 tiles)

`rpe_ter_correlations.csv`, `rpe_ter_per_well.csv`, `rpe_ter_per_tile.csv`,
`barrier_proxy_vs_measured.png`.

| Statistic (per-well mean) | Expected sign vs TER | Curated: Spearman (p) [95% CI] | Curated tile-level (well-permutation p) | Neural: Spearman (p) |
|---|---|---|---|---|
| permeability index (= G_eff / area) | negative | **-0.52 (0.13) [-0.92, 0.22]** | -0.40 (0.14) | -0.09 (0.80) |
| G_eff | negative | -0.52 (0.13) | -0.40 (0.14) | -0.09 (0.80) |
| mean g per edge | negative | -0.24 (0.51) | -0.17 (0.52) | -0.45 (0.19) |
| TJ (ZO-1) coverage | positive | 0.33 (0.35) | 0.31 (0.20) | 0.28 (0.43) |
| TJ continuity | positive | 0.28 (0.43) | 0.21 (0.37) | 0.22 (0.53) |
| ZO-1 strip intensity / tile median | positive | 0.14 (0.70) | 0.01 (0.97) | 0.15 (0.68) |
| cells per tile | none expected | -0.03 (0.93) | -0.12 (0.73) | 0.42 (0.23) |
| cell area CV | | 0.31 (0.39) | 0.26 (0.44) | -0.25 (0.49) |
| unlabelled pockets (curated gaps) | negative | -0.13 (0.73) | -0.18 (0.35) | none found |

Per well (curated): permeability index 0.041 to 0.060 per px^2, ZO-1 coverage 0.49 to 0.59.
The lowest predicted permeability (0.041, AMD1 B day 76) belongs to the highest-TER well
(975) and the leakiest predictions (0.060, AMD3 B day 73 and AMD1 B day 73) to wells at 847
to 892, which is the sign the model predicts; but AMD1 A day 75, the lowest TER (780), is
predicted tight (0.049), and the between-well TER differences are mostly within the
within-well SD. Curated and neural per-tile statistics agree well for the junction fields
(Spearman 0.87 coverage, 0.91 continuity, 0.99 intensity) but poorly for the conductances
(0.24), because G_eff is dominated by edge count and the neural decoder splits 60% more
cells than the masks: the barrier proxy inherits the reconstruction's cell count directly.

## Results: S-BIAD1169 vs Bromberger 2024 Fig 4

`sbiad1169_per_treatment_vs_paper.csv`, `sbiad1169_vs_paper_correlations.csv`.

| Treatment (n fields) | Paper: resistance change 1 h (ohm) | Paper: permeability 376 Da / 70 kDa (fold) | Significant in paper | Predicted G_eff / DMSO |
|---|---|---|---|---|
| DMSO (3) | 0 | 1.0 / 1.0 | | 1.00 |
| V1 / V10 / V100 (1 / 2 / 2) | -50 / -500 / -1700 | 1.0, 1.15, 2.2 / 1.0, 1.05, 1.7 | V10, V100 | 1.14 / 1.01 / 1.15 |
| D1 / D10 / D100 (1 each) | -50 / -100 / -600 | 1.0, 1.05, 1.3 / 1.0, 1.05, 1.1 | D100 | 1.24 / 0.86 / 0.79 |
| E1 / E10 / E100 (1 each) | -50 / -50 / -100 | 1.15, 1.1, 1.4 / 1.0, 1.0, 1.2 | none | 1.17 / 1.03 / 1.25 |
| P1 / P10 / P100 (1 each) | -100 / -250 / -1500 | 1.1, 1.2, 2.1 / 1.05, 1.15, 1.85 | P100 | 0.90 / 1.01 / 1.11 |

Spearman over the 17 fields: G_eff vs resistance change -0.03 (p 0.90; expected negative),
vs 376 Da permeability 0.20 (p 0.43), vs 70 kDa 0.07 (p 0.79). Mean G_eff of the six fields
whose condition has a significant measured barrier effect: 1020, of the other eleven: 1034
(Mann-Whitney p 0.96). Direction per condition: V100 (+15%) and P100 (+11%) predicted
leakier as measured; D100 predicted 21% tighter although the paper measures a 600 ohm drop;
E100 predicted 25% leakier although encorafenib has no measured effect. The 2024 paper's
own conclusion at 1 h is that vemurafenib and PLX8394 visibly interrupt claudin-5 and
VE-cadherin at high dose; our TJ coverage for V100 (0.31) and P100 (0.23) is indeed below
DMSO (0.38), but so is D1 (0.32) and E10 (0.30), and one field per condition from 8-bit
display exports cannot separate a real effect from field-to-field variation (the
earlier `runs/multijunction/REPORT.md` warning stands).

Verdict for S-BIAD1169: the predicted barrier change does not agree with the measured
one beyond the two highest doses of the two inhibitors with the largest effect, and it
contradicts it for dabrafenib and encorafenib. G_eff as computed is not a usable predictor
of ECIS resistance here.

## What was validated, what was not

- Validated: nothing about barrier function. The correct sign on the RPE wells (predicted
  permeability falls as TER rises, -0.52) is suggestive and worth a properly powered
  experiment (TER range wider than the within-well SD, junction stain of the same wells,
  more than 10 wells), but at n = 10 with p = 0.13 it does not meet the bar to flip
  `validated`.
- The junction-field statistics themselves are reproducible between curated and PiMorph
  reconstructions (coverage 0.87, continuity 0.91, intensity 0.99 Spearman over 1031 tiles).
- Not validated, and now with evidence against: G_eff / permeability index as a predictor
  of measured barrier change across treatments on S-BIAD1169.
- The proxy's weakest point is exposed by both datasets: it scales with the number of
  reconstructed interfaces, so segmentation granularity (cell count) moves it as much as
  junction coverage does. A coverage-only or per-interface-normalized index would be the
  next thing to test against TER.

## Provenance of every number

- TER: `data/rpe_nist/AMD_Data/AnalysisOfSegmentedData/AMD_TER-Data_Mean-SD.csv`, columns
  `ter.Mean`, `ter.StDev`, joined by the well mapping above.
- RPE tile statistics: `scripts/pimorph_function_rpe_ter.py`, outputs in this directory;
  checkpoint `models/pimorph_proposals_v6_pool.pt`, TTA on, MPS.
- S-BIAD1169 predictions: `runs/multijunction/per_field.csv` (ClassicalProposer on
  VE-cadherin with DAPI seeds, see that report). Paper values: Bromberger et al. 2024 LSA,
  Fig 4 read from `data/sbiad1169/paper/bromberger2024_lsa.pdf` page 8.
