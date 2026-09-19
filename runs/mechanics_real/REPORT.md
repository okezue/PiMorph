# Force inference against real mechanical readouts (2026-09-19)

Question: do the relative tensions of `pimorph.mechanics.force_inference.infer_tensions`
(vertex force balance, optional Laplace rows from edge curvature, ridge prior, mean tension
= 1 gauge) agree with an independent mechanical measurement on real tissue? Until now the
module had been checked only on synthetic vertex-model sheets (`runs/mechanics/REPORT.md`).

Summary of what real data gave:

| Dataset | Independent readout | Result | Status |
|---|---|---|---|
| Lang et al. 2019 laser nanoablations, Drosophila germband, 15 cuts of parasegment-boundary cables (Zenodo 3257654) | recoil velocity of the cut ends (manual kymograph tracks) | cut edge inferred above field mean in 14/15 fields (sign test p 5e-4); cable-line edges above off-cable edges in 15/15 fields (+0.35 relative, Wilcoxon p 1.5e-4); across cuts, cut-edge tension vs recoil velocity Spearman 0.64 (p 0.011, 95% CI 0.16 to 0.86) tension-only, 0.21 (n.s.) with pressures and curvature | partial validation: known boundary-cable tension recovered; cut-to-cut ranking significant in one of three settings only |
| DLITE ZO-1 hiPSC colonies (Vasan et al. 2019), 4 series, 102 frames, hand traces | none measured; DLITE and CellFIT are other inferences | PiMorph vs DLITE on the same edges Spearman 0.27 (n 652 edge-frames, frame-bootstrap CI 0.19 to 0.35); PiMorph temporal consistency 0.14 to 0.66 vs DLITE 0.58 to 0.72 | cross-method agreement, weak; not a validation |
| TissueMiner pupal wing demo, 15 frames, ~630 tracked cells | none measured; tissue axis known (PD = image x) | inferred stress axis within 3.2 deg of PD in 15/15 frames; PD-aligned edges carry 1.50x the tension of AP-aligned edges at t = 0, relaxing to 1.23x at 5.8 h in step with the database elongation (Spearman 0.98 over frames) | consistency only |
| FRET tension sensors, Sugimura/Ishihara wing-disc ablations, CellFIT and Bayesian-inference validation sets, Noll/Streichan germband | | no public deposit with images plus per-junction measurements found | not obtained |

Absolute tension, pressure scale and the elastic moduli remain unidentifiable; every number
below is relative to the field mean.

## 1. Laser ablation recoil (the only independent force readout obtained)

Data: Zenodo record 3257654, Lang, Dutta, Scarpa, Sanson, Schoenlieb, Etienne 2019
(CC BY-NC-SA 4.0), downloaded to `data/ablation_lang2019/`. 15 confocal time lapses of
E-cadherin:GFP Drosophila embryos (255 x 255 px = 42.2 um, 727.67 ms per frame, 60 to 100
frames) from Scarpa et al. 2018 Dev Cell; each shows one plasma-induced nanoablation of a
supracellular actomyosin cable at a parasegment boundary (PSB). Per movie: a Fiji line ROI
along the cut cable, kymographs resliced along it, and manual tracks of the two cut ends
(`cutend_L.txt`, `cutend_R.txt`: position along the line in px, frame). Loader:
`src/pimorph/io/ablation_lang2019.py` (ImageJ line ROI parser, recoil from the tracks).

Recoil velocity: separation of the two cut ends interpolated on their common frames, slope
of a least-squares line over the first four frames after the cut (px/frame; x 0.165 um/px
/ 0.728 s for um/s). Range over the 15 cuts: 1.51 to 3.46 px/frame (0.34 to 0.79 um/s),
cut at frame 5 to 6 in every movie.

Pipeline (`scripts/pimorph_mechanics_ablation.py`): mean of the 3 frames before the cut,
`NeuralProposer(models/pimorph_proposals_v6_pool.pt, tta=True)` + `ConstrainedDecoder`
(`vertex_weight 0.3`, proposer cell radius; 39 to 79 cells per field), `smooth_complex`,
`infer_tensions` in three settings, identifiability report. Cut edge = the cell-cell edge
nearest the tracked cut position (median distance 0.5 px, max 3.3 px) aligned with the
reslice line (|cos| >= 0.5; 0.90 to 1.00 in practice). Cable-line edges = cell-cell edges
within 4 px of the reslice line and aligned within 45 deg (2 to 10 per field).

### Results, zoom 1 (`ablation_summary_zoom1.csv`, `ablation_per_cut_zoom1.csv`)

| Setting | Cut-edge tension vs recoil: Spearman (p) [95% bootstrap CI] | Pearson (p) | Cut edge above field mean | Cable minus off-cable mean tension (Wilcoxon p, 15 fields) | Median residual RMS | Vertices balanced within 5% |
|---|---|---|---|---|---|---|
| tension only, ridge 0.1 | **0.64 (0.011) [0.16, 0.86]** | 0.52 (0.048) | 14 / 15 (p 5e-4) | +0.35 (1.5e-4) | 0.20 | 8% |
| pressures + curvature, ridge 0.1 | 0.21 (0.44) [-0.42, 0.75] | 0.17 (0.54) | 13 / 15 (p 4e-3) | +0.35 (2.1e-4) | 0.23 | 4% |
| pressures + curvature, ridge 0.01 | 0.18 (0.52) [-0.45, 0.73] | 0.16 (0.57) | 12 / 15 (p 0.018) | +0.37 (1.7e-3) | 0.23 | 2% |

Mean cut-edge tension 1.41 (field mean = 1), median percentile within its field 0.85.
Cable-line edges average 1.35 against 0.99 for the other edges; the cable is at or above the
field mean in every field and is the highest-tension structure in the field in the plotted
example (`ablation_recoil_vs_tension_zoom1.png`, right). Scarpa et al. 2018 measured PSB
cables to recoil about twice as fast as non-boundary junctions; the inference recovers the
sign and a 35% excess. Note that a straight cable through several vertices is exactly the
configuration in which a vertex balance is informative: at each cable vertex the two
collinear cable edges must balance the transverse pull of the side edges, so their tension
must exceed the side tensions whenever the side edges are not perpendicular. The 15/15
result is therefore partly geometry that any force-balance method would return, and the
recoil measurement is what makes it a check rather than a tautology.

Across cuts, only the tension-only setting ranks the 15 recoil velocities significantly
(0.64); with pressures and Laplace rows the correlation drops to 0.2. On these fields the
curvature rows add 100 to 200 equations of poorly determined sign (segmentation traces of
short 15 to 30 px edges) and the null space has 23 to 33 directions; the tension-only
system is nearly square (110 to 250 equations for 113 to 234 unknowns, rank 92 to 202) and
the ridge fills the rest. Three settings were run, so 0.64 is the best of three and its p
should be read as about 0.03 after that multiplicity; the 95% CI excludes zero in either
reading but is wide. Also: recoil velocity is absolute (um/s, one embryo each) while the
inferred tension is relative to each field's mean, so the comparison assumes comparable
mean tension and friction across embryos.

Robustness, zoom 2 (`ablation_summary_zoom2.csv`, image upsampled 2x before reconstruction,
47 to 137 cells): within-field results unchanged (cut edge above mean 13 to 14 / 15;
cable > off-cable 15 / 15, +0.38 to +0.50, p <= 6e-4); across cuts Spearman 0.48 (p 0.07)
tension-only, 0.25 (n.s.) with pressures.

What this does and does not validate: the inference identifies the known high-tension
structure in 14 to 15 of 15 real fields with an independent measurement behind it, and
ranks the recoil velocities of 15 different embryos with rho 0.64 in its simplest setting.
It does not validate the pressure or curvature terms on real data (they made the ranking
worse), nor any absolute scale, and n = 15 cuts of the same junction type is a narrow test.

## 2. DLITE ZO-1 time series (cross-method, no measurement)

Data: `data/dlite/DLITE-master/Notebooks/Data/ZO-1_data` from github.com/AllenCellModeling/DLITE
(Vasan, Maleckar, Williams, Rangamani 2019 Biophys J). Four time series of hiPSC colonies
with endogenous GFP-ZO-1 (1024 x 1024 max projections), interfaces hand-traced in NeuronJ
with curvature; 30, 31, 31 and 10 frames. The published validation of DLITE is temporal
consistency of inferred tensions (and agreement with Surface Evolver ground truth on
synthetic colonies, not re-run here). The DLITE package was run in our venv on its own
traces with two patches for numpy 2 (2-vector `np.cross`, `np.math`) plus one `!= []`
comparison in `cell_describe.py` (original kept as `.orig`); its CellFIT-mode pressure solve
raises a singular-matrix error on series 3 with current scipy, so series 3 has DLITE-solver
values only. Script: `scripts/pimorph_mechanics_dlite.py`.

PiMorph complexes: the traced colony rasterized to a label image (regions enclosed by the
traces; regions touching the image border are medium), `extract_complex`, `smooth_complex`,
`infer_tensions`. DLITE edges (NeuronJ splits an interface at intermediate nodes) are
matched to PiMorph edges by mean point-to-polyline distance <= 4 px (many-to-one; 25 to 29
of 48 to 62 DLITE edges per frame matched). PiMorph only balances interior vertices, and in a
6 to 10 cell colony most vertices touch the medium, so only 4 to 8 matched edges per frame
carry a PiMorph tension (DLITE assigns tensions to all edges, medium included). This is the
main limitation of the comparison and of PiMorph's force balance on small islands.

Agreement on the same edges, both normalised to the frame mean (`dlite_agreement.csv`):

| PiMorph setting vs DLITE solver | n edge-frames | Spearman pooled [frame-bootstrap 95% CI] | Pearson | per-frame Spearman mean (fraction of frames > 0) |
|---|---|---|---|---|
| tension only vs DLITE | 652 | 0.27 [0.19, 0.35] | 0.27 | 0.13 to 0.36 by series (0.64 to 0.80) |
| tension only vs CellFIT (series 1, 2, 4) | 399 | 0.32 [0.22, 0.42] | 0.25 | 0.24 to 0.36 (0.71 to 0.84) |
| pressures + curvature vs DLITE | 652 | 0.24 [0.15, 0.33] | 0.21 | 0.14 to 0.31 |
| pressures + curvature vs CellFIT | 399 | 0.24 [0.11, 0.36] | 0.17 | 0.20 to 0.28 |

Temporal consistency, Pearson r of normalised tensions between consecutive frames on edges
present in both (`dlite_consistency.csv`; DLITE tracks labels, PiMorph edges inherit them via
the match):

| Series | DLITE CellFIT solver | DLITE solver | PiMorph tension only | PiMorph pressures + curvature | edges per pair (DLITE / PiMorph) |
|---|---|---|---|---|---|
| 1 | 0.47 | 0.62 | 0.14 | -0.04 | 59 / 7 |
| 2 | 0.27 | 0.72 | 0.42 | 0.42 | 54 / 6 |
| 3 | failed | 0.60 | 0.18 | 0.24 | 59 / 8 |
| 4 | 0.44 | 0.58 | 0.66 | 0.26 | 46 / 8 |

The DLITE paper's claim (DLITE more consistent than per-frame CellFIT) reproduces on its
data (0.58 to 0.72 vs 0.27 to 0.47). PiMorph's per-frame inference is as inconsistent as
CellFIT or worse on three series, on 6 to 8 edges per pair; it carries no temporal prior, so
this is expected and is not evidence about correctness. Agreement with DLITE is positive
and significant but weak (0.27), and DLITE itself is an inference, not a measurement.

## 3. TissueMiner pupal wing: tissue-scale consistency (no measurement)

Data: `data/tissueminer/example_data/demo` (Etournay et al. 2016; 71 frames, 5.8 h, ~630
tracked cells per frame, hinge to the left so PD is the image x axis). Frames 0, 5, ...,
70: complex from the curated tracked labels, `infer_tensions`, tissue-averaged tension
stress tensor sum_e gamma_e l_e l_e^T / |l_e| / A over identifiable cell-cell edges (1,690
to 1,880 per frame), its anisotropy (lambda_1 - lambda_2) / (lambda_1 + lambda_2) and axis;
control with gamma_e = 1 (network geometry only). Script
`scripts/pimorph_mechanics_tissueminer.py`, outputs `tissueminer_stress_anisotropy.csv`,
`tissueminer_stress_summary.csv`.

| Quantity | frame 0 (0 h) | frame 70 (5.8 h) | axis to PD, median over 15 frames | Spearman with database cell elongation over frames |
|---|---|---|---|---|
| database cell elongation |Q| | 0.317 | 0.194 | 1.5 to -3.4 deg | |
| geometric anisotropy (gamma = 1) | 0.448 | 0.279 | 1.6 deg | 0.99 |
| inferred, tension only | 0.532 | 0.336 | 1.3 deg | 0.98 |
| inferred, pressures + curvature | 0.537 | 0.332 | 1.2 deg | 0.98 |
| mean tension, PD-aligned edges (< 30 deg) / AP-aligned edges (> 60 deg) | 1.145 / 0.761 = 1.50 | 1.121 / 0.908 = 1.23 | | |

The inferred stress axis is PD in every frame (max deviation 3.2 deg) and relaxes with the
cells, as the known PD stress of this stage should. Most of the anisotropy is already in the
network geometry (0.45); the inference adds a consistent 0.06 to 0.08 by assigning PD-aligned
edges 1.5x (falling to 1.2x) the tension of AP-aligned edges. Vertex residual RMS 0.11 to
0.15 on these curated complexes (vs 0.38 on the VE-strat neural reconstruction and 0.20 on
the germband). Since the elongation and the anisotropy come from the same geometry this is
consistency, not validation.

## 4. Not obtained

- FRET tension sensor image sets (VE-cadherin TS, E-cadherin TS) with per-junction FRET
  index: none deposited with images on Zenodo/figshare found by search.
- Sugimura and Ishihara 2013 wing-disc ablations, Kong/Brodland CellFIT validation, Kale et
  al. 2018, Xu et al. 2018 Bayesian force inference: no public image plus recoil deposits
  found; the Noll/Streichan tissueAnalysisSuite is code only.
- The Mendeley set 10.17632/78ng4tmj75.1 mentioned in search results as force inference
  versus recoil validation could not be listed (API returned no record).

## Provenance

- Recoil: `data/ablation_lang2019/*/*/cutend_{L,R}.txt`, `reslice.roi`; frame interval
  and field size from the Zenodo description. Segmentation: `models/pimorph_proposals_v6_pool.pt`,
  TTA on, MPS. All ablation numbers: `scripts/pimorph_mechanics_ablation.py --zoom 1|2`.
- DLITE: package at `data/dlite/DLITE-master` run through `scripts/pimorph_mechanics_dlite.py`
  (solver runtimes in `dlite_meta.json`, 40 to 180 s per series for the DLITE solver).
- TissueMiner: `pimorph.dynamics.tissueminer.load_tissueminer_demo`, database table
  `cells` for elongation.
