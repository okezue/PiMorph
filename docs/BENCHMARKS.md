# Benchmarks

`pimorph benchmark` scores reconstruction methods against label-image ground truth on the exact cell complex: faces, contacts with multiplicity, multicellular vertices, incident sets, and cyclic order, next to the usual instance and boundary scores. Vertices, incident sets, and cyclic order have ground truth wherever masks exist because they are derived exactly from the GT label image with `extract_complex` at evaluation time (`bench/datasets.py`).

The tables below are copied from `runs/pimorph_bench/*_summary.md` as produced on 2026-09-16 by the current classical proposer and the Cellpose-SAM baseline. They are development numbers on small samples (6 to 24 images), not headline results. Read the caveats before quoting any of them.

## Running

```bash
pimorph benchmark --dataset {cornea,nuinsseg,mcellseg,livecell,synth} \
    [--methods classical cellpose_sam gt] [--root PATH] [--max-items N] \
    [--out runs/pimorph_bench] [--tol-px 3.0]
```

- `--methods` defaults to `classical`. `cellpose_sam` is skipped with a message when cellpose or torch is not installed. `gt` scores the ground truth against itself (all structural scores 1.0) and is a useful sanity check of the loaders.
- `--root` defaults to `data/cornea_cells`, `data/NuInsSeg`, `data/mcellseg`, `data/LIVECell`, `data/tiles/synth_val` (`bench/datasets.py:DEFAULT_ROOTS`).
- `--tol-px` is the distance tolerance used for edge trace overlap, vertex matching, and boundary F1.
- Outputs per dataset: `<dataset>_per_image.csv` (one row per image and method with every metric), `<dataset>_summary.csv` (mean and median per method), `<dataset>_summary.json`, and `<dataset>_summary.md` (the table below). A method that raises on an image is recorded with an `error` column and excluded from the summary means (`n_errors` is reported).

Methods (`bench/run.py:METHODS`):

- `classical`: the geometry channel is inverted when boundaries are dark (`resolve_polarity`, decided from Sato bright versus dark ridge response without touching the GT), the tissue mask is computed on the original image, `ClassicalProposer()` runs with the item's nuclei channel when the loader provides one, and `ConstrainedDecoder.decode` runs with `DecoderParams(cell_radius_px=maps.meta["cell_radius_px"])`. Without a nuclei channel `fill_gaps=True` is set because gaps cannot be told from cells without seeds.
- `cellpose_sam`: `CellposeSAM()(geometry, nuclei)` with `pretrained_model="cpsam"`, `diameter=None`, `flow_threshold=0.4`, `cellprob_threshold=0.0`, device auto (cuda, then mps, then cpu).

## Metric definitions (short)

All from `metrics/structural.py` and `complex/matching.py`; details in `COMPLEX.md`, section "Matching two complexes".

| Column | Definition |
|---|---|
| Adjacency F1 (pair) | Unordered cell pairs in contact, predicted pairs mapped through the face matching (IoU >= 0.5), versus GT pairs. Multiplicity ignored |
| Adjacency F1 (component) | Matched cell-cell edges (trace overlap >= 0.5 within `tol_px`) over predicted and GT edge counts. Two disconnected contacts between the same cells count twice |
| Vertex F1 | Junction vertices (>= 3 incident cells) matched by mutual nearest neighbor within `tol_px` |
| Vertex loc. err. median (px) | Median over images of the per-image median matched-vertex distance |
| Incident-set acc. | Fraction of matched vertices whose mapped incident cell set equals the GT set |
| Cyclic-order acc. | Fraction of matched vertices whose mapped cell cycle is a rotation of the GT cycle in the same direction (implies set agreement) |
| PQ | Panoptic quality, SQ x RQ at IoU >= 0.5 |
| Boundary F1 | Boundary pixels (`find_boundaries(mode="inner")`) within `tol_px` of the other boundary set |
| Edit dist. (approx) | Unmatched predicted faces + unmatched GT faces + unmatched predicted edges + unmatched GT edges + incident-set mismatches |
| Valid | Fraction of predictions whose complex passes `validate()`; must be 1.0 by construction |
| Legacy 4-nbr adj. F1 | Adjacency F1 from `endopigraph.interfaces.compute_interfaces` (4-neighbor pixel contacts, `min_contact_px=10`) on both images, labels mapped by IoU >= 0.5 |
| Runtime (s) | Wall time of the method call per image, excluding scoring |

The per-image CSV also holds AP50, AP75, VI with its merge and split components, Hausdorff distances, split and merge counts, Euler residuals, and the arclength relative errors of matched edges. `tests/pimorph/test_metrics.py::test_missing_border_barely_moves_pixels_but_changes_adjacency` shows why the structural columns are there: merging one small cell changes under 2 % of pixels, leaves PQ above 0.97 and boundary F1 above 0.99, and still removes a face, at least two contacts, and the incident sets around it.

## Endothelial ground truth (HAEC, mCellSeg)

`docs/ENDOTHELIAL_RESULTS.md` holds the first endothelial accuracy numbers: held-out human aortic endothelial cells (GFP + Hoechst, Zenodo 4898011) and mCellSeg (expert HUVEC/HEK DIC masks) for `v2_endo`, Cellpose-SAM, the classical filters and earlier checkpoints.

## Scaled results (400 images per dataset, 8x H100)

`docs/SCALE_RESULTS.md` holds the large-sample tables run on xAI compute on 2026-09-17: Cellpose-SAM and classical baselines on LIVECell test (all 8 cell lines), NeurIPS 2022 CellSeg, cornea and 400 synthetic tiles, plus the first model trained on real instance ground truth (`models/pimorph_proposals_v1_multi.pt`). The small-sample tables below were produced earlier on the laptop and are kept because the docs and model cards cite them.

## Current tables (laptop, small samples)

### Synthetic tiles (exact ground truth)

Source: `runs/pimorph_bench/synth_summary.md`, 2026-09-17. 40 held-out tiles of 512 x 512 px from `data/tiles/synth_val` (seed 777, never used in training; junction channel as geometry, nuclei channel present). Both methods run through the same constrained decoder; only the proposal maps differ. `neural` is `models/pimorph_proposals_v0_synth.pt` (stage 1, synthetic training only, see `models/pimorph_proposals_v0_synth.md`).

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | 0.447 | 0.362 | 0.503 | 1.000 | 0.372 | 0.372 | 0.554 | 0.852 | 879.750 | 1.000 | 0.434 | 4.529 |
| neural | 0.889 | 0.739 | 0.733 | 1.000 | 0.860 | 0.860 | 0.891 | 0.981 | 398.425 | 1.000 | 0.854 | 3.575 |

Learned proposals double adjacency F1 and incident-set accuracy over the classical filters with an identical decoder. Synthetic validation is in-distribution for the neural model (same generator, different seeds), so this measures what the decoder can extract from good proposal maps, not real-data accuracy.

Stage 2 (`models/pimorph_proposals_v0_mixed.pt`, resumed from stage 1 with 175 real pseudo-label tiles added; `runs/pimorph_bench_v2/synth_summary.md`, same 40 tiles): adjacency F1 0.887, component F1 0.735, vertex F1 0.764, incident-set accuracy 0.857, PQ 0.893, boundary F1 0.984, validity 1.0. Adding real pseudo-labels did not cost synthetic accuracy.

### Real fluorescence fields (self-consistency only, no instance truth)

Same decoder, three proposal sources, 2026-09-17. `ratio` is the mean VE-cadherin signal on reconstructed cell-cell interfaces over the mean in cell interiors (blueprint audit statistic); `render_ll` is the renderer's mean per-pixel log-likelihood. Neither is accuracy; both should rise when boundaries land on junction signal.

| Field | Proposals | Cells | Gaps | Tricellular vertices | Degree histogram | ratio | render_ll |
|---|---|---|---|---|---|---|---|
| VE-strat Histamine_s2, 1024 px crop at 2x down | classical | 181 | 0 | 305 | {3: 358, 4: 1} | 1.54 | -5.517 |
| | neural (synth) | 174 | 1 | 299 | {2: 1, 3: 344, 4: 1} | 1.49 | -5.525 |
| | neural (mixed) | 195 | 0 | 338 | {3: 386, 4: 1} | 1.52 | -5.531 |
| S-BIAD1540 EGM2_regular_6dyn-24, 1024 px | classical | 65 | 0 | 40 | {2: 31, 3: 66} | 2.62 | -5.039 |
| | neural (synth) | 147 | 2 | 238 | {2: 1, 3: 294} | 2.31 | -4.941 |
| | neural (mixed) | 169 | 0 | 271 | {2: 2, 3: 332} | 1.97 | -4.974 |

On the flow-aligned S-BIAD1540 field the classical ridge filter is confused by cytoplasmic texture and leaves 31 isolated cells (degree-2 artificial vertices), while both neural models recover the elongated cells with almost all vertices trivalent and a higher render likelihood; the lower `ratio` for neural proposals there reflects that they also find dim boundaries the ratio statistic penalizes. Figures: `runs/pimorph_dev/sbiad1540_6dyn_neural_vs_classical.png`, `runs/pimorph_dev/ve_strat_neural_vs_classical.png`. On the dense VE-strat monolayer the three sources agree closely.

### LIVECell (phase contrast, COCO polygon ground truth)

Source: `runs/pimorph_bench/livecell_summary.md` and `livecell_per_image.csv`, 2026-09-16/17. 6 validation images of the BT474 line, 520 x 704 px, no nuclei channel. GT background slivers below 12 px filled. Mean GT cells per image 201.0; mean predicted cells 178.3 (Cellpose-SAM) and 639.2 (classical).

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cellpose_sam | 0.558 | 0.466 | 0.327 | 1.707 | 0.657 | 0.657 | 0.628 | 0.894 | 325.667 | 1.000 | 0.546 | 8.384 |
| neural (synthetic only) | 0.000 | 0.000 | 0.027 | 1.414 | 0.000 | 0.000 | 0.031 | 0.460 | 750.333 | 1.000 | 0.000 | 6.906 |
| classical | 0.006 | 0.003 | 0.046 | 2.059 | 0.004 | 0.004 | 0.044 | 0.506 | 2897.333 | 1.000 | 0.004 | 13.171 |

The synthetic-only neural model does not transfer to phase contrast: the generator renders fluorescent junction lines, and BT474 phase-contrast images look nothing like them. Cellpose-SAM, trained on broad real data, is the baseline to beat there. On fluorescence VE-cadherin (VE-strat crop, no instance truth) the same model produces a continuous boundary map and 43 cells with every vertex trivalent (`runs/pimorph_dev/ve_strat_neural_vs_classical.png`).

### Cornea (specular microscopy, derived instance ground truth)

Source: `runs/pimorph_bench/cornea_summary.md`, 2026-09-16. 6 crops of 500 x 500 px, dark boundaries, no nuclei channel. Mean GT instances per crop 570.7; mean predicted cells 561.5.

| Method | Adjacency F1 (pair) | Adjacency F1 (component) | Vertex F1 | Vertex loc. err. median (px) | Incident-set acc. | Cyclic-order acc. | PQ | Boundary F1 | Edit dist. (approx) | Valid | Legacy 4-nbr adj. F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | 0.000 | 0.000 | 0.023 | 2.236 | 0.000 | 0.000 | 0.001 | 0.517 | 4308.000 | 1.000 | 0.000 | 7.522 |

### Not yet run

NuInsSeg (loader present, `data/NuInsSeg`) and mCellSeg (loader present; `data/mcellseg` is absent in this workspace, it needs a Kaggle token) have no `pimorph benchmark` output yet. Neither neural model has been scored against expert instance truth on endothelial fluorescence, because no such dataset is on disk; that is the first thing the mCellSeg download unlocks.

## Caveats

- Cornea ground truth is derived and over-segmented. The cornea labels are semantic (1 interior, 2 border, 3 background) and the border class does not close cells, so `semantic_to_instance_watershed` seeds a watershed from distance-transform peaks 10 px apart. The loader sets `meta["gt_kind"] = "derived_from_semantic_oversegmented"` and `meta["instance_metrics_reliable"] = False`. Face, adjacency, vertex, and PQ columns against this GT are not meaningful; only boundary F1 against the border class is. The zeros in the cornea row say as much about the GT as about the method. The construction is kept because the legacy report (`scripts/benchmark_cornea_final.py`) used it.
- LIVECell (and mCellSeg) slivers. Polygon annotations rasterized independently per cell leave 1 to 3 px background slivers where cells meet, which would turn every tricellular vertex into a cell-cell-gap vertex. `fill_gt_slivers(labels, 12)` assigns enclosed background components smaller than 12 px to the neighboring cell with the most contact before scoring (`meta["gt_sliver_fill_px"] = 12`). Larger genuine gaps are untouched.
- The classical proposer fails on phase contrast without nuclei. It was built for bright junction ridges with a nuclei channel for seeds. On LIVECell it has neither: boundary polarity is guessed, seeds come from distance-to-boundary maxima, and `fill_gaps=True` is forced. It over-segments (639 predicted versus 201 GT cells) and its adjacency F1 is near zero. This is the expected failure mode, not a tuning accident, and it is the reason the neural proposer exists.
- Cellpose-SAM numbers are a baseline with default settings on 6 images of one cell line. They are competitive on boundaries and instances (boundary F1 0.894, PQ 0.628) and much weaker on vertices (vertex F1 0.327), which is exactly the quantity the complex metrics add. Runtimes are wall times on the machine that wrote the files (the local workstation; `infra/aws/aws_log.md` records no GPU instance at that point).
- Synthetic tiles come from the same generator family that trains the neural proposer. They give exact vertex and gap truth but do not measure domain transfer.
- Sample sizes are 6, 6, and 24 images. No confidence intervals are reported; per-image values are in the CSVs.

## What these numbers do and do not show

They show that the pipeline runs end to end on three modalities, that every reconstruction is a valid complex (Valid = 1.000, Euler residual 0 everywhere), and that the structural metrics separate methods that pixel metrics do not: on LIVECell, Cellpose-SAM and the classical proposer differ by a factor of about 1.8 in boundary F1 but by two orders of magnitude in adjacency F1. They also show where the classical proposer is usable (bright junction ridges with nuclei seeds, as on the synthetic tiles) and where it is not (phase contrast without nuclei).

They do not show accuracy on the endothelial data the project is about: S-BIAD1540 and VE-strat have no instance ground truth, and mCellSeg (the HUVEC set with masks) has not been downloaded or run. They do not compare against the legacy published F1 values in `README.md` (LIVECell 78.4 %, NuInsSeg 82.2 %, cornea 93.1 %), which were computed by a different method (the legacy hybrid pipeline with its own seeds) on different image counts and with the cornea GT caveat above. They contain no neural proposal result and no calibration result. Treat them as the first row of a table that the benchmark command will refill.
