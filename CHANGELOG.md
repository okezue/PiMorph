# Changelog

## v1.1.0.dev0 (unreleased)

### 2026-09-19: symmetric comparison, real-data validation, README rewritten as an explainer
- Cellpose-SAM fine-tuned on exactly the training fields used for `v6_pool` (`scripts/export_cellpose_training.py`, `runs/cellpose_ft/`, `PIMORPH_CELLPOSE_MODEL`); held-out comparison in `docs/CONFLUENT_BENCHMARK.md`: fine-tuned Cellpose-SAM leads hCEC adjacency and PQ (0.937 / 0.862 vs 0.867 / 0.816) and vertex F1 within noise (0.695 vs 0.680); PiMorph keeps vertex F1 on FlyWing (0.875 vs 0.852) and HAEC (0.352 vs 0.331) and vertex localization; one PiMorph model covers both culture densities where each fine-tuned Cellpose collapses across domains.
- Force inference against laser-ablation recoil (Spearman 0.64, p 0.011, tension-only; 15 / 15 fields recover the boundary-cable excess); barrier proxy against measured TER (Spearman -0.52, p 0.13, not validated). Loaders `io.rpe_nist`, `io.ablation_lang2019`.
- Third data hunt (`docs/DATASET_HUNT_2026-09-19.md`): no public confluent vascular endothelial monolayer with junction marker and expert 2-D masks; COVER zebrafish dorsal aorta contours are the closest.
- README rewritten: what PiMorph does, the symmetric table, what is and is not validated, quickstart, data, history; refuted legacy claims removed from the front page.

### 2026-09-19: data moved to Zenodo
- Checkpoints, complete benchmark outputs and training tiles live in the PiMorph Zenodo data record (`scripts/publish_zenodo.py` creates and versions it, `scripts/fetch_zenodo.py --models` fetches and md5-verifies; concept DOI 10.5281/zenodo.22839866, first version 1.1.0 at 10.5281/zenodo.22839867). `models/*.pt` is no longer tracked (`.gitattributes` drops the LFS rule, `.gitignore` ignores fetched checkpoints); the unpushed history was rewritten so that no new Git LFS object is needed. README "Data availability" section.

### 2026-09-19: frontier round for multicellular vertices, pooled data
- `scripts/make_vertex_miss_weights.py` writes per-tile `loss_weight` arrays around true vertices the current decode misses (x4) and spurious predicted vertices (x3); `TileDataset` honours them plus a `vertex_focus` weight near all true vertices. `v5_vertex` (not shipped) and `models/pimorph_proposals_v6_pool.pt` (+ RPE train stacks and 1,200 real PECAM-1 HUVEC pseudo-label tiles) with card.
- Decoder: `vertex_weight` (vertex head in the elevation, default 0.3), `nucleus_merge` (cells without a nuclear peak join the neighbour across their weakest boundary), `NeuralProposer(tta=True)`, benchmark method `neural_map`, env hooks `PIMORPH_DECODER_PARAMS` and `PIMORPH_NEURAL_TTA`. Held-out hCEC vertex F1 0.611 to 0.680 (Cellpose-SAM 0.635); FlyWing 0.875; alizarine 0.992; HAEC 0.352.
- New loaders and pools: `rpe_zo1` (`pimorph.bench.rpe`, NIH-NEI RPE monolayer polygons, stack-disjoint splits), `pimorph.io.jacquemet` (495 real PECAM-1 HUVEC fields, manifest, pseudo-label driver, self-check), `pimorph.dynamics.epicure` (curated movies; 59/59 and 13/13 isolated T1s conserve charge; residuals are free-edge effects). `pimorph_dynamics_eval.py --dataset epicure`.
- `infra/xai/run_frontier.sh`, `run_pool.sh`, `run_final_bench.sh`.

### 2026-09-18: vertices against real truth, later phases, shear re-test
- HAEC reference derivation corrected (8-connected body markers, no sub-30 px specks, 4-connected labels): about 40% of the former "cells" were 1 to 3 px fringe specks. All HAEC and mCellSeg held-out tables were recomputed (`runs/vertex/`, `docs/ENDOTHELIAL_RESULTS.md`).
- Decoder: pixels the signed-distance head places outside every cell are excluded from the flood (`DecoderParams.outside_px`, default 0). The gap head only knows enclosed background, so open background was flooded and cells that never touch were joined (HAEC field 0005: 1,167 predicted multicellular vertices against 81 true; now 89). Optional vertex-consistent merges (`max_merges`). Vectorized small-background fill.
- `NeuralProposer(nucleus_seeds=True)`: nuclear peaks with no neural seed nearby are added as seeds.
- New loaders with real instance truth on confluent tissue: `hcec` (manually traced human corneal endothelial monolayers, NCAM + DAPI), `alizarine` (expert-contoured porcine corneal endothelium), `flywing` (E-cadherin Drosophila epithelium); `BenchItem.roi` restricts scoring to the annotated region; `pimorph benchmark --dataset` accepts every registered loader. `docs/DATASET_HUNT_2026-09-18.md`, `docs/CONFLUENT_BENCHMARK.md`.
- Checkpoints `models/pimorph_proposals_v3_endo.pt` (v2 recipe on the corrected reference) and `v4_confluent` (fine-tuned on the hCEC / alizarine / FlyWing train splits) with cards.
- `pimorph.metrics.sensitivity`: conditional-null enrichment z for all-reticular 3-cliques and the tricellular realization fraction; `scripts/pimorph_sensitivity_egm2.py` runs the neural proposer over every field with shards; `NETWORK_DISCOVERIES.md` records the posterior re-test of the three shear findings.
- Later blueprint phases: `pimorph.dynamics`, `pimorph.mechanics`, `pimorph.complex3d`, `pimorph.fields.multichannel`, `pimorph.function`, `pimorph.io.sbiad1169` with 65 tests (`docs/LATER_PHASES.md`).
- Tooling: `scripts/tune_decoder.py`, `scripts/select_checkpoint_egm2.py`, `infra/xai/sync_code.sh`, `infra/xai/run_vertex.sh`.

New canonical package `src/pimorph/` next to the legacy `src/endopigraph/`. Package version `1.1.0.dev0`; both packages ship in one wheel (`hatch` packages `src/endopigraph`, `src/pimorph`); new console script `pimorph`; new extras `complex` (shapely, pyarrow, zarr, ome-zarr, hypothesis, numba), `torch` (torch, torchvision, cellpose>=4), `aws` (boto3); pytest markers `torch`, `slow`, `data`. Documentation: `docs/ARCHITECTURE.md`, `docs/COMPLEX.md`, `docs/NEURAL_PROPOSALS.md`, `docs/BENCHMARKS.md`, `docs/AWS_RUNBOOK.md`.

### Exact cell complex (`pimorph.complex`)
- `extract_complex`: label image to embedded half-edge complex via the crack graph on the doubled interpixel grid. Labels are split into 4-connected faces, enclosed background becomes gap faces, border-touching background is the single outer face, one edge per connected contact component, degree-3 and degree-4 vertices, artificial degree-2 vertices on closed loops, exact cyclic order. Numba-accelerated chain tracing when numba is installed.
- `boundary_matrices` (B1, B2) and `validate`: `B1 @ B2 == 0`, two faces per edge, loop and vertex-link consistency, Euler residual `V - E + F - (1 + c)`. Valid by construction for any input (Hypothesis property tests).
- `invariants`: Euler characteristic, boundary-corrected defect law residual, topological charge, Weaire sum rule residual, T1 charge delta.
- `geometry`: endpoint-fixed Laplacian smoothing, arclength, curvature, left normals, polygon area with holes, perimeter, centroid, pixel-count area, vertex refinement. Removes the staircase bias of pixel-count lengths (diagonal boundary: +41.4 % raw, +0.05 % smoothed; disk perimeter within 0.24 %).
- `events`: `t1_exchange`, `contact_death`, `contact_birth`, `divide`, `extrude`, `nucleate_gap`, `rupture`, `reseal` with preconditions, exact rewrite, validation, and asserted `(dV, dE, dF)`; provenance records events and division lineage.
- `dual`: multigraph and simple-graph projections, `clique_vertex_report` separating dual-graph 3-cliques from those realized by a physical vertex.
- `matching`: faces by Hungarian IoU with split and merge detection, edges by face pair plus trace overlap (pair-level and component-level adjacency), junction vertices by mutual nearest neighbor with incident-set and cyclic-order checks.

### Fields (`pimorph.fields`)
- Oriented strip sampling `P_e(s, r, c)` with a fixed left-side sign convention, arclength-binned `EdgeProfile` (occupancy, continuity, segments, width, left and right intensity), pooled Otsu threshold, `weighted_layer`, legacy `AJ_*` feature frame and heuristic morphology labels (documented as feature-derived states).

### Inference (`pimorph.infer`)
- `ProposalMaps` and `ClassicalProposer` (Sato ridge boundary, LoG nuclear seeds or distance-based seeds, tissue mask, gap map, local noise sigma, data-driven scale estimation).
- `ConstrainedDecoder`: marker-controlled watershed on the boundary map with gap exclusion, small-region handling, exact complex output, merge and split moves; cells labelled by master seed index.
- `energy`: `E = E_image + E_curve + E_seed + E_vertex + E_prior` with `EnergyWeights`; only representation validity is hard, biological terms are soft and zeroable.
- `renderer`: per-face, per-edge, per-vertex density fitting, Gaussian PSF rendering, heteroscedastic Gaussian log-likelihood with fitted gain and read noise, `boundary_interior_ratio`.
- `posterior`: hypothesis grid plus local moves, `PosteriorEnsemble` with softmax weights at the temperature that reaches a target effective sample size, contact and cell probabilities keyed by seed index, vertex credible radius, expectation, variance, credible intervals.
- `cellpose_sam`: Cellpose-SAM baseline wrapper (cellpose >= 4, device auto).
- `neural`: six-channel `MultiHeadUNet` (boundary, distance, seed, vertex, gap, log_sigma), `TileDataset` with channel dropout and field-wise splits, `MultiHeadLoss` (weighted BCE, focal, heteroscedastic L1, soft clDice), training loop with AMP, cosine schedule, checkpoints and JSONL log (`python -m pimorph.infer.neural.train`), consensus pseudo-labels with ignore masks, `NeuralProposer` with tiled inference. Two checkpoints committed via LFS (`models/pimorph_proposals_v0_synth.pt`, `models/pimorph_proposals_v0_mixed.pt`) with model cards; trained on AWS g5.xlarge (about 7 GPU-hours total). Held-out synthetic tiles through the constrained decoder: adjacency F1 0.89 (classical 0.45), vertex F1 0.73 to 0.76 (0.50), PQ 0.89 (0.55). Endothelial ground truth added (`haec`: Zenodo 4898011 HAEC GFP + Hoechst, 434 fields; `mcellseg`: 200 expert DIC HUVEC/HEK masks); `v2_endo` checkpoint fine-tuned on their train splits reaches held-out HAEC PQ 0.407 / boundary F1 0.842 vs Cellpose-SAM 0.347 / 0.707 (`docs/ENDOTHELIAL_RESULTS.md`).

### Synthetic data (`pimorph.synth`)
- Lloyd-relaxed anisotropic Voronoi tissues with boundary jitter, tricellular and bicellular gap carving, nuclei placement; forward rendering of junction, membrane, and nuclei channels with PSF, flat field, Poisson and read noise, broken junctions; exact targets and `make_dataset` (`pimorph synth`).

### Metrics and benchmarks (`pimorph.metrics`, `pimorph.bench`)
- `structural_metrics`: adjacency P/R/F1 with multiplicity, vertex localization, incident-set and cyclic-order accuracy, PQ, AP50/AP75, VI, boundary F1 and Hausdorff, validity fraction, Euler residual, approximate complex edit distance, legacy 4-neighbor adjacency F1 for comparison.
- `calibration`: ECE, Brier, log score, coverage, risk-coverage curve, reliability plot. `sensitivity`: posterior mean, sd, and credible interval of reticular fraction, all-reticular 3-clique fraction, area-degree Spearman.
- `pimorph benchmark` over cornea, NuInsSeg, mCellSeg, LIVECell, and synthetic tiles with `classical`, `cellpose_sam`, and `gt` methods; first tables in `runs/pimorph_bench/` (synth 24 tiles, LIVECell 6 images, cornea 6 crops).

### I/O (`pimorph.io`)
- Calibrated TIFF reading (OME-XML, TIFF and ImageJ resolution tags, manifest fallback, `units = "px"` when unknown), manifests with explicit `geometry`, `nuclei`, `junction` roles and `geometry_source`, tables for cells, interfaces, vertices, gaps, and provenance (CSV or parquet), legacy `cells.csv`, `edges.csv`, GraphML and JSON adapters for `scripts/harden_network_stats.py` and the labeler.

### CLI
- `pimorph version | validate | reconstruct | benchmark | synth`; subcommands import lazily so torch and cellpose are never required for the CLI to start.

### Legacy package and data (Phase 0)
- `scripts/harden_network_stats.py`: "triangle" renamed to "3-clique" (`all_reticular_3clique_pct`, `n_3cliques`; the old keys and `compute_per_image_triangle_stats` remain as deprecated aliases, to be removed in v1.2), conditional permutation null preserving each image's reticular-edge count (`--n-perm`, `--seed`), optional `--complex-dir` reporting the fraction of 3-cliques realized by a true vertex.
- `endopigraph.config.resolve_channel_roles`: explicit `channels: {geometry, nuclei, junction}` block, `pixel_size_um`, a warning when the geometry channel is a junction channel, and `geometry_source` written to run provenance; `examples/config_sbiad1540.yaml` updated.
- `scripts/make_ve_strat_paired_manifest.py` and `data/ve_strat/manifest_paired.csv`: VE-strat `w2` is VE-cadherin and `w4` is nuclei; the original manifest had this backwards, which is why v1.0 found zero contacts there.
- `NETWORK_DISCOVERIES.md` and `README.md`: measured graph quantities separated from hypothesized biology; 3-clique wording.

### Tests and infrastructure
- `tests/pimorph/`: extraction and property tests, events, fields, inference, matching, metrics, synth, I/O, neural (torch tests skipped when torch is absent). Legacy tests unchanged.
- `infra/aws/`: `provision.sh`, `push_code.sh`, `launch.sh` (spot then on-demand), `bootstrap.sh`, `cleanup.sh` (tag-scoped, bucket kept unless `--delete-bucket`), `aws_log.md`. Named profile only; no credentials in the repository.

### Not in this release
- Temporal tracking or event inference on movies, mechanochemical fitting, TJ/GJ/transcellular modules, curved or 3-D surfaces, labeler changes, a trained neural proposal checkpoint, mCellSeg benchmark numbers.

## v1.0.0 — 2026-04-27

First Zenodo-archivable release accompanying the PiMorph manuscript submission.

### Pipeline
- Three input modes: mask-driven, segmentation (Cellpose), and hybrid (nuclei seeds + watershed).
- Conservative contact inference with contact-length filter; precision-first design.
- Per-interface VE-cadherin feature extraction (occupancy, intensity stats, fragmentation, skeleton topology, thickness proxy, complexity score).
- Optional Golgi-nucleus polarity quantification with per-image R and signed V.
- AJMORPH morphology-state labels via unsupervised GMM clustering with BIC-selected k; bootstrap-resampled Adjusted Rand Index for stability.
- Blur-robust mode using stable feature subset (occupancy, skeleton length, intensity summaries).

### Validation suite
- Adjacency benchmarks on three modalities: LIVECell (phase-contrast, F1 = 78.4%), NuInsSeg (H&E, F1 = 82.2%), cornea cells (specular, F1 = 93.1%).
- Junction Mapper capability comparison.
- Robustness suite: 28/30 metric-perturbation pairs stable; intensity scaling perfectly absorbed; blur destabilizes only fragmentation features.
- Shear-stress demonstration on S-BIAD1540 EGM2 (95 fields of view, 22,175 interfaces).
- Manual-annotation infrastructure for AJMORPH validation (`scripts/sample_ajmorph_annotation.py`, `scripts/analyze_ajmorph_annotations.py`).
- Mixed-effects sensitivity analysis for batch and density confounds using biological replicate (3 reps, fully crossed with condition) as random intercept (`scripts/mixed_effects_sensitivity.py`); all three confirmed shear effects survive at p<0.002.

### Reproducibility
- 107 unit tests; CI on GitHub Actions.
- All randomness seeded; run metadata exported.
- Validation data bundle (Desktop, ~19 MB) contains all CSVs, JSONs, GraphML, reports.

### Known limitations
- Pipeline assumes confluent monolayers; non-confluent cultures (e.g., VE-strat) produce zero edges by design.
- AJMORPH labels are feature-derived states, not expert-validated biological classes; expert annotation infrastructure shipped, annotations themselves not yet collected.
- Polarity comparison limited to 5 fields per condition; descriptive only.
- High-shear (18-20 dyn cm⁻²) regime appears distinct from 6 dyn cm⁻² rather than a monotonic continuation.

### Citation
See `CITATION.cff`. Software DOI: [10.5281/zenodo.19831621](https://doi.org/10.5281/zenodo.19831621).
