# PiMorph architecture (v1.1, in development)

PiMorph (`src/pimorph/`) is the canonical package for endothelial cell-complex inference. It replaces the pixel-contact adjacency of the legacy `endopigraph` package with an exact embedded cell complex and adds proposals, a constrained decoder, an energy, and a posterior ensemble. The legacy package stays in place and is fed through adapters (`pimorph.io.legacy`), so `scripts/harden_network_stats.py` and the labeler keep reading `cells.csv` and `edges.csv`.

This document describes what exists in the code. Everything described here is implemented and tested under `tests/pimorph/` unless it is listed in the last section, "Not implemented yet".

## The five layers

| Layer | What it holds | Modules |
|---|---|---|
| 1. pixels | Multichannel microscopy with physical calibration and explicit channel roles | `io/images.py`, `io/manifest.py` |
| 2. geometry | An embedded half-edge cell complex K: faces (cells, gaps, outer), one edge per connected interface component, multicellular vertices, subpixel traces | `complex/halfedge.py`, `complex/extract.py`, `complex/incidence.py`, `complex/geometry.py`, `complex/invariants.py`, `complex/dual.py`, `complex/matching.py` |
| 3. state | Vector-valued molecular fields on edges, half-edges, and cells, sampled from the images along the complex | `fields/strip.py`, `fields/profile.py`, `fields/functionals.py` |
| 4. dynamics | Admissible topology rewrites with exact preconditions and (dV, dE, dF) checks | `complex/events.py` |
| 5. function | Prediction targets that other measurements must validate; nothing here is asserted by the code, only recorded as hypotheses | `NETWORK_DISCOVERIES.md` (measured vs hypothesized), `metrics/sensitivity.py` |

Layer 1 reads pixel size from OME-XML, then TIFF or ImageJ resolution tags, then the manifest, and otherwise leaves it `None` (`io/images.py:pixel_size_from_tiff`). Layer 2 is exact: any integer label image yields a valid complex by construction (`extract_complex`). Layer 3 samples oriented strips `P_e(s, r, c) = I_c(gamma_e(s) + r n_e(s))` along each interface and reduces them to arclength profiles `z_e(s)` and scalars. Layer 4 is a library of operations on the data structure, not a tracker. Layer 5 has no inference code in this release; it is the place where downstream biology gets stated as a hypothesis with the uncertainty produced by layers 2 to 4 attached.

## Hard constraints versus soft priors

Only representation validity is hard. `pimorph.complex.incidence.validate` returns a `ValidationReport` and asserts:

- `B1 @ B2 == 0` identically (oriented boundary matrices, outer face included);
- every edge has exactly two face incidences (`abs(B2).sum(axis=1) == 2`);
- every half-edge lies on exactly one face loop and its left face agrees with `edge_faces`;
- vertex links are consistent: around each vertex, the face left of an outgoing half-edge equals the face right of the next half-edge counter-clockwise;
- the Euler residual `V - E + F - (1 + c)` is zero, `c` being the number of connected components of the 1-skeleton.

Everything biological is a soft energy term in `infer/energy.py` with a weight in `EnergyWeights` that can be set to zero:

| Term | Function | What it encodes | Default weight |
|---|---|---|---|
| `E_image` | `renderer.render_log_likelihood` (negated) | The complex, rendered through a PSF with per-structure densities, explains the geometry channel | `image=1.0` |
| `E_curve` | `curve_energy` | Arclength-weighted `-log b` along cell-cell edges plus `beta * curvature^2` | `curve=1.0`, `curvature_beta=0.02` |
| `E_seed` | `seed_energy` | About one seed (nucleus) per cell: penalty 1 for none, `k - 1` for `k >= 2` | `seed=0.5` |
| `E_vertex` | `vertex_energy` | Fraction of regular vertices whose degree is not 3 | `vertex=0.2` |
| `E_prior` | `shape_prior_energy` | Fraction of cells that are log-area outliers (robust z-score above 2.5 in magnitude) or elongated beyond aspect 6 | `prior=0.2` |

`tests/pimorph/test_infer.py::test_energy_breakdown_and_soft_priors` verifies that setting `image`, `seed`, `vertex`, and `prior` to zero leaves only the curve term. Binucleation, mitosis, wounds, and sprouts are therefore representable; they cost energy under the defaults but are never forbidden.

## The inference chain

```
image channels
   |  ClassicalProposer (infer/proposals.py)  or  NeuralProposer (infer/neural/proposer.py)
   v
ProposalMaps: boundary, seed, tissue, gap, sigma, seed_points, seed_scores [, vertex, distance]
   |  ConstrainedDecoder.decode (infer/decoder.py)
   v
label image -> extract_complex -> smooth_complex        (valid by construction)
   |  complex_energy (infer/energy.py)
   v
Hypothesis(labels, cx, energy, breakdown, params)
   |  generate_hypotheses + PosteriorEnsemble.from_hypotheses (infer/posterior.py)
   v
weights w = softmax(-E / T), T chosen so the effective sample size >= ess_min (default 4)
   |
   v
contact_probabilities, cell_probabilities, vertex_credible_radius, expectation / variance / credible_interval of any statistic
```

Proposals. `ClassicalProposer` builds the boundary map from a Sato ridge filter on the geometry channel mixed with normalized intensity, seeds from LoG-smoothed nuclear maxima (or distance-to-boundary maxima when there is no nuclei channel), a permissive tissue mask, a gap map (far from seeds, dark, ridge-free), and a local noise scale from the MAD of a Laplacian residual. With a nuclei channel it estimates the nucleus radius and derives the other scales from it (`auto_scale`). `NeuralProposer` produces the same `ProposalMaps` container from a trained `MultiHeadUNet` checkpoint, so the decoder and posterior do not know which proposer ran (`ProposalMaps.source` records it).

Decoder. `ConstrainedDecoder` selects seeds by score threshold, floods a marker-controlled watershed on the boundary map (optionally smoothed, gamma-scaled, or mixed with distance to the nearest seed) inside the tissue mask minus pixels with `gap >= gap_threshold`, fills enclosed background smaller than `min_gap_area_px`, merges regions smaller than `min_cell_area_px` into their longest-contact neighbor, and calls `extract_complex`. Cell label = master seed index + 1. Two local moves exist: `merge_faces` (two seeds are one cell) and `split_face` (re-flood one cell from its seed plus an extra seed).

Energy and posterior. `generate_hypotheses` runs a grid over seed threshold, seed drop fraction, boundary smoothing, distance mixing, and gap threshold, deduplicates pixel-identical label images, then applies merge moves across the weakest-supported cell-cell edges and split moves on cells that contain a second seed candidate or are unusually large. `PosteriorEnsemble.from_hypotheses` sorts by energy, optionally keeps the top k, and sets the temperature by bisection so the effective sample size reaches `ess_min`.

How contact and vertex probabilities are defined. All hypotheses share one master seed list, so a cell is identified by its seed index across hypotheses and a contact by the unordered pair of seed indices. `contact_probabilities()` sums the weights of the hypotheses in which a pair shares at least one cell-cell edge. `vertex_credible_radius()` groups vertices by their incident cell set (three or more cells), and reports the posterior existence probability, weighted mean position, and RMS radius. No pixel-level matching between hypotheses is needed.

## Measured versus hypothesized

The code separates quantities it measures from interpretations it does not test. `pimorph.complex.dual.clique_vertex_report` distinguishes dual-graph 3-cliques (three pairwise contacts) from 3-cliques realized by a physical vertex incident to all three cells; the honeycomb test asserts they coincide there, and on real fields they need not. `fields/functionals.py` labels the heuristic morphology classes as "feature-derived states" that have not been benchmarked against expert annotation. `NETWORK_DISCOVERIES.md` lists the three surviving shear-stress findings as measured graph quantities and lists their biological readings (junction maturation, adhesion, barrier function, tricellular hotspots) under "hypothesized". `metrics/sensitivity.py` recomputes the reticular fraction, the all-reticular 3-clique fraction, and the area-degree Spearman per hypothesis so that reconstruction uncertainty can be attached to those numbers.

## Identifiability caveats made explicit in code

- `geometry_source`. When the geometry channel is the junction channel (S-BIAD1540 has GM130, DAPI, VE-cadherin and no membrane marker; VE-strat has VE-cadherin and nuclei), scoring junction continuity along boundaries found with the same signal is circular. `io/manifest.py` records `geometry_source = "junction_channel"` in that case, `endopigraph/config.py` warns and writes it into run provenance, and `pimorph reconstruct` copies it into `complex.provenance`.
- Units. Geometry is stored in pixels. Micrometer columns are filled only when `pixel_size_um` is known; `io/schema.py` writes `units = "px"` into provenance otherwise and never guesses.
- Cornea ground truth. The cornea labels are semantic (interior, border, background) and the border class does not close cells, so instances are derived by a distance-transform watershed (`bench/datasets.py:semantic_to_instance_watershed`). The loader marks `gt_kind = "derived_from_semantic_oversegmented"` and `instance_metrics_reliable = False`. Face, adjacency, and vertex metrics against that GT are not meaningful.
- LIVECell and mCellSeg ground truth. Polygon annotations rasterized per cell leave 1 to 3 px background slivers where cells meet; `fill_gt_slivers` assigns enclosed background below 12 px to the neighboring cell so that vertices are cell-cell-cell rather than cell-cell-gap (`gt_sliver_fill_px = 12` in `meta`).
- VE-strat channels. The original `data/ve_strat/manifest.csv` listed only the `w4` file and called it VE-cadherin; `w4` holds nuclei and `w2` holds the junction mesh (`scripts/make_ve_strat_paired_manifest.py`). This is why the legacy run found zero contacts. `data/ve_strat/manifest_paired.csv` pairs both files per site with `geometry_source = junction_channel`.

## Module map

| Module | Role |
|---|---|
| `pimorph/__init__.py` | Version and layer summary |
| `complex/halfedge.py` | `HalfEdgeComplex` container, `FaceKind`, `VertexKind`, rotation system, `he_next` |
| `complex/extract.py` | `extract_complex`: label image to complex via the crack graph; `rebuild_topology` |
| `complex/incidence.py` | `boundary_matrices` (B1, B2), `validate`, `euler_residual` |
| `complex/invariants.py` | Euler characteristic, defect law residual, topological charge, Weaire sum rule residual |
| `complex/geometry.py` | Endpoint-fixed Laplacian smoothing, arclength, curvature, normals, face area, perimeter, centroid, vertex refinement |
| `complex/events.py` | `t1_exchange`, `contact_death`, `contact_birth`, `divide`, `extrude`, `nucleate_gap`, `rupture`, `reseal` |
| `complex/dual.py` | `to_multigraph`, `to_simple_graph`, `clique_vertex_report` |
| `complex/matching.py` | `match_faces` (Hungarian on IoU), `match_edges`, `match_vertices` |
| `fields/strip.py` | Oriented strip sampling with `map_coordinates` |
| `fields/profile.py` | `EdgeProfile`, arclength-binned occupancy, continuity, width, left/right intensity, Otsu threshold |
| `fields/functionals.py` | `weighted_layer`, legacy `AJ_*` feature frame, heuristic morphology labels |
| `infer/proposals.py` | `ProposalMaps`, `ClassicalProposer`, noise and scale estimators |
| `infer/decoder.py` | `DecoderParams`, `ConstrainedDecoder`, merge and split moves |
| `infer/energy.py` | `EnergyWeights`, `complex_energy` and its soft terms |
| `infer/renderer.py` | `RenderModel`, density fitting, rendering, heteroscedastic log-likelihood, `boundary_interior_ratio` |
| `infer/posterior.py` | `Hypothesis`, `PosteriorEnsemble`, `generate_hypotheses`, `temperature_for_ess` |
| `infer/cellpose_sam.py` | `CellposeSAM` wrapper (cellpose >= 4, device auto) |
| `infer/neural/` | `MultiHeadUNet`, `TileDataset`, `MultiHeadLoss`, `train`, `make_pseudolabel_tiles`, `NeuralProposer` (see `NEURAL_PROPOSALS.md`) |
| `synth/tissue.py` | Lloyd-relaxed anisotropic Voronoi tissues with gaps, jitter, nuclei |
| `synth/render.py` | Junction, membrane, and nuclei channels with PSF, flat field, Poisson and read noise, broken junctions |
| `synth/targets.py` | Exact targets (boundary, signed distance, seed, vertex, gap, outer) and `make_dataset` |
| `metrics/structural.py` | `structural_metrics`, PQ, AP, VI, boundary F1 and Hausdorff, legacy adjacency F1 |
| `metrics/calibration.py` | ECE, Brier, log score, coverage, risk-coverage curve, reliability plot |
| `metrics/sensitivity.py` | Posterior mean, sd, and credible interval of network statistics |
| `bench/datasets.py` | Loaders: cornea, nuinsseg, mcellseg, livecell, synth |
| `bench/run.py` | `run_benchmark`, summary tables (see `BENCHMARKS.md`) |
| `io/images.py` | `CalibratedImage`, `read_tiff`, `stack_channel_files` |
| `io/manifest.py` | `FieldSpec`, `parse_manifest` with explicit channel roles |
| `io/schema.py` | `complex_tables` (cells, interfaces, vertices, gaps, provenance) |
| `io/legacy.py` | `cells.csv`, `edges.csv`, GraphML and JSON for the legacy consumers |
| `cli.py` | Subcommands `version`, `validate`, `reconstruct`, `benchmark`, `synth` |

The CLI registers subcommands lazily so that importing it does not pull in torch or cellpose. Training and pseudo-labelling are reached through `python -m pimorph.infer.neural.train` and `pimorph.infer.neural.pseudolabel.make_pseudolabel_tiles`; there is no `pimorph train` subcommand.

## Not implemented yet, and where the hooks are

- Temporal tracking and event inference on movies. `complex/events.py` implements each rewrite as an operation with preconditions, an exact rewrite, `validate()`, and an expected `(dV, dE, dF)`; `complex/matching.py` matches two complexes of the same shape. Neither is wired to a time series.
- Mechanochemical parameter fitting. There is no vertex-model or tension inference. `fields/profile.py` gives per-edge `z_e(s)` and `functionals.weighted_layer` turns any per-bin functional into an edge weight, which is where such a model would read its inputs.
- Tight junction, gap junction, and transcellular modules. `fields/strip.py` and `profile.py` accept any channel; only the adherens junction heuristics of the legacy package are wrapped in `functionals.py`.
- Curved and three-dimensional surfaces. `extract_complex` requires a 2-D label image.
- Changes to the labeler web app. `io/legacy.py` writes the columns the labeler already reads.
- A trained neural proposal checkpoint. The model, data, losses, training loop, and proposer exist and are tested with a tiny smoke run; no checkpoint is committed under `models/` (see `NEURAL_PROPOSALS.md`).
