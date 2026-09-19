# Later blueprint phases: dynamics, mechanics, 3-D, multi-junction fields, function

Status as of 2026-09-18. Each phase is a separate subpackage with its own tests; each was
built and evaluated against the data that actually exists, and every table below states
what was validated and what was not. Numbers come from the run directories named in each
section.

## Dynamics: tracking, exact event detection, admissibility (`pimorph.dynamics`)

- `tracking.link_frames` matches cells between consecutive label images by IoU with the
  Hungarian assignment and flags split, merge and division candidates; `tracking.track`
  builds track ids and a lineage over a stack; `complexes_over_time` extracts one complex
  per frame; `complex_to_labels` rasterizes a complex back to labels (exact round trip on
  crack-aligned complexes).
- `events_detect.snapshot` records the topology of one complex (cells, contacts, gaps,
  vertex incidence sets, V, E, F). `detect_events` compares two snapshots plus the lineage
  and emits typed events: `t1`, `division`, `extrusion`, `death_to_gap`, `contact_birth`,
  `contact_death`, `gap_nucleation`, `gap_closure`, `rupture`, `reseal`, and bookkeeping
  kinds for cells that enter or leave the image. Every event carries the exact
  `(dV, dE, dF)` of its rewrite in `pimorph.complex.events`; `admissibility_check` sums
  them and compares with the observed change, so a frame pair is "fully explained" only
  when the detected events account for the whole topological change.
- Loaders: Cell Tracking Challenge (`ctc.py`, silver-truth masks relabelled with gold
  track ids) and TissueMiner (`tissueminer.py`, decodes the tracked RGB label movie and
  reads the SQLite database so the same detector runs on the database topology).
- Tests (`tests/pimorph/test_dynamics.py`, 16): each event kind is generated with the
  `pimorph.complex.events` rewrite, rasterized and re-detected; residual 0 and charge
  delta 0 for isolated T1s; 5-frame tracking with 0 id switches.

Measured (`runs/dynamics/`):

| Data | Result |
|---|---|
| TissueMiner demo (Drosophila pupal wing, 71 frames, ~633 cells/frame, 27 divisions) | T1 detection against the database with ground-truth ids: precision 0.92, recall 0.90, F1 0.91 (ROI cells); with our IoU tracker 0.85 / 0.87 / 0.86. 761 of 761 isolated T1s conserve topological charge with the generic (-1, -1, +1, +1) side pattern. Image-derived contacts match the database exactly on the 531 common cells of frame 0. |
| TissueMiner divisions | ground-truth ids precision 0.64 / recall 1.00; the IoU tracker recall 0.96 but precision 0.19 (fast tissue flow, per-frame self-IoU often below 0.5; 685 id switches over 1,313 tracks). A motion-compensated linker is the open item. |
| CTC DIC-C2DH-HeLa 01 (84 frames, 8 divisions) | division precision 1.00 / recall 0.875 / F1 0.93 with our tracker; 1.00 / 1.00 with ground-truth ids; 18 id switches over 37 tracks. Sparse culture: no confluent regime, admissibility flags almost every frame pair (silver-truth 1 px holes and contacts between disconnected clusters). |

EpiCure curated movies (Zenodo 20607705, `pimorph.dynamics.epicure`, curated track ids, no
tracker; `runs/dynamics/epicure_movie{2,3}/`):

| Movie | Frames | Events | Isolated T1 charge check | Frame pairs fully explained |
|---|---|---|---|---|
| movie2, Drosophila abdomen histoblasts, 213x213, 0.275 um/px | 30 (167 to 109 cells) | t1 559, contact_birth 203, contact_death 122, extrusion 94, division 30 | 59 / 59 conserved and generic | 5 / 29 (5 / 9 of the pairs without free-edge events) |
| movie3, zebrafish telencephalon, 628x548 | 11 (173 to 196 cells) | division 33, t1 24, gap_nucleation 15, rupture 11, reseal 10 | 13 / 13 | 1 / 10 |

Every one of the ten largest unexplained residuals is a free-edge effect: only the histoblasts
are labelled, so cells at the edge of the annotated region enter, leave or flicker, and edge
"extrusions" close with one to three new contacts instead of one vertex. The admissibility
identity is doing its job (it flags exactly those pairs); a ROI-aware event grammar for
partially annotated tissue is the open item. EpiCure divisions are heuristic (no lineage file
was parsed).

Not validated: event detection on a confluent endothelial time lapse (none with truth is public);
the identity is exact, the linker is the weak part.

## Mechanics: vertex model and force inference (`pimorph.mechanics`)

- `vertex_model`: polygon vertex model on a complex (area elasticity `K_A`, perimeter
  elasticity `K_P`, per-edge line tensions), analytic forces, Armijo-backtracked
  Barzilai-Borwein relaxation with pinned boundary vertices, rejection of steps that flip
  a loop or change a cyclic order, Laplace-implied circular arcs written to `edge_smooth`.
- `force_inference.infer_tensions`: vertex force balance (chord tangents plus the shoelace
  pressure term) with optional Laplace rows `p_L - p_R = gamma kappa` from fitted edge
  curvatures, ridge regularization, identifiability report (rank, null space, gauge).
  `inferred_stress_tensor` gives the per-cell Batchelor stress.
- Tests (`tests/pimorph/test_mechanics.py`, 12).

Synthetic recovery (`runs/mechanics/REPORT.md`, 9 seeds, 60-cell Voronoi sheets, truth =
effective tension, all inferred values relative):

| Tension dispersion | Vertex noise | Ridge | Pearson (tension) | Pearson (pressure) |
|---|---|---|---|---|
| 0.3 | 0 px | 1e-3 | 1.000 | 1.000 |
| 0.6 | 0 px | 1e-3 | 0.999 | 1.000 |
| 0.3 | 0.5 px | 0.1 | 0.88 | 0.94 |
| 0.6 | 0.5 px | 0.1 | 0.80 | 0.86 |
| 0.3 | 1.0 px | 0.1 | 0.66 | 0.83 |

Without curvature rows the pressures are underdetermined on these sheets (182 equations vs
236 unknowns); tension-only inference reaches Spearman 0.74. On the real VE-strat field
Control_s1 (541 cells, 1,528 relative tensions) the vertex-balance residual is large
(RMS 0.38 tension units, 2% of interior vertices balanced within 5%), so the table in
`runs/mechanics/ve_strat_Control_s1_tensions.csv` is a diagnostic of how poorly a static
force balance describes that reconstruction, not a measurement. No absolute scale, no
validation against laser ablation or traction data.

## 3-D cell complex (`pimorph.complex3d`)

- `extract_complex3d(labels)` builds the crack complex of a 3-D label volume on a doubled
  grid: 3-cells (cells, enclosed gaps, one outer cell), interfaces (2-cells) between exactly
  two 3-cells, triple lines (1-cells) and 0-cells at quadruple points and dead ends, with the
  same artificial-vertex convention as the 2-D extractor for closed loops and closed
  interfaces.
- `boundary_matrices3d` returns sparse `B1, B2, B3`; `B1 B2 = 0` and `B2 B3 = 0` hold
  exactly because the coefficients are grouped fine-level boundary sums. `validate3d`
  checks the algebra, two-cell interfaces and the Euler residual against the exact cubical
  Euler characteristic of the occupied set (handles are flagged, not hidden).
- `per_cell_euler` (coarse from the complex, or exact from the voxel mask: 2 for a ball,
  0 for a torus, 4 for a ball with a cavity), `neighbor_counts`, `contact_multiplicity`.
- `surface_complex_from_labels(labels3d, lumen_label)` projects a curved monolayer around a
  lumen to a 2-D-style face/edge/vertex complex with 3-D vertex positions.
- Tests (`tests/pimorph/test_complex3d.py`, 17): hand cases (2x2x2 block with its degree-6
  quadruple point, two cubes, enclosed sphere, torus, hollow cylinder), watershed Voronoi
  volumes at 48^3 and 64^3, a hypothesis property test, and a 400-volume random stress run
  with zero algebraic failures. 64^3 extraction takes 0.2 s after the numba compile.

Not validated: no real 3-D microscopy volume with cell truth was available in this campaign.

## Multi-junction vector fields and function (`pimorph.fields.multichannel`, `pimorph.function`)

- `multichannel_profiles` samples every channel on the same resampled strip of each edge,
  giving per-bin occupancy, intensity and width per channel and per-edge co-occupancy,
  Jaccard, exclusive fractions and a discrete vector state (which markers are present).
  `joint_morphology_states` fits a Gaussian mixture over the per-edge features, picks the
  number of states by BIC and reports bootstrap stability (adjusted Rand index).
- `function.transport`: interface conductances from the junction fields, a sparse
  resistor network over cells and gaps (gaps as short circuits), effective in-plane
  conductance by a Dirichlet solve, adjoint sensitivities (checked against finite
  differences at 1e-4), Rayleigh monotonicity checked over 50 random trials.
  `function.barrier.barrier_report` returns the permeability index with
  `validated: False` and the note that no TEER or tracer data exists for these fields.
- `io.sbiad1169`: BioStudies file list, balanced resumable download, channel roles from
  the file-list attributes (`_c2` VE-cadherin, `_c3` F-actin, `_c4` claudin-5, `_c5`
  DAPI), scale from the burned-in 50 um bar (0.132 um/px; the TIFF tag is print dpi).
- Tests: `tests/pimorph/test_multijunction.py` (10), `tests/pimorph/test_transport.py` (10).

Measured on S-BIAD1169 (dermal microvascular endothelium, 17 unique fields, 267 cells,
546 cell-cell edges; `runs/multijunction/REPORT.md`):

| Statistic (arclength-weighted field mean) | DMSO (n = 3) | BRAFi (n = 14) | Mann-Whitney p |
|---|---|---|---|
| VE-cadherin (AJ) coverage | 0.414 | 0.394 | 0.68 |
| Claudin-5 (TJ) coverage | 0.376 | 0.358 | 1.00 |
| AJ/TJ co-occupancy | 0.248 | 0.235 | 0.68 |
| AJ/TJ Jaccard | 0.376 | 0.389 | 0.86 |
| Vector state TJ + actin (no AJ) | 0.057 | 0.038 | 0.021 |
| Effective conductance G_eff (prediction, unvalidated) | 981 | 1040 | 0.43 |

Joint states: BIC picked 4 states (sparse, AJ-dominant with actin, AJ+TJ co-junction,
dense triple-positive); the dense state is 17% of DMSO edges vs 11% of BRAFi edges. With 3
control fields and 21 tests without correction, nothing here is a finding; the pipeline is
what was delivered. Images are 8-bit display exports, so intensities are not comparable
across fields and thresholds are per-field Otsu.
