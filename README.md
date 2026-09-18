# EndoPiGraph-AJmorph v1

EndoPiGraph-AJmorph v1 is a validated tool for building **typed endothelial contact graphs** ("pi-graphs") and extracting **adherens junction (AJ) morphology features** from fluorescence microscopy.

It is designed to run on BioImage Archive datasets (e.g. **S-BIAD1540**) and to produce:

- instance segmentation masks (cells)
- a cell-cell contact graph (neighbors)
- edge attributes for junction markers (AJ/TJ/GJ/NJ if present)
- AJ morphology features per interface (occupancy, cluster density, etc.)
- data-driven junction morphology classification (GMM clustering)
- publication-ready QC figures and a lightweight HTML report

---

## Validation

Adjacency extraction validated on three independent datasets with ground-truth instance masks:

| Dataset | Modality | Images | F1 | Precision | Recall |
|---------|----------|--------|----|-----------|--------|
| **LIVECell** | Phase-contrast | 50 | **78.4%** | 98.0% | 67.4% |
| **NuInsSeg** | H&E histopathology | 80 | **82.2%** | 93.8% | 74.9% |
| **Cornea Cells** | Specular microscopy | 160 | **93.1%** | 99.7% | 87.4% |

EndoPiGraph outperforms competing approaches on the Cornea Cells benchmark:

| Method | F1 | Time (160 images) |
|--------|----|----|
| **EndoPiGraph** | **93.1%** | **10.4s** |
| Delaunay + verify | 80.1% | 2.7s |
| Dilation (2px) | 46.7% | 128.4s |
| Centroid distance | 35.6% | 4.3s |

Additional validation:
- **Blur robustness**: 93.3% label consistency under 1-2px Gaussian blur (vs 46.9% Junction Mapper)
- **Network statistics**: 3 network-level findings (per-image replicate testing: Mann-Whitney U, bootstrap CIs). These are measured graph quantities (reticular edge fraction, all-reticular 3-clique fraction, area-degree correlation); their biological interpretation (junction maturation, adhesion, barrier function) is hypothesized, not validated. See `NETWORK_DISCOVERIES.md`.
- **Unit tests**: 107 tests covering all modules (pytest + CI)

---

## PiMorph complex (v1.1, in development)

`src/pimorph/` is the new canonical package (import `pimorph`, CLI `pimorph`). It replaces the 4-neighbor
pixel-contact adjacency of `endopigraph` with an exact embedded cell complex and adds uncertainty-aware
inference. The legacy package stays and is fed through adapters: `pimorph.io.legacy` writes the same
`cells.csv`, `edges.csv`, and GraphML that `scripts/harden_network_stats.py` and the labeler read.
Everything below is implemented and tested under `tests/pimorph/`; the linked docs list what is not.

- **Exact half-edge cell complex** from any label image (`pimorph.complex.extract_complex`). Faces are
  cells, enclosed gaps, and one outer face; one edge per connected contact component (two disconnected
  contacts between the same cells are two edges); vertices where 3 or 4 cracks meet, with exact cyclic
  order. `validate()` checks `B1 @ B2 == 0`, two faces per edge, consistent vertex links, and a zero
  Euler residual. This is the only hard constraint in the system. Conventions: [`docs/COMPLEX.md`](docs/COMPLEX.md).
- **Subpixel geometry.** Endpoint-fixed smoothing removes the staircase bias of pixel-count contact
  lengths: a diagonal boundary measures 41 % too long as a crack trace and 0.05 % off after smoothing;
  a disk perimeter is within 0.24 %.
- **Exact identities as QC:** Euler residual, boundary-corrected defect law, Weaire sum rule, topological charge.
- **Event library** (`pimorph.complex.events`): T1 exchange, contact birth and death, division,
  extrusion (T2), gap nucleation, rupture, reseal, each with preconditions and an asserted `(dV, dE, dF)`.
  Operations only; no movie tracking yet.
- **Fields on the complex** (`pimorph.fields`): oriented strip sampling along each interface, arclength
  profiles `z_e(s)` (occupancy, continuity, width, left and right side intensity), and functionals that
  reproduce the legacy `AJ_*` features.
- **Inference chain** (`pimorph.infer`): proposal maps (classical filters or a neural multi-head UNet),
  a constrained watershed decoder that emits a valid complex by construction, an energy whose
  biological terms (one nucleus per cell, trivalent vertices, area and aspect priors) are soft and
  zeroable, and a posterior ensemble of legal complexes with contact probabilities, cell probabilities,
  vertex credible radii, and credible intervals for any statistic. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Structural metrics and benchmarks** (`pimorph.metrics`, `pimorph.bench`): adjacency P/R/F1 with
  multiplicity, vertex localization, incident-set and cyclic-order accuracy, PQ, VI, boundary F1,
  validity fraction, plus ECE, Brier, and risk-coverage curves for probabilities. Current tables and
  their caveats (cornea GT is derived and over-segmented, LIVECell slivers filled at 12 px, the classical
  proposer fails on phase contrast without nuclei): [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- **Synthetic tissues and neural proposals** (`pimorph.synth`, `pimorph.infer.neural`): Lloyd-relaxed
  anisotropic Voronoi sheets with gaps and broken junctions rendered through a PSF and noise model with
  exact targets; a six-channel UNet (geometry, nuclei, junction plus presence indicators) with boundary,
  distance, seed, vertex, gap, and log-sigma heads, trained with `python -m pimorph.infer.neural.train`.
  Three checkpoints are committed (LFS) with model cards under `models/`: a synthetic-only stage 1, a stage 2
  fine-tuned on real pseudo-labels, and `v1_multi`, trained on 8x H100 with real instance ground truth from
  LIVECell and NeurIPS CellSeg. Through the same decoder on held-out synthetic tiles the learned proposals
  reach adjacency F1 0.86 to 0.89 vs 0.37 to 0.45 for the classical filters; on held-out LIVECell phase
  contrast v1_multi reaches 0.23 vs 0.62 for Cellpose-SAM (400 images each; [`docs/SCALE_RESULTS.md`](docs/SCALE_RESULTS.md)).
  On held-out human aortic endothelial fields with instance truth (corrected reference), `v3_endo` reaches
  adjacency F1 0.51, vertex F1 0.26 (95 predicted vs 92 true vertices per field), PQ 0.60 and boundary F1 0.84
  vs 0.31 / 0.04 / 0.47 / 0.71 for Cellpose-SAM ([`docs/ENDOTHELIAL_RESULTS.md`](docs/ENDOTHELIAL_RESULTS.md)).
  See [`docs/NEURAL_PROPOSALS.md`](docs/NEURAL_PROPOSALS.md); GPU training on AWS follows
  [`docs/AWS_RUNBOOK.md`](docs/AWS_RUNBOOK.md) (named profile, tag-scoped cleanup, no keys in the repo).
- **Multicellular vertices against real truth on confluent monolayers** ([`docs/CONFLUENT_BENCHMARK.md`](docs/CONFLUENT_BENCHMARK.md)):
  three new truth sets (`hcec`: manually traced human corneal endothelial monolayers with NCAM + DAPI, 15 fields of
  about 1,700 cells and 3,000 tricellular vertices each; `alizarine`: expert-contoured porcine corneal endothelium;
  `flywing`: E-cadherin Drosophila epithelium). Vertex F1 is 0.96 to 0.99 on alizarine for every method (PiMorph
  fine-tuned 0.990, the best), PiMorph's zero-shot proposals give the best vertex F1 on FlyWing (0.868), and on the
  cultured hCEC monolayer Cellpose-SAM leads zero-shot (0.635 vs 0.334) while a fine-tune on 10 field-disjoint
  training fields (`v4_confluent`) reaches 0.611 on the 5 held-out fields with higher vertex recall and boundary F1
  than Cellpose-SAM. The decoder now
  excludes pixels the signed-distance head places outside every cell (open background was being flooded, which
  produced about 12 false vertices per true one on HAEC) and adds nuclear peaks as seeds where the seed head is silent.
  The search that found these sets, and the negative verdict on public VE-cadherin monolayers with expert masks, is
  in [`docs/DATASET_HUNT_2026-09-18.md`](docs/DATASET_HUNT_2026-09-18.md).
- **Later blueprint phases** ([`docs/LATER_PHASES.md`](docs/LATER_PHASES.md)): `pimorph.dynamics` (IoU/Hungarian
  tracking, exact event detection with summed `(dV, dE, dF)` admissibility; T1 F1 0.91 against the TissueMiner
  database), `pimorph.mechanics` (vertex model, force inference with curvature-pressure rows; tension Pearson 1.00
  noiseless, 0.88 at 0.5 px noise), `pimorph.complex3d` (3-D crack complex with exact `B1 B2 = 0`, `B2 B3 = 0`,
  quadruple points, per-cell Euler, curved-surface monolayers), `pimorph.fields.multichannel` and `pimorph.function`
  (VE-cadherin + claudin-5 + F-actin vector states on S-BIAD1169, resistor-network transport proxy flagged
  `validated: False`).
- **Shear findings re-tested with posteriors on all 102 EGM2 fields** (`NETWORK_DISCOVERIES.md`, top section):
  the reticular-fraction increase at 6 dyn cm^-2 survives (p = 2.6e-9, consistent in all three replicates, high shear
  distinct); the all-reticular 3-clique increase is explained entirely by the reticular fraction (conditional-null
  enrichment z 0.00 vs 0.03, p = 0.95); the area-degree correlation strengthening does not replicate (0.72 vs 0.75,
  p = 0.43). About 95% of graph 3-cliques are realized by a multicellular vertex.

Data notes that changed since v1.0. The VE-strat manifest had the channels backwards (`w2` is
VE-cadherin, `w4` is nuclei), which is why the legacy run found zero contacts;
`data/ve_strat/manifest_paired.csv` pairs both files per site. On S-BIAD1540 and VE-strat the junction
channel doubles as the geometry channel, and outputs record `geometry_source = junction_channel`
because scoring junction continuity along boundaries found with the same signal is circular.
`NETWORK_DISCOVERIES.md` now separates measured graph quantities from hypothesized biology and reports
3-cliques (not "triangles") against a conditional null.

Quickstart:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev,ml,complex]"          # add torch for neural proposals, aws for boto3
pimorph validate --labels path/to/labels.tif   # validity report, defect law and Weaire residuals
pimorph reconstruct --manifest data/ve_strat/manifest_paired.csv --out runs/pimorph_ve_strat --posterior
pimorph benchmark --dataset synth --root data/tiles/synth_val --methods classical --out runs/pimorph_bench
pimorph synth --n 200 --out data/tiles/synth_val --shape 512
pytest tests/pimorph -q                        # torch tests skip when torch is absent
```

---

## Installation

```bash
python -m venv .venv                 # or: uv venv --python 3.12 .venv  (recommended; system Python 3.14 has no torch or cellpose wheels)
source .venv/bin/activate
pip install -e ".[dev]"           # core + tests
pip install -e ".[cellpose]"      # optional: deep learning segmentation
pip install -e ".[complex]"       # pimorph complex extras: shapely, pyarrow, zarr, ome-zarr, hypothesis, numba
pip install -e ".[torch]"         # pimorph neural proposals and Cellpose-SAM: torch, torchvision, cellpose>=4
pip install -e ".[aws]"           # boto3 for the AWS training runbook
```

---

## Quickstart: run on S-BIAD1540

```bash
endopigraph download --accession S-BIAD1540 --out data/raw --method print
endopigraph make-manifest --input data/raw/S-BIAD1540 --out data/manifest.csv
cp examples/config_sbiad1540.yaml config.yaml
endopigraph run --config config.yaml
```

---

## AJmorph features (per interface)

Given a segmentation mask and an AJ marker channel, for each contacting pair of cells `(i, j)`:

- `contact_px` : shared boundary length (pixel units)
- `aj_mean`, `aj_median`, `aj_max`, `aj_std` : intensity statistics
- `aj_occupancy` : fraction of interface pixels above threshold
- `aj_cluster_count` : connected components (+ h-maxima robust variant)
- `aj_skeleton_len`, `aj_skeleton_endpoints`, `aj_skeleton_branch_points` : skeleton topology
- `aj_thickness_proxy` : area / skeleton length ratio
- `aj_complexity_score` : weighted topological complexity

Blur-stable subset (Cohen's d < 0.3): `mean_intensity`, `occupancy`, `median_intensity`

---

## AJ morphology classification

Two approaches available:

**1. Data-driven (recommended):** Gaussian Mixture Model clustering with BIC-selected k, bootstrap stability assessment, and GroupKFold cross-validation:

```python
from endopigraph import cluster_junctions_gmm
edges_df, meta = cluster_junctions_gmm(edges_df, prefix="AJ_", blur_robust=True)
```

**2. Heuristic (legacy):** Threshold-based rules mapping features to classes (straight, thick, reticular, fingers, etc.). Retained for backward compatibility but not recommended for publication.

---

## Running tests

```bash
pytest tests/ -v
ruff check src/ tests/
```

---

## Citation and status

Please credit Okezue Bell (okezue@stanford.edu) and Anthony Bell for this work when used.
