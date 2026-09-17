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

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"           # core + tests
pip install -e ".[cellpose]"      # optional: deep learning segmentation
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
