# PiMorph

PiMorph turns a fluorescence image of a cell monolayer into an **embedded cell complex**: every cell,
every gap, every cell-cell contact and every multicellular vertex, with exact incidence, cyclic order,
subpixel geometry and a posterior over the legal reconstructions of the field. It was built for
endothelial monolayers (VE-cadherin, PECAM-1, ZO-1, claudin-5 junctions), where the questions are about
the network (who touches whom, where three cells meet, how the junction protein is arranged along each
contact) rather than about masks. Any label image, from PiMorph's own proposal model or from Cellpose,
becomes a valid complex; the structure and the uncertainty are what PiMorph adds.

Code and small tables live here; checkpoints, complete benchmark outputs and training tiles live on
Zenodo: **[10.5281/zenodo.22839866](https://doi.org/10.5281/zenodo.22839866)** (see [Data](#data)).

## What it does

| Layer | What you get | Where |
|---|---|---|
| Complex | half-edge complex from any label image; faces are cells, enclosed gaps and one outer face; one edge per connected contact; vertices where three or more cracks meet, with cyclic order; `validate()` checks `B1 B2 = 0`, two faces per edge, consistent vertex links, zero Euler residual. Validity 1.0 on about 2,000 real and synthetic fields. | `pimorph.complex`, [`docs/COMPLEX.md`](docs/COMPLEX.md) |
| Geometry | endpoint-fixed smoothing of crack traces (a diagonal boundary measures 41% too long as pixels, 0.05% off after smoothing), subpixel vertex positions, per-cell and per-edge tables | `pimorph.complex.geometry` |
| Inference | proposal maps (neural multi-head UNet or classical filters), a constrained watershed decoder that emits a valid complex by construction, an energy whose biological terms are soft, a posterior ensemble with contact probabilities, vertex credible radii and credible intervals for any statistic | `pimorph.infer`, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Fields | oriented strip sampling along each contact, arclength profiles of junction markers, multichannel vector states (AJ, TJ, actin), junction morphology labels | `pimorph.fields` |
| Dynamics | frame linking, exact event detection (T1, division, extrusion, contact birth and death, gap nucleation and closure, rupture and reseal) with a summed `(dV, dE, dF)` admissibility identity | `pimorph.dynamics`, [`docs/LATER_PHASES.md`](docs/LATER_PHASES.md) |
| Mechanics, 3-D, function | vertex model and force inference; 3-D crack complex with `B1 B2 = B2 B3 = 0` and curved-surface monolayers; resistor-network transport proxy for barrier function | `pimorph.mechanics`, `pimorph.complex3d`, `pimorph.function` |
| Metrics and benchmarks | adjacency P/R/F1, multicellular vertex F1 with localization and incident-set accuracy, PQ, boundary F1, validity, calibration; loaders for nine public truth sets with field-disjoint splits | `pimorph.metrics`, `pimorph.bench`, [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |

## How good is it, honestly

Every number below is on held-out test fields that no model saw in training. The comparison is
symmetric: PiMorph's proposal model (`v6_pool`) and Cellpose-SAM were both fine-tuned on the same
training fields, and Cellpose-SAM masks are scored through PiMorph's own complex extractor, so the
structural metrics mean the same thing in every row. Vertex F1 counts multicellular vertices matched
within 3 px. Full tables and per-field CSVs: [`docs/CONFLUENT_BENCHMARK.md`](docs/CONFLUENT_BENCHMARK.md).

| Held-out test set (truth) | Metric | PiMorph v6_pool | Cellpose-SAM fine-tuned on the same fields | Cellpose-SAM zero-shot |
|---|---|---|---|---|
| hCEC, cultured human corneal endothelial monolayer, NCAM + DAPI, 5 fields, ~3,000 tricellular vertices each (manual tracings) | vertex F1 / adjacency F1 / PQ | 0.680 / 0.867 / 0.816 | **0.695 / 0.937 / 0.862** | 0.635 / 0.830 / 0.790 |
| Alizarine corneal endothelium in situ, 10 fields (expert contours) | vertex F1 / adjacency F1 / PQ | **0.992** / 0.987 / **0.920** | 0.991 / **0.992** / 0.903 | 0.984 / 0.989 / 0.898 |
| FlyWing E-cadherin epithelium, 10 fields (corrected labels) | vertex F1 / adjacency F1 / PQ | **0.875** / 0.911 / 0.780 | 0.852 / **0.966** / **0.793** | 0.849 / 0.966 / 0.795 |
| RPE monolayer, 20 tiles (manual polygons) | vertex F1 / adjacency F1 / PQ | 0.327 / 0.605 / 0.556 | **0.336 / 0.681 / 0.612** | 0.287 / 0.622 / 0.548 |
| HAEC, sub-confluent aortic endothelial culture, 86 fields (semantic truth to instances) | vertex F1 / adjacency F1 / PQ | **0.352** / 0.536 / 0.626 | 0.331 / **0.555 / 0.648** | 0.039 / 0.314 / 0.465 |

Reading it plainly:

- **As a segmenter, PiMorph's proposal model is competitive with a fine-tuned Cellpose-SAM, not better.**
  On the cultured endothelial monolayer the fine-tuned Cellpose-SAM leads on adjacency, PQ and, within
  noise (3 of 5 fields), on vertices; PiMorph keeps the better vertex localization (1.28 vs 1.40 px
  median) and a calibrated vertex count. PiMorph leads vertex F1 on FlyWing and HAEC, ties on alizarine,
  trails on RPE. The zero-shot column is what a user gets without any annotation; both fine-tunes used
  10 hCEC fields.
- **One model versus one model per culture.** `v6_pool` holds its numbers on confluent and sub-confluent
  data at once; the Cellpose-SAM fine-tuned on HAEC collapses on hCEC (vertex F1 0.098) and the one
  fine-tuned on the confluent pool collapses on HAEC (0.018).
- **The structure is the point.** Everything a Cellpose mask acquires in the table (vertices with
  incident sets, contact multiplicity, gaps, validity, credible intervals) it acquires by passing through
  PiMorph. The natural configuration for a new endothelial dataset is therefore a fine-tuned Cellpose-SAM
  or PiMorph proposal, whichever scores higher on a few annotated fields, followed by the PiMorph complex,
  fields, posterior and events.
- **Where vertices are solved and where they are not.** Clean borders in situ: 0.99. Confluent culture
  with weak membrane regions: 0.68 to 0.70 for every method, with the misses concentrated in two of the
  five fields (0.37 to 0.50 there, 0.84 to 0.92 on the other three). Sub-confluent culture: about 0.35,
  bounded by the reference itself deciding at one pixel whether two cells touch.

What the same machinery established on biology (`NETWORK_DISCOVERIES.md`): the three shear-stress
findings of the legacy pipeline were re-tested with 32-hypothesis posteriors on all 102 S-BIAD1540 EGM2
fields. The reticular junction fraction rising at 6 dyn cm^-2 survives (0.112 to 0.207, p = 2.6e-9, all
three replicates, distinct high-shear regime); the all-reticular 3-clique increase is explained by the
reticular fraction alone (conditional-null enrichment z 0.00 vs 0.03, p = 0.95); the area-degree
correlation strengthening does not replicate (0.72 vs 0.75, p = 0.43). Reconstruction ambiguity was
negligible for all three (credible half-widths 0.001 to 0.006).

## What is validated on real data, and what is not

- **Force inference** ([`docs/LATER_PHASES.md`](docs/LATER_PHASES.md), `runs/mechanics_real/`): against
  15 laser-ablation recoils of parasegment-boundary cables (Lang et al. 2019), the inferred tension of the
  cut edge is above the field mean in 14 of 15 fields, cable edges are above off-cable edges in 15 of 15
  (+35%, Wilcoxon p 1.5e-4), and cut-edge tension ranks recoil velocity with Spearman 0.64 (p 0.011) in the
  tension-only setting; adding pressure and curvature rows removes that correlation. Partial validation:
  relative tensions of one junction type, n = 15. Absolute scale and pressures are unvalidated.
- **Barrier function proxy** (`runs/function_real/`): against measured transepithelial resistance of 10
  iPSC-RPE wells with hand-corrected ZO-1 masks (NIST/NEI), the predicted permeability index correlates
  with the right sign but not significantly (Spearman -0.52, p 0.13); against the ECIS and tracer readouts
  of the S-BIAD1169 paper, no agreement. `barrier_report` carries `validated: False` and will until a
  dataset pairing junction images with per-well permeability at adequate power exists.
- **Event detection**: T1 F1 0.91 against the TissueMiner database with curated ids; all isolated T1s
  conserve topological charge on TissueMiner (761 / 761) and EpiCure (59 / 59, 13 / 13). Not tested on
  an endothelial time lapse, because none with curated ids is public.
- **The target domain is still unmeasured.** Three systematic searches
  ([`docs/DATASET_HUNT_2026-09-18.md`](docs/DATASET_HUNT_2026-09-18.md),
  [`docs/DATASET_HUNT_2026-09-19.md`](docs/DATASET_HUNT_2026-09-19.md): 98 BioStudies accessions, 95 Zenodo
  records, IDR, the BioImage Model Zoo, figshare, Dryad, OSF, Hugging Face, GitHub) found no public
  confluent vascular endothelial monolayer with a junction marker and expert 2-D instance masks. The
  closest is COVER (Seeler et al. 2024): zebrafish dorsal aorta, Pecam1-EGFP, 296 manually traced cells,
  but as 3-D contours on a 25 um tube. hCEC is the benchmark until someone traces a VE-cadherin or
  PECAM-1 monolayer; 495 real PECAM-1 HUVEC fields are already loaded (`pimorph.io.jacquemet`) and are the
  natural target for that effort.
- Junction morphology classes are heuristic states, not validated biology; the phase-contrast and DIC
  modalities (LIVECell, mCellSeg) are far behind Cellpose-SAM; the 3-D complex has no real volume with
  truth yet.

## Quickstart

```bash
git clone https://github.com/okezue/PiMorph && cd PiMorph
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev,ml,complex]" torch            # cellpose>=4 for the Cellpose-SAM methods
python scripts/fetch_zenodo.py --models                 # checkpoints into models/, md5-verified

pimorph validate --labels path/to/labels.tif            # validity report, defect law, Weaire residuals
pimorph reconstruct --manifest data/ve_strat/manifest_paired.csv --out runs/demo --posterior
PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v6_pool.pt PIMORPH_NEURAL_TTA=1 \
PIMORPH_DECODER_PARAMS='{"nucleus_merge": true, "boundary_smooth_sigma": 1.0, "vertex_weight": 0.6}' \
  pimorph benchmark --dataset hcec --methods neural cellpose_sam_filled --out runs/check
pytest tests/pimorph -q                                  # 192 tests; torch tests skip without torch
```

Benchmark datasets are read in place from their original sources (`src/pimorph/bench/datasets.py`,
`DEFAULT_ROOTS`); the hunt documents list every download URL and licence. `PIMORPH_CELLPOSE_MODEL`
points the Cellpose methods at a fine-tuned model; `scripts/export_cellpose_training.py` writes the
training pairs. GPU campaigns are scripted under `infra/xai/` (xAI devbox) and `infra/aws/`.

## Data

The repository holds the code, the tests and small derived tables. Everything else is in the PiMorph
data record on Zenodo, **[10.5281/zenodo.22839866](https://doi.org/10.5281/zenodo.22839866)** (concept
DOI, resolves to the newest version, currently 1.1.0, record
[10.5281/zenodo.22839867](https://doi.org/10.5281/zenodo.22839867)):

| file | size | contents |
|---|---|---|
| `pimorph_models.tar` | 536 MB | eight proposal checkpoints `pimorph_proposals_v0_synth.pt` to `v6_pool.pt` with cards, training logs and configurations; extracts into `models/` and `runs/neural/` |
| `pimorph_results.tar` | 997 MB | every benchmark output: held-out tables for HAEC, mCellSeg, hCEC, alizarine, FlyWing and RPE, decoder tuning tables, the shear re-test posteriors for all 102 EGM2 fields, legacy per-field outputs, the multi-junction, dynamics and mechanics reports, figures and result documents; extracts into `runs/` and `docs/` |
| `pimorph_training_tiles.tar` | 1,584 MB | 200 synthetic validation tiles, 177 S-BIAD1540 and VE-strat pseudo-label tiles, 1,200 real PECAM-1 HUVEC consensus pseudo-label tiles; extracts into `data/tiles/` |

Third-party images and their ground truth are not redistributed. The 8,000 synthetic training tiles are
regenerated exactly by `pimorph synth` with the seeds in `infra/xai/run_scale.sh`. New versions of the
record are made with `scripts/publish_zenodo.py`. The fine-tuned Cellpose-SAM weights of the symmetric
comparison (1.2 GB each) and the real-data validation results after version 1.1.0 will be added in the
next record version.

## History

PiMorph grew out of EndoPiGraph-AJmorph (v1.0.0, `src/endopigraph/`), a pixel-adjacency contact-graph
tool for S-BIAD1540 with junction morphology features and a GMM classifier. The legacy package remains
and is fed through adapters (`pimorph.io.legacy` writes the same `cells.csv`, `edges.csv` and GraphML).
Two of its published numbers were retracted by this work: the 93% cornea adjacency F1 measured
extraction from labels that cannot support instance metrics, and the area-degree shear finding did not
replicate under the posterior re-test. The VE-strat manifest had its channels swapped, which is why the
legacy run found zero contacts there; `data/ve_strat/manifest_paired.csv` fixes it. Its documentation is
in `docs/BLUR_STABILITY.md` and the `endopigraph` docstrings; `endopigraph run --config config.yaml`
still works.

## Citation and status

Please credit Okezue Bell (okezue@stanford.edu) and Anthony Bell for this work when used. Data:
Bell, A. O. (2026). PiMorph: proposal checkpoints, benchmark results and training tiles for endothelial
cell complexes (1.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22839867
