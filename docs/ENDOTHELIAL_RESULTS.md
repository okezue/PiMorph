# Endothelial ground-truth datasets and results (2026-09-17, corrected 2026-09-18)

## Correction (2026-09-18)

The tables under "Held-out results" below were computed against a flawed HAEC reference and with
the old decoder flood mask; they are kept for the record and superseded by `runs/vertex/SUMMARY.md`
and `docs/CONFLUENT_BENCHMARK.md`. Two things changed:

1. The HAEC instance derivation used 4-connected body components; along the ragged 1 px border
   class this split off 1 to 3 px specks that counted as cells (field 0005: 939 "cells", 386 under
   60 px; 522 after the fix). `haec_semantic_to_instance` now uses 8-connected markers, drops
   markers under 30 px and makes the labels 4-connected up front.
2. The decoder flooded open background (the gap head is trained on enclosed background only), so
   cells that never touch were joined and every false contact produced false vertices. Pixels the
   signed-distance head places outside every cell are now excluded (`DecoderParams.outside_px = 0`).

Corrected held-out numbers (same 86 HAEC and 40 mCellSeg test fields, all methods re-run):

| Dataset | Method | n | Adj F1 pair / component | Vertex F1 (pred / true vertices per field) | Incident-set acc. | PQ | AP50 | Boundary F1 |
|---|---|---|---|---|---|---|---|---|
| HAEC test | **v3_endo** (`models/pimorph_proposals_v3_endo.pt`) | 86 | **0.505 / 0.306** | **0.262** (95 / 92) | **0.677** | **0.600** | **0.616** | **0.844** |
| HAEC test | v2_endo | 86 | 0.504 / 0.308 | 0.261 (112 / 92) | 0.670 | 0.597 | 0.612 | 0.843 |
| HAEC test | Cellpose-SAM (filled) | 86 | 0.314 / 0.087 | 0.039 (110 / 92) | 0.607 | 0.465 | 0.487 | 0.708 |
| HAEC test | v1_multi | 86 | 0.001 / 0.000 | 0.003 | 0.581 | 0.011 | 0.009 | 0.235 |
| HAEC test | classical | 86 | 0.039 / 0.004 | 0.002 | 0.539 | 0.162 | 0.152 | 0.271 |
| mCellSeg test | **Cellpose-SAM (filled)** | 40 | **0.170 / 0.060** | **0.117** | 0.941 | **0.215** | **0.208** | 0.203 |
| mCellSeg test | v3_endo | 40 | 0.064 / 0.013 | 0.014 | 0.525 | 0.134 | 0.119 | **0.349** |
| mCellSeg test | v2_endo | 40 | 0.073 / 0.013 | 0.014 | 0.593 | 0.140 | 0.124 | 0.341 |

The corrected reference has 569 cells and 92 multicellular vertices per HAEC test field (was 1,151
and 124). The decoder fix, not the retraining, accounts for nearly all of the gain (v2_endo and
v3_endo agree within 0.01; the retrain calibrated the vertex count). Vertex F1 0.26 is bounded by
cell-level accuracy (PQ 0.60) and by the 1 px ambiguity of whether two cells in a sub-confluent
culture touch: the missed true vertices are topological (median 17 px from the nearest predicted
vertex), not mislocalized. Confluent monolayers with real truth, where this ambiguity is absent,
are in `docs/CONFLUENT_BENCHMARK.md`.

Until this campaign, no dataset in the project combined endothelial cells with instance ground truth, so every endothelial number was a self-consistency check. This document records the search for such data, what was found, and the first accuracy numbers. Raw outputs: `runs/endo/` (per-image CSVs, summaries, `splits.json`, `campaign_log.txt`).

## Search outcome

| Source | What it is | Ground truth | Verdict |
|---|---|---|---|
| mCellSeg (Alam et al. 2026; Zenodo 10.5281/zenodo.20174259, Kaggle) | 200 expert-annotated DIC images, 100 HUVEC + 100 HEK-293T, 16,199 cells, varied confluency | expert instance masks (uint16 TIFF) | **used**. Downloaded token-free via `kagglehub`. Transmitted light, not fluorescence. |
| Human aortic endothelial cell ground truth (Harrison, Wu, Fang, Huang; Zenodo 4898011, CC-BY 4.0) | 434 fields, 1200x1200, cytoplasmic Laconic-GFP reporter plus separate Hoechst channel | 4-class semantic (background, border, body, nucleus) from which instances derive (about 1,150 cells per field) | **used**. First endothelial fluorescence set with an independent whole-cell geometry channel and nuclei, exactly the input pairing the blueprint asks for. Sub-confluent (about 46% background). |
| VE-strat (Zenodo 13936923) | HUVEC VE-cadherin + nuclei, ImageXpress | 64x64 border patches with morphology classes; no instance masks | patches useful later for morphology validation; not instance GT |
| S-BIAD1540, S-BIAD1345, S-BIAD1169, S-BIAD1294 (BioImage Archive) | HUVEC/HAEC junction images under shear and perturbations | images only, or Cellpose-generated masks (S-BIAD1294 nuclei) | unlabelled endothelium; already used as pseudo-label sources |
| HCA metastable phenotypes (Chesnais et al. 2022, GitHub exr98) | >20,000 CDH5 HUVEC images, 70 annotated for Weka | annotations available only on request | not public |
| Corneal endothelium (Alizarine/Padova, Gavet, Rotterdam rod-rep.com, CLEAR-EC, opi-lab) | polygonal endothelial monolayers, specular microscopy | border maps or dots; Zenodo CLEAR-EC record returned no files; opi-lab repo holds 7 example PNGs; rod-rep.com unreachable | structurally relevant but nothing downloadable with instance masks was found |

## Protocol

- Splits are field-disjoint and fixed in `runs/endo/splits.json`: HAEC field id % 5 == 0 (86 test / 348 train); every 5th sorted mCellSeg image (40 test, 10 of them HUVEC / 160 train).
- HAEC geometry is inverted before any method (cytoplasm bright, borders dark); the loader records `boundary_polarity = "dark"`. A first pass run with the wrong polarity is kept under `runs/endo/first_pass_wrong_polarity/` and is not reported.
- `cellpose_sam_filled` fills sub-12 px background seams between touching Cellpose masks, the same rule applied to polygon ground truth, so vertex metrics compare geometry rather than mask format. On HAEC it changes nothing (0.037 vs 0.038 vertex F1), so Cellpose-SAM's vertex mismatch is real.
- Real-GT training tiles: `scripts/make_gt_tiles.py` on the train splits (3,132 HAEC tiles with the Hoechst channel, 2,914 mCellSeg tiles), exact targets from the masks.
- `v2_endo` = `v1_multi` fine-tuned 40 epochs on those tiles plus synthetic and pseudo-label tiles, DDP over 8 H100s, 21 minutes. Card: `models/pimorph_proposals_v2_endo.md`.

## Held-out results, 2026-09-17 (superseded: flawed reference and old flood mask, see the correction above)

| Dataset | Method | n | Adj F1 pair | Adj F1 component | Vertex F1 | Incident-set acc. | PQ | AP50 | Boundary F1 | Valid |
|---|---|---|---|---|---|---|---|---|---|---|
| HAEC test | **v2_endo** | 86 | 0.243 | **0.145** | **0.067** | 0.374 | **0.407** | **0.352** | **0.842** | 1.0 |
| HAEC test | Cellpose-SAM (filled) | 86 | **0.324** | 0.092 | 0.038 | **0.631** | 0.347 | 0.323 | 0.707 | 1.0 |
| HAEC test | Cellpose-SAM (raw) | 86 | 0.324 | 0.092 | 0.037 | 0.616 | 0.346 | 0.323 | 0.708 | 1.0 |
| HAEC test | v1_multi (no endothelial data) | 86 | 0.001 | 0.000 | 0.004 | 0.512 | 0.006 | 0.005 | 0.198 | 1.0 |
| HAEC test | classical | 86 | 0.040 | 0.005 | 0.002 | 0.511 | 0.109 | 0.097 | 0.273 | 1.0 |
| mCellSeg test | Cellpose-SAM (filled) | 40 | **0.169** | **0.060** | **0.117** | 0.951 | **0.214** | **0.207** | 0.203 | 1.0 |
| mCellSeg test | v2_endo | 40 | 0.051 | 0.011 | 0.011 | 0.544 | 0.131 | 0.115 | **0.354** | 1.0 |
| mCellSeg test | v1_multi | 40 | 0.001 | 0.000 | 0.002 | 0.500 | 0.011 | 0.009 | 0.203 | 1.0 |

First-pass numbers on 200 HAEC fields (train + test mixed, used only to choose the protocol) agree with the held-out table to within 0.01 for Cellpose-SAM (0.324 / 0.350) and the classical method (0.038 / 0.107).

## What this established on 2026-09-17 (points 2 and 4 are superseded by the correction above and by `docs/CONFLUENT_BENCHMARK.md`)

1. Endothelial fluorescence accuracy is now measured, not assumed. With about 3,000 real endothelial tiles the PiMorph model goes from unusable (PQ 0.006) to the best instance quality on this data (PQ 0.407, boundary F1 0.842), finding dim cells Cellpose-SAM misses (679 vs 753 predicted against 1,151 true cells, with far higher boundary agreement). Figure: `runs/pimorph_dev/haec_test_example.png`.
2. Multicellular vertices are still not solved by anyone here. v2_endo over-splits (about 1,060 predicted vertices per field against 124 in the ground truth), Cellpose-SAM places the right number in the wrong places (F1 0.038, median error 2.0 px). Vertex-aware training targets exist in the tiles but the soft priors and decoder moves that would suppress over-splitting were not tuned for this data.
3. Transmitted-light HUVEC (mCellSeg) is a different problem: v2_endo over-segments, Cellpose-SAM under-segments, and 10 HUVEC test images cannot separate the cell lines. More DIC data or a DIC-specific proposer is needed.
4. HAEC ground truth is derived from semantic classes and the cultures are sub-confluent, so these are lower bounds on a monolayer benchmark; a confluent VE-cadherin dataset with instance truth still does not exist publicly.

## Compute record

xAI devbox `okebell-pimorph` (`fou` / `h100-sandbox`, 8x H100). Campaign 21:26 to 23:20 box time including two relaunches (driver argument bug, polarity fix); DDP fine-tune 21 minutes; 19 benchmark jobs. No other job, queue or namespace was touched.
