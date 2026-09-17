# Endothelial ground-truth datasets and results (2026-09-17)

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

## Held-out results (all methods through the same constrained decoder)

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

## What this establishes

1. Endothelial fluorescence accuracy is now measured, not assumed. With about 3,000 real endothelial tiles the PiMorph model goes from unusable (PQ 0.006) to the best instance quality on this data (PQ 0.407, boundary F1 0.842), finding dim cells Cellpose-SAM misses (679 vs 753 predicted against 1,151 true cells, with far higher boundary agreement). Figure: `runs/pimorph_dev/haec_test_example.png`.
2. Multicellular vertices are still not solved by anyone here. v2_endo over-splits (about 1,060 predicted vertices per field against 124 in the ground truth), Cellpose-SAM places the right number in the wrong places (F1 0.038, median error 2.0 px). Vertex-aware training targets exist in the tiles but the soft priors and decoder moves that would suppress over-splitting were not tuned for this data.
3. Transmitted-light HUVEC (mCellSeg) is a different problem: v2_endo over-segments, Cellpose-SAM under-segments, and 10 HUVEC test images cannot separate the cell lines. More DIC data or a DIC-specific proposer is needed.
4. HAEC ground truth is derived from semantic classes and the cultures are sub-confluent, so these are lower bounds on a monolayer benchmark; a confluent VE-cadherin dataset with instance truth still does not exist publicly.

## Compute record

xAI devbox `okebell-pimorph` (`fou` / `h100-sandbox`, 8x H100). Campaign 21:26 to 23:20 box time including two relaunches (driver argument bug, polarity fix); DDP fine-tune 21 minutes; 19 benchmark jobs. No other job, queue or namespace was touched.
