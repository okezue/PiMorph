# Confluent monolayers with real instance truth: datasets, vertex decoding, results (2026-09-18)

This document answers the question left open on 2026-09-17: can multicellular vertices be
measured against real truth on a confluent monolayer, and does PiMorph solve them? The
dataset search is in `docs/DATASET_HUNT_2026-09-18.md`; the campaign log is
`runs/vertex/campaign.log`.

## Datasets found

| Loader | Data | Truth | Cells / tricellular vertices per field | Notes |
|---|---|---|---|---|
| `hcec` | Travers, Coulomb et al. 2025 (Sci Rep 15:31301, MOESM6): cultured human corneal endothelial cells, NCAM lateral membrane + DAPI, 15 fields 2048x2048, 0.65 um/px | manual tracing of every cell in the Cellpose GUI (4 to 8 h per field), shipped as closed border skeleton + ROI; filled to instances by `skeleton_roi_to_instance` | ~1,670 / ~3,000 | confluent endothelium with a nuclear channel; NCAM is a membrane marker, not a junction protein; signal patchy, annotators closed borders through weak regions using nuclei. Licence CC BY-NC-ND 4.0 (evaluation only). |
| `alizarine` | Padova BioImLab alizarine-red porcine corneal endothelium, 30 fields 576x768, phase contrast (GitHub mirror; original host offline) | expert 1 px closed contours + ROI + one marker per cell | ~340 / ~610 inside the ROI | in situ confluent endothelium, dark borders, hexagonal; the ROI covers 28% of the frame |
| `flywing` | FlyWing (Funke et al. 2018 tracking benchmark, DenoiSeg release Zenodo 5156991, CC BY 4.0), Drosophila wing disc E-cadherin:GFP, 42 fields 512x512 | Tissue Analyzer + manual correction, full field | ~760 / ~1,290 | confluent epithelium, bright junctions, no nuclei; 1 px label seams filled |

Also found, not yet loaded: NIH-NEI RPE Mask R-CNN set (iPSC retinal pigment epithelium, ZO-1 or
actin borders, manual VIA polygons per z-frame, 4.5 GB on figshare+ CC0), EpiCure curated
movies, ReSCU-Nets E-cadherin embryo epidermis (S-BIAD1410). The verdict on the original
question stands: no public confluent endothelial monolayer stained for a junction protein
(VE-cadherin, PECAM-1, ZO-1, claudin-5) ships expert instance masks. The Jacquemet PDAC_DL
"Landmark" HUVEC junction labels are Cellpose output on pix2pix virtual staining and are not
truth. hCEC is the closest substitute: endothelial, confluent, fluorescent membrane borders,
manual instance truth, independent nuclear channel.

Partially annotated fields (hCEC ROI, alizarine ROI) are scored inside the ROI only
(`pimorph.bench.run.restrict_to_roi`).

## What was wrong with the vertex numbers before

1. HAEC reference: the derivation flooded 4-connected body components through the border
   class; along ragged borders this created 1 to 3 px specks that counted as cells (field 0005:
   939 "cells", 386 of them under 60 px; 522 after the fix). Every method's HAEC numbers on
   2026-09-17 were computed against this reference, and the specks were also seed and boundary
   targets in the v2_endo training tiles.
2. Decoder flood mask: the gap head is trained on ENCLOSED background only (open background is
   the outer face and has no target), and the benchmark's permissive intensity mask covers the
   space between cells at the 15 px blur scale. The watershed therefore flooded open background
   and joined cells that never touch; every false contact spawned false multicellular vertices
   (HAEC field 0005: 1,167 predicted against 81 true). The signed-distance head does know
   outside from inside: excluding pixels with predicted distance below 0 (`outside_px = 0`)
   brings the count to 89 and is nearly free on confluent tissue (synthetic vertex F1 0.687 to
   0.734 on 60 tiles; 0.854 to 0.836 on the one fully confluent tile probed). Tuning on 40 HAEC
   TRAIN fields (`runs/vertex/tune_haec_train_v3endo.csv`) selected the value; 1 px carves
   slivers along real contacts and 0.5 px inside re-joins cells.
3. Nucleus-guided seeding: where the seed head is silent (weak membrane signal) but a nucleus
   is present, the nuclear peak is added as a seed (`NeuralProposer(nucleus_seeds=True)`), the
   one-nucleus-per-cell prior at proposal time. On two hCEC training fields adjacency F1 rose
   0.616 to 0.678 and 0.657 to 0.725.
4. Vertex-consistent merges (`DecoderParams.max_merges`, edges with neither boundary nor
   vertex-head support) are implemented and tested but gain under 0.01 everywhere, so they stay
   off by default.

## Held-out HAEC and mCellSeg, corrected reference (`runs/vertex/SUMMARY.md`)

| Dataset | Method | n | Adj F1 pair / comp. | Vertex F1 | pred / true vertices | Incident-set acc. | PQ | AP50 | Boundary F1 |
|---|---|---|---|---|---|---|---|---|---|
| HAEC test | **v3_endo** | 86 | **0.505 / 0.306** | **0.262** | 95 / 92 | **0.677** | **0.600** | **0.616** | **0.844** |
| HAEC test | v2_endo | 86 | 0.504 / 0.308 | 0.261 | 112 / 92 | 0.670 | 0.597 | 0.612 | 0.843 |
| HAEC test | Cellpose-SAM (filled) | 86 | 0.314 / 0.087 | 0.039 | 110 / 92 | 0.607 | 0.465 | 0.487 | 0.708 |
| HAEC test | classical | 86 | 0.039 / 0.004 | 0.002 | | 0.539 | 0.162 | 0.152 | 0.271 |
| mCellSeg test (DIC) | Cellpose-SAM (filled) | 40 | **0.170 / 0.060** | **0.117** | | 0.941 | **0.215** | **0.208** | 0.203 |
| mCellSeg test (DIC) | v3_endo | 40 | 0.064 / 0.013 | 0.014 | | 0.525 | 0.134 | 0.119 | **0.349** |

On HAEC the predicted vertex count is now calibrated (95 vs 92 per field) and vertex F1 is 6.7x
Cellpose-SAM's, but the absolute value stays at 0.26 because half of the true vertices involve a
cell the model misses or splits (PQ 0.60) and because whether two cells in a sub-confluent
culture touch is a 1 px decision (the missed vertices are topological, median 17 px from the
nearest prediction, not mislocalized). DIC (mCellSeg) remains a different problem.

## Zero-shot on the confluent truth sets (`runs/vertex/confluent_zero_shot_summary.csv`)

None of these sets was in any training run. All methods use the same decoder rules and the same
metrics (mutual-nearest vertex matching within 3 px).

| Dataset | Method | n | Adj F1 pair | Vertex F1 | Vertex prec. / rec. | Vertex loc. median (px) | PQ | Boundary F1 | pred / true cells |
|---|---|---|---|---|---|---|---|---|---|
| hCEC | **Cellpose-SAM (filled)** | 15 | **0.774** | **0.584** | 0.709 / 0.524 | **1.28** | **0.744** | **0.850** | 1651 / 1670 |
| hCEC | v2_endo | 15 | 0.696 | 0.329 | 0.359 / 0.304 | 1.92 | 0.655 | 0.796 | 1657 / 1670 |
| hCEC | v3_endo | 15 | 0.682 | 0.324 | 0.338 / 0.311 | 1.96 | 0.648 | 0.786 | 1627 / 1670 |
| hCEC | classical | 15 | 0.637 | 0.293 | 0.310 / 0.279 | 2.09 | 0.637 | 0.809 | 1480 / 1670 |
| alizarine | **Cellpose-SAM (filled)** | 30 | **0.987** | **0.981** | 0.975 / 0.987 | 1.00 | **0.888** | 1.000 | 348 / 342 |
| alizarine | classical | 30 | 0.908 | 0.962 | 0.979 / 0.946 | 1.00 | 0.856 | 0.990 | 331 / 342 |
| alizarine | v3_endo | 30 | 0.904 | 0.957 | 0.984 / 0.935 | 1.01 | 0.854 | 0.991 | 326 / 342 |
| FlyWing | **Cellpose-SAM (filled)** | 42 | **0.969** | 0.857 | 0.840 / 0.875 | 1.46 | **0.796** | 0.996 | 728 / 759 |
| FlyWing | **v3_endo** | 42 | 0.799 | **0.864** | 0.896 / 0.836 | 1.56 | 0.721 | 0.984 | 643 / 759 |
| FlyWing | v1_multi | 42 | 0.738 | 0.852 | 0.896 / 0.813 | 1.46 | 0.692 | 0.984 | 621 / 759 |
| FlyWing | classical | 42 | 0.734 | 0.818 | 0.833 / 0.805 | 1.54 | 0.680 | 0.969 | 670 / 759 |

Reading:

- Multicellular vertices are solved where the borders are clean: on in situ corneal
  endothelium (alizarine) every method reaches vertex F1 0.96 to 0.98 through the complex, and on
  FlyWing PiMorph's proposals give the best vertex F1 (0.864, precision 0.896) even though
  Cellpose-SAM has the better adjacency and PQ.
- On the cultured endothelial monolayer (hCEC), Cellpose-SAM's zero-shot masks are clearly
  better than PiMorph's zero-shot proposals (vertex F1 0.58 vs 0.33). Cellpose-SAM was trained
  on a very large and diverse corpus; PiMorph's proposal models saw HAEC (dark seams), DIC and
  synthetic junctions, and their boundaries on 5 to 6 px wide NCAM ridges are placed about 2 px
  off (median vertex error 1.9 vs 1.3 px), which the 3 px tolerance punishes. The in-domain
  fine-tune below measures how much of this gap is data.
- Every reconstruction is a valid complex (validity 1.0 on all 173 fields), so the vertex
  incidence sets, cyclic orders and contact multiplicities reported by PiMorph are exact
  properties of the segmentation, whichever method produced it. Cellpose-SAM masks reach those
  structures only through PiMorph's extractor (`cellpose_sam_filled` is Cellpose-SAM + PiMorph).

## In-domain fine-tune (field-disjoint splits, `runs/vertex/splits_confluent.json`)

`v4_confluent` = `v3_endo` fine-tuned on the TRAIN splits (10 hCEC fields, 20 alizarine, 32
FlyWing; 362 tiles mixed with the earlier real and synthetic tiles), evaluated on the TEST splits
(5 hCEC fields, 10 alizarine, 10 FlyWing) that no model saw. Results are appended below when the
run finishes (`runs/vertex/*_test_v4confluent`).
