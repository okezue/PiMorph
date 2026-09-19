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

`v4_confluent` = `v3_endo` fine-tuned to epoch 140 on the TRAIN splits (10 hCEC fields, 20
alizarine, 32 FlyWing; 362 tiles mixed with the earlier real and synthetic tiles), evaluated on
the TEST splits (5 hCEC fields, 10 alizarine, 10 FlyWing) that no model saw. Cellpose-SAM and
`v3_endo` are evaluated on the same test fields.

First attempt (`runs/vertex/v4_roi_as_background/`): tiles treated pixels outside the annotated
ROI as background. On hCEC test this already lifted vertex F1 from 0.334 to 0.618 (Cellpose-SAM
0.635) and adjacency from 0.712 to 0.816 (0.830), and HAEC test improved too (vertex F1 0.262 to
0.297, PQ 0.609). But on alizarine, where the ROI is 28% of the frame and the rest is full of
cells, the model learned that visible cells outside the ROI are background (test vertex F1 0.977
to 0.595, 203 of 320 cells found), and FlyWing vertex localization degraded (0.868 to 0.759).
Pixels outside a ROI are unknown, not background: `scripts/make_gt_tiles.py` now writes them
(plus a 3 px rim) into the tile `ignore` mask, which zeroes their loss weight. The second attempt
with those tiles is the reported one:

`v4_confluent` (`models/pimorph_proposals_v4_confluent.pt`, best epoch 137, 34 minutes on 8 H100)
on the held-out TEST splits (`runs/vertex/test_split_summary.csv`):

| Test split | Method | n | Adj F1 pair / comp. | Vertex F1 | Vertex prec. / rec. | pred / true vertices | Loc. median (px) | PQ | AP50 | Boundary F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| hCEC (5 fields) | Cellpose-SAM (filled) | 5 | **0.830** / **0.745** | **0.635** | 0.721 / 0.581 | 2298 / 2974 | **1.17** | **0.790** | 0.814 | 0.880 |
| hCEC | **v4_confluent** | 5 | 0.822 / 0.711 | 0.611 | 0.579 / **0.648** | 3377 / 2974 | 1.53 | 0.776 | **0.817** | **0.896** |
| hCEC | v3_endo (zero-shot) | 5 | 0.712 / 0.473 | 0.334 | 0.350 / 0.320 | 2750 / 2974 | 2.00 | 0.671 | 0.748 | 0.804 |
| alizarine (10 fields) | **v4_confluent** | 10 | 0.977 / 0.976 | **0.990** | 0.995 / 0.985 | 564 / 570 | 1.00 | **0.911** | 0.974 | 0.998 |
| alizarine | Cellpose-SAM (filled) | 10 | **0.989** / **0.989** | 0.984 | 0.979 / 0.989 | 576 / 570 | 1.00 | 0.898 | **0.984** | **1.000** |
| alizarine | v3_endo (zero-shot) | 10 | 0.941 / 0.939 | 0.977 | 0.993 / 0.961 | 550 / 570 | 1.00 | 0.888 | 0.945 | 0.995 |
| FlyWing (10 fields) | **Cellpose-SAM (filled)** | 10 | **0.966** / **0.940** | 0.849 | 0.834 / 0.864 | 1306 / 1260 | **1.47** | **0.795** | **0.950** | **0.996** |
| FlyWing | v3_endo (zero-shot) | 10 | 0.802 / 0.774 | **0.868** | 0.897 / 0.842 | 1185 / 1260 | 1.68 | 0.726 | 0.784 | 0.986 |
| FlyWing | v4_confluent | 10 | 0.894 / 0.865 | 0.833 | 0.831 / 0.836 | 1268 / 1260 | 2.00 | 0.741 | 0.875 | 0.994 |
| HAEC test (86 fields, for reference) | **v4_confluent** | 86 | **0.521** / **0.332** | **0.292** | 0.294 / 0.304 | 108 / 92 | 1.43 | **0.612** | **0.631** | **0.852** |
| HAEC test | v3_endo | 86 | 0.505 / 0.306 | 0.262 | 0.279 / 0.258 | 95 / 92 | 1.48 | 0.600 | 0.616 | 0.844 |

Reading:

- Ten manually traced hCEC fields (about 17,000 cells) fine-tune the proposal model from
  vertex F1 0.33 to 0.61 on the five held-out fields, within 0.024 of Cellpose-SAM (0.635), with
  higher vertex recall (0.648 vs 0.581), lower precision (it predicts 14% too many vertices)
  and boundary F1 0.896 vs 0.880. Cellpose-SAM keeps the better vertex localization (1.17 vs
  1.53 px) and PQ (0.790 vs 0.776). Cellpose-SAM was trained on orders of magnitude more data;
  that PiMorph's small in-domain fine-tune reaches parity on the structural metrics says the
  representation and decoder are not the bottleneck, data is.
- On in situ corneal endothelium (alizarine) the fine-tuned model has the best vertex F1
  (0.990) and PQ (0.911); on FlyWing its adjacency improved (0.802 to 0.894) but vertex
  localization degraded (1.68 to 2.00 px median) and vertex F1 fell to 0.833, below its own
  zero-shot 0.868; 32 FlyWing tiles at a different cell scale were not enough to keep both.
- The confluent data also improved the sub-confluent HAEC test (vertex F1 0.262 to 0.292, PQ
  0.612, boundary F1 0.852), the best PiMorph numbers on that set.
- Decoder observations on hCEC training fields (`runs/decoder_tuning/hcec_train_variants.csv`,
  v3_endo): `boundary_gamma` is a no-op (the watershed is order-based), `boundary_smooth_sigma =
  1` adds about 0.02 vertex F1, `distance_mix = 0.2` adds 0.04 to 0.05 adjacency F1, compactness
  hurts vertices. Defaults were not changed on this basis (they were tuned on HAEC train); the
  posterior grid already spans smoothing and distance mixing.

## Verdict on "does PiMorph solve multicellular vertices"

- Where borders are clean and the tissue is confluent, yes, for every method: vertex F1 0.98 to
  0.99 on corneal endothelium in situ and 0.83 to 0.87 on E-cadherin epithelium, with exact
  incidence sets and cyclic orders from the complex.
- On cultured endothelial monolayers with real truth (hCEC), the best vertex F1 is 0.61 to 0.64
  (PiMorph fine-tuned, Cellpose-SAM zero-shot) at a 3 px tolerance on 0.65 um pixels; the
  remaining errors are missed or split cells in weak-signal regions, not vertex placement.
- On sub-confluent cultures (HAEC) vertex F1 stays near 0.3 for the best method because whether
  two cells touch is a 1 px decision in the reference; the predicted vertex count is calibrated
  and 6.7x more accurate than Cellpose-SAM's, but the metric ceiling is the data.
- No public VE-cadherin or PECAM-1 monolayer with expert instance masks exists to close the last
  gap; the raw PECAM-1 HUVEC pairs from Zenodo 10611092 (484 fields, no masks) are the natural
  target for a small annotation effort, and `hcec` is the benchmark to beat until then.

## Frontier round (2026-09-19): training for topologically missed vertices, pooled data

Everything in this section is measured on the same held-out TEST splits as above; every decoder
setting was chosen on the TRAIN splits (`runs/frontier/tune_*.csv`). Full table:
`runs/frontier/final_summary.csv`; campaign log `runs/frontier/campaign.log`.

What was done, in order of measured effect on hCEC test vertex F1 (v4_confluent start 0.611):

1. **Hard-example mining of topological misses** (`scripts/make_vertex_miss_weights.py`): every
   real-truth training tile is decoded exactly as at test time, its vertices matched to the
   truth, and pixels within 8 px of a true vertex with no predicted counterpart get 4x loss
   weight (3x around spurious predicted vertices); all true vertices get 2x within 5 px
   (`vertex_focus`). v4 matched 67% of hCEC, 85% of FlyWing, 97% of alizarine and 36% of HAEC
   training vertices. `v5_vertex` = v4 + 40 epochs with these weights: 0.611 to 0.636 (plain
   decode), 0.833 to 0.871 on FlyWing, 0.292 to 0.337 on HAEC.
2. **Decoder and test-time changes** (+0.031 on hCEC): the vertex head enters the watershed
   elevation (`vertex_weight`, now 0.3 by default), nucleus-consistency merges remove cells
   without a nuclear peak across their weakest boundary (`nucleus_merge`; hCEC train adjacency
   0.82 to 0.85, cells 1,907 to 1,760 vs 1,673 true), 1 px boundary smoothing, and dihedral
   test-time augmentation (`NeuralProposer(tta=True)`, +0.018 alone). Distance mixing and
   plain boundary-support merges hurt and were dropped.
3. **Pooled data** (+0.013): `v6_pool` = v5 + 40 epochs with the NIH-NEI RPE monolayer train
   stacks (208 tiles, `rpe_zo1` loader, manual VIA polygons united over z) and 1,200
   consensus pseudo-label tiles from 300 real PECAM-1 HUVEC monolayer fields (Zenodo 10611092;
   Cellpose-SAM primary, v4 second opinion, 80% cell agreement, disputed boundaries ignored).
4. **Posterior MAP** (`neural_map`: 32 hypotheses scored by the full energy) gains 0.002 over
   the default decode at 15x the cost; the energy does not select better complexes than the
   tuned single decode on this data.

| Test split | Method | Adj F1 pair | Vertex F1 | Vertex prec. / rec. | pred / true vertices | Loc. median (px) | PQ | Boundary F1 |
|---|---|---|---|---|---|---|---|---|
| hCEC (5) | **v6_pool tuned** | **0.867** | **0.680** | 0.678 / 0.683 | 3014 / 2974 | 1.28 | **0.816** | **0.916** |
| hCEC | v5_vertex tuned | 0.861 | 0.667 | 0.661 / 0.674 | 3058 / 2974 | 1.53 | 0.807 | 0.911 |
| hCEC | v5_vertex posterior MAP | 0.863 | 0.669 | 0.664 / 0.674 | 3054 / 2974 | 1.53 | 0.809 | 0.912 |
| hCEC | v4_confluent tuned | 0.851 | 0.645 | 0.634 / 0.657 | 3103 / 2974 | 1.45 | 0.800 | 0.907 |
| hCEC | v5_vertex plain | 0.826 | 0.636 | 0.604 / 0.673 | 3357 / 2974 | 1.53 | 0.781 | 0.901 |
| hCEC | Cellpose-SAM (filled) | 0.830 | 0.635 | 0.721 / 0.581 | 2298 / 2974 | **1.17** | 0.790 | 0.880 |
| alizarine (10) | v5_vertex tuned | 0.988 | **0.993** | 0.995 / 0.991 | 567 / 570 | 1.00 | **0.920** | 0.999 |
| alizarine | v6_pool tuned | 0.987 | 0.992 | 0.994 / 0.990 | 568 / 570 | 1.00 | 0.920 | 0.999 |
| alizarine | Cellpose-SAM (filled) | **0.989** | 0.984 | 0.979 / 0.989 | 576 / 570 | 1.00 | 0.898 | **1.000** |
| FlyWing (10) | v6_pool tuned | 0.911 | **0.875** | 0.869 / 0.881 | 1278 / 1260 | 1.59 | 0.780 | 0.994 |
| FlyWing | v5_vertex tuned | 0.900 | 0.875 | 0.870 / 0.880 | 1274 / 1260 | 1.59 | 0.776 | 0.995 |
| FlyWing | Cellpose-SAM (filled) | **0.966** | 0.849 | 0.834 / 0.864 | 1306 / 1260 | **1.47** | **0.795** | 0.996 |
| RPE (20 tiles, 5 stacks) | v6_pool tuned | 0.605 | **0.327** | 0.291 / 0.381 | 265 / 209 | 1.73 | **0.556** | **0.731** |
| RPE | Cellpose-SAM (filled) | **0.622** | 0.287 | 0.254 / 0.331 | 269 / 209 | 1.77 | 0.548 | 0.686 |
| HAEC (86) | **v6_pool tuned** | **0.536** | **0.352** | 0.293 / 0.453 | 151 / 92 | 1.25 | **0.626** | **0.858** |
| HAEC | v4_confluent | 0.521 | 0.292 | 0.294 / 0.304 | 108 / 92 | 1.43 | 0.612 | 0.852 |
| HAEC | Cellpose-SAM (filled) | 0.314 | 0.039 | 0.039 / 0.043 | 110 / 92 | 1.40 | 0.465 | 0.708 |

Where this leaves the frontier:

- Cultured endothelial monolayer (hCEC): PiMorph leads on adjacency, vertex F1, PQ and boundary
  F1 with a calibrated vertex count; Cellpose-SAM keeps better vertex localization (1.17 vs 1.28
  px) and incident-set accuracy (0.91 vs 0.86). The remaining vertex errors are split and merged
  cells in weak-signal regions, at roughly equal rates now (precision 0.68, recall 0.68).
- In situ corneal endothelium and E-cadherin epithelium: vertices are at 0.99 and 0.875; on
  FlyWing Cellpose-SAM still has the better adjacency (0.966 vs 0.911).
- RPE with stress fibres in the border channel and sub-confluent HAEC remain hard for every
  method; PiMorph is ahead on both but at vertex F1 0.33 to 0.35.
- Real PECAM-1 HUVEC monolayers are now in the training pool, but only as pseudo-labels; their
  vertex accuracy is still unmeasured because no expert masks exist. A small annotation effort on
  those 495 fields (they are 0.65 um/px, ~500 cells each) is the single most valuable next step.

## Pooled datasets (2026-09-19)

| Loader / resource | Data | Role |
|---|---|---|
| `rpe_zo1` (`pimorph.bench.rpe`) | NIH-NEI RPE monolayer training set (figshare+ 28832501, CC0): 18 confocal stacks x 4 tiles, 130,201 manual VIA polygons, 6,883 cells; green border channel (upstream target "Actin": phalloidin or ZO-1 depending on the stack), red nuclei | benchmark (stack-disjoint splits, `runs/vertex/splits_rpe.json`) and training (13 stacks) |
| `pimorph.io.jacquemet` | 495 real PECAM-1 HUVEC monolayer fields (Zenodo 10611092, 1022x1024, 0.65 um/px) with a manifest | 1,200 consensus pseudo-label tiles (`data/tiles/pseudo_pecam` on the devbox); label-free self-check `runs/pecam_selfcheck/` (v4: 544 cells and 987 tricellular vertices per field, 1 gap; v3: 686 cells, 72 gaps) |
| `pimorph.dynamics.epicure` | EpiCure curated movies (Zenodo 20607705) | event calculus on curated ids (`docs/LATER_PHASES.md`) |
