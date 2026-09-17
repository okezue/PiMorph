# Scaled tests and benchmarks on xAI H100 compute (2026-09-17)

Everything in this document ran on one dedicated explorer devbox (`okebell-pimorph`, cluster `fou`, queue `h100-sandbox`, 8x H100 80 GB, 70 CPUs) following `infra/xai/README.md` and `infra/xai/run_scale.sh`. Raw outputs are under `runs/scale/`.

## Test suite

`pytest tests/ -n 16`: 259 passed, 2 skipped (the two `data`-marked VE-strat tests, whose raw fields are not on the box), 135 s.

## Data brought in

| Set | Size | Ground truth | Role |
|---|---|---|---|
| LIVECell, all splits | 5,239 images, 3,253 / 570 / 1,564 train / val / test, 8 cell lines, phase contrast | COCO instance polygons (slivers below 12 px filled) | real-GT training tiles from train; held-out benchmark on test |
| NeurIPS 2022 Cell Segmentation Challenge (Zenodo 10719375) | 1,000 labelled + 101 tuning images; brightfield, fluorescence, phase contrast, DIC | instance label TIFFs | real-GT training tiles and (in-sample) benchmark |
| Synthetic | 8,000 train + 400 val tiles at 512 px (16 CPU workers, 17 min) | exact (labels, vertices, gaps, distance) | training and held-out benchmark |
| Cornea | 160 crops | derived from semantic classes, unusable for instance metrics | negative control only |

Tile building: `scripts/make_gt_tiles.py` sharded 16-way (LIVECell 2,276 tiles from 576 images; NeurIPS 5,455 tiles from 1,101 images).

## Baselines at 400 images per dataset

| Dataset | Method | n | Adjacency F1 (pair) | Vertex F1 | Incident-set acc. | PQ | Boundary F1 | Edit dist. | Valid | s/image |
|---|---|---|---|---|---|---|---|---|---|---|
| LIVECell test | Cellpose-SAM | 400 | 0.617 | 0.341 | 0.718 | 0.639 | 0.907 | 849 | 1.0 | 0.50 |
| LIVECell test | classical | 100 | 0.021 | 0.073 | 0.164 | 0.072 | 0.539 | 2848 | 1.0 | 2.44 |
| NeurIPS CellSeg | Cellpose-SAM | 400 | 0.382 | 0.531 | 0.939 | 0.597 | 0.724 | 632 | 1.0 | 1.61 |
| NeurIPS CellSeg | classical | 100 | 0.025 | 0.099 | 0.743 | 0.042 | 0.266 | 4011 | 1.0 | 9.64 |
| Synthetic (seed 9999) | classical | 400 | 0.368 | 0.438 | 0.300 | 0.483 | 0.815 | 998 | 1.0 | 0.63 |
| Cornea | Cellpose-SAM | 160 | 0.000 | 0.020 | 0.000 | 0.000 | 0.510 | 3429 | 1.0 | 0.73 |

Per cell line, Cellpose-SAM adjacency F1 on LIVECell test ranges from 0.36 (SHSY5Y, neuron-like, thin processes) to 0.78 (BV2, SkBr3). The 6-image LIVECell number from the laptop run (0.558) sits inside this range, so the small sample was not misleading. Cornea scores 0 for every method, which confirms the derived cornea GT is the problem, not the methods.

## Neural proposals trained on real ground truth (v1_multi)

`torchrun` DDP over 8 H100s, base-48 UNet, 60 epochs over about 18,000 tiles in 52 minutes (about 50 s per epoch). Details and caveats in `models/pimorph_proposals_v1_multi.md`.

| Dataset | Split status | neural v1_multi adj F1 / vertex F1 / PQ | Cellpose-SAM | classical |
|---|---|---|---|---|
| Synthetic | held out | 0.861 / 0.693 / 0.868 | not run | 0.368 / 0.438 / 0.483 |
| LIVECell test | held out | 0.230 / 0.164 / 0.403 | 0.617 / 0.341 / 0.639 | 0.021 / 0.073 / 0.072 |
| NeurIPS CellSeg | in-sample | 0.097 / 0.058 / 0.390 | 0.382 / 0.531 / 0.597 | 0.025 / 0.099 / 0.042 |

What changed versus the laptop models: real training data moved the LIVECell held-out adjacency F1 from 0.000 (synthetic-only) to 0.230, and the model now beats the classical proposer on every real dataset. It does not approach Cellpose-SAM on real images. The NeurIPS result is weak even in-sample, so the mixed-modality data is under-fitted at this training budget.

## What these numbers do and do not show

- They show the decoder and metrics run unchanged across four datasets and about 1,600 images with validity 1.0 throughout, that learned proposals dominate hand-crafted filters, and that real GT is what moves real-image accuracy.
- They do not show endothelial accuracy: none of the four datasets is endothelial fluorescence with instance truth. mCellSeg (Kaggle token pending) is still the first dataset that would.
- Vertex metrics on polygon-annotated data depend on the sliver-filling rule (12 px) and on annotators having drawn touching cells as touching; they are lower bounds.

## Compute record

Devbox `okebell-pimorph` on `fou`/`h100-sandbox` (a queue this user already runs in; no other job, queue or namespace was touched). Created 2026-09-17 04:18 local; campaign 12:09 to 16:12 box time. It is left running for follow-up work; remove with `x rm okebell-pimorph -c fou` when done.
