# pimorph_proposals_v4_confluent.pt

The checkpoint file is not in the repository: `python scripts/fetch_zenodo.py --models` downloads it from the PiMorph Zenodo data record (md5-verified) into `models/`.

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 48, depth 4,
19,441,446 parameters). `v3_endo` fine-tuned on confluent monolayers with real instance truth.

- Resumed from `pimorph_proposals_v3_endo.pt` and trained to epoch 140 on the xAI devbox (8x
  H100, `torchrun` DDP, batch 8 per GPU, AdamW 1.0e-4 x 8 cosine, AMP), 34 minutes; `best.pt` is
  epoch 137 (best validation loss). Log: `runs/neural/v4_confluent/train_log.jsonl`.
- Training tiles: the TRAIN splits of `runs/vertex/splits_confluent.json`: 250 hCEC tiles (10
  manually traced human corneal endothelial fields, NCAM + DAPI, weighted 3x), 80 alizarine tiles
  (20 fields, weighted 2x), 32 FlyWing tiles (32 fields, weighted 2x), plus the HAEC, mCellSeg,
  synthetic and S-BIAD1540 pseudo-label tiles of `v3_endo`. Pixels outside an annotated ROI (plus
  a 3 px rim) carry zero loss weight; a first attempt that treated them as background collapsed
  on alizarine (`runs/vertex/v4_roi_as_background/`).
- Synthetic validation at epoch 137: boundary F1 0.749, seed peak recall 0.990, vertex peak
  recall 0.868, gap IoU 0.469, distance MAE 0.47 px.

Held-out TEST splits (no field seen in training; `runs/vertex/test_split_summary.csv`,
`docs/CONFLUENT_BENCHMARK.md`):

| Test split | Method | Adj F1 pair | Vertex F1 | Vertex prec. / rec. | PQ | Boundary F1 |
|---|---|---|---|---|---|---|
| hCEC (5 fields, ~1,665 cells and ~2,970 tricellular vertices each) | v4_confluent | 0.822 | 0.611 | 0.579 / 0.648 | 0.776 | 0.896 |
| | Cellpose-SAM (filled) | 0.830 | 0.635 | 0.721 / 0.581 | 0.790 | 0.880 |
| | v3_endo | 0.712 | 0.334 | 0.350 / 0.320 | 0.671 | 0.804 |
| alizarine (10 fields) | v4_confluent | 0.977 | 0.990 | 0.995 / 0.985 | 0.911 | 0.998 |
| | Cellpose-SAM (filled) | 0.989 | 0.984 | 0.979 / 0.989 | 0.898 | 1.000 |
| FlyWing (10 fields) | v4_confluent | 0.894 | 0.833 | 0.831 / 0.836 | 0.741 | 0.994 |
| | Cellpose-SAM (filled) | 0.966 | 0.849 | 0.834 / 0.864 | 0.795 | 0.996 |
| | v3_endo | 0.802 | 0.868 | 0.897 / 0.842 | 0.726 | 0.986 |
| HAEC test (86 fields) | v4_confluent | 0.521 | 0.292 | 0.294 / 0.304 | 0.612 | 0.852 |

Reading: on the cultured endothelial monolayer the fine-tune closes most of the gap to
Cellpose-SAM (vertex F1 0.61 vs 0.64, higher recall, lower precision, better boundary F1), it has
the best vertex F1 and PQ on corneal endothelium in situ, and it improves the sub-confluent HAEC
test as well. On FlyWing its vertex localization is worse than `v3_endo` (2.0 vs 1.7 px median),
so `v3_endo` remains the better choice for small-cell E-cadherin epithelia. Licence note: the hCEC
training images are CC BY-NC-ND 4.0; this checkpoint is for research evaluation.

Use: `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v4_confluent.pt pimorph benchmark --dataset hcec --methods neural ...`
