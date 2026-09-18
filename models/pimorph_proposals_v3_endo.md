# pimorph_proposals_v3_endo.pt

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 48, depth 4,
19,441,446 parameters). Same recipe as `v2_endo`, retrained after the HAEC reference
derivation was corrected.

- Resumed from `pimorph_proposals_v1_multi.pt` (epoch 60) and fine-tuned to epoch 100 on the
  xAI devbox (8x H100, `torchrun` DDP, batch 8 per GPU, AdamW 1.5e-4 x 8 cosine, AMP). Wall time
  19 minutes. Log: `runs/neural/v3_endo/train_log.jsonl`; `best.pt` is epoch 98 (best validation
  loss).
- Training tiles as for `v2_endo` (3,132 HAEC train tiles weighted twice, 2,914 mCellSeg train
  tiles, 2,000 synthetic, 175 pseudo-label tiles), except that the HAEC tiles were rebuilt from
  the corrected reference: body components 8-connected, no sub-30 px specks, 4-connected labels.
  The old derivation made about 40% of HAEC "cells" 1 to 3 px fringe specks that became seed
  and boundary targets.
- Synthetic validation at epoch 98: boundary F1 0.739, seed peak recall 0.990, vertex peak
  recall 0.853, gap IoU 0.428, distance MAE 0.53 px.

Held-out results through the constrained decoder with the outside-cell flood exclusion
(`DecoderParams.outside_px = 0`) and nucleus-guided seeding, all methods scored against the
corrected reference (`runs/vertex/`, `docs/ENDOTHELIAL_RESULTS.md`):

| Dataset (held out) | Method | n | Adj F1 pair / component | Vertex F1 (pred / true vertices) | PQ | AP50 | Boundary F1 |
|---|---|---|---|---|---|---|---|
| HAEC test | v3_endo | 86 | 0.505 / 0.306 | 0.262 (95 / 92) | 0.600 | 0.616 | 0.844 |
| | v2_endo | 86 | 0.504 / 0.308 | 0.261 (112 / 92) | 0.597 | 0.612 | 0.843 |
| | Cellpose-SAM (seams filled) | 86 | 0.314 / 0.087 | 0.039 (110 / 92) | 0.465 | 0.487 | 0.708 |
| mCellSeg test (DIC) | Cellpose-SAM (seams filled) | 40 | 0.170 / 0.060 | 0.117 | 0.215 | 0.208 | 0.203 |
| | v3_endo | 40 | 0.064 / 0.013 | 0.014 | 0.134 | 0.119 | 0.349 |

Reading: the corrected reference and the decoder fix account for nearly all of the change from
the `v2_endo` card (vertex F1 0.067 to 0.26, adjacency 0.243 to 0.505); retraining on corrected
tiles changed accuracy by less than 0.01 but calibrated the vertex count (95 predicted per field
against 92 true, v2_endo 112). On DIC (mCellSeg) the model still over-segments; Cellpose-SAM
remains ahead there. Zero-shot on confluent monolayers with real truth (hCEC NCAM, FlyWing
E-cadherin, alizarine corneal endothelium) see `docs/CONFLUENT_BENCHMARK.md`.

Use: `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v3_endo.pt pimorph benchmark --dataset haec --methods neural ...`
or `pimorph reconstruct --checkpoint models/pimorph_proposals_v3_endo.pt ...`.
