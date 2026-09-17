# pimorph_proposals_v2_endo.pt

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 48, depth 4, 19,441,446 parameters). First PiMorph model trained on endothelial ground truth.

- Resumed from `pimorph_proposals_v1_multi.pt` (epoch 60) and fine-tuned to epoch 100 (40 epochs) on the xAI devbox (8x H100, `torchrun` DDP, batch 8 per GPU, AdamW 1.5e-4 x 8 cosine, AMP). Wall time 21 minutes. Log: `runs/neural/v2_endo/train_log.jsonl`.
- Training tiles (about 11,000 per epoch): 3,132 HAEC train tiles (human aortic endothelial cells, inverted cytoplasmic GFP as geometry channel plus real Hoechst nuclei channel; 348 fields, weighted twice), 2,914 mCellSeg train tiles (DIC HUVEC and HEK-293T; 160 images), 2,000 synthetic tiles, 175 S-BIAD1540 / VE-strat pseudo-label tiles. Splits are field-disjoint (`runs/endo/splits.json`: HAEC field id % 5 == 0 and every 5th mCellSeg image are test).
- Synthetic validation at the best epoch 92: boundary F1 0.744, seed peak recall 0.990, vertex peak recall 0.850, gap IoU 0.437, distance MAE 0.52 px.

Held-out endothelial test results through the constrained decoder (`runs/endo/*_test_*`):

| Dataset (held out) | Method | n | Adj F1 pair / component | Vertex F1 | PQ | AP50 | Boundary F1 | Pred cells / GT cells |
|---|---|---|---|---|---|---|---|---|
| HAEC test (GFP + Hoechst) | v2_endo | 86 | 0.243 / 0.145 | 0.067 | 0.407 | 0.352 | 0.842 | 679 / 1151 |
| | Cellpose-SAM (seams filled) | 86 | 0.324 / 0.092 | 0.038 | 0.347 | 0.323 | 0.707 | 753 / 1151 |
| | v1_multi (no endothelial data) | 86 | 0.001 / 0.000 | 0.004 | 0.006 | 0.005 | 0.198 | 196 / 1151 |
| | classical | 86 | 0.040 / 0.005 | 0.002 | 0.109 | 0.097 | 0.273 | 426 / 1151 |
| mCellSeg test (DIC, 10 HUVEC + 30 HEK) | v2_endo | 40 | 0.051 / 0.011 | 0.011 | 0.131 | 0.115 | 0.354 | 266 / 89 |
| | Cellpose-SAM (seams filled) | 40 | 0.169 / 0.060 | 0.117 | 0.214 | 0.207 | 0.203 | 45 / 89 |
| | v1_multi | 40 | 0.001 / 0.000 | 0.002 | 0.011 | 0.009 | 0.203 | 1141 / 89 |

Reading:
- On HAEC, endothelial fine-tuning moved PQ from 0.006 (v1_multi) to 0.407 and boundary F1 to 0.842, ahead of Cellpose-SAM on instance quality (PQ 0.347) and on component-level adjacency (0.145 vs 0.092), while Cellpose-SAM keeps the higher pair-level adjacency F1 (0.324 vs 0.243). Both remain poor on multicellular vertices: v2_endo predicts about 1,060 vertices per field against 124 in the ground truth (158 splits per field), so its vertex precision is low; Cellpose-SAM predicts the right number (110) but only 0.038 F1, so its vertices sit in the wrong places (median localization error 2.0 px vs 1.4 px). HAEC cultures are sub-confluent (about 46% background), so the vertex ground truth is small.
- On mCellSeg (DIC), v2_endo over-segments (266 predicted vs 89 GT cells) and Cellpose-SAM under-segments (45); neither is adequate, and the 10 HUVEC test images are too few to separate cell lines.
- All outputs are valid complexes by construction (validity 1.0 everywhere).
- Ground truth caveat: HAEC instances are derived from a 4-class semantic annotation (body/border/nucleus/background) by flooding bodies through the border class; mCellSeg masks are expert instance annotations.

Use: `pimorph reconstruct --checkpoint models/pimorph_proposals_v2_endo.pt ...` or `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v2_endo.pt pimorph benchmark --dataset haec --methods neural ...`. For a cytoplasmic reporter the geometry channel must be inverted first (the `haec` loader marks `boundary_polarity = "dark"` so `pimorph benchmark` does this automatically; `pimorph reconstruct` currently assumes bright boundaries).
