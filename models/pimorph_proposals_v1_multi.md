# pimorph_proposals_v1_multi.pt

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 48, depth 4; about 19.4 M parameters). First model trained on real instance ground truth.

- Trained 2026-09-17 on an xAI devbox (`fou` cluster, 8x H100 80 GB) with `torchrun --nproc_per_node 8` DDP, 60 epochs, batch 8 per GPU (64 effective), crop 512, AdamW 3e-4 x 8 (linear scaling) cosine, AMP bf16. Wall time 52 minutes. Log: `runs/neural/v3_multi/train_log.jsonl`; campaign log `runs/scale/campaign.log`.
- Training tiles (about 18,000): 8,000 + 2,000 synthetic (`pimorph synth`, seeds 5000 to 5015 and 1000 to 1007), 2,276 LIVECell train tiles (real COCO instance masks, 576 images subsampled across the 8 cell lines), 5,455 NeurIPS 2022 CellSeg tiles (real instance masks, 1,101 labelled images across brightfield, fluorescence, phase contrast, DIC), 135 S-BIAD1540 and 40 VE-strat pseudo-label tiles. Real GT tiles built by `scripts/make_gt_tiles.py` (exact targets from masks, no nuclei channel).
- Validation (synthetic only, 400 tiles seed 9999) at the best epoch 59: boundary F1 0.739, seed peak recall 0.991, vertex peak recall 0.856, gap IoU 0.459, distance MAE 0.47 px.

Benchmarks through the constrained decoder (`runs/scale/bench_*`), 400 images each:

| Dataset | Split status | neural (v1_multi) adj F1 / vertex F1 / PQ | Cellpose-SAM adj F1 / vertex F1 / PQ | classical adj F1 / vertex F1 / PQ |
|---|---|---|---|---|
| Synthetic (seed 9999) | held out | 0.861 / 0.693 / 0.868 | not run | 0.368 / 0.438 / 0.483 |
| LIVECell test (8 cell lines) | held out (trained on train split) | 0.230 / 0.164 / 0.403 | 0.617 / 0.341 / 0.639 | 0.021 / 0.073 / 0.072 (n=100) |
| NeurIPS CellSeg | IN-SAMPLE (labelled images were in training) | 0.097 / 0.058 / 0.390 | 0.382 / 0.531 / 0.597 | 0.025 / 0.099 / 0.042 (n=100) |

Reading:
- Adding 2,276 real LIVECell tiles moved phase-contrast adjacency F1 from 0.000 (synthetic-only v0) to 0.230 on a held-out split, best on BV2 and SkBr3 (0.38 to 0.39), worst on Huh7 and SHSY5Y (0.11). Cellpose-SAM (trained on orders of magnitude more real data) remains far ahead on real images; this model is not a replacement for it there.
- The NeurIPS number is low even in-sample, so the mixed-modality set is under-fitted at 60 epochs with the real tiles at about 40% of the mix; it is reported for completeness, not as a generalization result.
- Synthetic accuracy dropped slightly from v0 (0.889 / 0.733 / 0.891 on 40 tiles) to 0.861 / 0.693 / 0.868 on 400 tiles; part of that is the larger, harder validation sample and part is capacity spent on real data.
- Validity is 1.0 everywhere by construction of the decoder.

Use: `pimorph reconstruct --checkpoint models/pimorph_proposals_v1_multi.pt ...` or `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v1_multi.pt pimorph benchmark --methods neural ...`. For endothelial fluorescence at 2048 px, downsample by 2 as for v0.
