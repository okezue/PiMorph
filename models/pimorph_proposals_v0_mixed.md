# pimorph_proposals_v0_mixed.pt

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 32, depth 4, 8,643,270 parameters). Stage 2 of two.

- Resumed from `pimorph_proposals_v0_synth.pt` (stage 1, 40 epochs synthetic) and trained 20 more epochs (to epoch 60) on 2000 synthetic tiles plus 135 S-BIAD1540 EGM2 pseudo-label tiles (`cellpose_primary` policy, 24% of pixels ignored) and 40 VE-strat pseudo-label tiles (consensus policy). Validation stayed synthetic-only (200 tiles, seed 777).
- Trained 2026-09-17 on AWS g5.xlarge (A10G), torch 2.7.1+cu128, AMP, AdamW cosine, batch 8, crop 512. Log: `runs/neural/v2_mixed/train_log.jsonl`.
- Best epoch 57 synthetic validation: boundary F1 0.772, seed peak recall 0.991, vertex peak recall 0.884, gap IoU 0.540, distance MAE 0.37 px.
- Through the constrained decoder on 40 held-out synthetic tiles (`runs/pimorph_bench_v2/synth_summary.md`): adjacency F1 0.887, vertex F1 0.764, incident-set accuracy 0.857, PQ 0.893, validity 1.0 (stage 1: 0.889 / 0.733 / 0.860 / 0.891).
- Real fluorescence fields (no instance truth, self-consistency only): S-BIAD1540 6-dyne field `EGM2_regular_6dyn-24`, 169 cells and 271 tricellular vertices (classical proposals: 65 cells, 40 tricellular vertices, 31 isolated cells); render log-likelihood per pixel -4.97 vs -5.04 classical. Figure: `runs/pimorph_dev/sbiad1540_6dyn_neural_vs_classical.png`.
- Pseudo-labels came from Cellpose-SAM masks, so this model inherits Cellpose's biases on real data; it is not expert-validated. Does not transfer to phase contrast (see the stage-1 card).
- Input scale: cells about 10 to 28 px radius; downsample 2048x2048 VE-strat fields by 2 (`pimorph reconstruct --downsample 2`).

Use: `pimorph reconstruct --checkpoint models/pimorph_proposals_v0_mixed.pt --downsample 2 --posterior ...` or `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v0_mixed.pt pimorph benchmark --methods neural ...`.
