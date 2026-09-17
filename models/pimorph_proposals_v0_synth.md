# pimorph_proposals_v0_synth.pt

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 32, depth 4, 8,643,270 parameters).

- Stage 1 only: 40 epochs on 2000 synthetic 512x512 tiles (`pimorph synth`, seeds 1000 to 1007), validated on 200 synthetic tiles (seed 777). No real data seen.
- Trained 2026-09-17 on AWS g5.xlarge (A10G), torch 2.7.1+cu128, AMP, AdamW 3e-4 cosine, batch 8, crop 512. Log: `runs/neural/v1_synth/train_log.jsonl`.
- Best epoch 34 synthetic validation: boundary F1 0.760, seed peak recall 0.991, vertex peak recall 0.884, gap IoU 0.448, distance MAE 0.38 px.
- Through the constrained decoder on 40 held-out synthetic tiles (`runs/pimorph_bench/synth_summary.md`): adjacency F1 0.889 (classical 0.447), vertex F1 0.733 (0.503), incident-set accuracy 0.860 (0.372), PQ 0.891 (0.554), validity 1.0.
- Does not transfer to phase contrast (LIVECell adjacency F1 0.000 vs Cellpose-SAM 0.558). Transfers qualitatively to fluorescence VE-cadherin (VE-strat crop: 43 cells, all vertices trivalent, boundary/interior ratio 1.63; figure `runs/pimorph_dev/ve_strat_neural_vs_classical.png`). No expert instance truth exists for that domain yet.
- Input scale: cells about 10 to 28 px radius; downsample 2048x2048 VE-strat fields by 2 first (as `make_pseudolabel_tiles` does).

Use: `pimorph reconstruct --checkpoint models/pimorph_proposals_v0_synth.pt ...` or `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v0_synth.pt pimorph benchmark --methods neural ...`.
