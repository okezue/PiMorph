# pimorph_proposals_v6_pool.pt

The checkpoint file is not in the repository: `python scripts/fetch_zenodo.py --models` downloads it from the PiMorph Zenodo data record (md5-verified) into `models/`.

Multi-head UNet proposal model (`pimorph.infer.neural.model.MultiHeadUNet`, base 48, depth 4,
19,441,446 parameters). The recommended checkpoint for confluent monolayers with a bright
membrane or junction channel. Lineage: `v1_multi` (synthetic + LIVECell + NeurIPS CellSeg) ->
`v3_endo` (+ HAEC, mCellSeg) -> `v4_confluent` (+ hCEC, alizarine, FlyWing train splits) ->
`v5_vertex` (hard-example weights for topologically missed vertices, vertex focus) -> `v6_pool`
(+ RPE monolayer train stacks, + 1,200 consensus pseudo-label tiles of real PECAM-1 HUVEC).

- `v5_vertex` (not shipped; `runs/neural/v5_vertex` on the devbox): `v4_confluent` resumed 40
  epochs (4x H100, 28 minutes, best epoch 176) with `scripts/make_vertex_miss_weights.py`
  weights (x4 around true vertices the v4 decode missed, x3 around spurious ones; on the
  training tiles v4 matched 67% of hCEC, 85% of FlyWing, 97% of alizarine and 36% of HAEC
  vertices) and `vertex_focus 1.0` within 5 px of every true vertex.
- `v6_pool`: `v5_vertex` resumed 40 epochs (4x H100, 33 minutes, best epoch 214) on the same
  tiles plus 208 RPE tiles (13 train stacks of the NIH-NEI monolayer set, weighted 2x, ROI
  ignored, mined weights) and 1,200 pseudo-label tiles from 300 real PECAM-1 HUVEC monolayer
  fields (Cellpose-SAM primary, v4 second opinion, disputed boundaries ignored).
- Synthetic validation at epoch 214: boundary F1 0.718, seed peak recall 0.990, vertex peak
  recall 0.885 (v4: 0.868), gap IoU 0.48, distance MAE 0.46 px.

Held-out TEST splits, no field seen in training, decoder settings chosen on the TRAIN splits
(`runs/frontier/final_summary.csv`, `docs/CONFLUENT_BENCHMARK.md`). "tuned" = TTA + vertex head
in the elevation (0.6 with nuclei, 0.3 without) + nucleus-consistency merges and 1 px smoothing
where a nuclear channel exists.

| Test split | Method | Adj F1 pair | Vertex F1 | Vertex prec. / rec. | pred / true vertices | PQ | Boundary F1 |
|---|---|---|---|---|---|---|---|
| hCEC (5 fields) | **v6_pool tuned** | **0.867** | **0.680** | 0.678 / 0.683 | 3014 / 2974 | **0.816** | **0.916** |
| | Cellpose-SAM (filled) | 0.830 | 0.635 | 0.721 / 0.581 | 2298 / 2974 | 0.790 | 0.880 |
| | v4_confluent (previous card) | 0.822 | 0.611 | 0.579 / 0.648 | 3377 / 2974 | 0.776 | 0.896 |
| alizarine (10) | v6_pool tuned | 0.987 | 0.992 | 0.994 / 0.990 | 568 / 570 | 0.920 | 0.999 |
| | Cellpose-SAM (filled) | 0.989 | 0.984 | 0.979 / 0.989 | 576 / 570 | 0.898 | 1.000 |
| FlyWing (10) | v6_pool tuned | 0.911 | **0.875** | 0.869 / 0.881 | 1278 / 1260 | 0.780 | 0.994 |
| | Cellpose-SAM (filled) | **0.966** | 0.849 | 0.834 / 0.864 | 1306 / 1260 | **0.795** | 0.996 |
| RPE monolayer (20 tiles, 5 stacks) | v6_pool tuned | 0.605 | **0.327** | 0.291 / 0.381 | 265 / 209 | **0.556** | **0.731** |
| | Cellpose-SAM (filled) | **0.622** | 0.287 | 0.254 / 0.331 | 269 / 209 | 0.548 | 0.686 |
| HAEC (86 fields, sub-confluent) | v6_pool tuned | **0.536** | **0.352** | 0.293 / 0.453 | 151 / 92 | **0.626** | **0.858** |
| | Cellpose-SAM (filled) | 0.314 | 0.039 | 0.039 / 0.043 | 110 / 92 | 0.465 | 0.708 |

Reading: on the cultured endothelial monolayer PiMorph now leads Cellpose-SAM on every
structural metric with a calibrated vertex count (3,014 predicted vs 2,974 true); Cellpose-SAM
keeps the better vertex localization (1.17 vs 1.28 px median) and incident-set accuracy (0.91
vs 0.86). On FlyWing PiMorph has the better vertices and Cellpose-SAM the better adjacency and
PQ. On HAEC the vertex head over-predicts (151 vs 92 vertices; precision 0.29) and F1 is bounded
by the sub-confluent reference. Licence note: hCEC training images are CC BY-NC-ND 4.0; RPE is
CC0; PECAM-1 pseudo-labels come from CC BY 4.0 images. Research evaluation checkpoint.

Use: `PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v6_pool.pt PIMORPH_NEURAL_TTA=1 \
PIMORPH_DECODER_PARAMS='{"nucleus_merge": true, "boundary_smooth_sigma": 1.0, "vertex_weight": 0.6}' \
pimorph benchmark --dataset hcec --methods neural ...`
