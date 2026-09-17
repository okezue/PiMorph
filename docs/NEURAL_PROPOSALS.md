# Neural proposals

`pimorph.infer.neural` trains a multi-head UNet that produces the dense evidence maps consumed by the constrained decoder. It is an alternative front end to `ClassicalProposer`: both return the same `ProposalMaps` container, and everything downstream (decoder, energy, posterior, benchmark) is unchanged.

Status: the model, dataset, losses, training loop, pseudo-labelling, and `NeuralProposer` are implemented and covered by `tests/pimorph/test_neural.py` (a tiny UNet trained for two steps on CPU). Two trained checkpoints are committed through git LFS, each with a model card next to it:

- `models/pimorph_proposals_v0_synth.pt` (stage 1): 40 epochs on 2000 synthetic tiles, no real data. Card: `models/pimorph_proposals_v0_synth.md`.
- `models/pimorph_proposals_v0_mixed.pt` (stage 2): resumed from stage 1, 20 more epochs with 135 S-BIAD1540 and 40 VE-strat pseudo-label tiles added. Card: `models/pimorph_proposals_v0_mixed.md`.

Training logs are in `runs/neural/v1_synth/` and `runs/neural/v2_mixed/`; benchmark numbers are in `docs/BENCHMARKS.md`. Both were trained on an AWS g5.xlarge following `docs/AWS_RUNBOOK.md`. Any number attributed to "neural proposals" must come from a run you can point to (`train_log.jsonl` plus `pimorph benchmark` output).

Torch is optional. Nothing outside `pimorph.infer.neural` imports it, and `tests/pimorph/test_neural.py` is skipped when torch is absent (`pytest.importorskip`, marker `torch`). Install with the `torch` extra.

## Inputs: six channels with presence indicators

`data.build_input(geometry, nuclei, junction, shape)` builds a `(6, H, W)` float32 tensor shared by training and inference:

| Index | Channel | Content |
|---|---|---|
| 0 | geometry | Robust-normalized ([1st, 99.8th percentile] to [0, 1]) channel used to place boundaries |
| 1 | nuclei | Robust-normalized nuclei channel, zeros when absent |
| 2 | junction | Robust-normalized junction channel, zeros when absent |
| 3 | present_geometry | Constant 1 (geometry is always given) |
| 4 | present_nuclei | Constant 1 when channel 1 is present, else 0 |
| 5 | present_junction | Constant 1 when channel 2 is present, else 0 |

The presence indicators let one network serve fields with and without a nuclei channel and with or without an independent membrane marker. During training `TileDataset` applies channel dropout: with probability `p_membrane_as_geometry` (default 0.5) the synthetic `membrane` channel is used as geometry instead of `junction`; nuclei are kept with probability `p_nuclei` (0.85) and the junction channel with probability `p_junction` (0.9). Evaluation mode is deterministic: geometry is the junction channel, nuclei and junction present whenever the tile has them, center crop.

## Heads

`model.MultiHeadUNet(in_channels=6, base=32, depth=4)` is a plain UNet (double 3x3 convolutions with GroupNorm and SiLU, max-pool down, bilinear up with skip concatenation) with one shared output block producing six channels in `HEADS` order:

| Head | Output | Target |
|---|---|---|
| `boundary` | logit | `find_boundaries(labels, connectivity=1, mode="inner")`: nonzero pixels with a 4-neighbor of a different label |
| `distance` | regression of `signed_distance / 16` | Euclidean distance from the pixel center to the nearest crack, positive in cells, negative in background, clipped to 16 px (`DISTANCE_SCALE`) |
| `seed` | logit | Unit-peak Gaussian heatmap (sigma 3 px) at nuclei, or cell centroids when nuclei are unknown |
| `vertex` | logit | Unit-peak Gaussian heatmap (sigma 2 px) at vertices with at least three incident cells, taken from `extract_complex` |
| `gap` | logit | Enclosed background (gap faces of the complex) |
| `log_sigma` | per-pixel log scale | Laplace scale for the distance head (no direct target) |

Targets are computed exactly from the label image by `pimorph.synth.targets.make_targets`, which also writes an `outer` mask (background connected to the border) that is stored but not trained on. The default network has between 8 and 10 million parameters (`test_model_forward_shape_and_parameter_count`); spatial sizes that are not multiples of `2**depth` are handled by interpolating to the skip size, and `NeuralProposer` pads tiles to the divisor anyway.

## Losses

`losses.MultiHeadLoss` combines per-head terms, each normalized by the sum of the weight map `w` so ignored pixels contribute nothing:

| Term | Function | Default weight | Notes |
|---|---|---|---|
| boundary | `bce_with_pos_weight` | 1.0 | `pos_weight = 3.0` |
| distance | `heteroscedastic_l1` | 1.0 | Laplace NLL `abs(e) * exp(-s) + s`, `s = log_sigma` clamped to [-6, 6] |
| seed | `bce_with_pos_weight` | 1.0 | `pos_weight = 5.0` |
| vertex | `bce_with_pos_weight` | 1.0 | `pos_weight = 5.0` |
| gap | `focal_bce` | 0.5 | focal gamma 2.0 |
| cldice | `soft_cldice` | 0.3 | soft clDice (Shit et al.) between `sigmoid(boundary) * w` and the boundary target, 3 skeletonization iterations |

`test_losses_finite_and_overfit_single_tile` checks that all terms are finite, that a tiny model overfits one tile, and that a zero weight map gives zero loss.

## Training loop and CLI

`train.train(TrainConfig)` uses AdamW, cosine learning-rate schedule with linear warmup (`warmup_fraction`, default 0.03), gradient clipping at `grad_clip = 1.0`, and automatic mixed precision on CUDA only (bfloat16 when supported, otherwise float16 with a GradScaler; `amp` is ignored on MPS and CPU). Every epoch it evaluates on the validation loader, appends one JSON line to `train_log.jsonl` (loss, per-head losses, and the metrics below), writes `last.pt` with optimizer and scheduler state, and writes `best.pt` when the validation loss improves. `config.json` holds the resolved `TrainConfig`. Checkpoints hold `{"model", "config", "epoch", "val"}` and, for `last.pt`, `optimizer`, `scheduler`, `best_val`; `--resume` restarts from one.

Validation metrics computed by `train._MetricAccumulator` on valid pixels: `boundary_f1`, `boundary_precision`, `boundary_recall` at probability 0.5; `seed_peak_recall` (GT maxima with a predicted peak within 4 px); `vertex_peak_recall` (within 3 px); `gap_iou`; `distance_mae_px`.

The CLI is `python -m pimorph.infer.neural.train`; there is no `pimorph train` subcommand. Flags mirror `TrainConfig` (verified with `--help`):

```
--train-dirs DIR [DIR ...]   directories or .npz files (required)
--val-dirs DIR [DIR ...]     optional; without it, --val-fraction of the training fields is held out by field
--out DIR                    output directory (alias --out-dir; default runs/neural)
--epochs N (20)  --batch-size N (8)  --crop N (256)  --lr F (3e-4)  --weight-decay F (1e-4)
--base N (32)  --depth N (4)  --amp | --no-amp (default on)  --num-workers N (0)  --seed N (0)
--device auto|cuda|mps|cpu   --log-every N (20)  --max-steps N  --resume PATH  --val-fraction F (0.1)
--p-membrane-as-geometry F (0.5)  --p-nuclei F (0.85)  --p-junction F (0.9)
--warmup-fraction F (0.03)  --grad-clip F (1.0)
```

Example (the docstring in `train.py`):

```bash
python -m pimorph.infer.neural.train --train-dirs data/tiles/synth_train \
    --val-dirs data/tiles/synth_val --out runs/neural/v1 --epochs 40
```

A local smoke run on Apple silicon uses `--device mps --no-amp` with a small `--base` and `--max-steps`.

## Data sources

Tiles are `.npz` files. Both sources carry `junction`, `labels`, and the targets `boundary`, `signed_distance`, `seed`, `vertex`, `gap`, `outer`.

Synthetic tiles (`pimorph synth`, `pimorph.synth.targets.make_dataset`). Each tile is a Lloyd-relaxed anisotropic Voronoi tissue (`synth/tissue.py`) rendered through a forward model (`synth/render.py`) into `junction`, `membrane`, and `nuclei` channels plus a `broken_mask`. Vertex and gap targets are exact because they come from the tissue's own label image. The default samplers draw cell area 600 to 2500 px, elongation 1 to 3, gap area fraction 0 to 0.08 (30 % of tiles gap-free), boundary jitter 0 to 2.5 px, binucleate fraction up to 0.06, anucleate fraction up to 0.04, PSF sigma 0.8 to 2.5 px, junction width 1 to 3 px, SNR 3 to 30 (log-uniform), photobleach gradient 0 to 0.5, and broken-junction fraction 0 to 0.4. `manifest.csv` records every drawn parameter per tile together with the realized cell, gap, vertex, and nucleus counts.

```bash
pimorph synth --n 2000 --out data/tiles/synth_train --shape 512 --seed 0
pimorph synth --n 200  --out data/tiles/synth_val   --shape 512 --seed 1
```

Pseudo-label tiles (`pimorph.infer.neural.pseudolabel.make_pseudolabel_tiles(manifest_csv, out_dir, tile=512, stride=512, max_fields=None, use_cellpose=True, ...)`). For each field of a manifest with explicit channel roles (`io/manifest.py`), `segment_field` runs `ClassicalProposer` plus `ConstrainedDecoder` and, when cellpose and torch are installed, Cellpose-SAM on the same field (fields whose classical cell radius exceeds 35 px are downsampled by 2 first). `consensus_labels(labels_a, labels_b, iou_thresh=0.7, boundary_tol_px=3.0)` matches faces through `complex.matching.match_faces` and keeps only the intersection of face pairs with IoU at least 0.7; the symmetric difference of matched faces, all unmatched faces, and pixels within 3 px of a boundary present in only one segmentation become `ignore`, which zeroes the loss weight there. Without Cellpose, `classical_only_ignore` masks label boundaries that the boundary map does not support (`support < 0.3`). Tiles are written as `field_XXX_rYYYY_cZZZZ.npz` with `labels`, `ignore`, `nuclei` (zeros with `has_nuclei=False` when absent), `membrane` when the geometry channel is an independent membrane marker, and a `manifest.csv` with `source_field`, `consensus` (`classical+cellpose` or `classical_only`), `ignore_frac`, `cell_radius_px`, `condition`, and `dataset`. `test_consensus_labels_ignores_merge_and_keeps_agreement` verifies that a merged cell pair is fully ignored while untouched cells keep one consistent label.

Pseudo-labels inherit the biases of both segmenters. They are for domain adaptation; held-out label-image ground truth is the arbiter (see Evaluation).

## Splits

`data.split_by_field(items, val_fraction, seed)` splits group-wise: tiles from one `source_field` never straddle the two splits (`test_split_by_field_keeps_fields_disjoint`). It accepts a manifest DataFrame (`file`, optional `source_field` and `path` columns) or plain paths (each path its own group). `train._resolve_splits` uses it when `--val-dirs` is empty. When you give explicit `--val-dirs`, hold out whole fields (or whole datasets) yourself; per-tile random splits of one field leak.

## How `NeuralProposer` plugs into the decoder

```python
from pimorph.infer import ConstrainedDecoder, DecoderParams
from pimorph.infer.neural import NeuralProposer

prop = NeuralProposer("runs/neural/v1/best.pt", device="auto", tile=512, overlap=64)
maps = prop(geometry, nuclei, junction)          # same call signature as ClassicalProposer
res = ConstrainedDecoder(pixel_size_um=px).decode(maps, DecoderParams(cell_radius_px=maps.meta["cell_radius_px"]))
```

`NeuralProposer.__call__` runs `tiled_predict` (overlapping tiles with cosine blending, reflect padding so no true border pixel sits at a tile edge, padding to the network divisor), applies sigmoids to `boundary`, `seed`, `vertex`, `gap`, rescales `seed` to peak 1, converts `distance` and `sigma` back to pixels, estimates the cell radius from the nucleus radius (`cell_to_nucleus_ratio = 2.5`) or, without nuclei, from the distance values at provisional seed peaks, detects seed points with `peak_local_max` at `min_distance = max(3, 0.6 * cell_radius)`, normalizes their scores by the 90th percentile, builds a tissue mask from `(gap < 0.5) | (distance > 0)`, and returns `ProposalMaps(source="neural")` with `vertex` and `distance` filled. `meta` records the checkpoint path, tile, overlap, device, `cell_radius_px`, `ridge_width_px`, and which channels were used. `test_neural_proposer_maps_and_decode` checks ranges, shapes, ordering of seed scores, and that the decoded complex validates.

The extra `vertex` and `distance` maps are carried but not yet consumed by `ConstrainedDecoder` or `complex_energy`; `geometry.refine_vertices` accepts any probability map and is the natural consumer of `vertex`.

## Evaluation protocol

1. Synthetic exact ground truth. `pimorph benchmark --dataset synth --root data/tiles/synth_val --methods classical` scores against the tissue's own labels, where vertices, gap faces, and incident sets are exact. Adding a neural method requires a `bench/run.py` method entry that wraps `NeuralProposer`; none is registered yet (`METHODS` has `gt`, `classical`, `cellpose_sam`).
2. Real label-image ground truth. LIVECell (phase contrast, COCO polygons) and mCellSeg (HUVEC, Kaggle) through `bench/datasets.py`; both have background slivers filled below 12 px so tricellular vertices are cell-cell-cell. mCellSeg is not downloaded in this workspace (`data/mcellseg` absent; needs `~/.kaggle/kaggle.json`).
3. Held out by field. Fields used for pseudo-labels must not appear in the evaluation set; use `split_by_field` on the pseudo-label manifest and keep at least one whole dataset unseen for a held-out-domain number.
4. Calibration. Edge existence probabilities from `PosteriorEnsemble.contact_probabilities()` against ground-truth adjacency go through `metrics.calibration` (ECE, Brier, log score, reliability diagram, risk-coverage curve).

Report the checkpoint path, the training log, the tile sets and their manifests, and the benchmark command with every number.
