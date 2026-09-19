# PiMorph models

This is the consolidated model card for PiMorph's neural proposal checkpoints. The
[study explainer](../EXPLAINER.md) describes the biological questions, reconstruction method,
complete evaluation, and limitations; the [repository README](../README.md) covers installation
and tool usage.

## Obtain and choose a checkpoint

Checkpoints, training configurations, and logs are distributed through the PiMorph data record,
[10.5281/zenodo.22839866](https://doi.org/10.5281/zenodo.22839866). The concept DOI follows new
versions; the study reported here uses version 1.1.0,
[10.5281/zenodo.22839867](https://doi.org/10.5281/zenodo.22839867).

```bash
python scripts/fetch_zenodo.py --models
```

The helper prints the resolved record and version, verifies the archive MD5, and extracts the
checkpoints into `models/` and supporting logs into `runs/neural/`. Record that resolved version
when reproducing a result. Historical root/docs/models Markdown is excluded during extraction by default, so the current model card is preserved. `models/*.pt` is ignored by Git. The small legacy junction-morphology
classifiers, `ajmorph_classifier*.joblib`, remain in the repository and belong to the legacy
pipeline, not the neural proposal model.

| Checkpoint | Approximate size | Intended use and position in the study |
|---|---:|---|
| `pimorph_proposals_v6_pool.pt` | 78 MB | Current starting point for confluent monolayers with a bright membrane or junction channel; pooled training and vertex-focused refinement |
| `pimorph_proposals_v5_vertex.pt` | 78 MB | Intermediate vertex-focused model; hard-example loss weights and vertex emphasis |
| `pimorph_proposals_v4_confluent.pt` | 78 MB | First in-domain confluent-monolayer fine-tune; useful for the training/decoder ablations |
| `pimorph_proposals_v3_endo.pt` | 78 MB | Endothelial training with the corrected HAEC reference; historical small-cell E-cadherin baseline |
| `pimorph_proposals_v2_endo.pt` | 78 MB | First endothelial fine-tune; trained with the older HAEC reference derivation |
| `pimorph_proposals_v1_multi.pt` | 78 MB | First real-instance model: synthetic, LIVECell, and NeurIPS CellSeg training |
| `pimorph_proposals_v0_mixed.pt` | 35 MB | Synthetic model adapted with real fluorescence pseudo-labels |
| `pimorph_proposals_v0_synth.pt` | 35 MB | Synthetic-only starting model |

The MD5-verified version 1.1.0 model archive contains all eight checkpoints above. `v5_vertex`
is included even though the earlier v6 card called it unshipped; it had no individual card. Its
configuration, logs, and evaluation are retained under `runs/neural/v5_vertex/` and
`runs/frontier/`. The archive checksum is `bebb467a5d85d47cf345a6252d7fb8dc`.

The recommendation is domain-specific. v6 improves several structural scores on hCEC and HAEC,
but does not win every metric or dataset. Cellpose-SAM remains stronger on the reported LIVECell
phase-contrast and mCellSeg DIC tests. Bright boundary fluorescence, DIC, phase contrast, and an
inverted cytoplasmic reporter are different input domains; select and validate a model against
representative truth for the intended experiment.

## Network and inference contract

All versions use `pimorph.infer.neural.model.MultiHeadUNet`, depth 4, with GroupNorm, SiLU,
bilinear upsampling, and skip connections. v0 uses base width 32 and 8,643,270 parameters;
v1 through v6 use base width 48 and 19,441,446 parameters. v1 was trained from scratch at the
larger width, rather than resumed from v0.

The six input channels are normalized geometry, nuclei, and junction images, followed by their
three presence indicators. Missing optional image channels are zeroed. Six output heads provide
boundary logits, signed distance, seed logits, vertex logits, gap logits, and the log scale of
the distance uncertainty model. Signed distance is scaled by 16 px during training. The proposal
maps are converted into an instance segmentation and cell complex by the constrained decoder.
A structurally valid complex is not evidence that its cells or contacts are biologically correct.

For fluorescence fields resembling the original VE-strat/S-BIAD1540 examples, the v0/v1 training
scale was approximately 10–28 px cell radius; the original 2048 × 2048 VE-strat fields were
reconstructed after 2× downsampling. For HAEC, cytoplasmic GFP must be inverted to make boundaries
bright. The `haec` benchmark loader marks dark-boundary polarity and handles this automatically;
`pimorph reconstruct` assumes bright boundaries, so prepare that input explicitly.

A basic reconstruction selects a checkpoint explicitly:

```bash
pimorph reconstruct --checkpoint models/pimorph_proposals_v6_pool.pt \
  --manifest path/to/manifest.csv --out runs/my_reconstruction
```

For the reported tuned hCEC evaluation, the split and decoder settings are part of the result:

```bash
PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v6_pool.pt \
PIMORPH_NEURAL_TTA=1 \
PIMORPH_ID_LIST=runs/vertex/splits_confluent.json:hcec_test \
PIMORPH_DECODER_PARAMS='{"nucleus_merge": true, "boundary_smooth_sigma": 1.0, "vertex_weight": 0.6}' \
  pimorph benchmark --dataset hcec --methods neural --out runs/hcec_v6_reproduction
```

The final benchmark uses test-time augmentation for each tuned neural run. hCEC and RPE use the
parameters above. FlyWing and alizarine use `{"vertex_weight": 0.3}`. HAEC uses
`{"vertex_weight": 0.6}` without the hCEC/RPE smoothing or merge settings; that selection is saved
in `runs/frontier/haec_decoder_params.json`. These settings were selected on training splits,
then used on field- or stack-disjoint test splits. See
[`run_final_bench.sh`](../infra/xai/run_final_bench.sh) for the exact dataset/split mapping.

## Current held-out evaluation

These are the archived v6 results from
[`runs/frontier/final_summary.csv`](../runs/frontier/final_summary.csv), scored through the
constrained decoder. Values are means over evaluated fields/tiles, not pooled counts. RPE's
20 tiles come from five held-out stacks; they are not 20 independent biological samples.
The Cellpose-SAM baseline uses seam filling before complex extraction. Metrics are rounded to
three decimals, and no difference here is a claim of statistical significance.

| Dataset | Method | Fields/tiles | Adjacency F1 | Vertex F1 | PQ | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|
| hCEC | v6 tuned | 5 | 0.867 | 0.680 | 0.816 | 0.916 |
| hCEC | Cellpose-SAM | 5 | 0.830 | 0.635 | 0.790 | 0.880 |
| Alizarine | v6 tuned | 10 | 0.987 | 0.992 | 0.920 | 0.999 |
| Alizarine | Cellpose-SAM | 10 | 0.989 | 0.984 | 0.898 | 1.000 |
| FlyWing | v6 tuned | 10 | 0.911 | 0.875 | 0.780 | 0.994 |
| FlyWing | Cellpose-SAM | 10 | 0.966 | 0.849 | 0.795 | 0.996 |
| RPE | v6 tuned | 20 | 0.605 | 0.327 | 0.556 | 0.731 |
| RPE | Cellpose-SAM | 20 | 0.622 | 0.287 | 0.548 | 0.686 |
| HAEC | v6 tuned | 86 | 0.536 | 0.352 | 0.626 | 0.858 |
| HAEC | Cellpose-SAM | 86 | 0.314 | 0.039 | 0.465 | 0.708 |

The accompanying vertex diagnostics prevent a higher F1 from concealing other errors. Predicted
and reference counts below are per-field/tile means, rounded to integers. Localization is the
summary's mean of field-level median localization errors; lower is better.

| Dataset | Method | Precision / recall | Predicted / reference vertices | Incident-set accuracy | Localization (px) |
|---|---|---:|---:|---:|---:|
| hCEC | v6 tuned | 0.678 / 0.683 | 3,014 / 2,974 | 0.858 | 1.283 |
| hCEC | Cellpose-SAM | 0.721 / 0.581 | 2,298 / 2,974 | 0.910 | 1.166 |
| Alizarine | v6 tuned | 0.994 / 0.990 | 568 / 570 | 0.985 | 1.000 |
| Alizarine | Cellpose-SAM | 0.979 / 0.989 | 576 / 570 | 0.991 | 1.000 |
| FlyWing | v6 tuned | 0.869 / 0.881 | 1,278 / 1,260 | 0.912 | 1.590 |
| FlyWing | Cellpose-SAM | 0.834 / 0.864 | 1,306 / 1,260 | 0.973 | 1.473 |
| RPE | v6 tuned | 0.291 / 0.381 | 265 / 209 | 0.741 | 1.728 |
| RPE | Cellpose-SAM | 0.254 / 0.331 | 269 / 209 | 0.759 | 1.766 |
| HAEC | v6 tuned | 0.293 / 0.453 | 151 / 92 | 0.700 | 1.253 |
| HAEC | Cellpose-SAM | 0.039 / 0.043 | 110 / 92 | 0.607 | 1.398 |

On hCEC, v6 improves adjacency F1, vertex F1, PQ, boundary F1, and vertex recall, while
Cellpose-SAM has higher vertex precision and incident-set accuracy and lower localization error.
The approximately correct v6 vertex count does not imply that individual vertices are correct.
On FlyWing, v6 has higher vertex F1, while Cellpose-SAM has higher adjacency F1 and PQ. On RPE,
the gains remain modest and absolute vertex accuracy is low. On sub-confluent HAEC, v6 predicts
151 vertices per field against 92 reference vertices; precision remains 0.293. The imperfect,
semantic-derived HAEC reference also limits interpretation.

## Training lineage and configurations

The two training branches are `v0_synth → v0_mixed` and
`v1_multi → v3_endo → v4_confluent → v5_vertex → v6_pool`.
`v2_endo` is a separate fine-tune of v1 using the older HAEC targets; v3 repeats that recipe
with corrected targets, rather than continuing v2's weights.

All runs use 512 px crops, batch size 8 per GPU, AdamW, weight decay 1e-4, a cosine schedule,
3% warmup, gradient clipping at 1.0, and AMP on CUDA. The stored learning rate is multiplied by
the distributed world size in the trainer. Channel augmentation uses membrane-as-geometry
probability 0.5, nuclei retention 0.85, and junction retention 0.9. v0 was trained on an AWS
A10G with torch 2.7.1+cu128; later stages used H100s, with bfloat16 AMP when supported.
The original cards date the initial v0/v1 training to 2026-09-17.

| Release name | Configuration/log directory under `runs/neural/` | Resume source in saved config | GPUs | Base LR × GPUs | Seed |
|---|---|---|---:|---:|---:|
| v0_synth | `v1_synth/` | None | 1 | 3e-4 × 1 | 0 |
| v0_mixed | `v2_mixed/` | `v1_synth/last.pt` | 1 | 3e-4 × 1 | 1 |
| v1_multi | `v3_multi/` | None | 8 | 3e-4 × 8 | 3 |
| v2_endo | `v2_endo/` | `v3_multi/best.pt` | 8 | 1.5e-4 × 8 | 5 |
| v3_endo | `v3_endo/` | `v3_multi/best.pt` | 8 | 1.5e-4 × 8 | 5 |
| v4_confluent | `v4_confluent/` | `v3_endo/best.pt` | 8 | 1e-4 × 8 | 7 |
| v5_vertex | `v5_vertex/` | `v4_confluent/best.pt` | 4 | 1e-4 × 4 | 11 |
| v6_pool | `v6_pool/` | `v5_vertex/best.pt` | 4 | 1e-4 × 4 | 13 |

The epoch indices below are zero-based and read directly from `train_log.jsonl`. `--epochs`
is the terminal epoch count, not the number of additional epochs after resuming. This distinction
matters: earlier cards described v5 and v6 as “40 epochs,” whereas the saved logs contain 42 and
43 epochs because training resumed from each predecessor's best checkpoint.

| Model | Logged epoch indices | Logged epochs | Best validation-loss epoch | Summed epoch time (minutes) |
|---|---:|---:|---:|---:|
| v0_synth | 0–39 | 40 | 34 | 63.6 |
| v0_mixed | 40–59 | 20 | 57 | 33.3 |
| v1_multi | 0–59 | 60 | 59 | 51.4 |
| v2_endo | 60–99 | 40 | 92 | 20.1 |
| v3_endo | 60–99 | 40 | 98 | 19.3 |
| v4_confluent | 61–139 | 79 | 137 | 33.7 |
| v5_vertex | 138–179 | 42 | 176 | 28.0 |
| v6_pool | 177–219 | 43 | 214 | 32.9 |

**Archive inconsistency:** v4's saved configuration names `v3_endo/best.pt`, but its log starts
at epoch 61, whereas the archived v3 log's best validation-loss epoch is 98. The available
configuration and logs establish the recorded parent path and run history, but do not explain
that mismatch. The table reports the archived evidence rather than inferring an unrecorded
restart. Summed epoch time excludes setup and is not identical to end-to-end campaign wall time;
the original cards reported approximately 52, 21, 19, 34, 28, and 33 minutes for v1 through v6.

### Training data and changes at each stage

| Stage | Training mixture and essential supervision details |
|---|---|
| v0_synth | 2,000 synthetic 512 × 512 tiles, seeds 1000–1007; exact synthetic labels and targets; 200 synthetic validation tiles, seed 777. No real images. |
| v0_mixed | Continues v0's final checkpoint with the same 2,000 synthetic tiles plus 135 S-BIAD1540 EGM2 pseudo-label tiles and 40 VE-strat consensus tiles. S-BIAD1540 used `cellpose_primary`, with 24% of pixels ignored. Validation remains synthetic-only. |
| v1_multi | 8,000 additional synthetic tiles (seeds 5000–5015), the original 2,000 synthetic tiles, 2,276 LIVECell training tiles from 576 fields across eight cell lines, 5,455 NeurIPS 2022 CellSeg tiles from 1,101 labelled images, and the 175 fluorescence pseudo-label tiles. Roughly 18,000 tiles total; validation uses 400 synthetic tiles, seed 9999. Real-mask targets were built by `scripts/make_gt_tiles.py`, without a nuclei channel. |
| v2_endo | 3,132 HAEC training tiles from 348 fields, repeated twice; 2,914 mCellSeg tiles from 160 images; 2,000 synthetic tiles; 175 S-BIAD1540/VE-strat pseudo-label tiles. HAEC uses inverted cytoplasmic GFP plus real Hoechst nuclei. About 11,000 sampled tiles per epoch. |
| v3_endo | Same mixture as v2, with rebuilt HAEC targets: 8-connected body components, removal of components under 30 px, and 4-connected output labels. The earlier derivation made about 40% of nominal HAEC instances into 1–3 px fringe specks. |
| v4_confluent | Adds 250 hCEC tiles from ten training fields, repeated 3×; 80 alizarine tiles from 20 fields, repeated 2×; and 32 FlyWing tiles from 32 fields, repeated 2×. The saved configuration also contains HAEC, mCellSeg, 2,000 synthetic, and S-BIAD1540 tiles, each once; it does not retain the v3 HAEC 2× weighting or VE-strat directory. Outside-ROI pixels and a 3 px rim have zero loss weight. Treating them as background in the first attempt caused alizarine performance to collapse (`runs/vertex/v4_roi_as_background/`). |
| v5_vertex | Same directories and sampling weights as v4, adding mined loss weights: 4× around missed true vertices and 3× around spurious vertices, generated by `scripts/make_vertex_miss_weights.py`. `vertex_focus=1.0` within 5 px of each true vertex. In the mining pass, v4 matched 67% of hCEC, 85% of FlyWing, 97% of alizarine, and 36% of HAEC training vertices. |
| v6_pool | Keeps v5's mixture, adds 208 RPE tiles from 13 NIH-NEI training stacks, repeated 2× with ROI exclusions and mined weights, and 1,200 pseudo-label tiles from 300 real PECAM-1 HUVEC fields. Cellpose-SAM supplies the primary pseudo-labels, v4 supplies a second opinion, and disputed boundaries are ignored. Vertex focus and loss weighting remain enabled. |

HAEC fields with ID divisible by five and every fifth sorted mCellSeg image are assigned to test;
exact lists are in `runs/endo/splits.json`. Confluent fields and RPE stacks use
`runs/vertex/splits_confluent.json` and `runs/vertex/splits_rpe.json`. Pseudo-labels inherit the
source segmenters' errors and are not independent expert truth.

### Synthetic validation-head diagnostics

These are the metrics at the best validation-loss epoch from each saved training log. They
measure dense heads before decoding. Seed/vertex peak recall is not decoded vertex F1, and the
synthetic validation data were also used for checkpoint selection.

| Model | Boundary F1 | Seed peak recall | Vertex peak recall | Gap IoU | Distance MAE (px) |
|---|---:|---:|---:|---:|---:|
| v0_synth | 0.760 | 0.991 | 0.884 | 0.448 | 0.381 |
| v0_mixed | 0.772 | 0.991 | 0.884 | 0.540 | 0.365 |
| v1_multi | 0.739 | 0.991 | 0.856 | 0.459 | 0.469 |
| v2_endo | 0.744 | 0.990 | 0.850 | 0.437 | 0.516 |
| v3_endo | 0.739 | 0.990 | 0.853 | 0.428 | 0.533 |
| v4_confluent | 0.749 | 0.990 | 0.868 | 0.469 | 0.466 |
| v5_vertex | 0.728 | 0.990 | 0.874 | 0.496 | 0.457 |
| v6_pool | 0.718 | 0.990 | 0.885 | 0.480 | 0.455 |

## Historical evaluation and corrections

Historical measurements below retain the checkpoint comparisons from the former individual
model cards. They are separated from the current evaluation because the data, reference
construction, sample size, and decoder changed during the study.

### v0: synthetic reconstruction and fluorescence adaptation

On 40 synthetic validation tiles, the decoder results were:

| Method | Adjacency F1 | Vertex F1 | Incident-set accuracy | PQ | Validity |
|---|---:|---:|---:|---:|---:|
| v0_synth | 0.889 | 0.733 | 0.860 | 0.891 | 1.000 |
| v0_mixed | 0.887 | 0.764 | 0.857 | 0.893 | 1.000 |
| Classical | 0.447 | 0.503 | 0.372 | 0.554 | 1.000 |

Sources are `runs/pimorph_bench/synth_summary.md` and
`runs/pimorph_bench_v2/synth_summary.md`. The former cards called these held-out tiles: they were
excluded from gradient training, but drawn from the validation set used for model selection,
so they are validation results rather than a separate final test set.

The synthetic-only model did not transfer to LIVECell phase contrast: adjacency F1 was 0.000
versus 0.558 for the Cellpose-SAM comparison in that initial experiment. A VE-strat fluorescence
crop produced 43 cells, trivalent vertices, and boundary/interior intensity ratio 1.63
(`runs/pimorph_dev/ve_strat_neural_vs_classical.png`), but had no expert instance truth.

For `EGM2_regular_6dyn-24`, v0_mixed reconstructed 169 cells and 271 tricellular vertices;
classical proposals yielded 65 cells, 40 tricellular vertices, and 31 isolated cells. Render
log-likelihood per pixel was −4.97 versus −5.04. These are self-consistency/qualitative
observations, not segmentation-accuracy measurements
(`runs/pimorph_dev/sbiad1540_6dyn_neural_vs_classical.png`).

### v1: mixed-modality generalization

The following comparisons each use 400 images except the 100-image real-data classical
baselines. Each metric triplet is adjacency F1 / vertex F1 / PQ. Sources: `runs/scale/bench_*`.

| Dataset and status | v1_multi | Cellpose-SAM | Classical |
|---|---:|---:|---:|
| Synthetic, seed 9999: validation set | 0.861 / 0.693 / 0.868 | Not run | 0.368 / 0.438 / 0.483 |
| LIVECell test: held out from PiMorph training | 0.230 / 0.164 / 0.403 | 0.617 / 0.341 / 0.639 | 0.021 / 0.073 / 0.072 |
| NeurIPS CellSeg: **in-sample** labelled images | 0.097 / 0.058 / 0.390 | 0.382 / 0.531 / 0.597 | 0.025 / 0.099 / 0.042 |

LIVECell adjacency improved from the synthetic-only model's 0.000 to 0.230, with BV2/SkBr3
around 0.38–0.39 and Huh7/SHSY5Y around 0.11; Cellpose-SAM remained substantially stronger.
NeurIPS performance was poor even in-sample, indicating inadequate fit under that training
mixture, not evidence of generalization. The lower synthetic scores than v0 were measured on a
different, larger validation set, so they do not isolate a cost of adding real data. Reported
complex validity remained 1.0 by construction.

### v2: original endothelial evaluation, superseded reference

This table preserves the original v2 card's evaluation against the **older HAEC reference and
decoder**. Do not compare its HAEC values directly with current scores. Cell counts are rounded
per-field means. mCellSeg's 40 test images contain ten HUVEC and 30 HEK-293T fields.

| Dataset / method | n | Adjacency F1 pair / component | Vertex F1 | PQ | AP50 | Boundary F1 | Predicted / reference cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAEC / v2 | 86 | 0.243 / 0.145 | 0.067 | 0.407 | 0.352 | 0.842 | 679 / 1,151 |
| HAEC / Cellpose-SAM filled | 86 | 0.324 / 0.092 | 0.038 | 0.347 | 0.323 | 0.707 | 753 / 1,151 |
| HAEC / v1 | 86 | 0.001 / 0.000 | 0.004 | 0.006 | 0.005 | 0.198 | 196 / 1,151 |
| HAEC / classical | 86 | 0.040 / 0.005 | 0.002 | 0.109 | 0.097 | 0.273 | 426 / 1,151 |
| mCellSeg / v2 | 40 | 0.051 / 0.011 | 0.011 | 0.131 | 0.115 | 0.354 | 266 / 89 |
| mCellSeg / Cellpose-SAM filled | 40 | 0.169 / 0.060 | 0.117 | 0.214 | 0.207 | 0.203 | 45 / 89 |
| mCellSeg / v1 | 40 | 0.001 / 0.000 | 0.002 | 0.011 | 0.009 | 0.203 | 1,141 / 89 |

Sources: `runs/endo/*_test_*`. In that HAEC evaluation, v2 predicted about 1,060 vertices per
field against 124 reference vertices, with about 158 splits per field. The Cellpose-SAM
comparison predicted about 110 vertices; its recorded median localization error was 2.0 px
versus 1.4 px for v2. HAEC was sub-confluent, with about 46% background. The labels were
instances derived by flooding bodies through the border class of body/border/nucleus/background
semantic annotations. mCellSeg instead supplied expert instance masks. The strong v2
HAEC improvement over v1 was domain adaptation, while mCellSeg still showed substantial
v2 over-segmentation and Cellpose-SAM under-segmentation.

### v3: corrected reference and decoder

All methods below are scored against the corrected HAEC reference, with outside-cell flood
exclusion (`DecoderParams.outside_px=0`) and nucleus-guided seeding. These historical results
are comparable within this table. Sources: `runs/vertex/*_test_*`.

| Dataset / method | n | Adjacency F1 pair / component | Vertex F1 | PQ | AP50 | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| HAEC / v3 | 86 | 0.505 / 0.306 | 0.262 | 0.600 | 0.616 | 0.844 |
| HAEC / v2 rescored | 86 | 0.504 / 0.308 | 0.261 | 0.597 | 0.612 | 0.843 |
| HAEC / Cellpose-SAM filled | 86 | 0.314 / 0.087 | 0.039 | 0.465 | 0.487 | 0.708 |
| mCellSeg / v3 | 40 | 0.064 / 0.013 | 0.014 | 0.134 | 0.119 | 0.349 |
| mCellSeg / Cellpose-SAM filled | 40 | 0.170 / 0.060 | 0.117 | 0.215 | 0.208 | 0.203 |

The corrected reference and decoder account for nearly all of the change from the original v2
HAEC result. Retraining alone changed the reported scores by less than 0.01, while improving
vertex count calibration: approximately 95 predicted vertices per field for v3, 112 for v2,
and 110 for Cellpose-SAM, against 92 reference vertices. mCellSeg remained a poor transfer
domain. The later confluent datasets were unseen by v3 during training.

### v4: first confluent fine-tune

These field-disjoint historical results use the pre-frontier decoder; they are not the tuned v6
pipeline. Source: `runs/vertex/test_split_summary.csv`.

| Test dataset / method | n | Adjacency F1 | Vertex F1 | Vertex precision / recall | PQ | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| hCEC / v4 | 5 | 0.822 | 0.611 | 0.579 / 0.648 | 0.776 | 0.896 |
| hCEC / Cellpose-SAM filled | 5 | 0.830 | 0.635 | 0.721 / 0.581 | 0.790 | 0.880 |
| hCEC / v3 | 5 | 0.712 | 0.334 | 0.350 / 0.320 | 0.671 | 0.804 |
| Alizarine / v4 | 10 | 0.977 | 0.990 | 0.995 / 0.985 | 0.911 | 0.998 |
| Alizarine / Cellpose-SAM filled | 10 | 0.989 | 0.984 | 0.979 / 0.989 | 0.898 | 1.000 |
| FlyWing / v4 | 10 | 0.894 | 0.833 | 0.831 / 0.836 | 0.741 | 0.994 |
| FlyWing / Cellpose-SAM filled | 10 | 0.966 | 0.849 | 0.834 / 0.864 | 0.795 | 0.996 |
| FlyWing / v3 | 10 | 0.802 | 0.868 | 0.897 / 0.842 | 0.726 | 0.986 |
| HAEC / v4 | 86 | 0.521 | 0.292 | 0.294 / 0.304 | 0.612 | 0.852 |

The hCEC fields average roughly 1,665 cells and 2,974 multicellular vertices. v4 closed much
of v3's hCEC gap, improved HAEC, and had higher vertex F1 and PQ than the alizarine baseline.
On FlyWing, however, v3 retained higher vertex F1 and better localization (about 1.7 versus
2.0 px). This was the basis of the older recommendation to use v3 for small-cell E-cadherin;
the current v6 comparison is reported above.

## Reproduce training and evaluation

The module entry point is `python -m pimorph.infer.neural.train`; there is no `pimorph train`
subcommand. Each directory in the configuration table contains the resolved `config.json` and
`train_log.jsonl`. Preserve the training data manifest, source split lists, checkpoint selection,
and decoder settings alongside any reproduced scores.

```bash
python -m pimorph.infer.neural.train \
  --train-dirs data/tiles/synth_train --val-dirs data/tiles/synth_val \
  --out runs/neural/v0_reproduction --epochs 40 --batch-size 8 --crop 512 \
  --base 32 --depth 4 --lr 3e-4 --device cuda --amp --num-workers 6 --seed 0
```

That command uses the already-generated original tile directories. Generating a new synthetic
set with different seeds is a new experiment, even if the number of tiles matches.

| Stage | Executable recipe and data preparation |
|---|---|
| v1 mixed real/synthetic | [`infra/xai/run_scale.sh`](../infra/xai/run_scale.sh), `tiles` / `tiles_gt` / `train`; `scripts/make_gt_tiles.py` |
| v2 endothelial | [`infra/xai/run_endothelial.sh`](../infra/xai/run_endothelial.sh), `tiles` / `train` / `bench_v2` |
| v3 corrected reference | [`infra/xai/run_vertex.sh`](../infra/xai/run_vertex.sh), `tiles` / `train` / `tune` / `bench` |
| v4 confluent | [`infra/xai/run_vertex.sh`](../infra/xai/run_vertex.sh), `finetune`, using `splits_confluent.json` |
| v5 hard vertices | [`infra/xai/run_frontier.sh`](../infra/xai/run_frontier.sh), `mine` / `train` / `tune`; `scripts/make_vertex_miss_weights.py` |
| v6 pooled data | [`infra/xai/run_pool.sh`](../infra/xai/run_pool.sh), `tiles` / `mine` / `train`, using `splits_rpe.json` |
| Final evaluation | [`infra/xai/run_final_bench.sh`](../infra/xai/run_final_bench.sh), with the archived split lists and train-selected decoder settings |

These campaign scripts preserve the original machine paths and GPU allocations; adjust the
repository path, Python executable, and devices before running them elsewhere. Consult the
saved configuration and epoch history when resuming, including the v4 discrepancy above.
The public checkpoint filenames and on-machine run directory names differ for v0 and v1, as
shown in the mapping table.

## Data and model-use provenance

The repository's [MIT license](../LICENSE) covers the code; it is not a blanket license for
source images, external models, or the trained checkpoints. The former v4/v6 cards describe
those checkpoints as research-evaluation checkpoints and record hCEC training images as
**CC BY-NC-ND 4.0**. The RPE Figshare source is documented as **CC0**, while the NIH-NEI software
repository has a separate software license and its release archive did not state a data license
in the inspected material. Fresh [Zenodo metadata](https://zenodo.org/records/10611092) lists the PECAM-1 HUVEC pseudo-label source under **MIT**, correcting the former card's CC BY attribution. The [NeurIPS CellSeg source](https://zenodo.org/records/10719375) is **CC BY-NC-ND 4.0**. These are recorded source terms, not a determination of downstream permissions.
Retain source attribution and consult the relevant records for the intended use. Full dataset
provenance and the distinctions between expert annotations, derived references, and pseudo-labels
are consolidated in the [study explainer](../EXPLAINER.md).
