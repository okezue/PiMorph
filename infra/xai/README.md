# PiMorph on xAI compute

Scaled tests, dataset ingestion, baselines and neural-proposal training run on a dedicated
explorer devbox. Nothing here touches any other job, queue or namespace.

## Devbox

| Item | Value |
|---|---|
| Name | `okebell-pimorph` |
| Cluster / queue | `fou` / `h100-sandbox` (a queue this user already runs in; 80 reserved GPUs) |
| Shape | 1 replica, 8x H100 80 GB, 70 CPUs, 2 TB RAM |
| Created | 2026-09-17 04:18 local (`x devbox okebell-pimorph -c fou -q h100-sandbox --machine h100 -g 8 -r 1 -t 3d`) |
| Working dir | `/data/okebell/pimorph/PiMorph` (VAST shared storage, survives pod restarts) |
| Image | default `jax:xlm2-*-cuda13p0`; our own uv venv with torch 2.7.1+cu128 |

Manage with `x ls -c fou`, `x ssh okebell-pimorph -c fou`, `x rm okebell-pimorph -c fou` when done.
`infra/xai/remote.sh "<cmd>"` wraps `x ssh`; `infra/xai/remote.sh cp <files>` copies into the working dir.

## Data on the box

- Code and tiles and checkpoints arrive as tarballs through presigned S3 URLs (`infra/aws/push_data.sh` writes `infra/aws/urls.txt`; `setup_devbox.sh urls.txt` fetches them). No AWS credentials on the box.
- Public datasets are fetched directly from their sources by `setup_devbox.sh`: LIVECell (all 5,239 images + train/val/test COCO annotations), NeurIPS 2022 CellSeg (Zenodo 10719375: 1,000 labelled training images + 101 tuning images across brightfield, fluorescence, phase contrast, DIC), cornea cells (GitHub).

## Campaign (`run_scale.sh`)

Phases, each writing under `runs/scale/`:

1. `tests`: full pytest suite with 16 workers.
2. `tiles`: 8,000 synthetic training tiles (16 workers, seeds 5000 to 5015) + 400 validation tiles (seed 9999); real ground-truth tiles from LIVECell train (up to 1,500 images) and NeurIPS CellSeg with exact targets via `scripts/make_gt_tiles.py`.
3. `bench_base`: classical and Cellpose-SAM on LIVECell test (400 images across the 8 cell lines), NeurIPS CellSeg (400), cornea (160), synthetic (400); one benchmark process per GPU.
4. `train`: `v3_multi`, base 48 UNet, `torchrun --nproc_per_node 8` DDP, 60 epochs on synthetic + LIVECell GT + NeurIPS GT + real pseudo-labels.
5. `bench_neural`: `v3_multi` on the same held-out sets.

Launch: `tmux new -d -s scale 'bash infra/xai/run_scale.sh'`; progress in `runs/scale/campaign.log`.

## Results pulled back

Copy `runs/scale/*/` and `runs/neural/v3_multi/{best.pt,train_log.jsonl,config.json}` with
`x ssh ... -- tar czf - runs/scale runs/neural/v3_multi | tar xzf - -C <local>` (see the session log for the exact command used).
