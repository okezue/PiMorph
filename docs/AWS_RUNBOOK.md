# AWS runbook for neural proposal training

Scripts live in `infra/aws/`. They create a small, fully tagged set of resources in the user's own AWS account, train the multi-head UNet on one GPU instance, sync checkpoints to S3, and tear everything down. Every create, modify, or delete call is logged in `infra/aws/aws_log.md`.

## Ground rules

1. Only session-tagged resources are ever touched. Everything the scripts create carries `Project=PiMorph`, `Owner=okebell-grok`, `Session=01a0ac4c`, plus a `Purpose` tag. `cleanup.sh` filters on `tag:Project` and `tag:Session` and deletes only the security group and key pair whose ids are in `env.sh`. Instances, volumes, buckets, and security groups that pre-exist in the account (several are listed in `aws_log.md` as observed, not ours) are never stopped, resized, or deleted, even if they look idle. This applies with no exceptions, including lack of quota or capacity.
2. No credentials in the repository. The CLI uses the named profile `pimorph` from `~/.aws/credentials` (`export AWS_PROFILE=pimorph`, also set by `env.sh`). No access key appears in any script, log, or document, and none should be added. `env.sh` holds only resource names and ids.
3. Log everything. The log file convention is below; a resource that is not in the log does not exist as far as this project is concerned.
4. Verify identity before creating anything: `aws sts get-caller-identity --profile pimorph`. The log records the account and IAM user this profile resolves to and that `servicequotas:GetServiceQuota` and `ce:GetCostAndUsage` are denied for it, so quota is discovered by launch attempt and spend is tracked by instance-hours.

## Files

| File | Runs where | What it does |
|---|---|---|
| `provision.sh` | local | Creates the S3 bucket, key pair, and security group (idempotent) and writes `env.sh` |
| `env.sh` | sourced by the other scripts | `AWS_PROFILE`, `AWS_DEFAULT_REGION`, `PIMORPH_BUCKET`, `PIMORPH_KEY_NAME`, `PIMORPH_KEY_FILE`, `PIMORPH_SG_ID`, `PIMORPH_SESSION`; `launch.sh` appends `PIMORPH_INSTANCE_ID` and `PIMORPH_INSTANCE_IP` |
| `push_code.sh` | local | Tars the tracked source and uploads it to the bucket; prints the S3 key |
| `launch.sh` | local | Launches one tagged GPU instance, spot first, on-demand fallback |
| `bootstrap.sh` | on the instance | Installs uv, unpacks the code, creates the venv, syncs tiles from S3 |
| `cleanup.sh` | local | Terminates tagged instances, deletes our security group and key pair, keeps the bucket unless `--delete-bucket` |
| `aws_log.md` | | The log |

## 1. Provision

```bash
AWS_PROFILE=pimorph infra/aws/provision.sh
```

Region defaults to `us-east-1` (`AWS_DEFAULT_REGION` overrides). The script:

- creates the bucket `pimorph-train-<account>-01a0ac4c` with the four tags and a full public-access block, skipping creation when it already exists;
- creates an ed25519 key pair `pimorph-01a0ac4c`, saving the private key to `~/.ssh/pimorph-01a0ac4c.pem` with mode 600 (outside the repository);
- creates the security group `pimorph-ssh-01a0ac4c` in the default VPC with one ingress rule, TCP 22 from this machine's current public IP (`checkip.amazonaws.com`) as a `/32`;
- appends one table row per created resource to `aws_log.md` and writes `env.sh`.

If your public IP changes, add a new `/32` rule to the security group yourself (and log it); the script does not modify an existing group.

## 2. Stage data and code

Tiles are produced locally (`pimorph synth`, `make_pseudolabel_tiles`, see [the consolidated model card](../models/README.md)) under `data/tiles/<set>/`. The IAM user cannot create instance profiles (`iam:*` is denied), so the instance never receives AWS credentials. Instead, `push_data.sh` tars each tile set and the source, uploads them to the bucket, and writes `infra/aws/urls.txt` with presigned GET URLs valid for 48 hours:

```bash
infra/aws/push_data.sh synth_train synth_val pseudo_ve_strat pseudo_sbiad1540
```

`urls.txt` is gitignored (the URLs grant read access to those objects until they expire). The source tarball contains tracked files plus untracked files under `src`, `tests`, `scripts`, `infra`, `examples`, minus `env.sh` and `urls.txt`.

## 3. Launch

```bash
infra/aws/launch.sh [instance_type] [spot|ondemand]
infra/aws/launch.sh                 # g6.xlarge, spot
infra/aws/launch.sh g5.xlarge       # A10G fallback, spot
```

The AMI defaults to the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) id hard-coded in the script (`PIMORPH_AMI` overrides it). The instance gets a 150 GB gp3 root volume with `DeleteOnTermination=true`, `--instance-initiated-shutdown-behavior terminate` so a `sudo shutdown` from inside ends the instance and its bill, the key pair and security group from `env.sh`, and the tags on both the instance and its volume. Spot requests are one-time with `InstanceInterruptionBehavior=terminate`; if the spot request fails (capacity or quota), the script re-executes itself with `ondemand`. On success it logs the instance id in both the resources table and the instance-hours table, waits for `running`, appends `PIMORPH_INSTANCE_ID` and `PIMORPH_INSTANCE_IP` to `env.sh`, and prints the ssh command:

```bash
ssh -i ~/.ssh/pimorph-01a0ac4c.pem ubuntu@<ip>
```

If the launch fails on both markets with an insufficient-capacity or quota message, stop. Do not free capacity by touching anything else in the account; try another instance type or region, or fall back to local MPS training.

## 4. Bootstrap on the instance

```bash
scp -i "$PIMORPH_KEY_FILE" infra/aws/bootstrap.sh infra/aws/urls.txt ubuntu@$PIMORPH_INSTANCE_IP:
ssh -i "$PIMORPH_KEY_FILE" ubuntu@$PIMORPH_INSTANCE_IP "bash bootstrap.sh urls.txt"
```

`bootstrap.sh` installs `libgl1` and `unzip`, installs uv, downloads every entry of `urls.txt` with `curl` (presigned URLs, no AWS credentials involved), unpacks the code into `~/work/PiMorph` and the tile tarballs into `data/tiles/`, creates `.venv` with Python 3.12, installs the package with the `dev`, `ml`, and `complex` extras plus `torch>=2.3`, `torchvision>=0.18`, `cellpose>=4.0`, `pycocotools`, and `imageio`, and prints the torch and CUDA device check.

## 5. Train

Run inside `tmux` so an ssh drop does not kill the job. The training entry point and its flags are in [the consolidated model card](../models/README.md); the flags below all exist in `python -m pimorph.infer.neural.train --help`.

```bash
cd ~/work/PiMorph && source .venv/bin/activate
tmux new -s train
python -m pimorph.infer.neural.train \
    --train-dirs data/tiles/synth_train \
    --val-dirs   data/tiles/synth_val \
    --out runs/neural/v1_synth --epochs 40 --batch-size 8 --crop 512 \
    --device cuda --amp --num-workers 4 --seed 0
```

For a second stage on mixed real and synthetic tiles, pass several directories to `--train-dirs` (for example `data/tiles/synth_train data/tiles/pseudo_ve_strat`) and keep a field-disjoint validation set in `--val-dirs`. `--resume runs/neural/v1_synth/last.pt` continues from a checkpoint with optimizer and scheduler state. `--max-steps` bounds a smoke run.

The instance has no AWS credentials, so checkpoints come back over ssh (run from the laptop, repeat during training to keep a copy of `last.pt` and `train_log.jsonl`):

```bash
source infra/aws/env.sh
rsync -az -e "ssh -i $PIMORPH_KEY_FILE" ubuntu@$PIMORPH_INSTANCE_IP:work/PiMorph/runs/neural/ runs/neural/
```

Cellpose-SAM baseline inference for the benchmark sets can run on the same instance (`pimorph benchmark --dataset livecell --methods cellpose_sam classical --root data/LIVECell --out runs/pimorph_bench`) provided the dataset tarball was included in `urls.txt`.

## 6. Pull results back

The `rsync` command above is the pull. After the final pull, upload the chosen run to the bucket from the laptop for persistence:

```bash
aws s3 sync runs/neural/ "s3://${PIMORPH_BUCKET}/runs/neural/" --only-show-errors
```

Copy the chosen checkpoint to `models/pimorph_proposals_v0.pt` only after its `train_log.jsonl` and a `pimorph benchmark` run have been recorded; as of this writing there is no such file.

## 7. Clean up

```bash
infra/aws/cleanup.sh                  # terminate tagged instances, delete our SG and key pair, keep the bucket
infra/aws/cleanup.sh --delete-bucket  # also remove the bucket and its contents
```

The instance filter is `tag:Project=PiMorph` and `tag:Session=01a0ac4c` in states pending, running, stopping, stopped. The script waits for termination, then deletes the security group `PIMORPH_SG_ID` and the key pair `PIMORPH_KEY_NAME` from `env.sh`, and logs each deletion. The bucket is kept by default because checkpoints live there and its contents are reproducible but not free to regenerate; pass `--delete-bucket` after the user confirms. Delete the local private key file yourself once the key pair is gone.

## Log file convention (`infra/aws/aws_log.md`)

- Header: profile name, account, IAM user, and the tag policy.
- A dated read-only section for identity checks and observations of pre-existing resources, written once and marked "NOT ours, never touched".
- Table "Resources created by this project": `| Created (UTC) | Type | ID | Region | Purpose | Destroyed (UTC) |`. `provision.sh`, `launch.sh`, and `cleanup.sh` append rows automatically with `date -u +%Y-%m-%dT%H:%M:%SZ` stamps; fill the "Destroyed" column when cleanup runs.
- Table "Instance-hours (cost proxy)": `| Instance | Type | Start | Stop | Hours | Est. USD |`. `launch.sh` appends the start row; fill stop time, hours, and estimate at teardown.
- Anything done by hand (a manual `authorize-security-group-ingress`, a manual `terminate-instances`) gets a row too, in the same format.

## Budget guidance

The soft budget for this push is about 40 USD: roughly 20 GPU-hours on spot plus S3. Because the profile cannot read Cost Explorer, cost is estimated from the instance-hours table using the price in effect at launch; read it with the read-only call `aws ec2 describe-spot-price-history --instance-types g6.xlarge --product-descriptions "Linux/UNIX" --max-items 5` (and the on-demand price from the EC2 pricing page) and write it into the "Est. USD" column. Practical rules:

- Smoke-test the exact command locally (`--device mps --max-steps 20 --base 8`) before launching, so the first GPU hour is not spent on a typo.
- Sync checkpoints every epoch so a spot interruption costs at most one epoch.
- Target under 4 hours of wall time per training run. If the projection exceeds the budget, stop and ask rather than continue.
- Terminate the instance as soon as checkpoints are in S3. The 150 GB gp3 root volume is deleted with the instance; only the bucket persists and it is cheap at the expected 5 to 10 GB of tiles.
