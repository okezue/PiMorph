# PiMorph AWS log

Every cloud command that creates, modifies, or deletes a resource is recorded here with its output.
Read-only observation of other resources is recorded once. Rule: only resources created in this
log are ever modified or deleted.

Profile: `pimorph` (named profile in `~/.aws/credentials`, IAM user `okezue`, account `545338549082`).
Tag policy for everything we create: `Project=PiMorph`, `Owner=okebell-grok`, `Session=01a0ac4c`, `Purpose=<text>`.

## 2026-09-16 M0 identity and permissions (read-only)

- `aws sts get-caller-identity` -> `arn:aws:iam::545338549082:user/okezue`
- `servicequotas:GetServiceQuota` denied (cannot read G/VT vCPU quota; will discover by launch attempt with dry run)
- `ce:GetCostAndUsage` denied (cannot read spend; cost tracked by instance-hours in this log)
- EC2 `describe-instances` allowed. Pre-existing instances observed in us-east-1 (all `stopped`, NOT ours, never touched):
  `i-041ed24e49e3cfe89` f2.6xlarge okezue; 8x t3.xlarge `cr234-scraper`; `i-00cbce98a2b91ec5f` g5.xlarge `wm-v3-llama-circuit`.
  us-west-2: none.
- S3 buckets pre-existing (NOT ours, never touched): `cr234-545338549082`, `flux-afi-545338549082`, `okezue`,
  `okezue-cs217-fpga`, `okezue-imp-aging-results`, `pimorph-glioma-545338549082`, `pimorph-spatial-545338549082`,
  `serif-fpga-builds`.

## Resources created by this project

| Created (UTC) | Type | ID | Region | Purpose | Destroyed (UTC) |
|---|---|---|---|---|---|

## Instance-hours (cost proxy)

| Instance | Type | Start | Stop | Hours | Est. USD |
|---|---|---|---|---|---|
| 2026-09-17T02:14:59Z | s3 bucket | pimorph-train-545338549082-01a0ac4c | us-east-1 | training tiles + checkpoints | |
| 2026-09-17T02:16:16Z | key pair | pimorph-01a0ac4c (/Users/okebell/.ssh/pimorph-01a0ac4c.pem) | us-east-1 | ssh to training instance | |
| 2026-09-17T02:16:16Z | security group | sg-0f1d1d76725853b7d (pimorph-ssh-01a0ac4c) | us-east-1 | ssh from 12.227.149.36/32 | |
| 2026-09-17T04:05:30Z | ec2 instance | i-0ffe4688aa1ae952b | us-east-1 | g5.xlarge ondemand, neural proposal training | |
| i-0ffe4688aa1ae952b | g5.xlarge (ondemand) | 2026-09-17T04:05:30Z | | | |
| 2026-09-17T04:08:34Z | sg ingress | sg-0f1d1d76725853b7d | us-east-1 | added 12.227.149.35/32 and current ip 12.227.149.36/32 (NAT pool flips) | |
| 2026-09-17T04:37:39Z | training job | tmux:train on i-0ffe4688aa1ae952b | us-east-1 | stage 1: 40 epochs synthetic (2000 tiles), out runs/neural/v1_synth | |
| 2026-09-17T05:56:22Z | training job | tmux:train2 on i-0ffe4688aa1ae952b | us-east-1 | stage 2: resume from v1_synth, synthetic + 135 S-BIAD1540 + 40 VE-strat pseudo-label tiles, to epoch 60 | |
