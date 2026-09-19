#!/usr/bin/env bash
# Pool round: RPE monolayer truth (train stacks) and real PECAM-1 HUVEC pseudo-labels join
# the training set; v6_pool = v5_vertex fine-tuned on everything with mined vertex weights.
# Usage: bash infra/xai/run_pool.sh [phases...]   (default: tiles mine train)
#   GPUS="4 5 6 7" limits mining/training to those GPUs (default all 8).
set -uo pipefail
cd /data/okebell/pimorph/PiMorph
PY=.venv/bin/python
OUT=runs/frontier
mkdir -p $OUT
LOG=$OUT/campaign.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(tiles mine train)
has() { for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
V5=runs/neural/v5_vertex/best.pt
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
NG=$(echo $GPUS | wc -w)
CUDA_LIST=$(echo $GPUS | tr ' ' ',')

if has tiles; then
  log "pool tiles: rpe_zo1 train stacks (52 tiles of 768^2 -> 512 crops, ROI ignored)"
  rm -rf data/tiles/gt_rpe_train
  for k in 0 1 2 3; do
    $PY scripts/make_gt_tiles.py --dataset rpe_zo1 --out data/tiles/gt_rpe_train --tile 512 --stride 256 --shard $k/4 \
      --id-list runs/vertex/splits_rpe.json:rpe_zo1_train > $OUT/gt_rpe_$k.log 2>&1 &
  done
  wait
  log "pool tiles: gt_rpe_train=$(find data/tiles/gt_rpe_train -name '*.npz' | wc -l) pseudo_pecam=$(find data/tiles/pseudo_pecam -name '*.npz' | wc -l)"
fi

if has mine; then
  log "pool mine: vertex-miss weights on gt_rpe_train with v5_vertex"
  k=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g PIMORPH_NEURAL_CKPT=$V5 $PY scripts/make_vertex_miss_weights.py --tiles data/tiles/gt_rpe_train \
      --shard $k/$NG > $OUT/mine_rpe_$k.log 2>&1 &
    k=$((k+1))
  done
  wait
  cat data/tiles/gt_rpe_train/vertex_miss_stats_shard*.csv 2>/dev/null | head -1 > /dev/null
  log "pool mine done: $(tail -q -n 1 $OUT/mine_rpe_*.log | head -1)"
fi

if has train; then
  log "pool train: v6_pool = v5_vertex + gt_rpe_train (x2) + pseudo_pecam, 40 epochs on $NG GPUs"
  CUDA_VISIBLE_DEVICES=$CUDA_LIST $PY -m torch.distributed.run --standalone --nproc_per_node $NG -m pimorph.infer.neural.train \
    --train-dirs data/tiles/gt_hcec_train data/tiles/gt_hcec_train data/tiles/gt_hcec_train data/tiles/gt_alizarine_train data/tiles/gt_alizarine_train data/tiles/gt_flywing_train data/tiles/gt_flywing_train data/tiles/gt_rpe_train data/tiles/gt_rpe_train data/tiles/gt_haec_train data/tiles/gt_mcellseg_train data/tiles/synth_train data/tiles/pseudo_sbiad1540 data/tiles/pseudo_pecam \
    --val-dirs data/tiles/synth_val \
    --out runs/neural/v6_pool --resume $V5 --epochs 220 --batch-size 8 --crop 512 --base 48 --depth 4 \
    --vertex-focus 1.0 --vertex-focus-radius-px 5 \
    --device cuda --amp --num-workers 6 --seed 13 --log-every 100 --lr 1.0e-4 > $OUT/train_v6_pool.log 2>&1
  log "pool train done: $(tail -1 runs/neural/v6_pool/train_log.jsonl | cut -c1-160)"
fi
log "pool campaign finished"
