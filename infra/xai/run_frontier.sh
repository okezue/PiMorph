#!/usr/bin/env bash
# Frontier round for multicellular vertices: hard-example mining of topologically missed
# vertices, v5_vertex fine-tune, decoder tuning with the vertex head and TTA on the TRAIN
# split, then the held-out test splits. GPUs 0 to 3 only (4 to 7 are used by a parallel job).
# Usage: bash infra/xai/run_frontier.sh [phases...]   (default: mine train tune bench)
set -uo pipefail
cd /data/okebell/pimorph/PiMorph
PY=.venv/bin/python
OUT=runs/frontier
mkdir -p $OUT
LOG=$OUT/campaign.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(mine train tune bench)
has() { for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
V4=runs/neural/v4_confluent/best.pt
V5=runs/neural/v5_vertex/best.pt
SPL=runs/vertex/splits_confluent.json
GPUS="0 1 2 3"
bench() {  # dataset gpu method ckpt outdir [extra cli args...]
  local ds=$1 gpu=$2 m=$3 ckpt=$4 out=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu PIMORPH_NEURAL_CKPT=$ckpt $PY -m pimorph.cli benchmark --dataset $ds --methods $m --out $out "$@" > $out.log 2>&1
}

# ------------------------------ 1. hard-example weights around missed / spurious vertices
if has mine; then
  log "mine: decode every real-truth training tile with v4_confluent, weight missed (x4) and spurious (x3) vertices"
  for d in data/tiles/gt_hcec_train data/tiles/gt_alizarine_train data/tiles/gt_flywing_train data/tiles/gt_haec_train; do
    rm -f $d/vertex_miss_stats*.csv
  done
  k=0
  for g in $GPUS; do
    for r in 0 1; do
      CUDA_VISIBLE_DEVICES=$g PIMORPH_NEURAL_CKPT=$V4 $PY scripts/make_vertex_miss_weights.py \
        --tiles data/tiles/gt_hcec_train data/tiles/gt_alizarine_train data/tiles/gt_flywing_train data/tiles/gt_haec_train \
        --shard $k/8 > $OUT/mine_$k.log 2>&1 &
      k=$((k+1))
    done
  done
  wait
  $PY - <<'EOF' | tee -a $LOG
import pandas as pd, glob
fs = glob.glob("data/tiles/gt_hcec_train/vertex_miss_stats_shard*.csv")
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d["set"] = d.tile.str.extract(r"gt_(\w+)_train")[0]
t = d.groupby("set")[["n_ref_vertices", "n_matched", "n_missed", "n_spurious"]].sum()
t["matched_frac"] = (t.n_matched / t.n_ref_vertices).round(3)
print(t.to_string())
d.to_csv("runs/frontier/vertex_miss_stats_v4.csv", index=False)
EOF
  log "mine done"
fi

# ---------------------------------------------- 2. v5_vertex = v4 + hard weights + vertex focus
if has train; then
  log "train: v5_vertex = v4_confluent resumed 40 epochs with loss_weight tiles and vertex_focus 1.0 (4 GPUs)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node 4 -m pimorph.infer.neural.train \
    --train-dirs data/tiles/gt_hcec_train data/tiles/gt_hcec_train data/tiles/gt_hcec_train data/tiles/gt_alizarine_train data/tiles/gt_alizarine_train data/tiles/gt_flywing_train data/tiles/gt_flywing_train data/tiles/gt_haec_train data/tiles/gt_mcellseg_train data/tiles/synth_train data/tiles/pseudo_sbiad1540 \
    --val-dirs data/tiles/synth_val \
    --out runs/neural/v5_vertex --resume $V4 --epochs 180 --batch-size 8 --crop 512 --base 48 --depth 4 \
    --vertex-focus 1.0 --vertex-focus-radius-px 5 \
    --device cuda --amp --num-workers 6 --seed 11 --log-every 100 --lr 1.0e-4 > $OUT/train_v5_vertex.log 2>&1
  log "train done: $(tail -1 runs/neural/v5_vertex/train_log.jsonl | cut -c1-160)"
fi

# --------------------------- 3. decoder tuning on the hCEC TRAIN split (vertex head, TTA)
if has tune; then
  log "tune: decoder variants (vertex_weight, smoothing, distance mix, TTA) on 10 hCEC train fields, v4 and v5"
  PIMORPH_ID_LIST=$SPL:hcec_train CUDA_VISIBLE_DEVICES=0 $PY scripts/tune_decoder.py --dataset hcec --max-items 10 --checkpoint $V5 \
    --variants runs/decoder_tuning/variants_frontier.json --out $OUT/tune_hcec_train_v5.csv > $OUT/tune_v5.log 2>&1 &
  PIMORPH_ID_LIST=$SPL:hcec_train CUDA_VISIBLE_DEVICES=1 $PY scripts/tune_decoder.py --dataset hcec --max-items 10 --checkpoint $V5 --tta \
    --variants runs/decoder_tuning/variants_frontier.json --out $OUT/tune_hcec_train_v5_tta.csv > $OUT/tune_v5_tta.log 2>&1 &
  PIMORPH_ID_LIST=$SPL:hcec_train CUDA_VISIBLE_DEVICES=2 $PY scripts/tune_decoder.py --dataset hcec --max-items 10 --checkpoint $V4 \
    --variants runs/decoder_tuning/variants_frontier.json --out $OUT/tune_hcec_train_v4.csv > $OUT/tune_v4.log 2>&1 &
  PIMORPH_ID_LIST=$SPL:flywing_train CUDA_VISIBLE_DEVICES=3 $PY scripts/tune_decoder.py --dataset flywing --max-items 16 --checkpoint $V5 \
    --variants runs/decoder_tuning/variants_frontier.json --out $OUT/tune_flywing_train_v5.csv > $OUT/tune_fw_v5.log 2>&1 &
  wait
  log "tune done"
fi

# ------------------------------------------------- 4. held-out test splits with v5
if has bench; then
  DP=${FRONTIER_DECODER_PARAMS:-'{}'}
  TTA=${FRONTIER_TTA:-0}
  log "bench: test splits with v5_vertex, decoder params $DP, tta $TTA"
  export PIMORPH_DECODER_PARAMS="$DP" PIMORPH_NEURAL_TTA=$TTA
  PIMORPH_ID_LIST=$SPL:hcec_test bench hcec 0 neural $V5 $OUT/hcec_test_v5 &
  PIMORPH_ID_LIST=$SPL:alizarine_test bench alizarine 1 neural $V5 $OUT/alizarine_test_v5 &
  PIMORPH_ID_LIST=$SPL:flywing_test bench flywing 2 neural $V5 $OUT/flywing_test_v5 &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 3 neural $V5 $OUT/haec_test_v5 &
  wait
  PIMORPH_ID_LIST=$SPL:hcec_test bench hcec 0 neural_map $V5 $OUT/hcec_test_v5_map &
  PIMORPH_ID_LIST=$SPL:hcec_test bench hcec 1 neural $V4 $OUT/hcec_test_v4_tuned &
  PIMORPH_ID_LIST=$SPL:flywing_test bench flywing 2 neural_map $V5 $OUT/flywing_test_v5_map &
  wait
  log "bench done"
fi
log "frontier campaign finished"
