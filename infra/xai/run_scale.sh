#!/usr/bin/env bash
# Scaled PiMorph campaign on the 8xH100 devbox. Each phase writes under runs/scale/.
# Phases can be selected: bash run_scale.sh tests tiles bench_base train bench_neural
# Default: all, in order. Assumes setup_devbox.sh finished.
set -euo pipefail
ROOT=/data/okebell/pimorph/PiMorph
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
OUT=runs/scale
mkdir -p "$OUT"
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(tests tiles bench_base train bench_neural)
has() { for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/campaign.log"; }

# ---------------------------------------------------------------- 1. full test suite
if has tests; then
  log "tests: full suite incl. torch and data-marked tests"
  $PY -m pytest tests/ -q -p no:cacheprovider -n 16 2>&1 | tail -3 | tee -a "$OUT/campaign.log" || true
fi

# --------------------------------------------- 2. big synthetic set + real GT tiles
if has tiles; then
  log "tiles: 8000 synthetic train tiles (16 parallel workers) + 400 val"
  mkdir -p data/tiles/synth_train_big
  for k in $(seq 0 15); do
    $PY -m pimorph.cli synth --n 500 --out data/tiles/synth_train_big/part$k --shape 512 --seed $((5000 + k)) > "$OUT/synth_part$k.log" 2>&1 &
  done
  $PY -m pimorph.cli synth --n 400 --out data/tiles/synth_val_big --shape 512 --seed 9999 > "$OUT/synth_val_big.log" 2>&1
  wait
fi
if has tiles || has tiles_gt; then
  log "tiles: real GT tiles from LIVECell train and NeurIPS CellSeg (exact targets from masks), 16 shards each"
  for k in $(seq 0 15); do
    $PY scripts/make_gt_tiles.py --dataset livecell --split train --out data/tiles/gt_livecell_train --tile 512 --max-items 1500 --shard $k/16 > "$OUT/gt_livecell_$k.log" 2>&1 &
  done
  wait
  for k in $(seq 0 15); do
    $PY scripts/make_gt_tiles.py --dataset neurips_cellseg --out data/tiles/gt_neurips --tile 512 --shard $k/16 > "$OUT/gt_neurips_$k.log" 2>&1 &
  done
  wait
  log "tiles: gt_livecell_train=$(find data/tiles/gt_livecell_train -name '*.npz' | wc -l) gt_neurips=$(find data/tiles/gt_neurips -name '*.npz' | wc -l)"
fi

# ----------------------------------------------- 3. baselines on every dataset
# Cellpose-SAM runs on GPUs 0-3 (a few GB each, coexists with training); the classical
# method is CPU-bound and known to fail on phase contrast, so it gets a smaller sample.
bench_base() {
  log "bench_base: Cellpose-SAM (400/dataset) + classical (100/dataset) on LIVECell test, NeurIPS CellSeg, cornea, synth_val_big"
  export PIMORPH_LIVECELL_SPLIT=test
  CUDA_VISIBLE_DEVICES=0 $PY -m pimorph.cli benchmark --dataset livecell --methods cellpose_sam --max-items 400 --out $OUT/bench_livecell_test > $OUT/bench_livecell_test.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY -m pimorph.cli benchmark --dataset neurips_cellseg --methods cellpose_sam --max-items 400 --out $OUT/bench_neurips > $OUT/bench_neurips.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY -m pimorph.cli benchmark --dataset cornea --methods cellpose_sam --max-items 160 --out $OUT/bench_cornea > $OUT/bench_cornea.log 2>&1 &
  $PY -m pimorph.cli benchmark --dataset synth --root data/tiles/synth_val_big --methods classical --max-items 400 --out $OUT/bench_synth_big > $OUT/bench_synth_big.log 2>&1 &
  $PY -m pimorph.cli benchmark --dataset livecell --methods classical --max-items 100 --out $OUT/bench_livecell_test_classical > $OUT/bench_livecell_test_classical.log 2>&1 &
  $PY -m pimorph.cli benchmark --dataset neurips_cellseg --methods classical --max-items 100 --out $OUT/bench_neurips_classical > $OUT/bench_neurips_classical.log 2>&1 &
  wait
  log "bench_base done"
}
if has bench_base; then
  bench_base &
  BENCH_PID=$!
fi

# ------------------------------------------------------------ 4. training at scale
# v3: big synthetic + LIVECell GT + NeurIPS GT + real pseudo-labels; larger model (base 48), 8 GPUs DDP
if has train; then
  log "train: v3_multi (base 48, DDP over 8 GPUs, mixed synthetic + real GT)"
  $PY -m torch.distributed.run --standalone --nproc_per_node 8 -m pimorph.infer.neural.train \
    --train-dirs data/tiles/synth_train_big data/tiles/synth_train data/tiles/gt_livecell_train data/tiles/gt_neurips data/tiles/pseudo_sbiad1540 data/tiles/pseudo_ve_strat \
    --val-dirs data/tiles/synth_val_big \
    --out runs/neural/v3_multi --epochs 60 --batch-size 8 --crop 512 --base 48 --depth 4 \
    --device cuda --amp --num-workers 6 --seed 3 --log-every 100 > $OUT/train_v3.log 2>&1
  log "train done: $(tail -1 runs/neural/v3_multi/train_log.jsonl | cut -c1-200)"
fi
if [ -n "${BENCH_PID:-}" ]; then wait "$BENCH_PID"; fi

# --------------------------------------------- 5. neural proposals on every dataset
if has bench_neural; then
  log "bench_neural: v3_multi vs Cellpose-SAM on LIVECell test, NeurIPS CellSeg, synth_val_big"
  export PIMORPH_LIVECELL_SPLIT=test
  export PIMORPH_NEURAL_CKPT=runs/neural/v3_multi/best.pt
  CUDA_VISIBLE_DEVICES=0 $PY -m pimorph.cli benchmark --dataset livecell --methods neural --max-items 400 --out $OUT/bench_livecell_test_neural > $OUT/bench_livecell_neural.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY -m pimorph.cli benchmark --dataset neurips_cellseg --methods neural --max-items 400 --out $OUT/bench_neurips_neural > $OUT/bench_neurips_neural.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY -m pimorph.cli benchmark --dataset synth --root data/tiles/synth_val_big --methods neural --max-items 400 --out $OUT/bench_synth_big_neural > $OUT/bench_synth_big_neural.log 2>&1 &
  wait
  log "bench_neural done"
fi
log "campaign finished"
