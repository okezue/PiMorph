#!/usr/bin/env bash
# Vertex campaign on the PiMorph devbox: corrected HAEC reference, v3_endo retrain,
# decoder tuning on the train split, held-out re-benchmark of every method against the
# corrected reference, and the EGM2 shear re-test with posteriors on every field.
# Usage: bash infra/xai/run_vertex.sh [phases...]   (default: all)
#   phases: tiles train tune bench bench_new shear
set -uo pipefail
cd /data/okebell/pimorph/PiMorph
PY=.venv/bin/python
OUT=runs/vertex
mkdir -p $OUT runs/shear_retest
LOG=$OUT/campaign.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(tiles train tune bench bench_new shear)
has() { for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
V1=runs/neural/v3_multi/best.pt   # v1_multi on the box
V2=runs/neural/v2_endo/best.pt
V3=runs/neural/v3_endo/best.pt
bench() {  # dataset gpu method ckpt outdir [extra cli args...]
  local ds=$1 gpu=$2 m=$3 ckpt=$4 out=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu PIMORPH_NEURAL_CKPT=$ckpt $PY -m pimorph.cli benchmark --dataset $ds --methods $m --out $out "$@" > $out.log 2>&1
}

# ------------------------------------------ 1. HAEC tiles from the corrected reference
if has tiles; then
  log "tiles: rebuilding gt_haec_train with the corrected reference (8-connected markers, no specks)"
  rm -rf data/tiles/gt_haec_train
  for k in $(seq 0 15); do
    $PY scripts/make_gt_tiles.py --dataset haec --out data/tiles/gt_haec_train --tile 512 --stride 400 --shard $k/16 --id-list runs/endo/splits.json:haec_train > $OUT/gt_haec_$k.log 2>&1 &
  done
  wait
  log "tiles: gt_haec_train=$(find data/tiles/gt_haec_train -name '*.npz' | wc -l)"
fi

# -------------------------------------------------------- 2. v3_endo = v2 recipe, corrected tiles
if has train; then
  log "train: v3_endo = v1_multi resumed on corrected HAEC + mCellSeg tiles + synthetic + pseudo (same recipe as v2_endo)"
  $PY -m torch.distributed.run --standalone --nproc_per_node 8 -m pimorph.infer.neural.train \
    --train-dirs data/tiles/gt_haec_train data/tiles/gt_mcellseg_train data/tiles/gt_haec_train data/tiles/synth_train data/tiles/pseudo_sbiad1540 data/tiles/pseudo_ve_strat \
    --val-dirs data/tiles/synth_val \
    --out runs/neural/v3_endo --resume $V1 --epochs 100 --batch-size 8 --crop 512 --base 48 --depth 4 \
    --device cuda --amp --num-workers 6 --seed 5 --log-every 100 --lr 1.5e-4 > $OUT/train_v3_endo.log 2>&1
  log "train done: $(tail -1 runs/neural/v3_endo/train_log.jsonl | cut -c1-160)"
fi

# ------------------------------------------------ 3. decoder tuning on the HAEC TRAIN split
if has tune; then
  log "tune: decoder variants on 40 HAEC train fields, v2_endo and v3_endo"
  CUDA_VISIBLE_DEVICES=0 PIMORPH_ID_LIST=runs/endo/splits.json:haec_train $PY scripts/tune_decoder.py --dataset haec --max-items 40 \
    --checkpoint $V2 --variants runs/decoder_tuning/variants_outside.json --out $OUT/tune_haec_train_v2endo.csv > $OUT/tune_v2.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 PIMORPH_ID_LIST=runs/endo/splits.json:haec_train $PY scripts/tune_decoder.py --dataset haec --max-items 40 \
    --checkpoint $V3 --variants runs/decoder_tuning/variants_outside.json --out $OUT/tune_haec_train_v3endo.csv > $OUT/tune_v3.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY scripts/tune_decoder.py --dataset synth --root data/tiles/synth_val --max-items 60 \
    --checkpoint $V3 --variants runs/decoder_tuning/variants_outside.json --out $OUT/tune_synth_v3endo.csv > $OUT/tune_synth.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 PIMORPH_ID_LIST=runs/endo/splits.json:mcellseg_train $PY scripts/tune_decoder.py --dataset mcellseg --max-items 40 \
    --checkpoint $V3 --variants runs/decoder_tuning/variants_outside.json --out $OUT/tune_mcellseg_train_v3endo.csv > $OUT/tune_mcellseg.log 2>&1 &
  wait
  log "tune done"
fi

# --------------------------- 4. held-out test splits, every method, corrected reference
if has bench; then
  log "bench: HAEC test (86) and mCellSeg test (40): v3_endo, v2_endo, cellpose_sam(_filled), classical, with the current decoder defaults"
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 0 neural $V3 $OUT/haec_test_v3endo &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 1 neural $V2 $OUT/haec_test_v2endo &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 2 cellpose_sam_filled none $OUT/haec_test_cellpose_filled &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 3 cellpose_sam none $OUT/haec_test_cellpose &
  PIMORPH_ID_LIST=runs/endo/splits.json:mcellseg_test bench mcellseg 4 neural $V3 $OUT/mcellseg_test_v3endo &
  PIMORPH_ID_LIST=runs/endo/splits.json:mcellseg_test bench mcellseg 5 neural $V2 $OUT/mcellseg_test_v2endo &
  PIMORPH_ID_LIST=runs/endo/splits.json:mcellseg_test bench mcellseg 6 cellpose_sam_filled none $OUT/mcellseg_test_cellpose_filled &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 7 neural $V1 $OUT/haec_test_v1multi &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test $PY -m pimorph.cli benchmark --dataset haec --methods classical --out $OUT/haec_test_classical > $OUT/haec_test_classical.log 2>&1 &
  wait
  log "bench done"
fi

# ------------------ 4b. new confluent sets with real truth (zero-shot: none is in training)
if has bench_new; then
  log "bench_new: hCEC (15 confluent corneal endothelium fields, NCAM), alizarine (30), FlyWing (42): v3_endo, v2_endo, v1_multi, cellpose_sam(_filled), classical"
  bench hcec 0 neural $V3 $OUT/hcec_v3endo &
  bench hcec 1 neural $V2 $OUT/hcec_v2endo &
  bench hcec 2 cellpose_sam_filled none $OUT/hcec_cellpose_filled &
  bench hcec 3 neural $V1 $OUT/hcec_v1multi &
  bench flywing 4 neural $V3 $OUT/flywing_v3endo &
  bench flywing 5 cellpose_sam_filled none $OUT/flywing_cellpose_filled &
  bench alizarine 6 neural $V3 $OUT/alizarine_v3endo &
  bench alizarine 7 cellpose_sam_filled none $OUT/alizarine_cellpose_filled &
  $PY -m pimorph.cli benchmark --dataset hcec --methods classical --out $OUT/hcec_classical > $OUT/hcec_classical.log 2>&1 &
  $PY -m pimorph.cli benchmark --dataset alizarine --methods classical --out $OUT/alizarine_classical > $OUT/alizarine_classical.log 2>&1 &
  $PY -m pimorph.cli benchmark --dataset flywing --methods classical --out $OUT/flywing_classical > $OUT/flywing_classical.log 2>&1 &
  wait
  bench flywing 4 neural $V1 $OUT/flywing_v1multi &
  bench alizarine 6 neural $V1 $OUT/alizarine_v1multi &
  bench hcec 2 cellpose_sam none $OUT/hcec_cellpose &
  wait
  log "bench_new done"
fi

# ------------- 4c. in-domain fine-tune on the confluent truth sets (field-disjoint splits)
SPL=runs/vertex/splits_confluent.json
V4=runs/neural/v4_confluent/best.pt
if has finetune; then
  log "finetune: tiles from hcec/alizarine/flywing TRAIN splits, v4_confluent = v3_endo resumed 40 epochs"
  rm -rf data/tiles/gt_hcec_train data/tiles/gt_alizarine_train data/tiles/gt_flywing_train
  for k in $(seq 0 9); do
    $PY scripts/make_gt_tiles.py --dataset hcec --out data/tiles/gt_hcec_train --tile 512 --stride 384 --shard $k/10 --id-list $SPL:hcec_train > $OUT/gt_hcec_$k.log 2>&1 &
  done
  for k in $(seq 0 3); do
    $PY scripts/make_gt_tiles.py --dataset alizarine --out data/tiles/gt_alizarine_train --tile 512 --stride 256 --shard $k/4 --id-list $SPL:alizarine_train > $OUT/gt_alizarine_$k.log 2>&1 &
    $PY scripts/make_gt_tiles.py --dataset flywing --out data/tiles/gt_flywing_train --tile 512 --stride 512 --shard $k/4 --id-list $SPL:flywing_train > $OUT/gt_flywing_$k.log 2>&1 &
  done
  wait
  log "finetune tiles: hcec=$(find data/tiles/gt_hcec_train -name '*.npz' | wc -l) alizarine=$(find data/tiles/gt_alizarine_train -name '*.npz' | wc -l) flywing=$(find data/tiles/gt_flywing_train -name '*.npz' | wc -l)"
  $PY -m torch.distributed.run --standalone --nproc_per_node 8 -m pimorph.infer.neural.train \
    --train-dirs data/tiles/gt_hcec_train data/tiles/gt_hcec_train data/tiles/gt_hcec_train data/tiles/gt_alizarine_train data/tiles/gt_alizarine_train data/tiles/gt_flywing_train data/tiles/gt_flywing_train data/tiles/gt_haec_train data/tiles/gt_mcellseg_train data/tiles/synth_train data/tiles/pseudo_sbiad1540 \
    --val-dirs data/tiles/synth_val \
    --out runs/neural/v4_confluent --resume $V3 --epochs 140 --batch-size 8 --crop 512 --base 48 --depth 4 \
    --device cuda --amp --num-workers 6 --seed 7 --log-every 100 --lr 1.0e-4 > $OUT/train_v4_confluent.log 2>&1
  log "finetune done: $(tail -1 runs/neural/v4_confluent/train_log.jsonl | cut -c1-160)"
  log "bench_ft: TEST splits of hcec (5), alizarine (10), flywing (10): v4_confluent, v3_endo, cellpose_sam_filled"
  PIMORPH_ID_LIST=$SPL:hcec_test bench hcec 0 neural $V4 $OUT/hcec_test_v4confluent &
  PIMORPH_ID_LIST=$SPL:hcec_test bench hcec 1 neural $V3 $OUT/hcec_test_v3endo &
  PIMORPH_ID_LIST=$SPL:hcec_test bench hcec 2 cellpose_sam_filled none $OUT/hcec_test_cellpose_filled &
  PIMORPH_ID_LIST=$SPL:alizarine_test bench alizarine 3 neural $V4 $OUT/alizarine_test_v4confluent &
  PIMORPH_ID_LIST=$SPL:alizarine_test bench alizarine 4 neural $V3 $OUT/alizarine_test_v3endo &
  PIMORPH_ID_LIST=$SPL:alizarine_test bench alizarine 5 cellpose_sam_filled none $OUT/alizarine_test_cellpose_filled &
  PIMORPH_ID_LIST=$SPL:flywing_test bench flywing 6 neural $V4 $OUT/flywing_test_v4confluent &
  PIMORPH_ID_LIST=$SPL:flywing_test bench flywing 7 neural $V3 $OUT/flywing_test_v3endo &
  wait
  PIMORPH_ID_LIST=$SPL:flywing_test bench flywing 0 cellpose_sam_filled none $OUT/flywing_test_cellpose_filled &
  PIMORPH_ID_LIST=runs/endo/splits.json:haec_test bench haec 1 neural $V4 $OUT/haec_test_v4confluent &
  wait
  log "bench_ft done"
fi

# ------------------------------------- 5. shear re-test: posteriors on every EGM2 field
if has shear; then
  CK=${SHEAR_CKPT:-$V3}
  log "shear: neural ($CK) posteriors + 1000-permutation conditional null on all 102 EGM2 fields, 8 shards"
  for k in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$k PIMORPH_NEURAL_CKPT=$CK $PY scripts/pimorph_sensitivity_egm2.py --proposer neural --checkpoint $CK \
      --per-condition 0 --n-perm 1000 --shard $k/8 --out runs/shear_retest --conditions static 6dyne high_shear > runs/shear_retest/shard_$k.log 2>&1 &
  done
  wait
  $PY scripts/pimorph_sensitivity_egm2.py --report-only --out runs/shear_retest --conditions static 6dyne high_shear > runs/shear_retest/report.log 2>&1
  log "shear done: $(grep -c . runs/shear_retest/per_field_posterior_stats.csv) rows"
fi
log "vertex campaign finished"
