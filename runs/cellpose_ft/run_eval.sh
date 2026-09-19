#!/usr/bin/env bash
# Evaluate fine-tuned Cellpose-SAM weights on the held-out TEST splits with the benchmark
# CLI (same metrics as every other row in docs/CONFLUENT_BENCHMARK.md). GPUs 0 to 3 only.
# Usage: bash runs/cellpose_ft/run_eval.sh [confluent|haec|hcec|cross ...]
set -uo pipefail
cd /data/okebell/pimorph/PiMorph
PY=.venv/bin/python
OUT=runs/frontier
FT=runs/cellpose_ft
SPL=runs/vertex/splits_confluent.json
RPE=runs/vertex/splits_rpe.json
ENDO=runs/endo/splits.json
LOG=$FT/eval.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(confluent haec hcec cross)
has() { for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
bench() {  # dataset gpu idlist model outdir
  local ds=$1 gpu=$2 ids=$3 model=$4 out=$5
  CUDA_VISIBLE_DEVICES=$gpu PIMORPH_ID_LIST=$ids PIMORPH_CELLPOSE_MODEL=$model \
    $PY -m pimorph.cli benchmark --dataset $ds --methods cellpose_sam_filled --out $out > $out.log 2>&1
  log "done $out: $(grep -c cellpose_sam_filled $out.log) items"
}

if has confluent; then
  M=$FT/cpsam_confluent/models/cpsam_confluent
  log "confluent: $M on hcec/alizarine/flywing/rpe_zo1 test"
  bench hcec 0 $SPL:hcec_test $M $OUT/hcec_test_cpsam_ft &
  bench alizarine 1 $SPL:alizarine_test $M $OUT/alizarine_test_cpsam_ft &
  bench flywing 2 $SPL:flywing_test $M $OUT/flywing_test_cpsam_ft &
  bench rpe_zo1 3 $RPE:rpe_zo1_test $M $OUT/rpe_zo1_test_cpsam_ft &
  wait
fi
if has haec; then
  M=$FT/cpsam_haec/models/cpsam_haec
  log "haec: $M on haec test"
  bench haec 0 $ENDO:haec_test $M $OUT/haec_test_cpsam_ft
fi
if has hcec; then
  M=$FT/cpsam_hcec/models/cpsam_hcec
  log "hcec-only: $M on hcec test"
  bench hcec 1 $SPL:hcec_test $M $OUT/hcec_test_cpsam_ft_hcec_only
fi
if has cross; then
  log "cross-domain: cpsam_confluent on haec test, cpsam_haec on hcec test"
  bench haec 2 $ENDO:haec_test $FT/cpsam_confluent/models/cpsam_confluent $OUT/haec_test_cpsam_ft_confluent &
  bench hcec 3 $SPL:hcec_test $FT/cpsam_haec/models/cpsam_haec $OUT/hcec_test_cpsam_ft_haec &
  wait
fi
log "eval finished: ${PHASES[*]}"
