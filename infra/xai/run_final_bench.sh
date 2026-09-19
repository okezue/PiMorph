#!/usr/bin/env bash
# Final held-out benchmark of the frontier round. Decoder settings per dataset come from the
# TRAIN-split tuning (runs/frontier/tune_*.csv): hCEC / RPE (nuclei present) use nucleus
# merges + smoothing 1 px + vertex head 0.6; FlyWing / alizarine use vertex head 0.3; HAEC is
# tuned here on 10 TRAIN fields first. TTA on for every neural run; "plain" rows are the
# ablation without TTA and without decoder changes.
set -uo pipefail
cd /data/okebell/pimorph/PiMorph
PY=.venv/bin/python
OUT=runs/frontier
LOG=$OUT/campaign.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
V4=runs/neural/v4_confluent/best.pt
V5=runs/neural/v5_vertex/best.pt
V6=runs/neural/v6_pool/best.pt
SPL=runs/vertex/splits_confluent.json
SPR=runs/vertex/splits_rpe.json
NUC='{"nucleus_merge": true, "boundary_smooth_sigma": 1.0, "vertex_weight": 0.6}'
VW='{"vertex_weight": 0.3}'
bench() {  # dataset gpu method ckpt outdir params tta idlist
  local ds=$1 gpu=$2 m=$3 ckpt=$4 out=$5 dp=$6 tta=$7 ids=$8
  PIMORPH_ID_LIST=$ids CUDA_VISIBLE_DEVICES=$gpu PIMORPH_NEURAL_CKPT=$ckpt PIMORPH_DECODER_PARAMS="$dp" PIMORPH_NEURAL_TTA=$tta \
    $PY -m pimorph.cli benchmark --dataset $ds --methods $m --out $out > $out.log 2>&1
}

log "final: HAEC train tune (v5, TTA, 10 fields) to pick the HAEC decoder setting"
PIMORPH_ID_LIST=runs/endo/splits.json:haec_train CUDA_VISIBLE_DEVICES=0 $PY scripts/tune_decoder.py --dataset haec --max-items 10 \
  --checkpoint $V5 --tta --variants runs/decoder_tuning/variants_haec_final.json --out $OUT/tune_haec_train_v5_tta.csv > $OUT/tune_haec_v5.log 2>&1 &
# the confluent sets do not depend on the HAEC choice; start them now
bench hcec 1 neural $V5 $OUT/hcec_test_v5_tuned "$NUC" 1 $SPL:hcec_test &
bench hcec 2 neural $V6 $OUT/hcec_test_v6_tuned "$NUC" 1 $SPL:hcec_test &
bench hcec 3 neural $V5 $OUT/hcec_test_v5_plain '{}' 0 $SPL:hcec_test &
bench flywing 4 neural $V5 $OUT/flywing_test_v5_tuned "$VW" 1 $SPL:flywing_test &
bench flywing 5 neural $V6 $OUT/flywing_test_v6_tuned "$VW" 1 $SPL:flywing_test &
bench alizarine 6 neural $V5 $OUT/alizarine_test_v5_tuned "$VW" 1 $SPL:alizarine_test &
bench alizarine 7 neural $V6 $OUT/alizarine_test_v6_tuned "$VW" 1 $SPL:alizarine_test &
wait
HP=$($PY - <<'EOF'
import json, pandas as pd
d = pd.read_csv("runs/frontier/tune_haec_train_v5_tta.csv")
best = d.groupby("variant")["vertex_f1"].mean().idxmax()
v = json.load(open("runs/decoder_tuning/variants_haec_final.json"))[best][1]
print(json.dumps(v))
EOF
)
log "final: HAEC decoder setting from train tune: $HP"
echo "$HP" > $OUT/haec_decoder_params.json
bench haec 0 neural $V5 $OUT/haec_test_v5_tuned "$HP" 1 runs/endo/splits.json:haec_test &
bench haec 1 neural $V6 $OUT/haec_test_v6_tuned "$HP" 1 runs/endo/splits.json:haec_test &
bench rpe_zo1 2 neural $V5 $OUT/rpe_test_v5_tuned "$NUC" 1 $SPR:rpe_zo1_test &
bench rpe_zo1 3 neural $V6 $OUT/rpe_test_v6_tuned "$NUC" 1 $SPR:rpe_zo1_test &
bench rpe_zo1 4 neural $V4 $OUT/rpe_test_v4_tuned "$NUC" 1 $SPR:rpe_zo1_test &
bench hcec 5 neural_map $V5 $OUT/hcec_test_v5_map "$NUC" 1 $SPL:hcec_test &
bench hcec 6 neural $V4 $OUT/hcec_test_v4_tuned "$NUC" 1 $SPL:hcec_test &
bench flywing 7 neural $V5 $OUT/flywing_test_v5_plain '{}' 0 $SPL:flywing_test &
wait
log "final bench done"
