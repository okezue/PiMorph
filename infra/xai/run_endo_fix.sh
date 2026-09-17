#!/usr/bin/env bash
# Relaunch only the bench_all jobs that failed on the first pass (HAEC GPU jobs and
# mCellSeg v1_multi), then continue with tiles, train, bench_v2 once the still-running
# first-pass mCellSeg jobs have finished.
set -euo pipefail
ROOT=/data/okebell/pimorph/PiMorph
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
OUT=runs/endo
V1=runs/neural/v3_multi/best.pt
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/campaign.log"; }
bench() {
  local ds=$1 gpu=$2 m=$3 ckpt=$4 out=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu PIMORPH_NEURAL_CKPT=$ckpt $PY -m pimorph.cli benchmark --dataset $ds --methods $m --out $out "$@" > $out.log 2>&1
}
log "fix: relaunch HAEC gpu jobs + mCellSeg v1_multi"
bench mcellseg 3 neural $V1 $OUT/mcellseg_v1multi &
bench haec 4 cellpose_sam none $OUT/haec_cellpose --max-items 200 &
bench haec 5 neural models/pimorph_proposals_v0_mixed.pt $OUT/haec_v0mixed --max-items 200 &
bench haec 6 neural $V1 $OUT/haec_v1multi --max-items 200 &
bench haec 7 neural models/pimorph_proposals_v0_synth.pt $OUT/haec_v0synth --max-items 200 &
# wait for the first-pass jobs still running under the old tmux session
while pgrep -f "pimorph.cli benchmark --dataset mcellseg --methods cellpose_sam" >/dev/null || pgrep -f "methods neural --out runs/endo/mcellseg_v0" >/dev/null || pgrep -f "methods classical" >/dev/null; do sleep 60; done
wait
log "bench_all done (after fix)"
bash infra/xai/run_endothelial.sh tiles train bench_v2
