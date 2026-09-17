#!/usr/bin/env bash
# Second-pass relauncher for the endothelial campaign: re-run the mCellSeg first-pass
# jobs that died with the original tmux shell (HAEC GPU jobs and mCellSeg v1_multi are
# already running from the previous fix), wait for everything, then tiles/train/bench_v2.
set -euo pipefail
ROOT=/data/okebell/pimorph/PiMorph
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
OUT=runs/endo
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/campaign.log"; }
bench() {
  local ds=$1 gpu=$2 m=$3 ckpt=$4 out=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu PIMORPH_NEURAL_CKPT=$ckpt $PY -m pimorph.cli benchmark --dataset $ds --methods $m --out $out "$@" > $out.log 2>&1
}
log "fix2: relaunch mCellSeg cellpose_sam, v0_synth, v0_mixed, classical"
bench mcellseg 0 cellpose_sam none $OUT/mcellseg_cellpose &
bench mcellseg 1 neural models/pimorph_proposals_v0_synth.pt $OUT/mcellseg_v0synth &
bench mcellseg 2 neural models/pimorph_proposals_v0_mixed.pt $OUT/mcellseg_v0mixed &
$PY -m pimorph.cli benchmark --dataset mcellseg --methods classical --out $OUT/mcellseg_classical > $OUT/mcellseg_classical.log 2>&1 &
# wait for the jobs launched by the first fix (different shell) as well as ours
while pgrep -f "pimorph.cli benchmark" >/dev/null; do sleep 60; done
log "bench_all done (after fix2)"
bash infra/xai/run_endothelial.sh tiles train bench_v2
