#!/usr/bin/env bash
# Endothelial ground-truth campaign on the 8xH100 devbox.
# Phases: fetch bench_all tiles train bench_v2   (default: all)
# Datasets: mCellSeg (200 expert DIC images, 100 HUVEC) and HAEC (434 fluorescence
# fields, GFP geometry + Hoechst nuclei, 4-class GT -> instances).
set -euo pipefail
ROOT=/data/okebell/pimorph/PiMorph
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
OUT=runs/endo
mkdir -p "$OUT"
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(fetch bench_all tiles train bench_v2)
has() { for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/campaign.log"; }

# --------------------------------------------------------------------- 0. fetch
if has fetch; then
  log "fetch: code + mCellSeg + HAEC from presigned URLs"
  while read -r name url; do
    [ -z "$name" ] && continue
    curl -sS -L --retry 3 -o "/tmp/$name" "$url"
  done < /data/okebell/pimorph/urls_endo.txt
  tar xzf /tmp/code.tar.gz
  mkdir -p data && tar xf /tmp/mcellseg.tar -C data && tar xf /tmp/haec_gt.tar -C data
  find data/mcellseg data/haec_gt -name '._*' -delete || true
  log "fetch: mcellseg images=$(ls data/mcellseg/labeled/images | wc -l) haec fields=$(ls data/haec_gt/GFP_original | wc -l)"
fi

# Held-out split by field id (deterministic): HAEC fields with id % 5 == 0 are test (87),
# mCellSeg every 5th sorted file is test (40, about 20 HUVEC). Written once as lists.
$PY - <<'EOF'
from pathlib import Path
import json
ids = sorted(int(p.stem) for p in Path("data/haec_gt/GFP_original").glob("*.tif"))
test = [i for i in ids if i % 5 == 0]; train = [i for i in ids if i % 5 != 0]
imgs = sorted(p.stem for p in Path("data/mcellseg/labeled/images").glob("*.tif"))
mtest = imgs[::5]; mtrain = [s for s in imgs if s not in set(mtest)]
Path("runs/endo").mkdir(parents=True, exist_ok=True)
json.dump({"haec_train": train, "haec_test": test, "mcellseg_train": mtrain, "mcellseg_test": mtest}, open("runs/endo/splits.json", "w"), indent=0)
print("splits: haec", len(train), len(test), "mcellseg", len(mtrain), len(mtest), "huvec test", sum('HUVEC' in s for s in mtest))
EOF

# ------------------------------------------------- 1. every method on both datasets
# One GPU per (dataset, method). Neural checkpoints: v0 synth, v0 mixed, v1 multi.
bench() {  # dataset gpu method ckpt outdir extra_env
  local ds=$1 gpu=$2 m=$3 ckpt=$4 out=$5; shift 5
  env "$@" CUDA_VISIBLE_DEVICES=$gpu PIMORPH_NEURAL_CKPT=$ckpt $PY -m pimorph.cli benchmark --dataset $ds --methods $m --out $out > $out.log 2>&1
}
if has bench_all; then
  log "bench_all: cellpose_sam, classical, neural(v0_synth, v0_mixed, v1_multi) on mCellSeg (200) and HAEC (200 sampled)"
  bench mcellseg 0 cellpose_sam none $OUT/mcellseg_cellpose &
  bench mcellseg 1 neural models/pimorph_proposals_v0_synth.pt $OUT/mcellseg_v0synth &
  bench mcellseg 2 neural models/pimorph_proposals_v0_mixed.pt $OUT/mcellseg_v0mixed &
  bench mcellseg 3 neural models/pimorph_proposals_v1_multi.pt $OUT/mcellseg_v1multi &
  bench haec 4 cellpose_sam none $OUT/haec_cellpose --max-items 200 &
  bench haec 5 neural models/pimorph_proposals_v0_mixed.pt $OUT/haec_v0mixed --max-items 200 &
  bench haec 6 neural models/pimorph_proposals_v1_multi.pt $OUT/haec_v1multi --max-items 200 &
  bench haec 7 neural models/pimorph_proposals_v0_synth.pt $OUT/haec_v0synth --max-items 200 &
  $PY -m pimorph.cli benchmark --dataset mcellseg --methods classical --out $OUT/mcellseg_classical > $OUT/mcellseg_classical.log 2>&1 &
  $PY -m pimorph.cli benchmark --dataset haec --methods classical --max-items 100 --out $OUT/haec_classical > $OUT/haec_classical.log 2>&1 &
  wait
  log "bench_all done"
fi

# --------------------------------------------- 2. real-GT tiles from the train splits
if has tiles; then
  log "tiles: HAEC train fields (GFP geometry + Hoechst nuclei) and mCellSeg train"
  for k in $(seq 0 15); do
    $PY scripts/make_gt_tiles.py --dataset haec --out data/tiles/gt_haec_train --tile 512 --stride 400 --shard $k/16 --id-list runs/endo/splits.json:haec_train > $OUT/gt_haec_$k.log 2>&1 &
  done
  wait
  for k in $(seq 0 7); do
    $PY scripts/make_gt_tiles.py --dataset mcellseg --out data/tiles/gt_mcellseg_train --tile 512 --stride 448 --shard $k/8 --id-list runs/endo/splits.json:mcellseg_train > $OUT/gt_mcellseg_$k.log 2>&1 &
  done
  wait
  log "tiles: gt_haec_train=$(find data/tiles/gt_haec_train -name '*.npz' | wc -l) gt_mcellseg_train=$(find data/tiles/gt_mcellseg_train -name '*.npz' | wc -l)"
fi

# --------------------------------------------------- 3. fine-tune v1_multi on endothelium
if has train; then
  log "train: v2_endo = v1_multi resumed on HAEC + mCellSeg train tiles + synthetic (DDP 8 GPUs, 40 epochs)"
  $PY -m torch.distributed.run --standalone --nproc_per_node 8 -m pimorph.infer.neural.train \
    --train-dirs data/tiles/gt_haec_train data/tiles/gt_mcellseg_train data/tiles/gt_haec_train data/tiles/synth_train data/tiles/pseudo_sbiad1540 data/tiles/pseudo_ve_strat \
    --val-dirs data/tiles/synth_val \
    --out runs/neural/v2_endo --resume runs/neural/v3_multi/best.pt --epochs 100 --batch-size 8 --crop 512 --base 48 --depth 4 \
    --device cuda --amp --num-workers 6 --seed 5 --log-every 100 --lr 1.5e-4 > $OUT/train_v2_endo.log 2>&1
  log "train done: $(tail -1 runs/neural/v2_endo/train_log.jsonl | cut -c1-160)"
fi

# ---------------------------------------------- 4. held-out test splits, all methods
if has bench_v2; then
  log "bench_v2: held-out HAEC test (87) and mCellSeg test (40) for v2_endo, v1_multi, cellpose_sam"
  export PIMORPH_ID_LIST=runs/endo/splits.json:haec_test
  bench haec 0 neural runs/neural/v2_endo/best.pt $OUT/haec_test_v2endo &
  bench haec 1 neural models/pimorph_proposals_v1_multi.pt $OUT/haec_test_v1multi &
  bench haec 2 cellpose_sam none $OUT/haec_test_cellpose &
  export PIMORPH_ID_LIST=runs/endo/splits.json:mcellseg_test
  bench mcellseg 3 neural runs/neural/v2_endo/best.pt $OUT/mcellseg_test_v2endo &
  bench mcellseg 4 neural models/pimorph_proposals_v1_multi.pt $OUT/mcellseg_test_v1multi &
  bench mcellseg 5 cellpose_sam none $OUT/mcellseg_test_cellpose &
  wait
  unset PIMORPH_ID_LIST
  log "bench_v2 done"
fi
log "endothelial campaign finished"
