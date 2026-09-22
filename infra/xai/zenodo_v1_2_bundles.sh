#!/usr/bin/env bash
# Build the version 1.2.0 bundles of the PiMorph Zenodo record ON THE DEVBOX, where the
# fine-tuned Cellpose weights and the Healthy-2 QBAM tiles live, then publish with
# scripts/publish_zenodo.py (token from ZENODO_TOKEN in the calling environment only).
# Usage on the box: ZENODO_TOKEN=... bash infra/xai/zenodo_v1_2_bundles.sh [--publish]
set -euo pipefail
cd /data/okebell/pimorph/PiMorph
OUT=output/zenodo_v1_2
mkdir -p $OUT

# 1. fine-tuned Cellpose-SAM weights of the symmetric comparison and the QBAM segmenter
tar cf $OUT/pimorph_cellpose_finetuned.tar \
  runs/cellpose_ft/cpsam_confluent/models runs/cellpose_ft/cpsam_confluent/train_info.json \
  runs/cellpose_ft/cpsam_hcec/models runs/cellpose_ft/cpsam_hcec/train_info.json \
  runs/cellpose_ft/cpsam_haec/models runs/cellpose_ft/cpsam_haec/train_info.json \
  runs/cellpose_ft/cpsam_qbam/models runs/cellpose_ft/cpsam_qbam/train_info.json \
  runs/cellpose_ft/train_cpsam.py \
  $(ls runs/cellpose_ft/*.log 2>/dev/null)

# 2. Healthy-2 QBAM tiles used for the TER test, with the decoded complexes when present
tar cf $OUT/pimorph_healthy2_qbam.tar \
  data/rpe_nist/healthy2/INVENTORY.md data/rpe_nist/healthy2/meta data/rpe_nist/healthy2/selection.csv \
  data/rpe_nist/healthy2/ter_long.csv data/rpe_nist/healthy2/tiles \
  $(ls -d data/rpe_nist/healthy2/labels data/rpe_nist/healthy2/decoded 2>/dev/null)

for f in $OUT/*.tar; do printf "%s  %s  %s\n" "$(md5sum $f | cut -d' ' -f1)" "$(stat -c%s $f)" "$(basename $f)"; done | tee $OUT/checksums.txt

if [ "${1:-}" = "--publish" ]; then
  .venv/bin/python scripts/publish_zenodo.py --version 1.2.0 \
    --file $OUT/pimorph_cellpose_finetuned.tar --file $OUT/pimorph_healthy2_qbam.tar \
    --file output/zenodo_v1_2/pimorph_results.tar --file output/zenodo_v1_2/README.md \
    --description-file output/zenodo_v1_2/description.html --publish
fi
