#!/usr/bin/env bash
# Upload the 1.2.0 files into an existing Zenodo draft with curl (one PUT per file, retries),
# then set the metadata and publish through scripts/publish_zenodo.py. Runs on the devbox.
# Usage: ZENODO_TOKEN=... bash infra/xai/zenodo_v1_2_upload.sh <draft_id>
set -uo pipefail
cd /data/okebell/pimorph/PiMorph
DRAFT=${1:?draft id}
OUT=output/zenodo_v1_2
LOG=/tmp/zenodo_upload.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
: "${ZENODO_TOKEN:?}"
BUCKET=$(curl -sS -H "Authorization: Bearer $ZENODO_TOKEN" "https://zenodo.org/api/deposit/depositions/$DRAFT" | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['links']['bucket'])")
log "draft $DRAFT bucket $BUCKET"

# one archive per fine-tuned model (a single 4.9 GB PUT is refused)
for m in cpsam_confluent cpsam_hcec cpsam_haec cpsam_qbam; do
  f=$OUT/pimorph_cellpose_ft_$m.tar
  [ -f $f ] || tar cf $f runs/cellpose_ft/$m/models runs/cellpose_ft/$m/train_info.json runs/cellpose_ft/train_cpsam.py $(ls runs/cellpose_ft/$m.log 2>/dev/null)
done
rm -f $OUT/pimorph_cellpose_finetuned.tar

upload() {  # file
  local f=$1 name; name=$(basename $f)
  local local_md5; local_md5=$(md5sum $f | cut -d' ' -f1)
  for attempt in 1 2 3 4; do
    log "PUT $name ($(stat -c%s $f) bytes) attempt $attempt"
    code=$(curl -sS -o /tmp/put_resp.json -w '%{http_code}' --retry 3 --retry-delay 20 -H "Authorization: Bearer $ZENODO_TOKEN" \
      -H "Content-Type: application/octet-stream" --upload-file $f "$BUCKET/$name")
    remote_md5=$(.venv/bin/python -c "import json; d=json.load(open('/tmp/put_resp.json')); print(d.get('checksum','').split(':')[-1])" 2>/dev/null)
    if [ "$code" = "200" ] || [ "$code" = "201" ]; then
      if [ "$remote_md5" = "$local_md5" ]; then log "ok $name md5 $local_md5"; return 0; fi
      log "checksum mismatch $name remote=$remote_md5 local=$local_md5"
    else
      log "http $code for $name: $(head -c 300 /tmp/put_resp.json)"
    fi
    sleep 30
  done
  return 1
}

FAIL=0
for f in $OUT/pimorph_cellpose_ft_cpsam_confluent.tar $OUT/pimorph_cellpose_ft_cpsam_hcec.tar $OUT/pimorph_cellpose_ft_cpsam_haec.tar \
         $OUT/pimorph_cellpose_ft_cpsam_qbam.tar $OUT/pimorph_healthy2_qbam.tar $OUT/pimorph_results.tar $OUT/README.md; do
  upload $f || FAIL=1
done
if [ $FAIL -ne 0 ]; then log "uploads incomplete; not publishing"; exit 1; fi

# the record README lists the split archives
.venv/bin/python - <<'EOF'
p = "output/zenodo_v1_2/README.md"
s = open(p).read()
old_start = s.index("| `pimorph_cellpose_finetuned.tar`")
old_end = s.index("\n", old_start) + 1
rows = (
    "| `pimorph_cellpose_ft_cpsam_confluent.tar` | 1.2 GB | (new in 1.2.0) Cellpose-SAM fine-tuned on the hCEC, alizarine, FlyWing and RPE training fields (the pool that trained PiMorph `v6_pool`), with `train_info.json`, the training script and log. Load with `PIMORPH_CELLPOSE_MODEL=runs/cellpose_ft/cpsam_confluent/models/cpsam_confluent` or `CellposeProposer(pretrained_model=...)`. Extracts into `runs/cellpose_ft/`. |\n"
    "| `pimorph_cellpose_ft_cpsam_hcec.tar` | 1.2 GB | (new in 1.2.0) Cellpose-SAM fine-tuned on the 10 hCEC training fields only. |\n"
    "| `pimorph_cellpose_ft_cpsam_haec.tar` | 1.2 GB | (new in 1.2.0) Cellpose-SAM fine-tuned on the 348 HAEC training fields. |\n"
    "| `pimorph_cellpose_ft_cpsam_qbam.tar` | 1.2 GB | (new in 1.2.0) Cellpose-SAM fine-tuned on 1,032 registered iPSC-RPE QBAM tiles (the segmenter of the barrier test; held-out wells AP50 0.741). |\n"
)
s = s[:old_start] + rows + s[old_end:]
open(p, "w").write(s)
d = "output/zenodo_v1_2/description.html"
t = open(d).read()
t = t.replace("<li><code>pimorph_cellpose_finetuned.tar</code> (4.7 GB): four Cellpose-SAM models",
              "<li><code>pimorph_cellpose_ft_cpsam_confluent.tar</code>, <code>pimorph_cellpose_ft_cpsam_hcec.tar</code>, <code>pimorph_cellpose_ft_cpsam_haec.tar</code>, <code>pimorph_cellpose_ft_cpsam_qbam.tar</code> (1.2 GB each): four Cellpose-SAM models")
open(d, "w").write(t)
EOF
upload $OUT/README.md || exit 1

log "publishing"
.venv/bin/python scripts/publish_zenodo.py --version 1.2.0 --publish --draft-id $DRAFT --description-file $OUT/description.html 2>&1 | tee -a $LOG
