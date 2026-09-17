#!/usr/bin/env bash
# Tar the tile caches and the source, upload them to the project bucket, and write
# infra/aws/urls.txt with presigned GET URLs (valid 48 h) for bootstrap.sh.
# Usage: infra/aws/push_data.sh [tile_dir ...]   (default: synth_train synth_val pseudo_ve_strat pseudo_sbiad1540)
set -euo pipefail
# keep macOS from adding ._* AppleDouble entries to the tarballs
export COPYFILE_DISABLE=1
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE/../.."
DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then DIRS=(synth_train synth_val pseudo_ve_strat pseudo_sbiad1540); fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
COMMIT=$(git rev-parse --short HEAD)
TMP=$(mktemp -d)
: > "$HERE/urls.txt"

# code
git ls-files src tests scripts infra examples pyproject.toml README.md > "$TMP/files.txt"
git ls-files --others --exclude-standard src tests scripts infra examples >> "$TMP/files.txt"
grep -v "infra/aws/env.sh\|infra/aws/urls.txt" "$TMP/files.txt" > "$TMP/files2.txt"
tar czf "$TMP/code.tar.gz" -T "$TMP/files2.txt"
KEY="code/pimorph_${COMMIT}_${STAMP}.tar.gz"
aws s3 cp "$TMP/code.tar.gz" "s3://${PIMORPH_BUCKET}/${KEY}" --only-show-errors
echo "code.tar.gz $(aws s3 presign "s3://${PIMORPH_BUCKET}/${KEY}" --expires-in 172800)" >> "$HERE/urls.txt"
echo "uploaded $KEY"

# tiles (uncompressed tar: npz files are already compressed)
for d in "${DIRS[@]}"; do
  if [ ! -d "data/tiles/$d" ]; then echo "skip missing data/tiles/$d"; continue; fi
  tar cf "$TMP/tiles_$d.tar" -C data/tiles "$d"
  KEY="tiles/${d}_${STAMP}.tar"
  aws s3 cp "$TMP/tiles_$d.tar" "s3://${PIMORPH_BUCKET}/${KEY}" --only-show-errors
  echo "tiles_$d.tar $(aws s3 presign "s3://${PIMORPH_BUCKET}/${KEY}" --expires-in 172800)" >> "$HERE/urls.txt"
  echo "uploaded $KEY ($(du -h "$TMP/tiles_$d.tar" | cut -f1))"
done
rm -rf "$TMP"
echo "wrote $HERE/urls.txt ($(wc -l < "$HERE/urls.txt") entries)"
