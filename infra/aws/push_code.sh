#!/usr/bin/env bash
# Tar the tracked source (no data, no runs) and upload it to the project bucket.
# Prints the S3 key to pass to bootstrap.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE/../.."
COMMIT=$(git rev-parse --short HEAD)
TMP=$(mktemp -d)
# tracked files plus any uncommitted source changes under src/tests/scripts/infra
git ls-files src tests scripts infra examples pyproject.toml README.md > "$TMP/files.txt"
git ls-files --others --exclude-standard src tests scripts infra examples >> "$TMP/files.txt"
tar czf "$TMP/code.tar.gz" -T "$TMP/files.txt"
KEY="code/pimorph_${COMMIT}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
aws s3 cp "$TMP/code.tar.gz" "s3://${PIMORPH_BUCKET}/${KEY}" --only-show-errors
echo "$KEY"
