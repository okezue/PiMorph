#!/usr/bin/env bash
# Stream the working-tree source (tracked + untracked src/tests/scripts/infra) to the
# devbox and unpack it over /data/okebell/pimorph/PiMorph. Data, runs and the venv on
# the box are untouched. Usage: infra/xai/sync_code.sh [extra paths...]
set -euo pipefail
export COPYFILE_DISABLE=1
export XAI_USER="${XAI_USER:-okebell}"
XAI_ROOT="${XAI_ROOT:-$HOME/workspace/xai}"
BOX="${PIMORPH_XAI_BOX:-okebell-pimorph}"
CLUSTER="${PIMORPH_XAI_CLUSTER:-fou}"
WORK="/data/${XAI_USER}/pimorph/PiMorph"
cd "$(dirname "$0")/../.."
TMP=$(mktemp)
{ git ls-files src tests scripts infra pyproject.toml; git ls-files --others --exclude-standard src tests scripts infra; printf '%s\n' "$@"; } \
  | grep -v "infra/aws/env.sh\|infra/aws/urls" | grep -v '^$' | sort -u > "$TMP"
tar czf - -T "$TMP" | "$XAI_ROOT/bin/explorer" ssh "$BOX" -c "$CLUSTER" -- "mkdir -p $WORK && cd $WORK && tar xzf - && echo synced \$(git -C $WORK rev-parse --short HEAD 2>/dev/null || echo no-git) \$(date -u +%H:%M:%SZ)"
rm -f "$TMP"
