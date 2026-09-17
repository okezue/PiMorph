#!/usr/bin/env bash
# Run a command on the PiMorph xAI devbox, or copy files to it.
# Usage: infra/xai/remote.sh "<command>"   |   infra/xai/remote.sh cp <local files...>
# Requires: XAI_ROOT, Tailscale connected, `x login` done.
set -uo pipefail
export XAI_USER="${XAI_USER:-okebell}"
XAI_ROOT="${XAI_ROOT:-$HOME/workspace/xai}"
BOX="${PIMORPH_XAI_BOX:-okebell-pimorph}"
CLUSTER="${PIMORPH_XAI_CLUSTER:-fou}"
HOST="explorer.${XAI_USER}.svc.${CLUSTER}.x.ai"
if [ "${1:-}" = "cp" ]; then
  shift
  # `x ssh` has no copy mode; go through the explorer ssh endpoint
  scp -q -o StrictHostKeyChecking=no -i "$HOME/.ssh/${XAI_USER}.id_explorer" -P 2222 "$@" "root@${HOST}:/data/${XAI_USER}/pimorph/" 2>/dev/null \
    || "$XAI_ROOT/bin/explorer" ssh "$BOX" -c "$CLUSTER" -- "mkdir -p /data/${XAI_USER}/pimorph" && \
       for f in "$@"; do "$XAI_ROOT/bin/explorer" ssh "$BOX" -c "$CLUSTER" -- "cat > /data/${XAI_USER}/pimorph/$(basename "$f")" < "$f"; done
else
  "$XAI_ROOT/bin/explorer" ssh "$BOX" -c "$CLUSTER" -- "$@"
fi
