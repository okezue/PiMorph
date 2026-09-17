#!/usr/bin/env bash
# Run a command on the training instance (or copy files with `remote.sh cp <src>... `).
# Usage: infra/aws/remote.sh "<command>"    |    infra/aws/remote.sh cp file1 file2 ...
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -i "$PIMORPH_KEY_FILE")
if [ "${1:-}" = "cp" ]; then
  shift
  scp -q "${OPTS[@]}" "$@" "ubuntu@${PIMORPH_INSTANCE_IP}:"
else
  ssh "${OPTS[@]}" "ubuntu@${PIMORPH_INSTANCE_IP}" "$@"
fi
