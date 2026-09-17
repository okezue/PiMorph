#!/usr/bin/env bash
# Run a command on the training instance (or copy files with `remote.sh cp <src>... `).
# Retries connection-level failures (exit 255) a few times because the local NAT drops
# some connections.
# Usage: infra/aws/remote.sh "<command>"    |    infra/aws/remote.sh cp file1 file2 ...
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i "$PIMORPH_KEY_FILE")
for attempt in 1 2 3 4 5 6; do
  if [ "${1:-}" = "cp" ]; then
    shift_args=("${@:2}")
    scp -q "${OPTS[@]}" "${shift_args[@]}" "ubuntu@${PIMORPH_INSTANCE_IP}:"
  else
    ssh "${OPTS[@]}" "ubuntu@${PIMORPH_INSTANCE_IP}" "$@"
  fi
  rc=$?
  if [ $rc -ne 255 ]; then exit $rc; fi
  echo "ssh connection failed (attempt $attempt), retrying" >&2
  sleep $((attempt * 5))
done
exit 255
