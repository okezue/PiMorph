#!/usr/bin/env bash
# Terminate and delete ONLY resources tagged Project=PiMorph AND Session=<this session>.
# The S3 bucket is kept unless --delete-bucket is passed (checkpoints live there).
# Usage: infra/aws/cleanup.sh [--delete-bucket]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
LOG="$HERE/aws_log.md"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FILTER=(--filters "Name=tag:Project,Values=PiMorph" "Name=tag:Session,Values=${PIMORPH_SESSION}")

IIDS=$(aws ec2 describe-instances "${FILTER[@]}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$IIDS" ]; then
  echo "terminating: $IIDS"
  aws ec2 terminate-instances --instance-ids $IIDS >/dev/null
  aws ec2 wait instance-terminated --instance-ids $IIDS
  for i in $IIDS; do echo "| ${STAMP} | terminated | ${i} | ${AWS_DEFAULT_REGION} | | ${STAMP} |" | tee -a "$LOG"; done
else
  echo "no tagged instances running"
fi

if aws ec2 describe-security-groups --group-ids "$PIMORPH_SG_ID" >/dev/null 2>&1; then
  aws ec2 delete-security-group --group-id "$PIMORPH_SG_ID" && echo "| ${STAMP} | deleted security group | ${PIMORPH_SG_ID} | ${AWS_DEFAULT_REGION} | | ${STAMP} |" | tee -a "$LOG"
fi
if aws ec2 describe-key-pairs --key-names "$PIMORPH_KEY_NAME" >/dev/null 2>&1; then
  aws ec2 delete-key-pair --key-name "$PIMORPH_KEY_NAME" && echo "| ${STAMP} | deleted key pair | ${PIMORPH_KEY_NAME} | ${AWS_DEFAULT_REGION} | | ${STAMP} |" | tee -a "$LOG"
fi
if [ "${1:-}" = "--delete-bucket" ]; then
  aws s3 rb "s3://${PIMORPH_BUCKET}" --force && echo "| ${STAMP} | deleted bucket | ${PIMORPH_BUCKET} | ${AWS_DEFAULT_REGION} | | ${STAMP} |" | tee -a "$LOG"
else
  echo "bucket kept: s3://${PIMORPH_BUCKET} (pass --delete-bucket to remove)"
fi
