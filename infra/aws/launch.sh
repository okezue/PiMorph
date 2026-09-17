#!/usr/bin/env bash
# Launch ONE tagged GPU instance for PiMorph training. Spot first, on-demand fallback.
# Usage: infra/aws/launch.sh [instance_type] [spot|ondemand]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
TYPE="${1:-g6.xlarge}"
MARKET="${2:-spot}"
AMI="${PIMORPH_AMI:-ami-012ba162b9cd2729c}"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260427
LOG="$HERE/aws_log.md"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TAGS="{Key=Project,Value=PiMorph},{Key=Owner,Value=okebell-grok},{Key=Session,Value=${PIMORPH_SESSION}},{Key=Purpose,Value=neural-proposal-training},{Key=Name,Value=pimorph-train-${PIMORPH_SESSION}}"

MARKET_OPTS=()
if [ "$MARKET" = "spot" ]; then
  MARKET_OPTS=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi

set +e
OUT=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$TYPE" \
  --key-name "$PIMORPH_KEY_NAME" --security-group-ids "$PIMORPH_SG_ID" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[${TAGS}]" "ResourceType=volume,Tags=[${TAGS}]" \
  "${MARKET_OPTS[@]}" \
  --query 'Instances[0].InstanceId' --output text 2>&1)
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "launch failed ($MARKET $TYPE): $OUT"
  if [ "$MARKET" = "spot" ]; then
    echo "retrying on-demand"
    exec "$0" "$TYPE" ondemand
  fi
  exit 1
fi
IID="$OUT"
echo "instance: $IID ($TYPE, $MARKET)"
echo "| ${STAMP} | ec2 instance | ${IID} | ${AWS_DEFAULT_REGION} | ${TYPE} ${MARKET}, neural proposal training | |" | tee -a "$LOG"
echo "| ${IID} | ${TYPE} (${MARKET}) | ${STAMP} | | | |" >> "$LOG"

aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "export PIMORPH_INSTANCE_ID=${IID}" >> "$HERE/env.sh"
echo "export PIMORPH_INSTANCE_IP=${IP}" >> "$HERE/env.sh"
echo "public ip: $IP"
echo "ssh -i $PIMORPH_KEY_FILE ubuntu@$IP"
