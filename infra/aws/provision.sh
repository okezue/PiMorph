#!/usr/bin/env bash
# Create the PiMorph training resources in the user's own AWS account.
# Every resource is tagged so cleanup.sh can find exactly what we made and nothing else.
# Usage: AWS_PROFILE=pimorph infra/aws/provision.sh
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-pimorph}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
SESSION="01a0ac4c"
TAGS="{Key=Project,Value=PiMorph},{Key=Owner,Value=okebell-grok},{Key=Session,Value=${SESSION}}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="pimorph-train-${ACCOUNT}-${SESSION}"
KEY_NAME="pimorph-${SESSION}"
SG_NAME="pimorph-ssh-${SESSION}"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
LOG="$(dirname "$0")/aws_log.md"
MY_IP="$(curl -s https://checkip.amazonaws.com)/32"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log() { echo "$1" | tee -a "$LOG"; }

echo "account=${ACCOUNT} region=${AWS_DEFAULT_REGION} ip=${MY_IP}"

# S3 bucket (idempotent)
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "bucket exists: $BUCKET"
else
  aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "TagSet=[{Key=Project,Value=PiMorph},{Key=Owner,Value=okebell-grok},{Key=Session,Value=${SESSION}},{Key=Purpose,Value=training-data-and-checkpoints}]"
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
  log "| ${STAMP} | s3 bucket | ${BUCKET} | ${AWS_DEFAULT_REGION} | training tiles + checkpoints | |"
fi

# Key pair (idempotent)
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "key pair exists: $KEY_NAME"
else
  aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519 --key-format pem \
    --tag-specifications "ResourceType=key-pair,Tags=[${TAGS},{Key=Purpose,Value=ssh}]" \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  log "| ${STAMP} | key pair | ${KEY_NAME} (${KEY_FILE}) | ${AWS_DEFAULT_REGION} | ssh to training instance | |"
fi

# Security group: SSH from this machine only (idempotent)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC}" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "PiMorph training ssh (session ${SESSION})" --vpc-id "$VPC" \
    --tag-specifications "ResourceType=security-group,Tags=[${TAGS},{Key=Purpose,Value=ssh-ingress}]" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP" >/dev/null
  log "| ${STAMP} | security group | ${SG_ID} (${SG_NAME}) | ${AWS_DEFAULT_REGION} | ssh from ${MY_IP} | |"
else
  echo "security group exists: $SG_ID"
fi

cat > "$(dirname "$0")/env.sh" <<EOF
export AWS_PROFILE=${AWS_PROFILE}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
export PIMORPH_BUCKET=${BUCKET}
export PIMORPH_KEY_NAME=${KEY_NAME}
export PIMORPH_KEY_FILE=${KEY_FILE}
export PIMORPH_SG_ID=${SG_ID}
export PIMORPH_SESSION=${SESSION}
EOF
echo "wrote $(dirname "$0")/env.sh"
