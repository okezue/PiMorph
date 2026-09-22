#!/usr/bin/env bash
# Relay large Zenodo uploads through AWS: the devbox pushes the bundles to the project S3
# bucket (fast), a small on-demand EC2 instance in us-east-1 downloads them and PUTs them
# into the Zenodo draft bucket (good route to CERN), logs to S3 and terminates itself.
# Nothing heavy runs on the laptop: only presigning and a handful of aws cli calls.
# Usage: ZENODO_TOKEN=... ZENODO_BUCKET=https://zenodo.org/api/files/<id> \
#        infra/aws/zenodo_relay.sh prepare  file1 file2 ...   # writes output/relay/manifest.tsv
#        infra/aws/zenodo_relay.sh launch                      # after the devbox has pushed
#        infra/aws/zenodo_relay.sh status
set -euo pipefail
cd "$(dirname "$0")/../.."
source infra/aws/env.sh
# the relay may run in another region (quota); S3 presigned URLs are region-independent
export AWS_DEFAULT_REGION="${RELAY_REGION:-$AWS_DEFAULT_REGION}"
OUT=output/relay
mkdir -p $OUT
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=infra/aws/aws_log.md
PREFIX="relay/${PIMORPH_SESSION}"

case "${1:-}" in
prepare)
  shift
  : > $OUT/manifest.tsv
  for f in "$@"; do
    name=$(basename "$f")
    key="$PREFIX/$name"
    put=$(.venv/bin/python - "$PIMORPH_BUCKET" "$key" <<'EOF'
import sys, boto3
b, k = sys.argv[1], sys.argv[2]
print(boto3.client("s3").generate_presigned_url("put_object", Params={"Bucket": b, "Key": k}, ExpiresIn=86400))
EOF
)
    get=$(aws s3 presign "s3://$PIMORPH_BUCKET/$key" --expires-in 86400)
    printf "%s\t%s\t%s\n" "$name" "$put" "$get" >> $OUT/manifest.tsv
  done
  logput=$(.venv/bin/python - "$PIMORPH_BUCKET" "$PREFIX/relay.log" <<'EOF'
import sys, boto3
b, k = sys.argv[1], sys.argv[2]
print(boto3.client("s3").generate_presigned_url("put_object", Params={"Bucket": b, "Key": k}, ExpiresIn=86400))
EOF
)
  echo "$logput" > $OUT/log_put_url.txt
  echo "manifest with $(wc -l < $OUT/manifest.tsv) files: $OUT/manifest.tsv (PUT urls for the devbox, GET urls for the relay)"
  ;;
launch)
  : "${ZENODO_TOKEN:?}" "${ZENODO_BUCKET:?}"
  AMI=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023*-kernel-*-x86_64" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  LOGPUT=$(cat $OUT/log_put_url.txt)
  {
    echo '#!/bin/bash'
    echo 'set -u; cd /mnt || cd /tmp; L=/tmp/relay.log; log(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; curl -sS -T $L "'"$LOGPUT"'" >/dev/null 2>&1 || true; }'
    echo "TOKEN='$ZENODO_TOKEN'; BUCKET='$ZENODO_BUCKET'"
    echo 'log "relay start $(df -h / | tail -1)"'
    while IFS=$'\t' read -r name put get; do
      echo "name='$name'; get='$get'"
      cat <<'EOS'
log "download $name"; curl -sS -L -o "$name" "$get" || { log "download failed $name"; continue; }
local_md5=$(md5sum "$name" | cut -d" " -f1); ok=0
for a in 1 2 3; do
  code=$(curl -sS -o resp.json -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/octet-stream" --upload-file "$name" "$BUCKET/$name")
  remote=$(python3 -c "import json;print(json.load(open('resp.json')).get('checksum','').split(':')[-1])" 2>/dev/null)
  if { [ "$code" = 200 ] || [ "$code" = 201 ]; } && [ "$remote" = "$local_md5" ]; then log "ok $name md5 $local_md5 ($(stat -c%s "$name") bytes)"; ok=1; break; fi
  log "attempt $a http $code remote=$remote local=$local_md5 $(head -c 200 resp.json)"; sleep 30
done
[ $ok = 1 ] || log "FAILED $name"; rm -f "$name"
EOS
    done < $OUT/manifest.tsv
    echo 'log "relay done"; shutdown -h now'
  } > $OUT/user_data.sh
  TAGS="{Key=Project,Value=PiMorph},{Key=Owner,Value=okebell-grok},{Key=Session,Value=${PIMORPH_SESSION}},{Key=Purpose,Value=zenodo-relay},{Key=Name,Value=pimorph-zenodo-relay-${PIMORPH_SESSION}}"
  ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type ${RELAY_TYPE:-t3.small} --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=24,VolumeType=gp3,DeleteOnTermination=true}' \
    --user-data file://$OUT/user_data.sh --tag-specifications "ResourceType=instance,Tags=[${TAGS}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "$ID" > $OUT/instance_id.txt
  echo "| ${STAMP} | launched | $ID ${RELAY_TYPE:-t3.small} on-demand ${AWS_DEFAULT_REGION}, self-terminating Zenodo relay (user-data), 24 GB gp3 | ~0.02 USD/h |" >> $LOG
  echo "launched $ID (self-terminating); status: infra/aws/zenodo_relay.sh status"
  ;;
status)
  ID=$(cat $OUT/instance_id.txt)
  aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text
  aws s3 cp "s3://$PIMORPH_BUCKET/$PREFIX/relay.log" - 2>/dev/null | tail -12 || echo "no log yet"
  ;;
*) echo "usage: prepare files... | launch | status"; exit 2 ;;
esac
