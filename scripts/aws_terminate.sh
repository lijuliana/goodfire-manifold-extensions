#!/usr/bin/env bash
# Terminate the instance in logs/aws_instance.json, verify, and log billed uptime to the ledger.
set -euo pipefail
cd "$(dirname "$0")/.."
ID=$(python3 -c "import json;print(json.load(open('logs/aws_instance.json'))['id'])")
START=$(python3 -c "import json;print(json.load(open('logs/aws_instance.json'))['started'])")
TYPE=$(python3 -c "import json;print(json.load(open('logs/aws_instance.json'))['type'])")
REGION=us-east-1
aws ec2 terminate-instances --region $REGION --instance-ids "$ID" --output text >/dev/null
aws ec2 wait instance-terminated --region $REGION --instance-ids "$ID"
STATE=$(aws ec2 describe-instances --region $REGION --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
END=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
echo "instance $ID state: $STATE"
uv run manifolds ledger add --provider aws --instance "$TYPE" --gpu "NVIDIA L40S" --start "$START" --end "$END" \
  --purpose "${PURPOSE:-AWS uptime (billing): debate extensions on pilot manifolds}" --notes "billing-uptime; on-demand"
uv run manifolds ledger summary
aws ec2 describe-instances --region $REGION --filters Name=instance-state-name,Values=running,pending,stopping --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' --output text
