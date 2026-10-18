#!/usr/bin/env bash
# Launch one g6e.xlarge (L40S 48 GB) from the saved DLAMI, wait for ssh, record state.
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=us-east-1
TYPE="${TYPE:-g6e.xlarge}"
AMI=ami-012ba162b9cd2729c
SG=sg-007ca52c2b6cedb1f
NAME="${NAME:-manifolds-l40s}"
START=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
# capacity is spotty: try each zone, then the next instance type (same L40S GPU)
ID=""
for T in $TYPE ${FALLBACK_TYPES:-g6e.2xlarge}; do
  for SUBNET in subnet-01c179c5b0acfb19c subnet-0fbe511037257797f subnet-0b0d7b21c97bca06a subnet-05f369dfd32dd4c39; do
    ID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type "$T" --key-name cs2881r \
      --security-group-ids $SG --subnet-id $SUBNET --count 1 \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=goodfire-manifolds}]" \
      --query 'Instances[0].InstanceId' --output text 2>/tmp/aws_launch_err || true)
    if [ -n "$ID" ] && [ "$ID" != "None" ]; then TYPE=$T; break 2; fi
    echo "no capacity for $T in $SUBNET: $(tr -d '\n' < /tmp/aws_launch_err | cut -c1-120)"
    ID=""
  done
done
[ -n "$ID" ] || { echo "no capacity anywhere"; exit 2; }
echo "launched $ID ($TYPE)"
aws ec2 wait instance-running --region $REGION --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
mkdir -p logs
printf '{"id":"%s","ip":"%s","type":"%s","started":"%s","region":"%s"}\n' "$ID" "$IP" "$TYPE" "$START" "$REGION" > logs/aws_instance.json
echo "running at $IP; waiting for ssh"
for i in $(seq 1 40); do scripts/aws_ssh.sh 'echo up' >/dev/null 2>&1 && break; sleep 10; done
scripts/aws_ssh.sh 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h / | tail -1'
