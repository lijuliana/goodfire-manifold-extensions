#!/usr/bin/env bash
# start -> sync -> run_depth.sh -> pull -> deallocate (verified) -> ledger
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
RG=rg-ditloracle-swedencentral; VM=ditloracle-t4-caption
START=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
az vm start -g $RG -n $VM -o none
for i in $(seq 1 20); do scripts/vm_ssh.sh 'echo up' >/dev/null 2>&1 && break; sleep 10; done
scripts/vm_ssh.sh --sync
scripts/vm_ssh.sh 'cd ~/goodfire-manifold-replication && bash scripts/rename_political.sh'
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_depth.sh}"
PURPOSE="${PURPOSE:-VM uptime session 3 (billing): political_bias_aug + depth sweep + lenses + steering}"
DONE_MARKER="${DONE_MARKER:-DEPTH_DONE}"
scripts/vm_ssh.sh "cd ~/goodfire-manifold-replication && export PATH=\$HOME/.local/bin:\$PATH && uv sync --frozen --extra dev >/dev/null 2>&1; (setsid nohup bash $RUN_SCRIPT > logs/cycle_run.log 2>&1 < /dev/null &); sleep 2; echo launched"
# poll until the marker appears or the job is gone (an SSH drop no longer kills the job)
for i in $(seq 1 720); do
  sleep 60
  state=$(scripts/vm_ssh.sh "cd ~/goodfire-manifold-replication; if grep -q '$DONE_MARKER' logs/cycle_run.log 2>/dev/null; then echo done; elif pgrep -f 'bash $RUN_SCRIPT' >/dev/null; then echo running; else echo dead; fi" 2>/dev/null | tail -1)
  [ "$state" = "running" ] || break
done
echo "remote job state: $state"
scripts/vm_ssh.sh "cd ~/goodfire-manifold-replication; tr '\r' '\n' < logs/cycle_run.log | grep -vE 'Warning|warn|it/s\]|%\||^\s*$' | tail -40"
scripts/vm_ssh.sh --pull
az vm deallocate -g $RG -n $VM -o none
STATE=$(az vm show -d -g $RG -n $VM --query powerState -o tsv); echo "VM state: $STATE"
END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
uv run manifolds ledger add --provider azure --instance $VM --gpu "Tesla T4" --start "${START%Z}+00:00" --end "${END%Z}+00:00" --purpose "$PURPOSE" --notes "billing-uptime"
uv run manifolds ledger summary
echo CYCLE_DONE
