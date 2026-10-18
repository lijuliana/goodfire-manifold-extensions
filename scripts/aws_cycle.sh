#!/usr/bin/env bash
# launch -> sync -> setup -> run_extensions.sh (detached) -> poll -> pull -> terminate (verified) -> ledger
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/aws_launch.sh
scripts/aws_ssh.sh --sync
scripts/aws_ssh.sh 'cd ~/goodfire-manifold-replication && bash scripts/aws_setup.sh'
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_extensions.sh}"
DONE_MARKER="${DONE_MARKER:-EXT_DONE}"
scripts/aws_ssh.sh "cd ~/goodfire-manifold-replication && (setsid nohup bash $RUN_SCRIPT > logs/cycle_run.log 2>&1 < /dev/null &); sleep 2; echo launched"
for i in $(seq 1 480); do
  sleep 60
  state=$(scripts/aws_ssh.sh "cd ~/goodfire-manifold-replication; if grep -q '$DONE_MARKER' logs/cycle_run.log 2>/dev/null; then echo done; elif pgrep -f 'bash $RUN_SCRIPT' >/dev/null; then echo running; else echo dead; fi" 2>/dev/null | tail -1)
  [ "$state" = "running" ] || break
done
echo "remote job state: ${state:-unknown}"
scripts/aws_ssh.sh "cd ~/goodfire-manifold-replication; tr '\r' '\n' < logs/cycle_run.log | grep -vE 'Warning|warn|it/s\]|%\||^\s*$' | tail -60"
scripts/aws_ssh.sh --pull
scripts/aws_terminate.sh
echo CYCLE_DONE
