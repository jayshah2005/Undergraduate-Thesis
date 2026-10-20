#!/bin/bash
# Controlled PMMFNet experiments (plan.md), one after another on the GPU.
#   ./run_all.sh smoke    smoke test of every experiment (tiny, fold 0) -> runs/smoke/<exp>/
#   ./run_all.sh          all experiments, 5 folds each -> runs/<exp>/, then runs/comparison.md
# Resumable: finished experiments are skipped and finished folds are reloaded.
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
EXPS="A1 A2 S1 S2 S3 L2"
if [ "$1" = "smoke" ]; then
  mkdir -p runs/smoke
  for e in $EXPS; do
    echo "=== smoke $e ($(date +%T))"
    $PY -u controlled.py --experiment $e --smoke > runs/smoke/$e.log 2>&1 || { echo "FAILED - see runs/smoke/$e.log"; exit 1; }
    grep "\[smoke\]" runs/smoke/$e.log | tail -2
  done
  exit 0
fi
for e in $EXPS; do
  mkdir -p runs/$e
  echo "=== $e ($(date +%T))"
  $PY -u controlled.py --experiment $e >> runs/$e/log.txt 2>&1 || { echo "FAILED - see runs/$e/log.txt"; exit 1; }
done
$PY -u compare.py > runs/compare.log 2>&1
echo "=== done ($(date +%T)): runs/comparison.md"
