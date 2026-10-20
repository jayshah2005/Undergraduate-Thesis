#!/bin/bash
# Runs the whole study, one step after another (sequential on purpose: the CPU
# scripts already use all cores, and running GPU scripts at the same time slows both).
#
#   ./experiments/run_all.sh          full run   (~40 min on a 14-core CPU + RTX GPU)
#   ./experiments/run_all.sh quick    smoke test (~3 min); writes to results/quick/
#
# Each step prints one line here and writes its full output to experiments/logs/.
# Final tables: experiments/results/summary.md and paired_tests.md
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
Q=""; [ "$1" = "quick" ] && Q="--quick"
LOGS=logs; [ -n "$Q" ] && LOGS=logs/quick       # smoke-test logs never overwrite full-run logs
mkdir -p "$LOGS"

# step "<title>" <logfile> <command...>
step() {
  local title=$1 log=$2; shift 2
  echo "=== $title  ($(date +%T))"
  "$@" > "$LOGS/$log" 2>&1 || { echo "FAILED - see experiments/$LOGS/$log"; exit 1; }
}

[ -f cache/meta.parquet ] || step "00 build cache (reads the 4 GB h5 once)" 00.log $PY -u 00_build_cache.py
step "01 variety-prior baseline"            01.log          $PY -u 01_baseline_variety_prior.py
step "02 classical, leaf-mean spectra"      02.log          $PY -u 02_classical_leaf_level.py $Q
step "02 classical, per-variety models"     02_byvariety.log $PY -u 02_classical_leaf_level.py $Q --by-variety
step "02 classical, leave-one-variety-out"  02_lovo.log     $PY -u 02_classical_leaf_level.py $Q --lovo
step "03 classical on pixels"               03.log          $PY -u 03_pixel_level_classical.py $Q
step "04 pixel MLP / 1D CNN"                04.log          $PY -u 04_pixel_mlp_cnn1d.py $Q
step "05 attention MIL"                     05.log          $PY -u 05_attention_mil.py $Q
step "06 ResNet-18 pretrained vs scratch"   06.log          $PY -u 06_resnet_transfer.py $Q
step "06b ResNet-18 with FIXED pseudo-RGB adaptor (control)" 06b.log $PY -u 06_resnet_transfer.py $Q --modes pretrained --adaptor fixed_rgb
step "07 foundation models (DOFA, DINOv2)"  07.log          $PY -u 07_foundation_models.py $Q
step "08 self-supervised pretraining"       08.log          $PY -u 08_self_supervised_pretraining.py $Q
[ -n "$Q" ] || step "10 diagnostics"        10.log          $PY -u 10_diagnostics.py
step "09 compare everything"                09.log          $PY -u 09_compare.py $Q
echo "=== done ($(date +%T)). Results: experiments/results/summary.md"
