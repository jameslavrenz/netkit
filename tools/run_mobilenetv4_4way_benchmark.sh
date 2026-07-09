#!/usr/bin/env bash
# Run netkit float32/int8 and TFLM float32/int8 MobileNetV4-small host benchmarks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPORT="${ROOT}/benchmark/mobilenetv4_4way_results.txt"
: >"$REPORT"

run_and_capture() {
  local label="$1"
  shift
  echo "=== $label ===" | tee -a "$REPORT"
  "$@" 2>&1 | tee -a "$REPORT"
  echo | tee -a "$REPORT"
}

run_and_capture "netkit float32" make -C benchmark/netkit run-mobilenetv4
run_and_capture "netkit int8" make -C benchmark/netkit run-mobilenetv4-int8

if make -C benchmark/tflm build-mobilenetv4 2>/dev/null; then
  run_and_capture "TFLM float32" make -C benchmark/tflm run-mobilenetv4
else
  echo "TFLM float32 build skipped (TFLM tree missing)" | tee -a "$REPORT"
fi

if [[ -f benchmark/tflm/generated/mobilenetv4_small_int8.tflite ]]; then
  if make -C benchmark/tflm build-mobilenetv4-int8 2>/dev/null; then
    run_and_capture "TFLM int8" make -C benchmark/tflm run-mobilenetv4-int8
  fi
else
  echo "TFLM int8 skipped (run: python3 benchmark/tflm/tools/export_mobilenetv4_int8.py)" | tee -a "$REPORT"
fi

echo "Wrote $REPORT"
