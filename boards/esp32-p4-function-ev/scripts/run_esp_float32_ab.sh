#!/usr/bin/env bash
# ESP32-P4-Function-EV float32 peer A/B: netkit vs TFLM (CNN + DS-CNN).
# ESP-NN has no float API — both sides are reference float. P4 has FPU.
# Methodology: 10×10, discard first invoke; order swaps nk↔tflm.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PORT="${PORT:-}"
LOGDIR="${LOGDIR:-$ROOT/benchmark/mcu_ab_logs/esp32_p4_ev}"
MODELS="${MODELS:-cnn cnn_dw}"
TIMEOUT="${TIMEOUT:-600}"
BAUD="${BAUD:-115200}"

pick_port() {
  local path
  for path in /dev/cu.usbmodem* /dev/ttyACM*; do
    [[ -e "$path" ]] && { echo "$path"; return 0; }
  done
  return 1
}
[[ -n "$PORT" ]] || PORT="$(pick_port)" || { echo "Set PORT=..." >&2; exit 1; }
mkdir -p "$LOGDIR"
echo "PORT=$PORT LOGDIR=$LOGDIR (float32)"

board_for() {
  case "$1-$2" in
    netkit-cnn) echo "esp32-p4-function-ev-cnn" ;;
    netkit-cnn_dw) echo "esp32-p4-function-ev-cnn-dw" ;;
    tflm-cnn) echo "esp32-p4-function-ev-tflm-cnn" ;;
    tflm-cnn_dw) echo "esp32-p4-function-ev-tflm-cnn-dw" ;;
    *) echo "unknown $1 $2" >&2; exit 1 ;;
  esac
}

capture_one() {
  local runtime="$1" model="$2" order_tag="$3"
  local board; board="$(board_for "$runtime" "$model")"
  local logfile="$LOGDIR/f32_${order_tag}_${runtime}_${model}.log"
  echo "=== build $board ==="
  make -C "$ROOT/boards/$board" all
  echo "=== flash $board → $PORT ==="
  make -C "$ROOT/boards/$board" flash PORT="$PORT"
  echo "=== capture → $logfile ==="
  python3 - "$PORT" "$BAUD" "$TIMEOUT" "$logfile" <<'PY'
import sys, time, serial
port, baud, timeout, logfile = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ser = serial.Serial(port, baud, timeout=0.3)
time.sleep(0.2)
ser.reset_input_buffer()
ser.dtr = False; ser.rts = True; time.sleep(0.1); ser.rts = False; time.sleep(0.1)
buf = bytearray(); deadline = time.time() + timeout
with open(logfile, "wb") as f:
    while time.time() < deadline:
        data = ser.read(4096)
        if data:
            f.write(data); f.flush(); buf.extend(data)
            sys.stdout.buffer.write(data); sys.stdout.flush()
            if b"BENCHMARK_SUMMARY" in buf and b"DONE" in buf:
                break
            if b"ERR " in buf or b"Guru Meditation" in buf or b"assert failed" in buf or b"Stack protection" in buf:
                sys.stderr.write("\n[firmware error; aborting]\n"); sys.exit(3)
    else:
        sys.stderr.write("\n[timeout]\n"); sys.exit(2)
ser.close()
PY
}

for model in $MODELS; do
  echo; echo "######## model=$model order=nk_then_tflm (float32) ########"
  capture_one netkit "$model" "nk_then_tflm"
  capture_one tflm "$model" "nk_then_tflm"
  echo; echo "######## model=$model order=tflm_then_nk (float32) ########"
  capture_one tflm "$model" "tflm_then_nk"
  capture_one netkit "$model" "tflm_then_nk"
done
echo; echo "=== BENCHMARK_SUMMARY (float32) ==="
rg -h "^BENCHMARK_SUMMARY" "$LOGDIR"/f32_*.log || true
echo "Logs in $LOGDIR (f32_*.log)"
