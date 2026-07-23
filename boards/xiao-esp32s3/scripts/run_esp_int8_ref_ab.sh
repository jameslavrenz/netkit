#!/usr/bin/env bash
# XIAO ESP32S3 int8 peer A/B — ESP-NN OFF (reference) for netkit + TFLM.
#
# Same MCU methodology as ESP-NN-on A/B: 10 runs × 10 images, discard first
# invoke each run; order swaps nk→tflm and tflm→nk.
#
# Usage:
#   PORT=/dev/cu.usbmodem21101 ./boards/xiao-esp32s3/scripts/run_esp_int8_ref_ab.sh
#   MODELS="cnn" TIMEOUT=900 ./boards/xiao-esp32s3/scripts/run_esp_int8_ref_ab.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PORT="${PORT:-}"
LOGDIR="${LOGDIR:-$ROOT/benchmark/mcu_ab_logs/xiao_esp32s3}"
MODELS="${MODELS:-cnn cnn_dw}"
# Reference int8 is much slower than ESP-NN; allow ~15 min per firmware capture.
TIMEOUT="${TIMEOUT:-900}"
BAUD="${BAUD:-115200}"
PIO_ENV="${PIO_ENV:-xiao_esp32s3_ref}"

pick_port() {
  local path
  for path in /dev/cu.usbmodem* /dev/ttyACM*; do
    if [[ -e "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if [[ -z "$PORT" ]]; then
  PORT="$(pick_port)" || {
    echo "Set PORT=/dev/cu.usbmodem… (XIAO USB Serial/JTAG)" >&2
    exit 1
  }
fi

mkdir -p "$LOGDIR"
echo "PORT=$PORT LOGDIR=$LOGDIR PIO_ENV=$PIO_ENV (ESP-NN off / reference)"

board_for() {
  local runtime="$1" model="$2"
  case "$runtime-$model" in
    netkit-cnn) echo "xiao-esp32s3-cnn-int8" ;;
    netkit-cnn_dw) echo "xiao-esp32s3-cnn-dw-int8" ;;
    tflm-cnn) echo "xiao-esp32s3-tflm-cnn-int8" ;;
    tflm-cnn_dw) echo "xiao-esp32s3-tflm-cnn-dw-int8" ;;
    *) echo "unknown $runtime $model" >&2; exit 1 ;;
  esac
}

capture_one() {
  local runtime="$1" model="$2" order_tag="$3"
  local board
  board="$(board_for "$runtime" "$model")"
  local logfile="$LOGDIR/ref_${order_tag}_${runtime}_${model}.log"

  echo "=== fullclean+build $board env=$PIO_ENV ==="
  # Switching ESP-NN on/off needs a clean IDF cmake cache.
  make -C "$ROOT/boards/$board" PIO_ENV="$PIO_ENV" fullclean || true
  make -C "$ROOT/boards/$board" PIO_ENV="$PIO_ENV" all

  echo "=== flash $board → $PORT ==="
  make -C "$ROOT/boards/$board" PIO_ENV="$PIO_ENV" flash PORT="$PORT"

  echo "=== capture → $logfile (timeout ${TIMEOUT}s) ==="
  python3 - "$PORT" "$BAUD" "$TIMEOUT" "$logfile" <<'PY'
import sys, time, serial

port, baud, timeout, logfile = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ser = serial.Serial(port, baud, timeout=0.3)
time.sleep(0.2)
ser.reset_input_buffer()
# USB-JTAG reset pulse
ser.dtr = False
ser.rts = True
time.sleep(0.1)
ser.rts = False
time.sleep(0.1)
buf = bytearray()
deadline = time.time() + timeout
with open(logfile, "wb") as f:
    while time.time() < deadline:
        data = ser.read(4096)
        if data:
            f.write(data)
            f.flush()
            buf.extend(data)
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
            if b"BENCHMARK_SUMMARY" in buf and b"DONE" in buf:
                break
            # Fail fast on firmware abort (don't burn the full TIMEOUT).
            if b"ERR model load" in buf or b"ERR probe" in buf or b"Guru Meditation" in buf:
                sys.stderr.write("\n[firmware error; aborting capture]\n")
                sys.exit(3)
    else:
        sys.stderr.write("\n[timeout waiting for DONE]\n")
        sys.exit(2)
ser.close()
PY
}

summarize() {
  echo
  echo "=== BENCHMARK_SUMMARY lines (reference) ==="
  rg -h "^BENCHMARK_SUMMARY" "$LOGDIR"/ref_*.log || true
}

for model in $MODELS; do
  echo
  echo "######## model=$model order=nk_then_tflm (ESP-NN off) ########"
  capture_one netkit "$model" "nk_then_tflm"
  capture_one tflm "$model" "nk_then_tflm"

  echo
  echo "######## model=$model order=tflm_then_nk (ESP-NN off) ########"
  capture_one tflm "$model" "tflm_then_nk"
  capture_one netkit "$model" "tflm_then_nk"
done

summarize
echo
echo "Logs in $LOGDIR (ref_*.log)"
echo "ImageNet/MobileNetV4 skipped: weights exceed ESP32-S3 factory app partition (~1 MiB)."
