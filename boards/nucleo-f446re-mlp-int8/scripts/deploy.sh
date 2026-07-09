#!/usr/bin/env bash
# MNIST CNN int8 MCU deploy — discrete steps, no long blocking in one shot.
# Usage:
#   ./scripts/deploy.sh export   # fast quantize from models/mnist_cnn.nk (~seconds)
#   ./scripts/deploy.sh build    # cross-compile firmware
#   ./scripts/deploy.sh flash    # ST-Link program
#   ./scripts/deploy.sh capture  # read UART 30s (after flash + reset)
#   ./scripts/deploy.sh all      # export + build + flash + capture
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
ELF="$ROOT/build/mnist_mlp_int8_nucleo_f446re.elf"
CAPTURE_SEC="${CAPTURE_SEC:-120}"

step_export() {
  echo "=== [1/4] export mnist_cnn_int8.nk + int8 test images ==="
  if [[ ! -f "$REPO/models/mnist_cnn.nk" ]]; then
    echo "Missing $REPO/models/mnist_cnn.nk — run: make -C $REPO export-mnist-cnn" >&2
    exit 1
  fi
  make -C "$REPO" export-mnist-mlp-int8
  ls -la "$REPO/models/mnist_mlp_int8.nk"
  ls -la "$REPO/benchmark/tflm/generated/mnist_mlp_int8_test_images.cc"
}

  step_build() {
  echo "=== [2/4] build firmware (interpreter embed by default) ==="
  make -C "$ROOT"
  "$ROOT/scripts/setup-toolchain.sh" >/dev/null
  TOOLCHAIN_BIN="$("$ROOT/scripts/setup-toolchain.sh")"
  "$TOOLCHAIN_BIN/arm-none-eabi-size" --format=berkeley "$ELF"
}

step_flash() {
  echo "=== [3/4] flash via ST-Link ==="
  if [[ ! -f "$ELF" ]]; then
    echo "ELF missing — run: ./scripts/deploy.sh build" >&2
    exit 1
  fi
  "$ROOT/scripts/flash.sh" "$ELF"
}

pick_port() {
  local candidates=()
  if [[ "$(uname -s)" == "Darwin" ]]; then
    candidates=(/dev/cu.usbmodem* /dev/tty.usbmodem*)
  else
    candidates=(/dev/ttyACM* /dev/serial/by-id/*STLINK*)
  fi
  local path
  for path in "${candidates[@]}"; do
    if [[ -e "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

step_capture() {
  echo "=== [4/4] capture UART (${CAPTURE_SEC}s) — press RESET on NUCLEO if silent ==="
  PORT="${PORT:-}"
  if [[ -z "$PORT" ]]; then
    PORT="$(pick_port)" || true
  fi
  if [[ -z "$PORT" ]]; then
    echo "No ST-Link VCP found. Set PORT=/dev/cu.usbmodem..." >&2
    exit 1
  fi
  echo "Reading $PORT @ 115200 for ${CAPTURE_SEC}s ..."
  python3 - "$PORT" "$CAPTURE_SEC" <<'PY'
import sys
import time

try:
    import serial
except ImportError:
    print("pip install pyserial", file=sys.stderr)
    sys.exit(1)

port, duration = sys.argv[1], float(sys.argv[2])
deadline = time.time() + duration
buf = bytearray()
with serial.Serial(port, 115200, timeout=0.3) as ser:
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            buf.extend(chunk)
            if b"DONE" in buf:
                break
        else:
            time.sleep(0.05)
PY
}

cmd="${1:-all}"
case "$cmd" in
  export) step_export ;;
  build)  step_build ;;
  flash)  step_flash ;;
  capture) step_capture ;;
  all)
    step_export
    step_build
    step_flash
    step_capture
    ;;
  *)
    echo "Usage: $0 {export|build|flash|capture|all}" >&2
    exit 1
    ;;
esac

echo "=== deploy.sh $cmd done ==="
