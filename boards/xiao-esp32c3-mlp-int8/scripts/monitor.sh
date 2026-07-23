#!/usr/bin/env bash
# USB Serial/JTAG monitor for XIAO ESP32C3 (VID:PID 303A:1001).
set -euo pipefail
BOARD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BAUD="${BAUD:-115200}"

pick_port() {
  local candidates=()
  if [[ "$(uname -s)" == "Darwin" ]]; then
    candidates=(/dev/cu.usbmodem*)
  else
    candidates=(/dev/ttyACM* /dev/ttyUSB*)
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

PORT="${PORT:-}"
if [[ -z "$PORT" ]]; then
  PORT="$(pick_port)" || true
fi

cd "$BOARD_ROOT"
if [[ -n "$PORT" ]]; then
  exec make monitor PORT="$PORT"
fi
exec make monitor
