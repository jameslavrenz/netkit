#!/usr/bin/env bash
# Flash XIAO ESP32C3 firmware via PlatformIO / esptool (USB Serial/JTAG).
set -euo pipefail
BOARD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-}"
cd "$BOARD_ROOT"
if [[ -n "$PORT" ]]; then
  exec make flash PORT="$PORT"
fi
exec make flash
