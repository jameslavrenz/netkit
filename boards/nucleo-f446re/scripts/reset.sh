#!/usr/bin/env bash
# Reset NUCLEO-F446RE via onboard ST-Link (no flash).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENOCD_BOARD_CFG="$ROOT/openocd/nucleo_f446re.cfg"

if ! command -v openocd >/dev/null 2>&1; then
  echo "openocd not found — install with: brew install openocd" >&2
  exit 1
fi

openocd -f interface/stlink.cfg -f "$OPENOCD_BOARD_CFG" \
  -c "init; reset run; shutdown"
