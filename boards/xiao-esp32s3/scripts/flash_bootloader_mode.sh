#!/usr/bin/env bash
# Flash netkit CNN int8 while board is held in ROM download mode (BOOT held + RESET).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BOARD="${BOARD:-$ROOT/boards/xiao-esp32s3-cnn-int8}"
ENV="${PIO_ENV:-xiao_esp32s3}"
PORT="${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)}"
export PATH="${HOME}/.platformio/penv/bin:${PATH}"
echo "Flashing $BOARD ($ENV) on PORT=$PORT"
echo "Enter download mode now: hold BOOT, tap RESET, keep BOOT until connect succeeds."
cd "$BOARD"
# Prefer no-reset once user has already entered bootloader
pio run -t upload -e "$ENV" --upload-port "$PORT"
