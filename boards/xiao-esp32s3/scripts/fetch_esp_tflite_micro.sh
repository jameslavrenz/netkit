#!/usr/bin/env bash
# Fetch Espressif esp-tflite-micro into a TFLM XIAO board components/ tree.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="${1:-$ROOT/boards/xiao-esp32s3-tflm-cnn-int8/components/esp-tflite-micro}"
TAG="${ESP_TFLITE_MICRO_TAG:-v1.3.7}"
mkdir -p "$(dirname "$DEST")"
if [[ -f "$DEST/CMakeLists.txt" ]]; then
  echo "esp-tflite-micro already present: $DEST"
  exit 0
fi
git clone --depth 1 --branch "$TAG" https://github.com/espressif/esp-tflite-micro.git "$DEST"
echo "cloned $TAG → $DEST"
