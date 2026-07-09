#!/usr/bin/env bash
# Fetch CMSIS packs required by TFLM cortex_m_generic cross-build (curl fallback for wget).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TFLM_ROOT="$ROOT/benchmark/tflm/third_party/tflite-micro"
DL="$TFLM_ROOT/tensorflow/lite/micro/tools/make/downloads"
CMSIS="$DL/cmsis"

if [[ -d "$CMSIS/CMSIS" && -d "$CMSIS/Cortex_DFP" ]]; then
  echo "CMSIS already present"
  exit 0
fi

mkdir -p "$DL"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

CMSIS_ZIP=5782d6f8057906d360f4b95ec08a2354afe5c9b9
DFP_ZIP=c2c70a97a20fb355815e2ead3d4a40e35a4a3cdf

curl -fsSL -o "$TMP/cmsis.zip" \
  "https://github.com/ARM-software/CMSIS_6/archive/${CMSIS_ZIP}.zip"
unzip -qo "$TMP/cmsis.zip" -d "$TMP"
rm -rf "$CMSIS"
mv "$TMP/CMSIS_6-${CMSIS_ZIP}" "$CMSIS"

curl -fsSL -o "$TMP/dfp.zip" \
  "https://github.com/ARM-software/Cortex_DFP/archive/${DFP_ZIP}.zip"
unzip -qo "$TMP/dfp.zip" -d "$TMP"
rm -rf "$CMSIS/Cortex_DFP"
mv "$TMP/Cortex_DFP-${DFP_ZIP}" "$CMSIS/Cortex_DFP"

echo "CMSIS installed under $CMSIS"
