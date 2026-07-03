#!/usr/bin/env bash
# Fetch CMSIS-Core headers (Apache-2.0) for MCU CMSIS-NN/DSP cross builds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/CMSIS-Core/Include"
URL="https://github.com/ARM-software/CMSIS_5.git"
PIN="main"

if [[ -f "$DEST/cmsis_compiler.h" ]]; then
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 --filter=blob:none --sparse "$URL" "$TMP/cmsis5"
git -C "$TMP/cmsis5" sparse-checkout set CMSIS/Core/Include
mkdir -p "$(dirname "$DEST")"
cp -R "$TMP/cmsis5/CMSIS/Core/Include" "$(dirname "$DEST")/"
echo "CMSIS-Core ready at third_party/CMSIS-Core/Include"
