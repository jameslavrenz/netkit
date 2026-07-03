#!/usr/bin/env bash
# Fetch ARM CMSIS-Core headers (Apache-2.0) for MCU CMSIS-NN/DSP firmware builds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/CMSIS-Core"
HEADER="$DEST/CMSIS/Core/Include/cmsis_compiler.h"
URL="https://github.com/ARM-software/CMSIS_6.git"
# CMSIS 6.3.0 — Core(M) headers under CMSIS/Core/Include.
PIN="45dab712ad84f8cbbf2b7bfc089c19088507df6f"

if [[ -f "$HEADER" ]]; then
  if [[ -e "$DEST/.git" ]]; then
    git -C "$DEST" fetch --depth 1 origin "$PIN" 2>/dev/null || git -C "$DEST" fetch origin
    git -C "$DEST" checkout --detach "$PIN"
  fi
  exit 0
fi

if [[ -e "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth 1 origin "$PIN" 2>/dev/null || git -C "$DEST" fetch origin
  git -C "$DEST" checkout --detach "$PIN"
  exit 0
fi

if [[ -e "$DEST" ]]; then
  echo "Removing incomplete $DEST" >&2
  rm -rf "$DEST"
fi

git clone --depth 1 "$URL" "$DEST"
git -C "$DEST" fetch --depth 1 origin "$PIN"
git -C "$DEST" checkout --detach "$PIN"

echo "CMSIS-Core ready at $PIN (CMSIS/Core/Include)"
