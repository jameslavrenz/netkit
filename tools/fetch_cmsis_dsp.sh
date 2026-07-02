#!/usr/bin/env bash
# Fetch ARM CMSIS-DSP (Apache-2.0) for optional NETKIT_CMSIS_DSP builds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/CMSIS-DSP"
URL="https://github.com/ARM-software/CMSIS-DSP.git"
PIN="4fb9ef734dacb2d677aa9910f9a0b5d067771f3d"

if [[ -d "$DEST/.git" ]]; then
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

echo "CMSIS-DSP ready at $PIN"
