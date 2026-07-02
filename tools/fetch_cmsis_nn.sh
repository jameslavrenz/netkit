#!/usr/bin/env bash
# Fetch ARM CMSIS-NN (Apache-2.0) for optional NETKIT_CMSIS_NN builds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/CMSIS-NN"
URL="https://github.com/ARM-software/CMSIS-NN.git"
# Pinned to a commit with experimental float32 kernels (main, 2026-04).
PIN="dbf45dbfcc515421dd6099037d3e2637b90748c8"

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

echo "CMSIS-NN ready at $PIN"
