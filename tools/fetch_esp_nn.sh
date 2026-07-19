#!/usr/bin/env bash
# Fetch Espressif ESP-NN (Apache-2.0) for optional NETKIT_ESP_NN builds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/ESP-NN"
URL="https://github.com/espressif/esp-nn.git"
# Pin master tip used when integrating; bump intentionally when updating.
PIN="${NETKIT_ESP_NN_PIN:-d45b843ca5f873ca9d0706ab0e7f14eafd132e98}"

if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth 1 origin "$PIN" 2>/dev/null || git -C "$DEST" fetch origin
  git -C "$DEST" checkout --detach "$PIN"
  "$ROOT/tools/sync_third_party_licenses.sh" || true
  echo "ESP-NN ready at $PIN"
  exit 0
fi

if [[ -e "$DEST" ]]; then
  echo "Removing incomplete $DEST" >&2
  rm -rf "$DEST"
fi

git clone --depth 1 "$URL" "$DEST"
git -C "$DEST" fetch --depth 1 origin "$PIN"
git -C "$DEST" checkout --detach "$PIN"

"$ROOT/tools/sync_third_party_licenses.sh" || true
echo "ESP-NN ready at $PIN"
