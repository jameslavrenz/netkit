#!/usr/bin/env bash
# Fetch Nuclei NMSIS (Apache-2.0) for optional NETKIT_NMSIS_NN builds (NN + Core headers).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/NMSIS"
URL="https://github.com/Nuclei-Software/NMSIS.git"
# NMSIS 1.6.0 — bump intentionally when updating.
PIN="${NETKIT_NMSIS_PIN:-dc728b988b6562868d311a2e14b5d9f8ebbd26b0}"

if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth 1 origin "$PIN" 2>/dev/null || git -C "$DEST" fetch origin
  git -C "$DEST" checkout --detach "$PIN"
  "$ROOT/tools/sync_third_party_licenses.sh" || true
  echo "NMSIS ready at $PIN"
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
echo "NMSIS ready at $PIN"
