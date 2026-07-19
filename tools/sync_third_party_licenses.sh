#!/usr/bin/env bash
# Refresh vendored license texts under third_party/licenses/ from fetched trees.
# Safe to run when some trees are missing (skips those). See THIRD_PARTY_NOTICES.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIC="$ROOT/third_party/licenses"
mkdir -p "$LIC"

copy_if() {
  local src="$1" dest="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dest"
    echo "synced $(basename "$dest")"
  fi
}

# Runtime backends (linked into libnetkit when enabled)
copy_if "$ROOT/third_party/CMSIS-Core/LICENSE" "$LIC/CMSIS-Core.Apache-2.0.txt"
copy_if "$ROOT/third_party/CMSIS-NN/LICENSE" "$LIC/CMSIS-NN.Apache-2.0.txt"
copy_if "$ROOT/third_party/ESP-NN/LICENSE" "$LIC/ESP-NN.Apache-2.0.txt"
copy_if "$ROOT/third_party/NMSIS/LICENSE" "$LIC/NMSIS.Apache-2.0.txt"
copy_if "$ROOT/third_party/XNNPACK/LICENSE" "$LIC/XNNPACK.BSD-3-Clause.txt"

# XNNPACK transitive deps (present after make xnnpack-init / CMake fetch)
for build in "$ROOT/third_party/XNNPACK/build" "$ROOT/third_party/XNNPACK/build_linux_aarch64"; do
  [[ -d "$build" ]] || continue
  copy_if "$build/pthreadpool-source/LICENSE" "$LIC/pthreadpool.BSD.txt"
  copy_if "$build/cpuinfo-source/LICENSE" "$LIC/cpuinfo.BSD.txt"
  copy_if "$build/FXdiv-source/LICENSE" "$LIC/FXdiv.MIT.txt"
  if [[ -f "$build/cpuinfo-source/deps/clog/LICENSE" ]]; then
    copy_if "$build/cpuinfo-source/deps/clog/LICENSE" "$LIC/clog.BSD.txt"
  fi
  copy_if "$build/kleidiai-source/LICENSES/Apache-2.0.txt" "$LIC/KleidiAI.Apache-2.0.txt"
  copy_if "$build/kleidiai-source/LICENSES/BSD-3-Clause.txt" "$LIC/KleidiAI.BSD-3-Clause.txt"
  break
done

# Peer: TFLM (optional local fetch)
copy_if "$ROOT/benchmark/tflm/third_party/tflite-micro/LICENSE" \
  "$LIC/tflite-micro.Apache-2.0.txt"

echo "License sync done → $LIC"
echo "Peer/Python texts (numpy, LiteRT, torch, …) are maintained separately; see THIRD_PARTY_NOTICES.md."
