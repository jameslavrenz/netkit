#!/usr/bin/env bash
# Fetch and build Google XNNPACK (BSD-3) for optional NETKIT_XNNPACK builds.
# Produces a static library under third_party/XNNPACK/build/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/XNNPACK"
URL="https://github.com/google/XNNPACK.git"
# Pin to the XNNPACK commit embedded by ai_edge_litert (LiteRT) peer benches.
# LiteRT v2.1.6 → TensorFlow b8a17154 → tflite/.../xnnpack.cmake GIT_TAG below
# ("Sync with tensorflow/workspace2.bzl"). Bump together with the LiteRT wheel.
PIN="${NETKIT_XNNPACK_PIN:-c2e81f01b01fca3327d4b3aa070b56085f2603bd}"
BUILD_DIR="$DEST/build"
JOBS="${NETKIT_XNNPACK_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

if [[ ! -d "$DEST/.git" ]]; then
  if [[ -e "$DEST" ]]; then
    echo "Removing incomplete $DEST" >&2
    rm -rf "$DEST"
  fi
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --depth 1 origin "$PIN" 2>/dev/null || git -C "$DEST" fetch origin "$PIN" || git -C "$DEST" fetch origin
git -C "$DEST" checkout --detach "$PIN"

# Ensure nested deps (cpuinfo, pthreadpool, …) are present.
if [[ -f "$DEST/.gitmodules" ]]; then
  git -C "$DEST" submodule update --init --depth 1
fi

# Clean rebuild when the pin changes (stale CMake cache / objects otherwise linger).
STAMP="$DEST/netkit_lib/.xnnpack_pin"
PREV="$(cat "$STAMP" 2>/dev/null || true)"
if [[ -d "$BUILD_DIR" ]] && [[ "$PREV" != "$PIN" ]]; then
  echo "XNNPACK pin changed (${PREV:-unknown} -> $PIN); wiping $BUILD_DIR" >&2
  rm -rf "$BUILD_DIR" "$DEST/netkit_lib" "$DEST/netkit_include"
fi

# Match TF Lite / LiteRT CMake defaults for this pin: tests/benchmarks off,
# all microkernels + assembly on (XNNPACK CMake defaults; TF does not override).
# Same host drivers / opt as benchmark/common/tflite_host_flags.mk (gcc/g++).
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
cmake -S "$DEST" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="-O3 -DNDEBUG" \
  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
  -DXNNPACK_LIBRARY_TYPE=static \
  -DXNNPACK_BUILD_TESTS=OFF \
  -DXNNPACK_BUILD_BENCHMARKS=OFF \
  -DXNNPACK_BUILD_ALL_MICROKERNELS=ON \
  -DXNNPACK_ENABLE_ASSEMBLY=ON

cmake --build "$BUILD_DIR" -j"$JOBS"

# Locate the built static library (CMake may place it under build/ or build/src/).
LIB=""
for cand in \
  "$BUILD_DIR/libXNNPACK.a" \
  "$BUILD_DIR/libxnnpack.a" \
  "$BUILD_DIR/src/libXNNPACK.a"
do
  if [[ -f "$cand" ]]; then
    LIB="$cand"
    break
  fi
done
if [[ -z "$LIB" ]]; then
  LIB="$(find "$BUILD_DIR" -name 'libXNNPACK.a' -o -name 'libxnnpack.a' | head -1 || true)"
fi
if [[ -z "$LIB" || ! -f "$LIB" ]]; then
  echo "XNNPACK build finished but libXNNPACK.a was not found under $BUILD_DIR" >&2
  exit 1
fi

# Stamp a stable path for Make/CMake consumers.
mkdir -p "$DEST/netkit_lib" "$DEST/netkit_include"
cp -f "$LIB" "$DEST/netkit_lib/libXNNPACK.a"

# Collect dependency static libs XNNPACK needs at link time.
for dep in pthreadpool cpuinfo fxdiv kleidiai; do
  found="$(find "$BUILD_DIR" -name "lib${dep}.a" 2>/dev/null | head -1 || true)"
  if [[ -n "$found" ]]; then
    cp -f "$found" "$DEST/netkit_lib/"
  fi
done
for mk in xnnpack-microkernels-prod xnnpack-microkernels-all; do
  found="$(find "$BUILD_DIR" -name "lib${mk}.a" 2>/dev/null | head -1 || true)"
  if [[ -n "$found" ]]; then
    cp -f "$found" "$DEST/netkit_lib/"
  fi
done

# Dependency headers (xnnpack.h includes <pthreadpool.h>).
for hdr_name in pthreadpool.h cpuinfo.h; do
  found="$(find "$BUILD_DIR" -name "$hdr_name" 2>/dev/null | head -1 || true)"
  if [[ -n "$found" ]]; then
    cp -f "$found" "$DEST/netkit_include/"
  fi
done

mkdir -p "$DEST/netkit_lib"
echo "$PIN" > "$DEST/netkit_lib/.xnnpack_pin"

"$ROOT/tools/sync_third_party_licenses.sh" || true
echo "XNNPACK ready at $PIN (LiteRT peer pin)"
echo "  headers: $DEST/include + $DEST/netkit_include"
echo "  libs:    $DEST/netkit_lib/"
ls -la "$DEST/netkit_lib/"
ls -la "$DEST/netkit_include/"
