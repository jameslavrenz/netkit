#!/usr/bin/env bash
# Install or locate an arm-none-eabi GCC toolchain with newlib (required for C/C++ firmware).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$ROOT/.toolchain}"
MARKER="$TOOLCHAIN_ROOT/.ready"

toolchain_bin() {
  if [[ -n "${ARM_GNU_TOOLCHAIN:-}" && -x "$ARM_GNU_TOOLCHAIN/bin/arm-none-eabi-gcc" ]]; then
    echo "$ARM_GNU_TOOLCHAIN/bin"
    return 0
  fi
  local candidate
  for candidate in \
    "$TOOLCHAIN_ROOT"/*/bin \
    /Applications/ArmGNUToolchain/*/arm-none-eabi/bin \
    /opt/homebrew/opt/gcc-arm-embedded/bin; do
    if [[ -x "$candidate/arm-none-eabi-gcc" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    dirname "$(command -v arm-none-eabi-gcc)"
    return 0
  fi
  return 1
}

verify_toolchain() {
  local bin="$1"
  local cc="$bin/arm-none-eabi-gcc"
  local cxx="$bin/arm-none-eabi-g++"
  if [[ ! -x "$cc" || ! -x "$cxx" ]]; then
    return 1
  fi
  if "$cc" --version 2>&1 | grep -Eiq 'clang|llvm'; then
    return 1
  fi
  if "$cxx" --version 2>&1 | grep -Eiq 'clang|llvm'; then
    return 1
  fi
  if ! "$cc" --version 2>&1 | grep -qi 'gcc'; then
    return 1
  fi
  "$cc" -xc -c - -o /dev/null <<'EOF' >/dev/null 2>&1
#include <stdint.h>
int x;
EOF
}

if bin="$(toolchain_bin)" && verify_toolchain "$bin"; then
  echo "$bin"
  exit 0
fi

if [[ -f "$MARKER" ]]; then
  rm -f "$MARKER"
fi

mkdir -p "$TOOLCHAIN_ROOT"
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS-$ARCH" in
  Darwin-arm64)
    XPACK_URL="https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v14.2.1-1.1/xpack-arm-none-eabi-gcc-14.2.1-1.1-darwin-arm64.tar.gz"
    ;;
  Darwin-x86_64)
    XPACK_URL="https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v14.2.1-1.1/xpack-arm-none-eabi-gcc-14.2.1-1.1-darwin-x64.tar.gz"
    ;;
  Linux-x86_64)
    XPACK_URL="https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v14.2.1-1.1/xpack-arm-none-eabi-gcc-14.2.1-1.1-linux-x64.tar.gz"
    ;;
  Linux-aarch64)
    XPACK_URL="https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v14.2.1-1.1/xpack-arm-none-eabi-gcc-14.2.1-1.1-linux-arm64.tar.gz"
    ;;
  *)
    echo "Unsupported host $OS-$ARCH — install Arm GNU toolchain manually and set ARM_GNU_TOOLCHAIN" >&2
    exit 1
    ;;
esac

ARCHIVE="$TOOLCHAIN_ROOT/xpack-arm-none-eabi-gcc.tar.gz"
echo "Downloading xPack arm-none-eabi-gcc..."
curl -L "$XPACK_URL" -o "$ARCHIVE"
tar -xf "$ARCHIVE" -C "$TOOLCHAIN_ROOT"
rm -f "$ARCHIVE"
touch "$MARKER"

bin="$(toolchain_bin)"
if ! verify_toolchain "$bin"; then
  echo "Toolchain install failed verification" >&2
  exit 1
fi
echo "$bin"
