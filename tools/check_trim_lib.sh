#!/usr/bin/env bash
# Verify libnetkit_trim.a omits unused layer-op bodies (linker DCE via per-op TUs).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f libnetkit.a || ! -f libnetkit_trim.a ]]; then
  echo "Run: make lib trim-lib" >&2
  exit 1
fi

defined_symbols() {
  nm "$1" 2>/dev/null | c++filt 2>/dev/null | grep ' T ' || true
}

if defined_symbols libnetkit_trim.a | grep -q 'NkEvalAvgPool2D'; then
  echo "FAIL: trim library still defines NkEvalAvgPool2D" >&2
  exit 1
fi

if defined_symbols libnetkit_trim.a | grep -q 'NkEvalBatchNorm2d'; then
  echo "FAIL: trim library still defines NkEvalBatchNorm2d" >&2
  exit 1
fi

if ! defined_symbols libnetkit.a | grep -q 'NkEvalAvgPool2D'; then
  echo "FAIL: full library missing NkEvalAvgPool2D definition" >&2
  exit 1
fi

if ! defined_symbols libnetkit_trim.a | grep -q 'NkEvalConv2D'; then
  echo "FAIL: trim library missing NkEvalConv2D definition" >&2
  exit 1
fi

full_size=$(stat -f%z libnetkit.a)
trim_size=$(stat -f%z libnetkit_trim.a)
echo "libnetkit.a:      ${full_size} bytes"
echo "libnetkit_trim.a: ${trim_size} bytes"

if [[ "$trim_size" -ge "$full_size" ]]; then
  echo "FAIL: trim library should be smaller than full library" >&2
  exit 1
fi

echo "PASS trim library omits avg-pool and batch-norm op bodies"
