#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ELF="${1:-$ROOT/build/mnist_cnn_int8_tvm_nucleo_f446re.elf}"
exec "$ROOT/../nucleo-f446re/scripts/flash.sh" "$ELF"
