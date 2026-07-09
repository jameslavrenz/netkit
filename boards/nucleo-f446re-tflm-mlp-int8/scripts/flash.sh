#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
../nucleo-f446re/scripts/flash.sh "$ROOT/build/mnist_mlp_int8_tflm_nucleo_f446re.elf"
