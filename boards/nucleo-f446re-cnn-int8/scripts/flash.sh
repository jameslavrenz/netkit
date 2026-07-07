#!/usr/bin/env bash
# Flash NUCLEO-F446RE MNIST CNN int8 firmware via onboard ST-Link (USB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$(cd "$(dirname "$0")/../../nucleo-f446re/scripts" && pwd)/flash.sh" \
  "${1:-$ROOT/build/mnist_cnn_int8_nucleo_f446re.elf}"
