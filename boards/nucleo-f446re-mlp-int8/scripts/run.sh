#!/usr/bin/env bash
# Build, flash, and open the UART monitor for the NUCLEO-F446RE MNIST CNN int8 benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

make "$@"
./scripts/flash.sh
./scripts/monitor.sh
