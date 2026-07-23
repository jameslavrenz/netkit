#!/usr/bin/env bash
set -euo pipefail
BOARD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BOARD_ROOT"
make all
./scripts/flash.sh
./scripts/monitor.sh
