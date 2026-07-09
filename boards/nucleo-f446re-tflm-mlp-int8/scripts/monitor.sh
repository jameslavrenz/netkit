#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")/.." && pwd)/../nucleo-f446re/scripts/monitor.sh" "$@"
