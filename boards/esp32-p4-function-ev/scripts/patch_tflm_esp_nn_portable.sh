#!/usr/bin/env bash
# After fetching esp-tflite-micro (+ managed esp-nn), apply P4 peer policy:
#   1) portable ESP-NN opt on ESP32-P4 (PIO gas rejects some PIE immediates)
#   2) gate -DESP_NN via -DNETKIT_TFLM_USE_ESP_NN=OFF for reference A/B
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

BOARD_ROOT="${1:-$ROOT/boards/esp32-p4-function-ev-tflm-cnn-int8}"
if [[ -f "$BOARD_ROOT/include/esp_nn.h" ]]; then
  ESP_NN_COMP="$BOARD_ROOT"
  TFLM_CMAKE=""
else
  ESP_NN_COMP="$BOARD_ROOT/managed_components/espressif__esp-nn"
  TFLM_CMAKE="$BOARD_ROOT/components/esp-tflite-micro/CMakeLists.txt"
fi

HDR="$ESP_NN_COMP/include/esp_nn.h"
CMAKE="$ESP_NN_COMP/CMakeLists.txt"

if [[ -f "$HDR" && -f "$CMAKE" ]]; then
  if ! grep -q 'NETKIT_ESP_NN_USE_P4_ASM' "$HDR"; then
    perl -i -0pe 's/#ifdef CONFIG_IDF_TARGET_ESP32P4\n#define ARCH_ESP32_P4 1\n#endif/#ifdef CONFIG_IDF_TARGET_ESP32P4\n#if defined(NETKIT_ESP_NN_USE_P4_ASM) \&\& NETKIT_ESP_NN_USE_P4_ASM\n#define ARCH_ESP32_P4 1\n#endif\n#endif/s' "$HDR"
  fi
  # Force-empty p4_srcs on P4 (do not leave gated list — cmake/ninja can keep stale SRCS).
  if grep -q 'set(p4_srcs' "$CMAKE"; then
    python3 - <<'PY' "$CMAKE"
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
t = p.read_text()
# Replace any ESP32P4 p4_srcs block with an empty set.
pat = re.compile(
    r'if\(CONFIG_IDF_TARGET_ESP32P4(?:[^\n]*)\)\n'
    r'    set\(p4_srcs\n(?:        "[^"]+"\n)+'
    r'        "[^"]+"\)\n'
    r'endif\(\)',
    re.M,
)
repl = (
    'if(CONFIG_IDF_TARGET_ESP32P4)\n'
    '    # NETKIT: PlatformIO riscv32-esp gas rejects some PIE immediates;\n'
    '    # use portable opt kernels (see esp_nn.h ARCH_ESP32_P4 gate).\n'
    '    set(p4_srcs)\n'
    'endif()'
)
nt, n = pat.subn(repl, t, count=1)
if n == 0 and 'set(p4_srcs)' not in t:
    # Already gated with NETKIT_ESP_NN_USE_P4_ASM — still force empty.
    pat2 = re.compile(
        r'if\(CONFIG_IDF_TARGET_ESP32P4 AND DEFINED NETKIT_ESP_NN_USE_P4_ASM AND NETKIT_ESP_NN_USE_P4_ASM\)\n'
        r'    set\(p4_srcs\n(?:        "[^"]+"\n)+'
        r'        "[^"]+"\)\n'
        r'endif\(\)',
        re.M,
    )
    nt, n = pat2.subn(repl, t, count=1)
if n == 0 and 'NETKIT: PlatformIO' not in t:
    sys.exit(f'failed to rewrite p4_srcs in {p}')
p.write_text(nt)
print(f'rewrote p4_srcs → empty in {p}')
PY
  fi
  echo "Patched $ESP_NN_COMP for portable ESP-NN on ESP32-P4"
elif [[ -n "${1:-}" ]]; then
  echo "esp-nn component not found at $ESP_NN_COMP (build once to fetch)" >&2
fi

if [[ -n "${TFLM_CMAKE:-}" && -f "$TFLM_CMAKE" ]]; then
  if grep -q 'NETKIT_TFLM_USE_ESP_NN' "$TFLM_CMAKE"; then
    # Repair a prior broken perl patch that dropped ${COMPONENT_LIB}.
    if grep -q 'target_compile_options( PRIVATE -DESP_NN)' "$TFLM_CMAKE"; then
      python3 - <<'PY' "$TFLM_CMAKE"
from pathlib import Path
import sys
p = Path(sys.argv[1])
t = p.read_text().replace(
    'target_compile_options( PRIVATE -DESP_NN)',
    'target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)',
)
p.write_text(t)
print(f'repaired COMPONENT_LIB in {p}')
PY
    else
      echo "TFLM CMakeLists already gated: $TFLM_CMAKE"
    fi
  elif grep -q 'PRIVATE -DESP_NN)' "$TFLM_CMAKE"; then
    python3 - <<'PY' "$TFLM_CMAKE"
from pathlib import Path
import sys
p = Path(sys.argv[1])
old = (
    "# enable ESP-NN optimizations by Espressif\n"
    "target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)"
)
new = (
    "# ESP-NN on by default. Peer A/B reference: -DNETKIT_TFLM_USE_ESP_NN=OFF\n"
    "if(NOT DEFINED NETKIT_TFLM_USE_ESP_NN)\n"
    "  set(NETKIT_TFLM_USE_ESP_NN ON)\n"
    "endif()\n"
    "if(NETKIT_TFLM_USE_ESP_NN)\n"
    "  target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)\n"
    "endif()"
)
t = p.read_text()
if old not in t:
    raise SystemExit(f'ESP_NN block not found in {p}')
p.write_text(t.replace(old, new, 1))
print(f'Patched {p} for NETKIT_TFLM_USE_ESP_NN gate')
PY
  else
    echo "WARN: unexpected esp-tflite-micro CMakeLists (no -DESP_NN line): $TFLM_CMAKE" >&2
  fi
fi
