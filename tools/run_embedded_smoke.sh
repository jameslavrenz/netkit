#!/usr/bin/env bash
# Build and run tests/embedded_smoke for MCU/MPU + NETKIT_ARCH + CMSIS profiles.
#
# Host execution validates linking and inference paths before on-device bring-up.
# Requires CMSIS trees (make cmsis-init) when NETKIT_CMSIS_* profiles are used.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v make >/dev/null 2>&1; then
  echo "make is required" >&2
  exit 1
fi

if [[ ! -f models/test_mlp.nk || ! -f models/cnn_4x4_single.nk ]]; then
  echo "missing bundled models under models/ — run from repo root" >&2
  exit 1
fi

run_profile() {
  local name="$1"
  shift

  echo ""
  echo "============================================================"
  echo " embedded smoke profile: ${name}"
  echo " make $* lib embedded-smoke"
  echo "============================================================"

  make clean >/dev/null
  make NETKIT_HOST_SMOKE=1 "$@" lib embedded-smoke
  ./tests/embedded_smoke
}

# Reference kernels only (no CMSIS fetch required).
run_profile "mcu" NETKIT_TARGET=mcu
run_profile "mpu" NETKIT_TARGET=mpu

if [[ ! -f third_party/CMSIS-NN/Include/arm_nnfunctions.h ]]; then
  echo "CMSIS-NN not found — run: make cmsis-init" >&2
  exit 1
fi
if [[ ! -f third_party/CMSIS-DSP/Include/arm_math.h ]]; then
  echo "CMSIS-DSP not found — run: make cmsis-init" >&2
  exit 1
fi

run_profile "mcu+cmsis" NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1
run_profile "mcu+cm4+cmsis" NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1
run_profile "mcu+m33+cmsis" NETKIT_TARGET=mcu NETKIT_ARCH=M33 NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1
run_profile "mpu+cmsis" NETKIT_TARGET=mpu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1
run_profile "mpu+a32+cmsis" NETKIT_TARGET=mpu NETKIT_ARCH=A32 NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1

echo ""
echo "All embedded smoke profiles passed."
