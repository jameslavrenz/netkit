#!/usr/bin/env bash
# Build and run tests/embedded_smoke for MCU/MPU ISA profiles + CMSIS-NN / ESP-NN.
#
# Host execution validates linking and inference paths before on-device bring-up.
# Requires CMSIS-NN/Core (make cmsis-init) when CMSIS-NN profiles are used.
# Requires ESP-NN (make esp-nn-init) when ESP-NN profiles are used.
# CMSIS-DSP is not used as a netkit backend.
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

# Reference kernels only (no CMSIS / ESP-NN fetch required).
run_profile "mcu_arm" NETKIT_TARGET=mcu_arm NETKIT_CMSIS_NN=0
run_profile "mpu_arm" NETKIT_TARGET=mpu_arm NETKIT_CMSIS_NN=0 NETKIT_XNNPACK=0
run_profile "mcu_risc" NETKIT_TARGET=mcu_risc
run_profile "mpu_risc" NETKIT_TARGET=mpu_risc

if [[ ! -f third_party/CMSIS-NN/Include/arm_nnfunctions.h ]]; then
  echo "CMSIS-NN not found — run: make cmsis-init" >&2
  exit 1
fi

run_profile "mcu_arm+cm4+cmsis-nn" NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1
run_profile "mcu_arm+m33+cmsis-nn" NETKIT_TARGET=mcu_arm NETKIT_ARCH=M33 NETKIT_CMSIS_NN=1

if [[ ! -f third_party/ESP-NN/include/esp_nn.h ]]; then
  echo "ESP-NN not found — run: make esp-nn-init" >&2
  exit 1
fi

run_profile "mcu_esp+esp32c6+esp-nn" NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C6 NETKIT_ESP_NN=1
run_profile "mcu_esp+esp32s3+esp-nn" NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 NETKIT_ESP_NN=1

echo ""
echo "All embedded smoke profiles passed."
