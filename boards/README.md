# Board firmware and peer setups

Hardware bring-up trees for **on-device** netkit builds and peer A/B benches.
Software targets without a board tree yet use the library API +
[docs/PLATFORMS.md](../docs/PLATFORMS.md) and host smoke
(`make test-embedded-smoke-matrix`).

**Espressif note:** C3 / C6 / P4 are RISC-V silicon but always use **`mcu_esp` +
ESP-NN** (same as S3). `mcu_risc` is only for non-Espressif RISC-V + NMSIS-NN.
Several Espressif boards can coexist — each `boards/…` tree pins its own
`NETKIT_ARCH`. Details: [PLATFORMS.md — Target ≠ CPU ISA](../docs/PLATFORMS.md#target--cpu-isa).

| Hardware | Class | netkit target | Status |
|----------|-------|---------------|--------|
| [STM32 NUCLEO-F446RE](#stm32-nucleo-f446re) | Arm MCU | `mcu_arm` + `NETKIT_ARCH=CM4` | Peer-benched (CMSIS-NN / reference vs TFLM / microTVM) |
| [Seeed XIAO ESP32C3](#seeed-studio-xiao-esp32c3) | Espressif MCU (RISC-V → still `mcu_esp`) | `mcu_esp` + `NETKIT_ARCH=ESP32C3` | Peer-benched int8 CNN / DS-CNN vs TFLM (ESP-NN); MLP firmware too |
| [Raspberry Pi Zero 2 W](#raspberry-pi-zero-2-w) | Arm MPU | `mpu_arm` | Peer-benched (XNNPACK ON/OFF vs TF Lite) |
| Other Espressif ESP32* (S3 / P4 / …) | MCU | `mcu_esp` + matching `NETKIT_ARCH` | Runtime + host smoke; add a `boards/…` tree per board — [PLATFORMS.md](../docs/PLATFORMS.md#mcu_esp--espressif-mcu) |
| RISC-V MCU (Nuclei / RV32) | MCU | `mcu_risc` | Runtime + host smoke; **no `boards/` tree yet** — [PLATFORMS.md](../docs/PLATFORMS.md#mcu_risc--risc-v-mcu) |

Canonical latency tables: [../README.md](../README.md#peer-benchmarks-mcu--mpu--cpu), [docs/STATUS.md](../docs/STATUS.md).

---

## STM32 NUCLEO-F446RE

**Chip:** STM32F446RET6 · Cortex-M4F · 180 MHz · 512 KiB flash / 128 KiB SRAM  
**Toolchain / flash:** each board README covers `make`, OpenOCD / ST-Link, UART capture.

### netkit firmware (production peers)

| Board directory | Model / dtype | Setup & build |
|-----------------|---------------|---------------|
| [`nucleo-f446re/`](nucleo-f446re/README.md) | MNIST MLP **float32** | CMSIS-NN / reference, lowered AOT |
| [`nucleo-f446re-mlp-int8/`](nucleo-f446re-mlp-int8/README.md) | MNIST MLP **int8** | CMSIS-NN, interpreter embed |
| [`nucleo-f446re-cnn-int8/`](nucleo-f446re-cnn-int8/README.md) | MNIST CNN **int8** | CMSIS-NN / reference; lowered or `NETKIT_EMBED=1` |
| [`nucleo-f446re-cnn-dw-int8/`](nucleo-f446re-cnn-dw-int8/README.md) | MNIST DS-CNN **int8** | Same pattern as CNN int8 |

Typical flow (see the board README for flags):

```bash
cd boards/nucleo-f446re-cnn-int8
make            # or: make deploy-lowered / NETKIT_EMBED=1
make flash      # ST-Link
# UART capture — details in that README
```

Repo-root helper for CNN int8: `make flash-mnist-cnn-int8`.

### Peer baselines (same board / vectors)

| Board directory | Stack |
|-----------------|-------|
| [`nucleo-f446re-tflm/`](nucleo-f446re-tflm/README.md) | TFLM MNIST MLP f32 |
| [`nucleo-f446re-tflm-mlp-int8/`](nucleo-f446re-tflm-mlp-int8/README.md) | TFLM MNIST MLP int8 |
| [`nucleo-f446re-tflm-cnn-int8/`](nucleo-f446re-tflm-cnn-int8/README.md) | TFLM MNIST CNN int8 |
| [`nucleo-f446re-tflm-cnn-dw-int8/`](nucleo-f446re-tflm-cnn-dw-int8/README.md) | TFLM MNIST DS-CNN int8 |
| [`nucleo-f446re-tvm-cnn-int8/`](nucleo-f446re-tvm-cnn-int8/README.md) | microTVM AOT CNN int8 |
| [`nucleo-f446re-tvm-cnn-dw-int8/`](nucleo-f446re-tvm-cnn-dw-int8/README.md) | microTVM AOT DS-CNN int8 |

Logs: [`benchmark/mcu_ab_logs/`](../benchmark/mcu_ab_logs/).

Float32 MNIST CNN / DS-CNN exceed 512 KiB flash on this part — on-device digit peers for those models are **int8 only** ([STATUS.md](../docs/STATUS.md)).

---

## Seeed Studio XIAO ESP32C3

**Chip:** ESP32-C3 · RISC-V · 160 MHz · ~400 KB SRAM / 4 MB flash  
**netkit:** `NETKIT_TARGET=mcu_esp` + `NETKIT_ARCH=ESP32C3` + **ESP-NN** (not NMSIS-NN)  
**Toolchain / flash:** PlatformIO ESP-IDF · onboard USB Serial/JTAG  
**Index:** [`xiao-esp32c3/README.md`](xiao-esp32c3/README.md)

| Board directory | Runtime | Model |
|-----------------|---------|-------|
| [`xiao-esp32c3-mlp-int8/`](xiao-esp32c3-mlp-int8/README.md) | netkit | MNIST MLP int8 |
| [`xiao-esp32c3-cnn-int8/`](xiao-esp32c3-cnn-int8/README.md) | netkit | MNIST CNN int8 |
| [`xiao-esp32c3-cnn-dw-int8/`](xiao-esp32c3-cnn-dw-int8/README.md) | netkit | MNIST DS-CNN int8 |
| [`xiao-esp32c3-tflm-cnn-int8/`](xiao-esp32c3-tflm-cnn-int8/README.md) | TFLM | MNIST CNN int8 |
| [`xiao-esp32c3-tflm-cnn-dw-int8/`](xiao-esp32c3-tflm-cnn-dw-int8/README.md) | TFLM | MNIST DS-CNN int8 |

Methodology: **10×10**, discard first invoke; order swaps `nk→tflm` / `tflm→nk`.  
Results: [`benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ab_results.txt).  
Runner: [`xiao-esp32c3/scripts/run_esp_int8_ab.sh`](xiao-esp32c3/scripts/run_esp_int8_ab.sh).

**ImageNet / MobileNetV4:** not on this part (weights exceed 1 MiB app partition).

```bash
PORT=/dev/cu.usbmodem* ./boards/xiao-esp32c3/scripts/run_esp_int8_ab.sh
```

---

## Raspberry Pi Zero 2 W

**ISA:** `linux/aarch64` (Cortex-A53) · **netkit:** `NETKIT_TARGET=mpu_arm` + XNNPACK  

No bare-metal firmware tree — cross-build on the host (Docker `linux/arm64`), deploy over SSH.

**Setup / rebuild / run A/B:** [`pi-zero-2w/README.md`](pi-zero-2w/README.md)

Covers prerequisites, `tools/build_mpu_pi_aarch64.sh`, SSH helpers, and float32 + int8 suites vs TF Lite (XNNPACK ON/OFF).

---

## Targets without a board tree yet

| Target | How to build / integrate |
|--------|---------------------------|
| `mcu_esp` (ESP-NN) | Board: [xiao-esp32c3-mlp-int8](xiao-esp32c3-mlp-int8/README.md) · [PLATFORMS.md — Espressif](../docs/PLATFORMS.md#mcu_esp--espressif-mcu) · `make esp-nn-init` |
| `mcu_risc` (NMSIS-NN) | [PLATFORMS.md — RISC-V MCU](../docs/PLATFORMS.md#mcu_risc--risc-v-mcu) · `make nmsis-init` · link into your RISC-V BSP |
| `mpu_risc` | [PLATFORMS.md — RISC-V MPU](../docs/PLATFORMS.md#mpu_risc--risc-v-mpu) · XNNPACK like other MPUs |
| Host `cpu` | [Getting Started](../docs/GETTING_STARTED.md) |

Host lean-runtime smoke for MCU/MPU profiles (including CMSIS / ESP-NN / NMSIS-NN):

```bash
make cmsis-init esp-nn-init nmsis-init
make test-embedded-smoke-matrix
```

---

## Related

- [docs/PLATFORMS.md](../docs/PLATFORMS.md) — `NETKIT_TARGET` / `NETKIT_ARCH` / fetch / CMake for every profile  
- [docs/BUILD_TARGETS.md](../docs/BUILD_TARGETS.md) — arena and backend flags  
- [docs/ARENA.md](../docs/ARENA.md) — MCU static arena / no-heap rules  
- [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) — CMSIS / TFLM / TVM / … licenses  
