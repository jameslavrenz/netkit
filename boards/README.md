# Board firmware and peer setups

Hardware bring-up trees for **on-device** netkit builds and peer A/B benches.
Software targets without a board tree yet (Espressif, RISC-V MCU) use the library
API + [docs/PLATFORMS.md](../docs/PLATFORMS.md) and host smoke
(`make test-embedded-smoke-matrix`).

| Hardware | Class | netkit target | Status |
|----------|-------|---------------|--------|
| [STM32 NUCLEO-F446RE](#stm32-nucleo-f446re) | Arm MCU | `mcu_arm` + `NETKIT_ARCH=CM4` | Peer-benched (CMSIS-NN / reference vs TFLM / microTVM) |
| [Raspberry Pi Zero 2 W](#raspberry-pi-zero-2-w) | Arm MPU | `mpu_arm` | Peer-benched (XNNPACK ON/OFF vs TF Lite) |
| Espressif ESP32 family | MCU | `mcu_esp` | Runtime + host smoke; **no `boards/` tree yet** — [PLATFORMS.md](../docs/PLATFORMS.md#mcu_esp--espressif-mcu) |
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

## Raspberry Pi Zero 2 W

**ISA:** `linux/aarch64` (Cortex-A53) · **netkit:** `NETKIT_TARGET=mpu_arm` + XNNPACK  

No bare-metal firmware tree — cross-build on the host (Docker `linux/arm64`), deploy over SSH.

**Setup / rebuild / run A/B:** [`pi-zero-2w/README.md`](pi-zero-2w/README.md)

Covers prerequisites, `tools/build_mpu_pi_aarch64.sh`, SSH helpers, and float32 + int8 suites vs TF Lite (XNNPACK ON/OFF).

---

## Targets without a board tree yet

| Target | How to build / integrate |
|--------|---------------------------|
| `mcu_esp` (ESP-NN) | [PLATFORMS.md — Espressif](../docs/PLATFORMS.md#mcu_esp--espressif-mcu) · `make esp-nn-init` · link `libnetkit.a` into your ESP-IDF app |
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
