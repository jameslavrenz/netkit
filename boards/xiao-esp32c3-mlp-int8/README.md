# Seeed Studio XIAO ESP32C3 — MNIST MLP int8

ESP-IDF firmware (via PlatformIO) for the **Seeed Studio XIAO ESP32C3**.

| Item | Value |
|------|--------|
| Board | [Seeed XIAO ESP32C3](https://wiki.seeedstudio.com/XIAO_ESP32C3_Getting_Started/) |
| Chip | ESP32-C3 · **RISC-V** · 160 MHz · 400 KB SRAM · 4 MB flash |
| netkit target | `NETKIT_TARGET=mcu_esp` + `NETKIT_ARCH=ESP32C3` |
| Backend | **ESP-NN** (int8 production) — **not** NMSIS-NN |
| Model | MNIST MLP int8, quant lowered AOT |
| Console | USB Serial/JTAG (onboard, no UART adapter) |

ESP32-C3 is RISC-V silicon, but Espressif chips use the **`mcu_esp` / ESP-NN**
profile in netkit (same target as ESP32-S3 / P4). Use `mcu_risc` + NMSIS-NN only
for Nuclei / generic RV32 BSPs — not for any ESP32*. See
[PLATFORMS.md — Target ≠ CPU ISA](../../docs/PLATFORMS.md#target--cpu-isa).

## Prerequisites

```bash
# Repo root — once
make esp-nn-init
make export-mnist-mlp-int8

# Host tools
# PlatformIO CLI: https://platformio.org/install/cli
# Optional: pip install pyserial
```

Connect the XIAO with a **data-capable** USB-C cable. On macOS you should see something like:

```text
/dev/cu.usbmodem21101   # USB JTAG/serial debug unit, VID:PID=303A:1001
```

## Build / flash / monitor

```bash
cd boards/xiao-esp32c3-mlp-int8
make
PORT=/dev/cu.usbmodem21101 make flash
PORT=/dev/cu.usbmodem21101 make monitor
# or:
PORT=/dev/cu.usbmodem21101 make run
```

Press the board **RESET** button (or unplug/replug) to re-run the benchmark while the monitor is open.

## Example UART output

```text
netkit XIAO ESP32C3 MNIST MLP int8 benchmark
  backend:     esp-nn (MCU ESP32C3, quant lowered AOT)
  ...
BENCHMARK_SUMMARY runtime=netkit model=mlp_int8 backend=esp-nn-int8 mean_us=... runs=10

DONE
```

## Library-only (no board flash)

Host ANSI smoke of the same profile:

```bash
make -C ../.. esp-nn-init
make -C ../.. NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C3 NETKIT_HOST_SMOKE=1 lib embedded-smoke
```

Cross-compile the lean static library for device linking (needs the Espressif RISC-V toolchain from PlatformIO / ESP-IDF):

```bash
make -C ../.. NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C3 lib
```

Details: [docs/PLATFORMS.md](../../docs/PLATFORMS.md#mcu_esp--espressif-mcu).
