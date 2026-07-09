# NUCLEO-F446RE — MNIST MLP f32 benchmark firmware

Bare-metal firmware for the **STM32 NUCLEO-F446RE** (STM32F446RET6, Cortex-M4F, 512 KiB flash / 128 KiB SRAM).

Runs the **same MNIST MLP benchmark** as `benchmark/netkit/`:

- 10 hard-coded test images (one digit 0–9 each)
- 100 runs × 10 images, discard first invoke per run
- Mean invoke latency printed on **USART2** (ST-Link virtual COM port, 115200 baud)

## netkit build profile

| Setting | Value |
|---------|--------|
| Target | `NETKIT_TARGET_MCU` |
| Arch | `NETKIT_ARCH=CM4` (Cortex-M4F + hard float) |
| CMSIS | **CMSIS-DSP** enabled (`NETKIT_USE_CMSIS_DSP=1`); inputs staged in SRAM before timed invoke |
| Weights | **Flash** — lowered AOT embeds coef arrays in `.rodata` (no SRAM copy; `NETKIT_WEIGHTS_IN_RAM=0` policy) |
| Deployment | **Lowered AOT** — static `Kernels::` FC chain (no runtime `.nk` loader) |

Arena for activations: **640 bytes** (see generated `mnist_mlp_aot.hpp`).

## Verified on-device results (NUCLEO-F446RE @ 180 MHz)

Captured after `make && ./scripts/run.sh` with the default profile above:

| Metric | Value |
|--------|-------|
| Mean invoke | **10,684.7 µs** (~10.7 ms) |
| Accuracy | 10/10 (final run) |
| Flash (text + data) | **477,312 B** (~466 KiB) |
| SRAM (data + bss) | **2,697 B** (~2.6 KiB) |

Compare with host `benchmark/netkit/` and TFLM on the same 10 images via [benchmark/README.md](../../benchmark/README.md).

## Prerequisites (host)

```bash
# From repo root — once
make cmsis-init
make export-mnist          # if models/mnist_mlp.nk missing

# Toolchain + OpenOCD (macOS examples)
brew install openocd
# arm-none-eabi GCC with newlib (auto-fetched on first build):
cd boards/nucleo-f446re && ./scripts/setup-toolchain.sh

# Optional: UART monitor
pip install pyserial
```

The board Makefile downloads **xPack arm-none-eabi-gcc** into `.toolchain/` if no suitable toolchain is found. Homebrew `arm-none-eabi-gcc` alone is incomplete (no newlib headers).

## Build

```bash
cd boards/nucleo-f446re
make
```

Outputs:

- `build/mnist_mlp_nucleo_f446re.elf`
- `build/mnist_mlp_nucleo_f446re.bin`
- `build/mnist_mlp_nucleo_f446re.map`

`make` regenerates AOT sources under `generated/` from `models/mnist_mlp.nk`.

## Flash (ST-Link USB)

Connect the NUCLEO via USB (onboard ST-Link). Programming uses **SWD**, not UART.

```bash
./scripts/flash.sh
# or
make flash
```

Requires [OpenOCD](https://openocd.org/) (`interface/stlink.cfg` + `openocd/nucleo_f446re.cfg` at **1800 kHz** SWD).

## Read benchmark results (UART)

The ST-Link exposes a **virtual COM port** wired to USART2 (PA2/PA3) at **115200 8N1**.

```bash
./scripts/monitor.sh
```

Press the black **RESET** button on the board to re-run the benchmark (or `./scripts/reset.sh` via OpenOCD while the monitor is open).

One-shot build + flash + monitor:

```bash
./scripts/run.sh
```

On Linux, set `PORT=/dev/ttyACM0` if auto-detect fails. On macOS, look for `/dev/cu.usbmodem*`.

## Example UART output

```
netkit NUCLEO-F446RE MNIST MLP benchmark
  backend:     cmsis-dsp (MCU CM4, lowered AOT)
  weights:     flash (embedded coef arrays)
  images:      10 per run
  runs:        100 (discard first invoke each run)
  arena bytes: 640
  sysclk:      180000000 Hz
  accuracy:    10/10 on final run

netkit MNIST mlp benchmark summary (cmsis-dsp)
  method:      100 runs x 10 images, discard first invoke each run
  per-run avg: avg of images 1-9 (us)

  mean:  10684.683 us (10.685 ms)
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=cmsis-dsp mean_us=10684.683 runs=100

DONE
```

Timing uses the **DWT cycle counter** at **180 MHz** SYSCLK (HSE 8 MHz → PLL; falls back to HSI PLL if HSE is absent).

## Flash budget

MNIST MLP weights dominate image size (~470 KiB `.text` + `.rodata`). This fits **512 KiB** parts with `-O2` and LTO but has little headroom. For smaller parts, use a smaller model (e.g. `test_mlp.nk`) or external flash.

## Layout

```
boards/nucleo-f446re/
  Makefile
  linker/STM32F446RETx_FLASH.ld
  startup/startup_stm32f446xx.S
  src/main.cpp          # benchmark loop
  src/uart.c            # USART2 @ 115200
  src/dwt_time.c        # cycle-accurate timing
  scripts/
    setup-toolchain.sh  # fetch xPack GCC if needed
    flash.sh            # OpenOCD + ST-Link
    monitor.sh          # pyserial UART reader
    run.sh              # build + flash + monitor
  generated/            # AOT output (gitignored)
```

## Related docs

- [benchmark/README.md](../../benchmark/README.md) — host MNIST vs TFLM methodology
- [docs/BUILD_TARGETS.md](../../docs/BUILD_TARGETS.md) — MCU / CMSIS flags
- [docs/ARENA.md](../../docs/ARENA.md) — `NETKIT_WEIGHTS_IN_RAM` tradeoffs
