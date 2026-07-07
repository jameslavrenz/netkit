# NUCLEO-F446RE — MNIST CNN int8 benchmark firmware

Bare-metal firmware for the **STM32 NUCLEO-F446RE** (STM32F446RET6, Cortex-M4F, 512 KiB flash / 128 KiB SRAM).

Runs the **same MNIST CNN benchmark** as `benchmark/netkit/` and `benchmark/tflm/` (10 test images, 10 runs), using **int8** weights, activations, and prequantized inputs end-to-end.

## netkit build profile

| Setting | Value |
|---------|--------|
| Target | `NETKIT_TARGET_MCU` |
| Arch | `NETKIT_ARCH=CM4` (Cortex-M4F) |
| CMSIS | **CMSIS-DSP** + **CMSIS-NN** enabled |
| Weights | **Flash** — static `.rodata` via quant lowered AOT (`NETKIT_WEIGHTS_IN_RAM=0`) |
| Deployment | **Quant lowered AOT** — static `CmsisQuantPlan` call chain (no runtime `.nk` loader) |
| Dtype | int8 weights / activations; int8 softmax output; prequantized int8 test inputs |

Int8 conv/pool/dense/softmax use CMSIS-NN kernels on Cortex-M4 (`QuantOps` + `CmsisQuantPlan`).

## Verified on-device results (NUCLEO-F446RE @ 180 MHz)

| Metric | Value |
|--------|-------|
| Mean invoke | **~137 ms** |
| Accuracy | **10/10** (with TFLite-aligned layer-0 input quant) |
| Arena | **64 bytes** (activation ping-pong + workspace in BSS) |
| Flash (text + data) | **~309 KiB** |

Compare with TFLM int8 on the same board: [nucleo-f446re-tflm-cnn-int8](../nucleo-f446re-tflm-cnn-int8/README.md).

## Prerequisites (host)

```bash
# From repo root — once
make cmsis-init
make export-mnist-cnn
make export-mnist-cnn-int8

# Refresh int8 test vectors after re-exporting the .nk model
python3 benchmark/tflm/tools/export_int8_test_images.py

# Toolchain + OpenOCD (macOS examples)
brew install openocd
cd boards/nucleo-f446re-cnn-int8 && ./scripts/setup-toolchain.sh

# Optional: UART monitor
pip install pyserial
```

## Deploy (recommended)

```bash
cd boards/nucleo-f446re-cnn-int8
chmod +x scripts/*.sh

./scripts/deploy.sh export   # quantize from models/mnist_cnn.nk (~seconds)
./scripts/deploy.sh build    # cross-compile + regenerate AOT
./scripts/deploy.sh flash    # ST-Link (do not run while monitor is open)
./scripts/deploy.sh capture  # UART — press RESET if silent

# Or all at once:
./scripts/deploy.sh all
```

Export uses **existing float weights** (`models/mnist_cnn.nk`) by default — no 20-epoch retrain.
When `benchmark/tflm/generated/mnist_cnn_int8.tflite` exists, export aligns **layer-0 input quant** with TFLite (`1/255`, `-128`) for shared test images; weight and output scales are calibrated from netkit weights.

## Build (manual)

```bash
cd boards/nucleo-f446re-cnn-int8
make
```

Outputs:

- `build/mnist_cnn_int8_nucleo_f446re.elf`
- `build/mnist_cnn_int8_nucleo_f446re.bin`

## Flash + collect results

Connect the NUCLEO via USB (onboard ST-Link). USART2 virtual COM port @ **115200 8N1**.

```bash
./scripts/flash.sh
./scripts/monitor.sh    # start monitor first, then press RESET (or ../nucleo-f446re/scripts/reset.sh)
```

Press the black **RESET** button to re-run the benchmark. Avoid flashing while the serial monitor holds the VCP port open. ST-Link SWD runs at **1800 kHz** (`boards/nucleo-f446re/openocd/nucleo_f446re.cfg`).

From repo root:

```bash
make flash-mnist-cnn-int8
```

## Example UART output

```
netkit NUCLEO-F446RE MNIST CNN int8 benchmark
  backend:     cmsis-nn int8 (MCU CM4, quant lowered AOT)
  weights:     flash (static .rodata)
  dtype:       int8 end-to-end (softmax int8, prequantized inputs)
  arena bytes: 64
  workspace:   1152 bytes
  probe:       label=0 pred=0 i8[0..3]=127,-128,-128,-128
  accuracy:    10/10 on final run
  mean:   145286.741 us (145.287 ms)
BENCHMARK_SUMMARY runtime=netkit model=cnn_int8 backend=cmsis-nn-int8 mean_us=145286.741 runs=10

DONE
```

## Related docs

- [NUCLEO-F446RE MNIST MLP (f32)](../nucleo-f446re/README.md)
- [MNIST CNN (host)](../../docs/MNIST_CNN.md)
- [Build targets](../../docs/BUILD_TARGETS.md)
- [Benchmarks](../../benchmark/README.md)
