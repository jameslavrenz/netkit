# NUCLEO-F446RE — MNIST CNN int8 benchmark firmware

Bare-metal firmware for the **STM32 NUCLEO-F446RE** (STM32F446RET6, Cortex-M4F, 512 KiB flash / 128 KiB SRAM).

Runs the **same MNIST CNN benchmark** as `benchmark/netkit/` and `benchmark/tflm/` (10 test images, 10 runs), using **int8** weights, activations, and prequantized inputs end-to-end.

**Default build = interpreter embed** (embedded `.nk` + runtime loader) for a fair comparison with TFLM `MicroInterpreter`. Optional `NETKIT_LOWERED=1` switches to quant lowered deployment (static `CmsisQuantPlan` chain).

## netkit build profile (default)

| Setting | Value |
|---------|--------|
| Target | `NETKIT_TARGET_MCU` |
| Arch | `NETKIT_ARCH=CM4` (Cortex-M4F) |
| CMSIS | **CMSIS-DSP** + **CMSIS-NN** enabled |
| Weights | **Flash** — embedded `.nk` blob in `.rodata` (`NETKIT_WEIGHTS_IN_RAM=0`) |
| Deployment | **Interpreter embed** — `NkLoader` + `NkOpsResolver` (same class as TFLM blob + interpreter) |
| Dtype | int8 weights / activations; int8 softmax output; prequantized int8 test inputs |

Int8 conv/pool/dense/softmax use CMSIS-NN kernels on Cortex-M4 (`QuantOps` + `CmsisQuantPlan`).

## Verified on-device results (NUCLEO-F446RE @ 180 MHz, interpreter embed)

| Metric | Value |
|--------|-------|
| Mean invoke | **~144 ms** |
| Accuracy | **10/10** |
| Arena | **~105 KiB** recommended (static buffer in firmware) |
| Flash (text + data) | **~351 KiB** |

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
./scripts/deploy.sh build    # cross-compile + regenerate embed sources
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
make                    # interpreter embed (default)
```

Outputs:

- `build/mnist_cnn_int8_nucleo_f446re.elf`
- `build/mnist_cnn_int8_nucleo_f446re.bin`

Generated embed sources: `generated/mnist_cnn_int8_aot.{hpp,cpp}` (historical `*_aot` filename; see [PHILOSOPHY.md](../../docs/PHILOSOPHY.md#terminology-embed-vs-lowered)).

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

## Build modes

| Mode | Command | Deployment |
|------|---------|------------|
| **Interpreter embed** (default) | `make` | Embedded `.nk` blob + runtime loader (fair vs TFLM `MicroInterpreter`) |
| **Quant lowered** (deployment) | `make NETKIT_LOWERED=1` | Static `CmsisQuantPlan` call chain, tiny arena |

Regenerate embed sources after changing mode:

```bash
rm -f generated/.embed_stamp && make              # interpreter
rm -f generated/.embed_stamp && make NETKIT_LOWERED=1   # lowered
```

## Example UART output (interpreter embed)

```
netkit NUCLEO-F446RE MNIST CNN int8 benchmark
  backend:     cmsis-nn int8 (MCU CM4, .nk loader)
  weights:     flash (embedded .nk blob)
  dtype:       int8 end-to-end (softmax int8, prequantized inputs)
  arena bytes: 107456
  nk bytes:    258440
  probe:       label=0 pred=0 pred_i8=127 out_i8=127,-128,-128,-128,-128,-128,-128,-128,-128,-128

  per-digit results (final run, int8 only — dequant in Python):
    image  label  pred  pred_i8  ok
        0      0     0      127  yes
        1      1     1      126  yes
        ...
DIGIT_SUMMARY runtime=netkit model=cnn_int8 image=0 label=0 pred=0 pred_i8=127 ok=1 out_i8=127,-128,-128,-128,-128,-128,-128,-128,-128,-128
...

Dequantized confidence is **not** printed on-device. Parse the log offline:

```bash
python3 benchmark/tools/parse_mcu_cnn_int8_log.py boards/nucleo-f446re-cnn-int8/uart.log
python3 benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit.log tflm.log
```

  accuracy:    10/10 on final run
  mean:   144141.967 us (144.142 ms)
BENCHMARK_SUMMARY runtime=netkit model=cnn_int8 backend=cmsis-nn-int8 mean_us=144141.967 runs=10

DONE
```

## Related docs

- [NUCLEO-F446RE MNIST MLP (f32)](../nucleo-f446re/README.md)
- [MNIST CNN (host)](../../docs/MNIST_CNN.md)
- [Build targets](../../docs/BUILD_TARGETS.md)
- [Benchmarks](../../benchmark/README.md)
- [Embed vs lowered terminology](../../docs/PHILOSOPHY.md#terminology-embed-vs-lowered)
