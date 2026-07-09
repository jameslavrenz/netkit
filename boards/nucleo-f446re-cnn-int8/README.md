# NUCLEO-F446RE — MNIST CNN int8 benchmark firmware

Bare-metal firmware for the **STM32 NUCLEO-F446RE** (STM32F446RET6, Cortex-M4F, 512 KiB flash / 128 KiB SRAM).

Runs the **same MNIST CNN benchmark** as `benchmark/netkit/` and `benchmark/tflm/` (10 test images, 10 runs), using **int8** weights, activations, and prequantized inputs end-to-end.

**Default build = interpreter embed** (embedded `.nk` + runtime loader) for a fair comparison with TFLM `MicroInterpreter`. Optional `NETKIT_LOWERED=1` switches to quant lowered deployment (static `CmsisQuantPlan` chain).

## netkit build profile (default)

| Setting | Value |
|---------|--------|
| Target | `NETKIT_TARGET_MCU` |
| Arch | `NETKIT_ARCH=CM4` (Cortex-M4F) |
| CMSIS | **CMSIS-NN** + **CMSIS-DSP** (q7 copy/max utils; layer kernels are CMSIS-NN) |
| Weights | **Flash** — embedded `.nk` blob in `.rodata` (`NETKIT_WEIGHTS_IN_RAM=0`) |
| Deployment | **Interpreter embed** — `NkLoader` + `NkOpsResolver` (same class as TFLM blob + interpreter) |
| Dtype | int8 weights / activations; prequantized int8 test inputs; output = logits (Softmax omitted) |

Int8 conv/pool/dense use CMSIS-NN kernels on Cortex-M4 (`CmsisQuantPlan`). Final Softmax is omitted for classification (`--omit-final-softmax`); firmware argmaxes logits. Benchmark firmware copies each test image into SRAM (`g_input_staging`) before the timed forward pass — same pattern as TFLM’s input tensor copy.

## Verified on-device results (NUCLEO-F446RE @ 180 MHz, interpreter embed)

| Metric | Value |
|--------|-------|
| Mean invoke | **~95 ms** (typical **94.9–97.0 ms** across captures; interpreter embed, CMSIS-NN s8) |
| Accuracy | **10/10** |
| Arena | **64 KiB** static buffer (fixed; verified 10/10 on-device) |
| Flash (text + data) | **~334 KiB** (of 512 KiB) |
| SRAM (bss + data) | **~75 KiB** (of 128 KiB; ~53 KiB headroom) |

Compare with TFLM int8 on the same board: [nucleo-f446re-tflm-cnn-int8](../nucleo-f446re-tflm-cnn-int8/README.md).

## Memory budget (STM32F446RE, interpreter embed)

The board has **512 KiB flash** and **128 KiB SRAM**. Default firmware uses **interpreter embed** with flash-backed weights (`NETKIT_WEIGHTS_IN_RAM=0`).

| Region | Approx. size | Notes |
|--------|--------------|-------|
| `.text` + `.rodata` (code + embedded `.nk`) | ~334 KiB | ~258 KiB is the embedded `mnist_cnn_int8.nk` blob |
| `.data` + `.bss` (SRAM) | ~75 KiB | Includes **64 KiB** interpreter arena + output scratch |
| SRAM headroom | ~53 KiB | Stack, heap (none used), ST-Link/USB buffers |

**Arena sizing:** embed codegen may emit a larger `kArenaBytesRecommended` (host probe + headroom). This firmware **fixes the arena at 64 KiB** in `src/main.cpp` — enough for load + ping-pong activations on this model with ~53 KiB SRAM to spare. To add margin, bump `kArenaCapacity` (e.g. 72–80 KiB) and re-check linker RAM + on-device accuracy.

**Quant lowered** (`NETKIT_LOWERED=1`) uses a different layout: static ping-pong activation buffers (`g_act_a` / `g_act_b`) and CMSIS workspace in **BSS** (~28 KiB for this model), with a **tiny** bump arena (composite-block scratch only). See [ARENA.md](../../docs/ARENA.md#quant-lowered-vs-interpreter-embed-on-mcu).

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
| **Quant lowered** (deployment) | `make NETKIT_LOWERED=1` | Static `CmsisQuantPlan` call chain; activations in static BSS, tiny arena |

Compiler/linker flags match TFLM kernel speed via `boards/nucleo-f446re/mcu_tflm_toolchain.mk` (CORE / KERNEL / THIRD_PARTY all `-O2`, `-flto` link with `--gc-sections`). `NETKIT_CPPFLAGS` is passed on the final link so CMSIS macros survive LTO.

**CMSIS always on for this board:** the Makefile uses `override NETKIT_CMSIS_DSP := 1` and `override NETKIT_CMSIS_NN := 1` so host/CI env vars (`NETKIT_CMSIS_NN=0`, `GITHUB_ACTIONS=true`, etc.) cannot accidentally link CMSIS **stub** kernels that fail forward at runtime.

**Optional kernel trace (bring-up only):** `make NETKIT_QUANT_TRACE=1` links `quant_trace.cpp` and prints a CMSIS vs reference summary over UART after the probe forward. Default `make` uses zero-cost inline stubs — no trace overhead on the hot path.

Regenerate embed sources after changing mode:

```bash
rm -f generated/.embed_stamp && make              # interpreter
rm -f generated/.embed_stamp && make NETKIT_LOWERED=1   # lowered
```

## Example UART output (interpreter embed)

```
netkit NUCLEO-F446RE MNIST CNN int8 benchmark
  backend:     cmsis-nn int8 + cmsis-dsp utils (MCU CM4, .nk loader)
  weights:     flash (embedded .nk blob)
  dtype:       int8 end-to-end (weights, activations, inputs; logits out)
  arena bytes: 65536
  nk bytes:    258440
  sysclk:      180000000 Hz
  model:       loaded
  probe:       label=0 pred=0 pred_i8=127 out_i8=127,-128,-128,-128,-128,-128,-128,-128,-128,-128

  per-digit results (final run, int8 only — dequant in Python):
    image  label  pred  pred_i8  ok
        0      0     0      127  yes
        ...
DIGIT_SUMMARY runtime=netkit model=cnn_int8 image=0 label=0 pred=0 pred_i8=127 ok=1 out_i8=127,-128,-128,-128,-128,-128,-128,-128,-128,-128
...

  accuracy:    10/10 on final run

netkit MNIST cnn int8 benchmark summary
  mean:   94904.878 us (94.905 ms)
BENCHMARK_SUMMARY runtime=netkit model=cnn_int8 backend=cmsis-nn-int8 mean_us=94904.878 runs=10

DONE
```

Single-capture means vary by roughly **±2%** (e.g. 94.9–97.0 ms) on the same firmware — normal for DWT timing on this board. Average several captures for regression tracking.

Dequantized confidence is **not** printed on-device. Parse the log offline:

```bash
python3 benchmark/tools/parse_mcu_cnn_int8_log.py boards/nucleo-f446re-cnn-int8/uart.log
python3 benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit.log tflm.log
```

## Related docs

- [NUCLEO-F446RE MNIST MLP (f32)](../nucleo-f446re/README.md)
- [MNIST CNN (host)](../../docs/MNIST_CNN.md)
- [Build targets](../../docs/BUILD_TARGETS.md)
- [Benchmarks](../../benchmark/README.md)
- [Embed vs lowered terminology](../../docs/PHILOSOPHY.md#terminology-embed-vs-lowered)
