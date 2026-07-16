# NUCLEO-F446RE — TFLM MNIST CNN int8 benchmark firmware

Bare-metal **TensorFlow Lite Micro** firmware with a **full int8** MNIST CNN graph
(int8 input/output, CMSIS-NN optimized kernels).

## Benchmark parity

| Item | Value |
|------|--------|
| Model | `benchmark/tflm/generated/mnist_cnn_int8.tflite` (full int8 I/O) |
| Images | 10 prequantized int8 digits from `mnist_cnn_int8_test_images.*` (TFLite input quant; `memcpy` at invoke) |
| Runs | **10** outer × 10 images; discard first invoke each run |
| Metric | Mean of per-run averages (images 1–9), DWT µs @ 180 MHz |
| UART | USART2 @ 115200 (ST-Link VCP) |

## Build

```bash
# Prerequisites (once)
make -C ../../benchmark/tflm fetch-tflm
pip install -r ../../benchmark/tflm/requirements-export.txt
make export-cnn-int8
../nucleo-f446re/scripts/setup-toolchain.sh

cd boards/nucleo-f446re-tflm-cnn-int8
make setup-deps
make
./scripts/flash.sh
./scripts/monitor.sh   # press RESET on board
```

## Resource notes (STM32F446RE: 512 KiB flash / 128 KiB SRAM)

- TFLite int8 model: embedded in flash (`.rodata`)
- Tensor arena: **120 KiB** SRAM (`kTensorArenaSize` in `src/main.cc`)
- TFLM microlite: `OPTIMIZED_KERNEL_DIR=cmsis_nn` (default); `make TFLM_OPT_KERNEL=` for reference

## Verified on-device results (NUCLEO-F446RE @ 180 MHz)

Matched toolchain with netkit/microTVM; 10×10, discard first invoke — [STATUS.md](../../docs/STATUS.md).

| Mode | TFLM | netkit embed | microTVM |
|------|-----:|-------------:|---------:|
| CMSIS-NN | **95.5 ms** | 95.3 ms | 112.3 ms |
| reference (`TFLM_OPT_KERNEL=`) | **2593.5 ms** | 336.2 ms | 343.0 ms |

All **10/10**.

```text
BENCHMARK_SUMMARY runtime=tflm model=cnn_int8 backend=cmsis-nn mean_us=95513 runs=10
BENCHMARK_SUMMARY runtime=tflm model=cnn_int8 backend=reference mean_us=2593540 runs=10
```

## Per-digit confidence (offline)

Firmware prints raw int8 softmax outputs (`pred_i8`, `out_i8=...` on `DIGIT_SUMMARY` lines).
Dequantize and compare in Python:

```bash
python3 ../../benchmark/tools/parse_mcu_cnn_int8_log.py uart.log
python3 ../../benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit_uart.log tflm_uart.log
```
