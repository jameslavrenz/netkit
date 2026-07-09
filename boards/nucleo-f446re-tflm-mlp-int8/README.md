# NUCLEO-F446RE — TFLM MNIST MLP int8 benchmark firmware

Bare-metal **TensorFlow Lite Micro** firmware with a **full int8** MNIST MLP graph
(int8 input/output, CMSIS-NN optimized kernels).

## Benchmark parity

| Item | Value |
|------|--------|
| Model | `benchmark/tflm/generated/mnist_mlp_int8.tflite` (full int8 I/O) |
| Images | 10 prequantized int8 digits from `mnist_mlp_int8_test_images.*` |
| Runs | **10** outer × 10 images; discard first invoke each run |
| Metric | Mean of per-run averages (images 1–9), DWT µs @ 180 MHz |
| UART | USART2 @ 115200 (ST-Link VCP) |

## Verified on-device results (NUCLEO-F446RE @ 180 MHz)

| Backend | Mean invoke | Accuracy |
|---------|------------:|----------|
| CMSIS-NN (default) | **~3.36 ms** | **10/10** |
| Reference (`TFLM_OPT_KERNEL=` empty) | **~13.3 ms** | **10/10** |

## Build

```bash
# Prerequisites (once)
make -C ../../benchmark/tflm fetch-tflm
make -C ../.. export-mnist-mlp-int8
../nucleo-f446re/scripts/setup-toolchain.sh

cd boards/nucleo-f446re-tflm-mlp-int8
make setup-deps
make
./scripts/flash.sh
./scripts/monitor.sh   # press RESET on board
```

## Compare with netkit int8 MLP

```text
BENCHMARK_SUMMARY runtime=tflm model=mlp_int8 backend=cmsis-nn mean_us=... runs=10
BENCHMARK_SUMMARY runtime=netkit model=mlp_int8 backend=cmsis-nn-int8 mean_us=... runs=10
```

See [nucleo-f446re-mlp-int8](../nucleo-f446re-mlp-int8/README.md) for the netkit firmware.
