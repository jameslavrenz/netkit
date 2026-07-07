# MNIST CNN Regression Test

Trained **784→CNN→10** classifier on MNIST using a stack common in Keras/TensorFlow tutorials. Runs as part of `make test` alongside the [MNIST MLP suite](MNIST.md).

## Architecture

| Stage | Config | Activation |
|-------|--------|------------|
| Input | 28×28×1 (NHWC) | — |
| Conv2D | 3×3, 32 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Conv2D | 3×3, 64 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Flatten | 5×5×64 → 1600 | — |
| Dense | 128 units | ReLU |
| Dense | 10 units | Softmax |

## Accuracy

| Metric | netkit (committed export) | Typical baseline |
|--------|---------------------------|------------------|
| Test accuracy | **99.02%** | ~98.5–99.3% (Keras/TF CNN tutorials) |
| Training set | **60,000** images | 60,000 |
| Optimizer | Adam, lr=0.001, **20 epochs** | Similar |

The [MNIST MLP suite](MNIST.md) on the same data reaches **98.06%** test accuracy.

## Files

| Path | Purpose |
|------|---------|
| `models/mnist_cnn.onnx` | Source graph (ONNX parity) |
| `models/mnist_cnn.nk` | Float32 runtime model + 10 embedded TCAS cases |
| `models/mnist_cnn_int8.nk` | Int8 quantized model (MCU / CMSIS-NN) |
| `tools/export_mnist_cnn.py` | Train + export float script |
| `tools/export_mnist_cnn_int8.py` | Quantize float `.nk` to int8 (optional TFLite input-quant alignment) |
| `benchmark/tflm/tools/export_int8_test_images.py` | Preqantized int8 benchmark test vectors (separate from float images) |
| `tools/compare_nk_tflite_quant.py` | Compare activation quant params vs TFLite |

The MNIST CNN suite uses a **4 MiB** dedicated arena in `src/nk_regression.cpp`. See [ARENA.md](ARENA.md).

## Running

Part of `make test` / `./netkit test` — see [TESTING.md](TESTING.md).

## Benchmarks

Same 10 TCAS vectors as the MLP suite; compared against TFLM in [benchmark/README.md](../benchmark/README.md). CNN invoke and per-op profile tables (Conv2D vs pool vs FC) are produced by `./benchmark/compare.sh`. On host builds both runtimes use reference conv kernels — CMSIS-NN applies only on MCU targets.

### Int8 on-device (NUCLEO-F446RE)

```bash
make export-mnist-cnn-int8
python3 benchmark/tflm/tools/export_int8_test_images.py
make -C boards/nucleo-f446re-cnn-int8
cd boards/nucleo-f446re-cnn-int8 && ./scripts/flash.sh && ./scripts/monitor.sh
```

Verified: **10/10** accuracy, **~144 ms** mean invoke (interpreter embed, default `make`). See [boards/nucleo-f446re-cnn-int8/README.md](../boards/nucleo-f446re-cnn-int8/README.md).

Optional quant lowered deployment build: `make NETKIT_LOWERED=1` (~137 ms in prior lowered runs).

UART captures `DIGIT_SUMMARY` lines with raw int8 softmax (`pred_i8`, `out_i8=...`). Dequantized per-digit confidence is computed offline — not on device:

```bash
python3 benchmark/tools/parse_mcu_cnn_int8_log.py uart.log
python3 benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit_uart.log tflm_uart.log
```

Int8 export aligns **layer-0 input quant** with TFLite when `benchmark/tflm/generated/mnist_cnn_int8.tflite` is present (`input_scale=1/255`, `zero_point=-128`). Weight and per-layer output scales are calibrated from netkit float weights. Benchmark inputs are prequantized in `mnist_cnn_int8_test_images.*` — no float→int8 conversion at invoke on netkit or TFLM MCU firmware.

## Regenerating

```bash
make export-mnist-cnn
make export-mnist-cnn-int8   # optional: int8 quant from float .nk
make export-onnx-test
make test
```

Commit `models/mnist_cnn.nk`, `models/mnist_cnn.onnx`, and `models/mnist_cnn_int8.nk` after regenerating so tests stay offline (no training at test time).
