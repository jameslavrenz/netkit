# TFLM MNIST benchmarks (MLP + CNN)

Compare netkit's trained MNIST models (`models/mnist_mlp.nk`, `models/mnist_cnn.nk`) against **TensorFlow Lite Micro (TFLM)** on the host (macOS/Linux).

Each benchmark:

- Embeds **float32 weights** converted from the same ONNX graphs as netkit
- Runs **10 MNIST test images** (one per digit) as static input tensors
- Calls `MicroInterpreter::Invoke()` for each image
- Times each invoke with `std::chrono::steady_clock`
- Prints per-image latency and the **average invoke time**

Profile targets (`run-mlp-profile`, `run-cnn-profile`) use TFLM `MicroProfiler` for per-op breakdown — see [../README.md](../README.md).

## Layout

```
benchmark/tflm/
  Makefile              # wrapper around TFLM's upstream Makefile
  Makefile.inc          # registered into TFLM additional_tests.inc
  src/main.cc           # benchmark entry (TFLM builds .cc, not .cpp)
  src/main.cpp          # symlink to main.cc
  generated/
    mnist_mlp.tflite    # converted model (from models/mnist_mlp.onnx)
    mnist_test_images.* # 10 embedded MNIST vectors from .nk TCAS cases
  third_party/tflite-micro/   # cloned by tools/fetch_tflm.sh (gitignored)
  tools/
    export_assets.py    # refresh .tflite + test image arrays
    fetch_tflm.sh       # clone TFLM
```

## Prerequisites

1. **GNU Make ≥ 3.82** (TFLM requirement). On macOS:
   ```bash
   brew install make
   ```
   The wrapper uses `gmake` from Homebrew when available.

2. **netkit MNIST model** (once):
   ```bash
   make export-mnist
   ```

3. **Asset export** (once, or after retraining):
   ```bash
   python3 -m venv benchmark/tflm/.venv
   benchmark/tflm/.venv/bin/pip install onnx onnx2tf
   python3 benchmark/tflm/tools/export_assets.py
   ```
   This converts `models/mnist_mlp.onnx` → `generated/mnist_mlp.tflite` via [onnx2tf](https://github.com/PINTO0309/onnx2tf) and regenerates the embedded test vectors.

## Build and run

```bash
cd benchmark/tflm
make fetch-tflm    # first time only — clones google/tflite-micro
make run           # build with TFLM flags + run benchmark
```

`make run` uses Google's TFLM Makefile (`tensorflow/lite/micro/tools/make/Makefile`):

| Setting | Default (host) |
|---------|----------------|
| `TARGET` | `osx` or `linux` (auto-detected) |
| `TOOLCHAIN` | `gcc` |
| `CXX` | `g++` |
| `CC` | `gcc` |
| `BUILD_TYPE` | `default` (TFLM error-checking build) |
| Flags | `-DTF_LITE_STATIC_MEMORY`, `-fno-rtti`, `-fno-exceptions`, `-O2` core / `-O2` kernels |

Inspect the exact flags TFLM selected:

```bash
gmake -C third_party/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile config_info
```

## Example output

```
TFLM MNIST MLP benchmark
  model bytes: 408576
  images:      10
  arena bytes: 131072

image  0  label=7  pred=7  invoke=581.083 us (0.581 ms)  MNIST digit 7 (test idx 0)
...
Average invoke time: 160.167 us (0.160 ms) over 10 images
```

The first invoke is often slower (cold caches). All 10 cases should match their labels when using the bundled weights.

## Notes

- Source file is `main.cc` because TFLM's Makefile only compiles `.cc` C++ sources; `main.cpp` is a symlink for convenience.
- TFLM downloads third-party deps (flatbuffers, gemmlowp, kissfft, pigweed, …) on first build under `third_party/tflite-micro/tensorflow/lite/micro/tools/make/downloads/`.
- To compare against netkit on the same images, run:
  ```bash
  make -C benchmark/netkit run
  ```
  See [../README.md](../README.md) for side-by-side instructions.
