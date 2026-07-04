# netkit — Neural Network Kit

netkit is a C++26 neural network kit for on-device inference on MCUs and MPUs. It is developed and validated on the desktop, then deployed to embedded targets. Companion to [memkit](https://github.com/jameslavrenz/memkit) for memory management.

Models are loaded from binary **`.nk`** files (single-file architecture + weights). Convert from ONNX with `python -m netkit convert`, or embed a `.nk` in firmware with `python -m netkit aot`. **Inference is float32-only today**; float16, int16, int8, and int4 are on the roadmap — see [docs/DATATYPES.md](docs/DATATYPES.md).

## Documentation

| Guide | Description |
|-------|-------------|
| **[Philosophy](docs/PHILOSOPHY.md)** | Phase 1 runtime vs Phase 2 packager; design principles |
| **[Getting Started](docs/GETTING_STARTED.md)** | Build, test, CLI, and first inference for new users |
| **[API Overview](docs/API.md)** | C vs C++ APIs, linking, memory model |
| **[Build Targets](docs/BUILD_TARGETS.md)** | CPU / MCU / MPU flags and arena defaults |
| **[CLI Reference](docs/CLI.md)** | `test`, `run`, and `inspect` (CPU build) |
| **[Arena Memory](docs/ARENA.md)** | Bump allocator — sizing, alignment, reset |
| **[Data Types](docs/DATATYPES.md)** | Float32 today; float16 / int16 / int8 / int4 roadmap |
| **[ONNX Import](docs/ONNX.md)** | Python packager (ONNX → `.nk`); parity tests in Python |
| **[Binary .nk Format](docs/NK_FORMAT.md)** | Single-file models — Python packager + C++ loader |
| **[Python packager](python/README.md)** | `python -m netkit convert` (ONNX → `.nk`), `aot` (embed `.nk` in C/C++) |
| **[Testing](docs/TESTING.md)** | Regression suites, Make targets, manual CI (`workflow_dispatch`) |
| **[C API Reference](docs/c-api.md)** | `netkit.h` (C23) |
| **[C++ API Reference](docs/cpp-api.md)** | Headers in `include/` (C++26) |
| **[API Parity Policy](docs/API_PARITY.md)** | C ↔ C++ symbol map and contribution rules |
| **[MNIST MLP Test](docs/MNIST.md)** | Trained 784→128→10 MLP on handwritten digits |
| **[MNIST CNN Test](docs/MNIST_CNN.md)** | Tutorial-style conv+pool CNN on MNIST |
| **[ResNet-18](docs/RESNET18.md)** | Fused BasicBlock + full ResNet-18 backbone fixture |
| **[ConvNeXt V2](docs/CONVNEXTV2.md)** | Fused block + LayerNorm2d + full Atto backbone fixture |
| **[MobileNetV4](docs/MOBILENETV4.md)** | Fused UIB block + full MNv4-Conv-Small backbone fixture |
| **[MLP Background](docs/nn.md)** | Optional theory (training/backprop); netkit is inference-only |

## Language standards

| Code | Standard | Role |
|------|----------|------|
| C++ engine | **C++26** | All implementation, primary API, CLI, C++ tests |
| C API | **C23** | `netkit.h` bridge + `tests/test_c_api.c` |

Application code is C++26. C23 is limited to the C header, the `extern "C"` bridge (`src/netkit_api.cpp`), and the C API test harness.

## Features

- **Dual API** — C23 (`netkit.h`) and C++26 (native headers)
- **CLI** — `test`, `run`, and `inspect` commands for desktop development
- **MLP & CNN** — conv (with padding), max/avg pool, batch norm, flatten, dense; `.nk` loading
- **Arena allocator** — Bump-pointer memory with aligned allocation (no heap in layer paths)
- **Regression tests** — 86 embedded `.nk` cases (C++/C) plus Python AOT/unit tests via `make test`; full ONNX parity (82) and backbone tests via `make test-full`
- **GitHub Actions CI** — manual `workflow_dispatch` only (`gh workflow run ci.yml`)
- **Embedded smoke** — MCU/MPU + `NETKIT_ARCH` + CMSIS bring-up harness on host (`test_mlp`, `cnn_4x4_single`; `make test-embedded-smoke-matrix`; local only)
- **Float32 inference** — all tensors, weights, and math use IEEE-754 single precision (`float`)
- **Optional CMSIS backends** — CMSIS-NN (MCU + Cortex-M): conv, depthwise, pool, FC, BN, activations, GELU; CMSIS-DSP: MatMul, add/mul, LayerNorm, GRN; reference fallback always linked ([KERNELS.md](docs/KERNELS.md))

## Quick start

```bash
make              # build netkit CLI + libnetkit.a (NETKIT_TARGET=cpu)
make test         # default: C++/C embedded regression + fast Python (~1 min)
make test-full    # full suite incl. ONNX parity (82) + backbone tests (manual / pre-release)
./netkit run models/test_mlp.nk --input 1,2
make example-cpp    # C++26 usage demo
make example-c      # C23 usage demo
```

See [Getting Started](docs/GETTING_STARTED.md) for full details.

## CLI

Full reference: [docs/CLI.md](docs/CLI.md)

```bash
./netkit help                              # print usage (-h / --help also work)
./netkit test                              # C++ API regression suite
./netkit run models/test_mlp.nk --input 1,2
./netkit inspect models/test_mlp.nk      # boxed network summary
./netkit inspect models/test_mlp.nk --full   # + arena sizing after forward
```

## Examples

| Demo | Language | Build | Run |
|------|----------|-------|-----|
| `examples/infer_cpp.cpp` | C++26 | `make example-cpp` | `./examples/infer_cpp models/test_mlp.nk 1 2` |
| `examples/infer_c.c` | C23 | `make example-c` | `./examples/infer_c models/test_mlp.nk 1 2` |

Both load a `.nk` model and print input/output tensors (stack buffers up to `NK_MAX_CASE_FLOATS` / 16384 floats, same as CLI). See [Getting Started](docs/GETTING_STARTED.md) for minimal code snippets and linking.

## Project structure

```
netkit/
├── include/
│   ├── netkit.h            # C23 public API
│   ├── arena.hpp           # Memory management
│   ├── tensor.hpp          # Tensor definitions
│   ├── mlp.hpp / cnn.hpp   # Network abstractions
│   ├── nk_loader.hpp       # .nk model loader
│   └── ...
├── src/                    # C++26 implementation
├── python/netkit/          # ONNX → .nk packager
├── examples/
│   ├── infer_cpp.cpp       # C++26 usage example
│   └── infer_c.c           # C23 usage example
├── tests/
│   ├── test_c_api.c        # C23 API regression tests
│   └── embedded_smoke.c    # MCU/MPU lean-runtime smoke (no CLI)
├── models/                 # bundled .nk models + matching .onnx sources
├── tools/
│   ├── export_mnist_mlp.py
│   ├── export_mnist_cnn.py
│   └── run_embedded_smoke.sh       # MCU/MPU + CMSIS host smoke (local)
└── docs/                   # Guides and API reference
    ├── TESTING.md
    ├── GETTING_STARTED.md
    ├── NK_FORMAT.md
    ├── c-api.md / cpp-api.md
    └── API_PARITY.md
```

## Model files

| File | Purpose |
|------|---------|
| `model.nk` | Single-file model (architecture + float32 weights) |
| `model.onnx` | Source graph for `python -m netkit convert` |
| Embedded tests (optional) | Regression cases in `.nk` `TCAS` section — see [NK_FORMAT.md](docs/NK_FORMAT.md) |

Regenerate `.nk` from ONNX: `make export-nk`. Arena buffer size is **not** in the model file — you provide a caller-owned buffer sized for weights + ping-pong activations. See [docs/ARENA.md](docs/ARENA.md).

Format spec: [docs/NK_FORMAT.md](docs/NK_FORMAT.md). Regression tests: [docs/TESTING.md](docs/TESTING.md).

## Building

### Requirements

- C++26 compiler (clang++ 17+, g++ 14+)
- C23 compiler for C examples (clang 17+, gcc 14+)
- GNU Make (primary); CMake 3.16+ (optional)

### Targets

```bash
make              # netkit CLI + libnetkit.a (NETKIT_TARGET=cpu, heap arena default)
make NETKIT_TARGET=mcu lib   # lean embedded runtime
make NETKIT_TARGET=mpu lib   # lean embedded runtime
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all   # desktop, static arena
make build-all    # cpu: netkit + examples + C API test binary
make test         # default: C++/C embedded regression + fast Python (~1 min)
make test-full    # full suite incl. ONNX parity (82) + backbone tests (manual / pre-release)
make test-cpp     # C++ embedded .nk cases only (86)
make test-c       # C API regression only
make test-python  # fast Python subset (same as in make test)
make test-python-full  # ONNX parity (82) + AOT compile tests (requires libnetkit.a)
make test-embedded-smoke-matrix  # MCU/MPU + NETKIT_ARCH + CMSIS (host smoke; local only)
make example-cpp  # C++26 usage demo
make example-c    # C23 usage demo
make cmsis-init   # fetch CMSIS-NN + CMSIS-DSP (optional backends)
make export-mnist # regenerate MNIST MLP model (requires PyTorch: pip install -e "python[train]")
make export-mnist-cnn # regenerate MNIST CNN model (requires PyTorch)
make clean
make rebuild
```

### Optional CMSIS backends and architecture

```bash
make cmsis-init
make NETKIT_CMSIS_DSP=1 test-cpp                   # CMSIS-DSP (desktop / MPU / MCU)
make NETKIT_ARCH=CM4 NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib   # MCU + NN

# Host smoke before on-device bring-up (7 profiles; sets NETKIT_HOST_SMOKE=1)
make cmsis-init
make test-embedded-smoke-matrix
```

Set **`NETKIT_ARCH`** when cross-compiling (e.g. `CM4`, `M33`, `M55`, `NEON`). Leave unset for native desktop builds. Full flag table: [docs/BUILD_TARGETS.md](docs/BUILD_TARGETS.md).

### CMake (optional)

```bash
cmake -B cmake-build
cmake --build cmake-build
./cmake-build/netkit test
```

CMake options mirror Make: `NETKIT_TARGET`, `NETKIT_ARCH`, `NETKIT_CMSIS_NN`, `NETKIT_CMSIS_DSP`, arena flags.

See [docs/BUILD_TARGETS.md](docs/BUILD_TARGETS.md) for CPU vs MCU vs MPU builds and [docs/TESTING.md](docs/TESTING.md) for the regression layout.

## Testing

Full guide: [docs/TESTING.md](docs/TESTING.md)

```bash
make test       # default: C++/C embedded cases + fast Python
make test-full  # full suite incl. ONNX parity (manual)
make test-cpp   # ./netkit test
make test-c     # ./tests/test_c_api
make test-python
make test-python-full
make test-embedded-smoke-matrix   # lean MCU/MPU profiles (see docs/TESTING.md)
```

| Suite | Language | Entry point | Cases |
|-------|----------|-------------|-------|
| C++ embedded | C++26 | `./netkit test` → `src/test.cpp` | 86 (19 hand + 20 MNIST + 17 op matrix + 27 ONNX import extensions + 1 MobileNetV4 Small + 1 ResNet-18 + 1 ConvNeXt V2-Atto) |
| C API | C23 | `tests/test_c_api.c` | Same 86 via `nk_run_all_tests()` + API smoke tests (`nk_run_model_tests` on composite/import fixtures) |
| ONNX parity | Python | `python/tests/test_onnx_parity.py` | 82 (.nk vs ONNX Runtime on bundled sidecars) |
| Timm backbone parity | Python | `test_torch_backbone_pack.py`, `test_torch_backbone_runtime_parity.py` | Pack timm ResNet-18 / ConvNeXt V2-Atto / MobileNetV4 Small; C++ runtime vs PyTorch (see [TESTING.md](docs/TESTING.md)) |
| AOT compile | Python | `python/tests/test_aot_compile.py` | Generates C/C++ from `.nk`, builds, runs vs reference |
| Embedded smoke | C23 | `tests/embedded_smoke.c` | `test_mlp`, `cnn_4x4_single` load/run on 7 MCU/MPU host profiles (`make test-embedded-smoke-matrix`; local only) |

CI is **manual only** (`gh workflow run ci.yml`). See [TESTING.md](docs/TESTING.md).

Regression cases are embedded in each bundled `.nk` file ([NK_FORMAT.md](docs/NK_FORMAT.md)).  
MNIST MLP: [MNIST.md](docs/MNIST.md). MNIST CNN: [MNIST_CNN.md](docs/MNIST_CNN.md).

## Design principles

See [PHILOSOPHY.md](docs/PHILOSOPHY.md) for the full narrative. In brief:

- **Phase 1 (today)** — Interpreter-style C++ runtime: load `.nk` (file or embedded memory), execute layer graph with generic kernels; Python packager converts ONNX → `.nk` and can AOT-embed `.nk` bytes into C/C++26 or C23 source
- **Phase 2 (planned)** — Python packager optimizations: fusion, layout, quantization-aware export
- **Lightweight** — Standard C/C++ only, no external dependencies in the engine
- **Memory-conscious** — Arena bump allocator; target-specific defaults (CPU 4 MiB / MCU 64 KiB / MPU 128 KiB)
- **Single-threaded** — Sequential forward pass
- **Inference-only** — No training

## Roadmap

**Phase 1 (today):** float32 interpreter runtime — `.nk` load, MLP/CNN + fused blocks (ResNet, MobileNet, ConvNeXt), depthwise conv, asymmetric padding, residual adds, optional CMSIS-NN/DSP with reference fallback ([KERNELS.md](docs/KERNELS.md)).

**Phase 2 (planned):**

- **Numeric types:** float16, int16, int8, int4 ([DATATYPES.md](docs/DATATYPES.md))
- **Packager:** quantized `.nk` export, broader fusion ([PHILOSOPHY.md](docs/PHILOSOPHY.md))
- **Import / runtime:** broader ONNX op coverage (e.g. non-depthwise grouped conv), int8 inference kernels

## License

MIT — see [LICENSE](LICENSE).

## Contributing

- C++ sources: C++26
- C sources and `netkit.h`: C23
- All tests must pass (`make test`; run `make test-full` before release)
- Update docs when changing public API
- **New C++ public API requires a matching C entry in `netkit.h`** — see [API_PARITY.md](docs/API_PARITY.md)
