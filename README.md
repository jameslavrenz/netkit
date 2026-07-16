# netkit — Neural Network Kit

netkit is a **multi-modal inference engine** (image / vision today; voice next) with an **embedded-first** design for **MCUs, MPUs, and NPUs**. Primary API is **C++26**; firmware and FFI use a matching **C23** API. Develop on the desktop, then deploy the lean runtime to embedded targets. Companion to [memkit](https://github.com/NetKit-Labs/memkit) for memory management.

**Status:** **Float32** and **int8** inference are **complete** on **Arm MCU**, **Arm MPU**, and **cpu** (RISC MCU on fast generic kernels; RISC MPU via XNNPACK). The inference engine is **peer-benched end-to-end** across **MCU** (NUCLEO-F446RE vs TFLM and microTVM), **MPU** (Raspberry Pi Zero 2 W vs TF Lite), and **CPU** (Apple M4 vs TF Lite) for latency and flash/RAM — see [docs/STATUS.md](docs/STATUS.md) and the gallery below. **YOLOX** detection (MobileNetV4 + PAFPN) is supported and latency-competitive on host; **detector accuracy still needs more training / calibration**. Next: voice fixtures and broader quantization.

Models are loaded from binary **`.nk`** files (single-file architecture + weights). Convert from ONNX with `python -m netkit convert`, or embed a `.nk` in firmware with `python -m netkit aot`.

Use netkit as an **`NkOpsResolver` interpreter** (load `.nk`, dispatch layers at runtime) for development and flexible deployment, or **compile for maximum speed** (AOT embed, packager graph optimizations, trimmed op tables, CMSIS backends) for production firmware. See [docs/PHILOSOPHY.md](docs/PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

## Peer benchmarks (MCU · MPU · CPU)

Fair A/B vs TFLM + microTVM (MCU) and TF Lite (MPU/CPU). Full tables and methodology: [docs/STATUS.md](docs/STATUS.md). Suite infographics:

| Int8 suite | Float32 suite |
|------------|---------------|
| ![netkit int8 peer suite](benchmark/linkedin/netkit_linkedin_int8_suite.png) | ![netkit float32 peer suite](benchmark/linkedin/netkit_linkedin_float32_suite.png) |

Raw logs and scripts: [benchmark/README.md](benchmark/README.md), `benchmark/linkedin/`.

## Documentation

| Guide | Description |
|-------|-------------|
| **[Philosophy](docs/PHILOSOPHY.md)** | Interpreter vs compiled deployment; Phase 1 runtime vs Phase 2 packager |
| **[Status](docs/STATUS.md)** | Dtype + platform maturity; MCU / MPU / CPU peer-bench results |
| **[Getting Started](docs/GETTING_STARTED.md)** | Build, test, CLI, and first inference for new users |
| **[API Overview](docs/API.md)** | C vs C++ APIs, linking, memory model |
| **[Build Targets](docs/BUILD_TARGETS.md)** | CPU / MCU / MPU flags and arena defaults |
| **[CLI Reference](docs/CLI.md)** | `test`, `run`, and `inspect` (CPU build) |
| **[Arena Memory](docs/ARENA.md)** | Bump allocator — sizing, alignment, reset |
| **[Data Types](docs/DATATYPES.md)** | Float32 + int8 (cpu / Arm / RISC); float16 / int16 / int4 roadmap |
| **[ONNX Import](docs/ONNX.md)** | Python packager (ONNX → `.nk`); parity tests in Python |
| **[Binary .nk Format](docs/NK_FORMAT.md)** | Single-file models — overview |
| **[`.nk` File Specification](docs/NK_FILE_SPECIFICATION.md)** | Byte-level `.nk` layout, offsets, hex inspection |
| **[Python packager](python/README.md)** | `python -m netkit convert` (ONNX → `.nk`), `aot` (embed `.nk` in C/C++) |
| **[Testing](docs/TESTING.md)** | Regression suites, Make targets, CI on push/PR + manual full suite |
| **[MNIST benchmarks](benchmark/README.md)** | Host invoke latency + per-op profiles: netkit vs TFLM |
| **[Peer-suite infographics](benchmark/linkedin/)** | MCU / MPU / CPU float32 + int8 A/B images |
| **[NUCLEO-F446RE firmware](boards/nucleo-f446re/README.md)** | On-device MNIST MLP f32 benchmark (CMSIS-NN / reference, lowered AOT) |
| **[NUCLEO-F446RE CNN int8](boards/nucleo-f446re-cnn-int8/README.md)** | On-device MNIST CNN int8 (CMSIS-NN / reference, interpreter embed) |
| **[NUCLEO-F446RE MLP int8](boards/nucleo-f446re-mlp-int8/README.md)** | On-device MNIST MLP int8 benchmark (CMSIS-NN, interpreter embed) |
| **[NUCLEO-F446RE TFLM CNN int8](boards/nucleo-f446re-tflm-cnn-int8/README.md)** | Same CNN int8 vectors via TFLite Micro (comparison baseline) |
| **[NUCLEO-F446RE microTVM CNN int8](boards/nucleo-f446re-tvm-cnn-int8/README.md)** | Same CNN int8 via microTVM AOT (CMSIS-NN / pure C) |
| **[NUCLEO-F446RE TFLM MLP int8](boards/nucleo-f446re-tflm-mlp-int8/README.md)** | Same MLP int8 vectors via TFLite Micro (comparison baseline) |
| **[C API Reference](docs/c-api.md)** | `netkit.h` (C23) |
| **[C++ API Reference](docs/cpp-api.md)** | Headers in `include/` (C++26) |
| **[API Parity Policy](docs/API_PARITY.md)** | C ↔ C++ symbol map and contribution rules |
| **[MNIST MLP Test](docs/MNIST.md)** | Trained 784→128→10 MLP on handwritten digits |
| **[MNIST CNN Test](docs/MNIST_CNN.md)** | Tutorial CNN + depthwise-separable peer on MNIST |
| **[ResNet-18](docs/RESNET18.md)** | Fused BasicBlock + full ResNet-18 backbone fixture |
| **[ConvNeXt V2](docs/CONVNEXTV2.md)** | Fused block + LayerNorm2d + full Atto backbone fixture |
| **[MobileNetV4](docs/MOBILENETV4.md)** | Fused UIB block + full MNv4-Conv-Small backbone fixture |
| **[YOLOX detector](docs/YOLOX.md)** | YOLOX PAFPN + decoupled head on MobileNetV4 (latency ready; accuracy needs more training) |
| **[MLP Background](docs/nn.md)** | Optional theory (training/backprop); netkit is inference-only |

## Language standards

| Code | Standard | Role |
|------|----------|------|
| C++ engine | **C++26** | All implementation, primary API, CLI, C++ tests |
| C API | **C23** | `netkit.h` bridge + `tests/test_c_api.c` |

Application code is C++26. C23 is limited to the C header, the `extern "C"` bridge (`src/netkit_api.cpp`), and the C API test harness.

## Features

- **Multi-modal (image / vision first)** — classification and detection fixtures today; voice planned (embedded-first for MCU, MPU, NPU)
- **Peer-benched inference** — float32 + int8 A/B on Arm MCU, Arm MPU, and CPU vs TFLM / TF Lite ([STATUS.md](docs/STATUS.md))
- **Interpreter or compiled** — `NkOpsResolver` + `.nk` load for flexibility; AOT embed + packager optimizations + trimmed ops for production speed ([PHILOSOPHY.md](docs/PHILOSOPHY.md#deployment-modes-interpreter-or-compiled))
- **Dual API** — C23 (`netkit.h`) and C++26 (native headers, modern patterns and type safety)
- **CLI** — `test`, `run`, and `inspect` commands for desktop development
- **MLP & CNN** — conv (with padding), depthwise, max/avg pool, batch norm, flatten, dense; fused ResNet / MobileNetV4 / ConvNeXt / YOLOX blocks; `.nk` loading
- **Detection** — YOLOX decoupled head + PAFPN on MobileNetV4 (latency ready; accuracy needs more training)
- **Arena allocator** — Bump-pointer memory; **MCU: static arena only — no heap ever**
- **Regression tests** — 88 embedded `.nk` cases (C++/C) plus Python AOT/unit tests via `make test`; full ONNX parity (82) and backbone tests via `make test-full`
- **GitHub Actions CI** — fast suite on push/PR (`make test`); full suite manual only (`gh workflow run test-full.yml`)
- **Embedded smoke** — MCU/MPU + `NETKIT_ARCH` + CMSIS bring-up harness on host (`test_mlp`, `cnn_4x4_single`; `make test-embedded-smoke-matrix`; local only)
- **Float32 inference** — complete on cpu / MCU / MPU
- **Int8 inference** — complete end-to-end int8 I/O (MNIST CNN/MLP MCU CMSIS-NN; host/MPU XNNPACK qs8 or QuantOps; ImageNet MNv4 int8)
- **Optional backends** — CMSIS-NN (Arm MCU int8); XNNPACK (cpu + any MPU, forbidden on MCU); reference everywhere else. CMSIS-DSP is not used. ([STATUS.md](docs/STATUS.md), [KERNELS.md](docs/KERNELS.md))

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

Format overview: [docs/NK_FORMAT.md](docs/NK_FORMAT.md). Byte-level spec and inspection: [docs/NK_FILE_SPECIFICATION.md](docs/NK_FILE_SPECIFICATION.md). Regression tests: [docs/TESTING.md](docs/TESTING.md).

## Building

### Requirements

- C++26 compiler (clang++ 17+, g++ 14+)
- C23 compiler for C examples (clang 17+, gcc 14+)
- GNU Make (primary); CMake 3.16+ (optional)

### Targets

```bash
make              # netkit CLI + libnetkit.a (NETKIT_TARGET=cpu, heap arena default)
make NETKIT_TARGET=mcu_arm lib   # lean embedded runtime
make NETKIT_TARGET=mpu_arm lib   # lean embedded runtime
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all   # desktop, static arena
make build-all    # cpu: netkit + examples + C API test binary
make test         # default: C++/C embedded regression + fast Python (~1 min)
make test-full    # full suite incl. ONNX parity (82) + backbone tests (manual / pre-release)
make test-cpp     # C++ embedded .nk cases only (88)
make test-c       # C API regression only
make test-python  # fast Python subset (same as in make test)
make test-python-full  # ONNX parity (82) + AOT compile tests (requires libnetkit.a)
make test-embedded-smoke-matrix  # MCU/MPU + NETKIT_ARCH + CMSIS (host smoke; local only)
make example-cpp  # C++26 usage demo
make example-c    # C23 usage demo
make cmsis-init   # fetch CMSIS-NN + CMSIS-Core (optional backends)
make export-mnist # regenerate MNIST MLP model (requires PyTorch: pip install -e "python[train]")
make export-mnist-cnn # regenerate MNIST CNN model (requires PyTorch)
make export-mnist-cnn-int8 # quantize MNIST CNN to int8 .nk + prequantized test vectors
make flash-mnist-cnn-int8  # build + flash + UART capture on NUCLEO-F446RE
make clean
make rebuild
```

### Optional backends and architecture

Backends: **reference** + **XNNPACK** (cpu / MPU) + **CMSIS-NN** (Arm MCU int8). CMSIS-DSP is **not** used. CMSIS-NN is **opt-in** via `NETKIT_CMSIS_NN=1` (or CMake `-DNETKIT_CMSIS_NN=ON`). `NETKIT_ARCH` only sets `ARM_MATH_*` tuning flags.

**Profile defaults** (override on the command line, e.g. `make NETKIT_CMSIS_NN=0`):

| `NETKIT_TARGET` | Default CMSIS-NN | Default XNNPACK |
|-----------------|------------------|-----------------|
| `cpu` | off | on (any host ISA) |
| `mcu_arm` | on (Cortex-M `NETKIT_ARCH`; int8 production) | forbidden |
| `mpu_arm` | off | on |
| `mcu_risc` | off (forbidden) | forbidden |
| `mpu_risc` | off (forbidden) | on |

Float32 on MCU uses portable/reference kernels only — there is no plan for an optimized float32 MCU build.

```bash
make cmsis-init
make xnnpack-init                     # once, for cpu / MPU LayerFast
make test-cpp                         # cpu: XNNPACK preferred
make NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 lib   # mcu_arm: CMSIS-NN

# Host smoke before on-device bring-up (sets NETKIT_HOST_SMOKE=1)
make test-embedded-smoke-matrix
```

Set **`NETKIT_ARCH`** when cross-compiling (e.g. `CM4`, `M33`, `M55`, `NEON`). Leave unset for native desktop builds. Full flag table: [docs/BUILD_TARGETS.md](docs/BUILD_TARGETS.md).

### CMake (optional)

```bash
cmake -B cmake-build
cmake --build cmake-build
./cmake-build/netkit test
```

CMake options mirror Make: `NETKIT_TARGET`, `NETKIT_ARCH`, `NETKIT_CMSIS_NN`, arena flags.

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
| C++ embedded | C++26 | `./netkit test` → `src/test.cpp` | 88 (19 hand + 20 MNIST + 17 op matrix + 27 ONNX import extensions + 1 MobileNetV4 Small + 2 YOLOX + 1 ResNet-18 + 1 ConvNeXt V2-Atto) |
| C API | C23 | `tests/test_c_api.c` | Same 88 via `nk_run_all_tests()` + API smoke tests (`nk_run_model_tests` on composite/import fixtures) |
| ONNX parity | Python | `python/tests/test_onnx_parity.py` | 82 (.nk vs ONNX Runtime on bundled sidecars) |
| Timm backbone parity | Python | `test_torch_backbone_pack.py`, `test_torch_backbone_runtime_parity.py` | Pack timm ResNet-18 / ConvNeXt V2-Atto / MobileNetV4 Small; C++ runtime vs PyTorch (see [TESTING.md](docs/TESTING.md)) |
| AOT compile | Python | `python/tests/test_aot_compile.py` | Generates C/C++ from `.nk`, builds, runs vs reference |
| Embedded smoke | C23 | `tests/embedded_smoke.c` | `test_mlp`, `cnn_4x4_single` load/run on 7 MCU/MPU host profiles (`make test-embedded-smoke-matrix`; local only) |

CI runs the fast suite on **push to `main`** and **pull requests**; full regression is manual (`gh workflow run test-full.yml`). See [TESTING.md](docs/TESTING.md).

Regression cases are embedded in each bundled `.nk` file ([NK_FORMAT.md](docs/NK_FORMAT.md)).  
MNIST MLP: [MNIST.md](docs/MNIST.md). MNIST CNN: [MNIST_CNN.md](docs/MNIST_CNN.md).

## Design principles

See [PHILOSOPHY.md](docs/PHILOSOPHY.md) for the full narrative — including [interpreter vs compiled deployment](docs/PHILOSOPHY.md#deployment-modes-interpreter-or-compiled). In brief:

- **Interpreter or compiled** — `NkOpsResolver` + `.nk` load for flexibility; AOT embed + packager optimizations + trimmed ops for production speed
- **Phase 1 (today)** — Float32 and int8 inference complete; MCU/MPU/CPU peer benches done; ONNX → `.nk` packager; desktop CLI; CMSIS-NN MCU int8; YOLOX path in (accuracy training open)
- **Phase 2 (planned)** — Broader quantization (float16, int16, int4), fusion, layout, NPU offload; YOLOX accuracy / voice fixtures
- **Lightweight** — Standard C/C++ only, no external dependencies in the engine
- **Memory-conscious** — Arena bump allocator; target-specific defaults (MCU 64 KiB / MPU·CPU 64 MiB; overridable)
- **Single-threaded** — Sequential forward pass
- **Inference-only** — No training

## Roadmap

**Phase 1 (today):** **Float32** and **int8** inference are **complete** for image/vision — `.nk` load or AOT embed, MLP/CNN + fused blocks (ResNet, MobileNetV4, ConvNeXt), depthwise conv, asymmetric padding, and **YOLOX** detection (PAFPN). **Peer A/B benchmarking of the inference engine is finished** across **Arm MCU** (NUCLEO-F446RE vs TFLM), **Arm MPU** (Pi Zero 2 W vs TF Lite), and **CPU** (vs TF Lite). **Arm MCU** uses CMSIS-NN for int8 (no heap; static arena); **MPU / cpu** use XNNPACK; **RISC MCU** stays on fast generic kernels until ISA-tuned kernels exist — [STATUS.md](docs/STATUS.md), [KERNELS.md](docs/KERNELS.md).

**Phase 2 (next):**

- **YOLOX / detection accuracy** — more training and calibration (runtime and latency path already land)
- **Voice modality** fixtures
- **Numeric types:** float16, int16, int4; broader **int8** model coverage ([DATATYPES.md](docs/DATATYPES.md))
- **Packager:** fusion, layout, target-specific profiles ([PHILOSOPHY.md](docs/PHILOSOPHY.md))
- **Import / runtime:** broader ONNX op coverage, NPU offload paths

## Repository topics

GitHub topics for discoverability: `embedded`, `embedded-systems`, `inference`, `multimodal`, `computer-vision`, `edge-ai`, `machine-learning`, `neural-network`, `onnx`, `firmware`, `aot`, `cmsis`, `mcu`, `mpu`, `cpp26`, `c23`.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

- C++ sources: C++26
- C sources and `netkit.h`: C23
- All tests must pass (`make test`; run `make test-full` before release)
- Update docs when changing public API
- **New C++ public API requires a matching C entry in `netkit.h`** — see [API_PARITY.md](docs/API_PARITY.md)
