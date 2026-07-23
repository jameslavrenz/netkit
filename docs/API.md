# API Overview

netkit is a **multi-modal inference engine** (voice, image, vision) with an **embedded-first** design for **MCUs, MPUs, and NPUs**. It exposes two language interfaces over the same **C++26** engine — a type-safe **C++26 API** and a **C23** mirror. **Float32** and **int8** inference are complete on Arm MCU/MPU, Espressif MCU, RISC-V MCU (NMSIS-NN), RISC MPU (XNNPACK), and host cpu — [STATUS.md](STATUS.md).

**Deployment:** use the **`NkOpsResolver` interpreter** (load `.nk`, runtime layer dispatch) or the **compiled path** (AOT **lower** by default — static kernel / plan chain + weight arrays; `--no-lower` for interpreter embed). Both share the same kernels — [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled). New users start with [GETTING_STARTED.md](GETTING_STARTED.md).

| API | Header | Language | Use when |
|-----|--------|----------|----------|
| **C API** | `include/netkit.h` | C23 | Embedded firmware, FFI, minimal dependencies at the call site |
| **C++ API** | `include/*.hpp` | C++26 | Application code, tests, extending layers and ops |

Both APIs share:

- Bump-pointer **arena** memory management (MCU: static buffer only — no heap; MPU/CPU may use heap arena backing)
- **`.nk`** single-file model loading (MCU: prefer flash/buffer; mmap forbidden)
- **MLP** and **CNN** forward-only inference (conv with symmetric padding, max/avg pool, batch norm, flatten, dense, fused blocks, YOLOX)
- **NHWC** tensor layout for convolutions
- **Float32 and int8** today — float16, int16, int4 planned ([DATATYPES.md](DATATYPES.md)); C uses `nk_model_run` vs `nk_model_run_int8`

Core inference, loading, tensor/ops, MLP/CNN construction (including FeatureTap / PAFPN), regression, and CLI entry points have documented C equivalents — see [API_PARITY.md](API_PARITY.md). Some C++ helpers (block introspection, op trimming, timed forward) remain C++-only.

**MCU peers:** NUCLEO-F446RE **int8** CNN/DS-CNN vs TFLM and microTVM (CMSIS-NN / reference); XIAO ESP32C3 **int8** CNN/DS-CNN vs TFLM (ESP-NN on + off; interpreter embed); XIAO ESP32-S3 + ESP32-P4-Function-EV **int8** + **float32** CNN/DS-CNN vs TFLM (float peers: lowered AOT — [KNOWN_ISSUES KI-001](KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device) on **P4 and S3**). Gallery: [README.md](../README.md#peer-benchmarks-mcu--mpu--cpu).

## Documentation map

| Document | Contents |
|----------|----------|
| [PHILOSOPHY.md](PHILOSOPHY.md) | Interpreter vs compiled deployment; Phase 1 runtime vs Phase 2 packager; memory and roadmap |
| [STATUS.md](STATUS.md) | Dtype + platform maturity; recent peer-bench results |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Open bugs, anomalies, deferred follow-ups |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Clone, build, CLI, integrate C/C++ |
| [BUILD_TARGETS.md](BUILD_TARGETS.md) | `NETKIT_TARGET` profiles, arena flags, backend defaults |
| [PLATFORMS.md](PLATFORMS.md) | Per-device configuration (cpu / Arm / RISC / Espressif) |
| [CLI.md](CLI.md) | `netkit test`, `run`, `inspect`, help |
| [ARENA.md](ARENA.md) | Bump allocator, sizing, alignment |
| [DATATYPES.md](DATATYPES.md) | Float32 + int8 today; float16/int16/int4 roadmap |
| [NK_FORMAT.md](NK_FORMAT.md) | `.nk` overview + embedded tests |
| [NK_FILE_SPECIFICATION.md](NK_FILE_SPECIFICATION.md) | Byte-level `.nk` specification and inspection |
| [TESTING.md](TESTING.md) | Regression suites, Make targets, manual CI |
| [MNIST.md](MNIST.md) / [MNIST_CNN.md](MNIST_CNN.md) | Trained MNIST bundles |
| [API_PARITY.md](API_PARITY.md) | C ↔ C++ symbol map |
| [KERNELS.md](KERNELS.md) | CRTP kernel backends (CMSIS / ESP / NMSIS / XNNPACK) |
| [c-api.md](c-api.md) | Full C23 reference |
| [cpp-api.md](cpp-api.md) | Full C++26 reference |

## Build targets and memory defaults

| Target | Command | CLI | `NK_ARENA_DEFAULT_CAPACITY` |
|--------|---------|-----|----------------------------|
| CPU | `make` | Yes | **64 MiB** |
| MCU_ARM | `make NETKIT_TARGET=mcu_arm lib` | No | **64 KiB** |
| MPU_ARM | `make NETKIT_TARGET=mpu_arm lib` | No | **64 MiB** |
| MCU_RISC | `make NETKIT_TARGET=mcu_risc NETKIT_ARCH=N300 lib` | No | **64 KiB** |
| MPU_RISC | `make NETKIT_TARGET=mpu_risc lib` | No | **64 MiB** |
| MCU_ESP | `make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 lib` | No | **64 KiB** |

**Arena backing flags** (see [BUILD_TARGETS.md](BUILD_TARGETS.md)):

| Flag | Effect |
|------|--------|
| *(CPU default)* | Heap arena — `nk_arena_init_heap` / CLI default 64 MiB |
| `NETKIT_GLOBAL_ARENA=1` (CPU) | Static/global arena only |
| `NETKIT_HEAP_ARENA=1` (**MPU only**) | Compile in optional heap arena API (forbidden on MCU) |

## Quick comparison

### Load and run (C23)

```c
alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
nk_arena_t arena;
nk_model_t model;
nk_arena_init(&arena, memory, sizeof(memory));
nk_model_load("models/test_mlp.nk", &arena, &model);
nk_model_run(&model, &arena, input, n, output, cap, &out_n);
```

Full example: [`examples/infer_c.c`](../examples/infer_c.c)

### Load and run (C++26)

```cpp
Arena arena;
arena.init(buffer, sizeof(buffer));
MLPNetwork* net = nullptr;
std::array<uint32_t, kMaxTensorRank> shape{};
uint32_t rank = 0;
NkLoader::LoadMLP("models/test_mlp.nk", arena, net, shape, rank);
net->forward(input, output, arena);
```

Full example: [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp)

## CLI (CPU build only)

The `netkit` binary is a desktop development tool (C++26). See [CLI.md](CLI.md).

| Command | Description |
|---------|-------------|
| `netkit test` | Run embedded `.nk` regression tests (89 cases) |
| `netkit run <model.nk> --input a,b,c` | Single inference |
| `netkit inspect <model.nk>` | Boxed network summary (`--full` for arena sizing) |
| `netkit help`, `-h`, `--help` | Print CLI usage |

## Language standards

| Code | Standard | Role |
|------|----------|------|
| C++ engine + headers | **C++26** | Implementation, primary API, CLI |
| C API | **C23** | `netkit.h`, examples, firmware integration |

## Linking

```bash
make lib          # or make (CPU) for CLI + lib
clang -std=c23 -Iinclude -c app.c -o app.o
clang++ -std=c++26 -o app app.o libnetkit.a
```

Build the library with `make lib`.

## Error handling

| API | Pattern |
|-----|---------|
| C | Functions return `nk_status_t`; call `nk_last_error()` for detail |
| C++ | `NkLoader::LoadResult` with `LoadStatus` and `message` |

## Memory model

Full guide: [ARENA.md](ARENA.md). Data types: [DATATYPES.md](DATATYPES.md).

Both APIs use a **caller-provided arena buffer** (or heap backing when `NETKIT_ARENA_HEAP` is enabled). Size is **not** in the model file — it depends on weights plus ping-pong activation buffers at load.

**Defaults:** MCU **64 KiB**; CPU and MPU **64 MiB**. CLI override: `./netkit --arena <size>`.

**Sizing:** `./netkit inspect <model.nk> --full` or `nk_inspect_model()` → `arena_bytes_after_forward` → add headroom.

**Alignment:** `alignas(max_align_t)` backing buffers; `nk_arena_alloc(arena, size, alignment)` with power-of-two alignment.

netkit implements its own minimal arena rather than linking [memkit](https://github.com/NetKit-Labs/memkit); alignment behavior matches memkit’s bump policy.

## Supported model format

Runtime models are **`.nk` v3** single files — [NK_FORMAT.md](NK_FORMAT.md).

Convert ONNX → `.nk` with `python -m netkit convert` or `make export-nk`. Supported ONNX ops: [ONNX.md](ONNX.md).

## Optional CMSIS / ESP-NN / NMSIS-NN / XNNPACK backends

Backends are **not** inferred from `NETKIT_ARCH` alone — set flags explicitly or use **profile defaults** (`cpu`: XNNPACK on; `mcu_arm`: CMSIS-NN; `mcu_esp`: ESP-NN; `mcu_risc`: NMSIS-NN; `mpu_arm` / `mpu_risc`: XNNPACK). Platform maturity: [STATUS.md](STATUS.md).

| Backend | When enabled | Targets |
|---------|----------------|---------|
| **CMSIS-NN** | `NETKIT_CMSIS_NN=1` + `NETKIT_TARGET=mcu_arm` + Cortex-M `NETKIT_ARCH` | Arm MCU firmware (CM4, M33, …) |
| **ESP-NN** | `NETKIT_ESP_NN=1` + `NETKIT_TARGET=mcu_esp` + `NETKIT_ARCH=ESP32*` | Espressif MCU (ESP32 / S3 / C3 / C6 / P4) |
| **NMSIS-NN** | `NETKIT_NMSIS_NN=1` + `NETKIT_TARGET=mcu_risc` + Nuclei/RV32 `NETKIT_ARCH` | RISC-V MCU firmware (N300, RV32IMAC, …) |
| **XNNPACK** | `NETKIT_XNNPACK=1` | `cpu` + any MPU LayerFast (**forbidden on MCU**) |
| **Generic / reference** | always linked | Fallback everywhere; float LayerFast on ESP/NMSIS MCU; int8 when accel off |

CMSIS-DSP is not used. Backend selection is compile-time CRTP — see [KERNELS.md](KERNELS.md), [BUILD_TARGETS.md](BUILD_TARGETS.md#cmsis-backends), and [PLATFORMS.md](PLATFORMS.md).

## Testing

Both API test suites run **89 embedded `.nk` regression cases** on CPU builds — [TESTING.md](TESTING.md). Full Python ONNX parity (**82** cases) runs via `make test-full` (`make test-python-full`). MCU/MPU bring-up: `make test-embedded-smoke-matrix` (`test_mlp`, `cnn_4x4_single` on seven host profiles).

```bash
make test       # default: C++ then C then fast Python (cpu only)
make test-full  # full suite incl. ONNX parity (manual)
make test-cpp   # ./netkit test
make test-c     # ./tests/test_c_api
make test-python
make test-python-full
make test-embedded-smoke-matrix   # MCU/MPU + CMSIS host smoke
```
