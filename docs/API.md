# API Overview

netkit exposes two language interfaces over the same **C++26 inference engine**. For the product vision (Phase 1 interpreter runtime, Phase 2 packager optimizations), see [PHILOSOPHY.md](PHILOSOPHY.md). New users start with [GETTING_STARTED.md](GETTING_STARTED.md).

| API | Header | Language | Use when |
|-----|--------|----------|----------|
| **C API** | `include/netkit.h` | C23 | Embedded firmware, FFI, minimal dependencies at the call site |
| **C++ API** | `include/*.hpp` | C++26 | Application code, tests, extending layers and ops |

Both APIs share:

- Bump-pointer **arena** memory management (no heap in layer code paths on MCU/MPU)
- **`.nk`** single-file model loading
- **MLP** and **CNN** forward-only inference (conv with symmetric padding, max/avg pool, batch norm, flatten, dense)
- **NHWC** tensor layout for convolutions
- **Float32 only (today)** — float16, int16, int8, int4 planned ([DATATYPES.md](DATATYPES.md))

Every stable C++ public symbol has a documented C equivalent except desktop-only diagnostics — see [API_PARITY.md](API_PARITY.md).

## Documentation map

| Document | Contents |
|----------|----------|
| [PHILOSOPHY.md](PHILOSOPHY.md) | Phase 1 runtime vs Phase 2 packager; memory and roadmap |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Clone, build, CLI, integrate C/C++ |
| [BUILD_TARGETS.md](BUILD_TARGETS.md) | `NETKIT_TARGET=cpu\|mcu\|mpu`, arena flags, defaults |
| [CLI.md](CLI.md) | `netkit test`, `run`, `inspect`, help |
| [ARENA.md](ARENA.md) | Bump allocator, sizing, alignment |
| [DATATYPES.md](DATATYPES.md) | Float32 today; float16/int roadmap |
| [NK_FORMAT.md](NK_FORMAT.md) | `.nk` layout + embedded tests |
| [TESTING.md](TESTING.md) | Regression suites, Make targets, CI |
| [MNIST.md](MNIST.md) / [MNIST_CNN.md](MNIST_CNN.md) | Trained MNIST bundles |
| [API_PARITY.md](API_PARITY.md) | C ↔ C++ symbol map |
| [KERNELS.md](KERNELS.md) | CRTP kernel backends and CMSIS dispatch |
| [c-api.md](c-api.md) | Full C23 reference |
| [cpp-api.md](cpp-api.md) | Full C++26 reference |

## Build targets and memory defaults

| Target | Command | CLI | `NK_ARENA_DEFAULT_CAPACITY` |
|--------|---------|-----|----------------------------|
| CPU | `make` | Yes | **4 MiB** |
| MCU | `make NETKIT_TARGET=mcu lib` | No | **64 KiB** |
| MPU | `make NETKIT_TARGET=mpu lib` | No | **128 KiB** |

**Arena backing flags** (see [BUILD_TARGETS.md](BUILD_TARGETS.md)):

| Flag | Effect |
|------|--------|
| *(CPU default)* | Heap arena — `nk_arena_init_heap` / CLI model-sized buffers |
| `NETKIT_GLOBAL_ARENA=1` (CPU) | Static/global arena only |
| `NETKIT_HEAP_ARENA=1` (MCU/MPU) | Compile in optional heap arena API |

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
| `netkit test` | Run embedded `.nk` regression tests (73 cases) |
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

**Defaults:** CPU **4 MiB** constant; MCU **64 KiB**; MPU **128 KiB**. CLI/regression on CPU use model-sized heap (64 KiB / 2 MiB / 4 MiB).

**Sizing:** `./netkit inspect <model.nk> --full` or `nk_inspect_model()` → `arena_bytes_after_forward` → add headroom.

**Alignment:** `alignas(max_align_t)` backing buffers; `nk_arena_alloc(arena, size, alignment)` with power-of-two alignment.

netkit implements its own minimal arena rather than linking [memkit](https://github.com/jameslavrenz/memkit); alignment behavior matches memkit’s bump policy.

## Supported model format

Runtime models are **`.nk` v3** single files — [NK_FORMAT.md](NK_FORMAT.md).

Convert ONNX → `.nk` with `python -m netkit convert` or `make export-nk`. Supported ONNX ops: [ONNX.md](ONNX.md).

## Optional CMSIS backends

| Backend | When enabled | Targets |
|---------|----------------|---------|
| **CMSIS-NN** | `NETKIT_CMSIS_NN=1` + `NETKIT_TARGET=mcu` + Cortex-M `NETKIT_ARCH` | MCU firmware (CM4, M33, …) |
| **CMSIS-DSP** | `NETKIT_CMSIS_DSP=1` | Desktop, MCU, MPU |

On **cpu** or **mpu**, `NETKIT_CMSIS_NN=1` prints a Make warning and is ignored — reference kernels (and optional CMSIS-DSP) apply. Backend selection is compile-time CRTP — see [KERNELS.md](KERNELS.md) and [BUILD_TARGETS.md](BUILD_TARGETS.md#cmsis-backends).

## Testing

Both API test suites run **73 embedded `.nk` regression cases** on CPU builds — [TESTING.md](TESTING.md). Python ONNX parity covers **69** embedded inputs (`make test-python`). MCU/MPU bring-up: `make test-embedded-smoke-matrix`.

```bash
make test       # C++ then C then Python ONNX parity (cpu only)
make test-cpp   # ./netkit test
make test-c     # ./tests/test_c_api
make test-python
make test-embedded-smoke-matrix   # MCU/MPU + CMSIS host smoke
```
