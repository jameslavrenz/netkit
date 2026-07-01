# API Overview

netkit exposes two language interfaces over the same **C++26 inference engine**:

| API | Header | Language | Use when |
|-----|--------|----------|----------|
| **C API** | `include/netkit.h` | C23 | Embedded firmware, FFI, minimal dependencies at the call site |
| **C++ API** | `include/*.hpp` | C++26 | Application code, tests, extending layers and ops |

Both APIs share:

- Bump-pointer **arena** memory management (no heap in layer code paths)
- **JSON + `.bin`** model loading
- **MLP** and **CNN** forward-only inference (including conv / pool / flatten / dense CNN pipelines)
- **NHWC** tensor layout for convolutions
- **Float32 only (today)** — all inference tensors, weights, and math use IEEE-754 single precision; float16, int16, int8, and int4 planned ([DATATYPES.md](DATATYPES.md))

Every stable C++ public symbol has a documented C equivalent except CLI-only diagnostics — see [API_PARITY.md](API_PARITY.md).

## Documentation map

| Document | Contents |
|----------|----------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, test, first inference, examples |
| [ARENA.md](ARENA.md) | Bump allocator memory model |
| [DATATYPES.md](DATATYPES.md) | Float32 today; float16/int roadmap |
| [CLI.md](CLI.md) | `netkit test`, `run`, `inspect`, help, network summary |
| [MODEL_FORMAT.md](MODEL_FORMAT.md) | JSON schema, `.bin` weight layout |
| [TESTING.md](TESTING.md) | Regression suites, Make targets, CI |
| [VECTORS_TESTS.md](VECTORS_TESTS.md) | Hand `*.vectors.json` format |
| [MNIST.md](MNIST.md) | Trained MNIST MLP test bundle |
| [MNIST_CNN.md](MNIST_CNN.md) | Trained MNIST CNN test bundle |
| [API_PARITY.md](API_PARITY.md) | C ↔ C++ symbol map and parity policy |
| [c-api.md](c-api.md) | Full C23 reference (`netkit.h`) |
| [cpp-api.md](cpp-api.md) | Full C++26 reference (headers in `include/`) |

## Quick comparison

### Load and run (C23)

```c
nk_arena_t arena;
nk_model_t model;
nk_arena_init(&arena, memory, size);
nk_model_load("models/test_mlp.json", &arena, &model);
nk_model_run(&model, &arena, input, n, output, cap, &out_n);
```

Full example: [`examples/infer_c.c`](../examples/infer_c.c)

### Load and run (C++26)

```cpp
Arena arena;
arena.init(buffer, size);
MLPNetwork* net = nullptr;
ModelLoader::LoadMLP("models/test_mlp.json", arena, net, shape, rank);
net->forward(input, output, arena);
```

Full example: [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp)

## CLI

The `netkit` binary is a desktop development tool (C++26). See [CLI.md](CLI.md).

| Command | Description |
|---------|-------------|
| `netkit test` | Run all registered `*.vectors.json` regression tests |
| `netkit run <model.json> --input a,b,c` | Single inference |
| `netkit inspect <model.json>` | Boxed network summary (`--full` for weights/arena sizing) |
| `netkit help`, `netkit -h`, `netkit --help` | Print CLI usage |

Full option reference: [CLI.md](CLI.md).

## Language standards

| Code | Standard | Role |
|------|----------|------|
| C++ engine | **C++26** | All implementation, primary API, CLI, C++ tests |
| C API | **C23** | `netkit.h`, `examples/infer_c.c`, `tests/test_c_api.c` |

Application code is C++26. C23 is limited to the C header, the `extern "C"` bridge (`src/netkit_api.cpp`), C examples, and the C API test harness.

## Linking

`libnetkit.a` contains C++ object code. Link C applications with a C++-aware linker:

```bash
clang -std=c23 -Iinclude -c my_app.c -o my_app.o
clang++ -std=c++26 -o my_app my_app.o libnetkit.a
```

C++ applications:

```bash
clang++ -std=c++26 -Iinclude -o my_app my_app.cpp libnetkit.a
```

Build the library with `make lib`.

## Error handling

| API | Pattern |
|-----|---------|
| C | Functions return `nk_status_t`; call `nk_last_error()` for detail |
| C++ | `ModelLoader::LoadResult` with `LoadStatus` and `message` |

## Memory model

Full guide: [ARENA.md](ARENA.md). Data type constraints: [DATATYPES.md](DATATYPES.md).

Both APIs require a **caller-provided buffer** for the arena. You pass the buffer and its size to `Arena::init()` / `nk_arena_init()`. The size is **not** in model JSON — it depends on weights plus two ping-pong activation buffers allocated at load (see [ARENA.md](ARENA.md)).

**Default constant:** 64 KiB (`Arena::kDefaultCapacity` / `NK_ARENA_DEFAULT_CAPACITY`) is used in examples, CLI, and hand tests. MNIST models need multi-MiB buffers; the test suite uses 2 MiB (MLP) and 4 MiB (CNN).

**Sizing workflow:** `./netkit inspect <model.json> --full` or `nk_inspect_model()` → read `arena_bytes_after_forward` → allocate static storage with headroom.

**Backing buffer:** declare with `alignas(max_align_t)` / `alignas(std::max_align_t)` so the arena base address satisfies the platform’s strictest alignment.

**Aligned bump allocation:** `Arena::alloc(size, alignment)` and `nk_arena_alloc(arena, size, alignment)` insert padding when the current offset is not a multiple of `alignment`. `alignment` must be a power of two. Returns `nullptr` on overflow or invalid arguments.

| Allocation kind | Typical alignment |
|-----------------|-------------------|
| float tensors, weight `.bin` blobs | `alignof(float)` (4) |
| Structs, placement `new`, pointer arrays | `alignof(T)` (usually 8 on 64-bit) |

The engine uses these rules internally so odd-sized weight files (e.g. an odd float count) do not misalign network structs. Direct C callers using `nk_arena_alloc` must pass the correct alignment themselves.

Size the buffer using `./netkit inspect --full` or `nk_inspect_model()`. When allocation fails, functions return an arena overflow error — there is no automatic growth.

Call `nk_arena_reset()` / `Arena::reset()` between inference batches to reuse the same buffer.

netkit implements its own minimal arena (~86 lines) rather than linking [memkit](https://github.com/jameslavrenz/memkit); alignment behavior matches memkit’s `static_arena` bump policy.

## Supported model format

Summary — full details in [MODEL_FORMAT.md](MODEL_FORMAT.md):

- JSON `version` must be `1`
- `network`: `"mlp"` or `"cnn"`
- Activations: `none`, `relu`, `sigmoid`, `tanh`, `leaky_relu`, `relu6`, `softmax`
- Weights: float32 little-endian in companion `.bin` file

## Testing

Both API test suites run **36 inference regression cases** (16 hand vector + 10 MNIST MLP + 10 MNIST CNN). See [TESTING.md](TESTING.md), [VECTORS_TESTS.md](VECTORS_TESTS.md), [MNIST.md](MNIST.md), and [MNIST_CNN.md](MNIST_CNN.md).

```bash
make test       # C++ then C
make test-cpp   # ./netkit test
make test-c     # ./tests/test_c_api
```
