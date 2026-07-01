# API Overview

netkit exposes two language interfaces over the same **C++26 inference engine**:

| API | Header | Language | Use when |
|-----|--------|----------|----------|
| **C API** | `include/netkit.h` | C23 | Embedded firmware, FFI, minimal dependencies at the call site |
| **C++ API** | `include/*.hpp` | C++26 | Application code, tests, extending layers and ops |

Both APIs share:

- Bump-pointer **arena** memory management (no heap in layer code paths)
- **JSON + `.bin`** model loading
- **MLP** and **CNN** forward-only inference
- **NHWC** tensor layout for convolutions
- **Float32 only** — all tensors, weights, and math use IEEE-754 single precision (`float`); no float64 inference path

## Documentation map

| Document | Contents |
|----------|----------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, test, first inference, examples |
| [CLI.md](CLI.md) | `netkit test`, `run`, `inspect` |
| [MODEL_FORMAT.md](MODEL_FORMAT.md) | JSON schema, `.bin` weight layout |
| [VECTORS_TESTS.md](VECTORS_TESTS.md) | Regression test file format |
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
| `netkit inspect <model.json>` | Architecture, weights, arena sizing |

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

Both APIs require a caller-provided buffer for the arena. Default size is 64 KiB (`Arena::kDefaultCapacity` / `NK_ARENA_DEFAULT_CAPACITY`).

Size the buffer using `./netkit inspect` or `nk_inspect_model()`. When allocation fails, functions return an arena overflow error — there is no automatic growth.

Call `nk_arena_reset()` / `Arena::reset()` between inference batches to reuse the same buffer.

## Supported model format

Summary — full details in [MODEL_FORMAT.md](MODEL_FORMAT.md):

- JSON `version` must be `1`
- `network`: `"mlp"` or `"cnn"`
- Activations: `none`, `relu`, `sigmoid`, `tanh`, `leaky_relu`, `relu6`, `softmax`
- Weights: float32 little-endian in companion `.bin` file

## Testing

Both API test suites cover the same eight vector models. See [VECTORS_TESTS.md](VECTORS_TESTS.md).

```bash
make test       # C++ then C
make test-cpp   # ./netkit test
make test-c     # ./tests/test_c_api
```
