# Getting Started

This guide gets you from clone to running inference on the desktop in a few minutes.

**Related docs:** [CLI](CLI.md) · [Model format](MODEL_FORMAT.md) · [Vectors tests](VECTORS_TESTS.md) · [C API](c-api.md) · [C++ API](cpp-api.md)

## Requirements

| Component | Standard | Compiler |
|-----------|----------|----------|
| Core engine + C++ API | **C++26** | clang++ 17+, g++ 14+, or MSVC `/std:c++latest` |
| C API (`netkit.h`) | **C23** | clang 17+ or gcc 14+ with `-std=c23` |
| Build | Make | Any Unix-like environment |

No external dependencies beyond the standard library.

All inference tensors, weights (`.bin`), and math use **float32** (`float`). There is no float64 path in the engine.

## Build

```bash
git clone https://github.com/jameslavrenz/netkit.git
cd netkit
make
```

This produces:

- **`netkit`** — CLI tool for tests, one-off inference, and inspection
- **`libnetkit.a`** — static library (C++ core + C API bridge)

Optional usage demos:

```bash
make example-cpp   # C++26: examples/infer_cpp
make example-c     # C23:    examples/infer_c
```

## Run the test suite

```bash
make test        # C++ API tests, then C API tests
make test-cpp    # C++26 only: ./netkit test
make test-c      # C23 only:  ./tests/test_c_api
```

Each suite validates the same eight vector models. The C API suite adds direct smoke tests for arena, tensor, ops, and model load/run.

## Run inference from the CLI

See [CLI.md](CLI.md) for full command reference.

```bash
# MLP: 2 inputs -> 2 outputs
./netkit run models/test_mlp.json --input 1,2

# CNN: 16 inputs (4x4x1)
./netkit run models/cnn_4x4_single.json --input=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

## Inspect a model

```bash
./netkit inspect models/test_mlp.json
```

Prints architecture, weight file info, and arena memory usage after load and forward pass.

## Use the C API (C23)

Build and run the full example:

```bash
make example-c
./examples/infer_c models/test_mlp.json 1 2
```

Source: [`examples/infer_c.c`](../examples/infer_c.c). Minimal integration pattern:

```c
#include "netkit.h"

alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
nk_arena_t arena;
nk_model_t model;

nk_arena_init(&arena, memory, sizeof(memory));
nk_model_load("models/test_mlp.json", &arena, &model);

float input[] = {1.0f, 2.0f};
float output[2];
uint32_t output_count = 0;

nk_model_run(&model, &arena, input, 2, output, 2, &output_count);
```

Link with `libnetkit.a` using a C++ linker driver (the library contains C++ object code):

```bash
clang -std=c23 -Iinclude -c my_app.c -o my_app.o
clang++ -std=c++26 -o my_app my_app.o libnetkit.a
```

See [c-api.md](c-api.md) for the full C reference.

## Use the C++ API (C++26)

Build and run the full example:

```bash
make example-cpp
./examples/infer_cpp models/test_mlp.json 1 2
```

Source: [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp). Minimal integration pattern:

```cpp
#include "arena.hpp"
#include "model_loader.hpp"
#include "tensor_factory.hpp"

alignas(std::max_align_t) unsigned char buffer[Arena::kDefaultCapacity];
Arena arena;
arena.init(buffer, sizeof(buffer));

MLPNetwork* network = nullptr;
std::array<uint32_t, kMaxTensorRank> input_shape{};
uint32_t input_rank = 0;

ModelLoader::LoadMLP("models/test_mlp.json", arena, network, input_shape, input_rank);

Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
TensorFactory::Fill(input, {1.0f, 2.0f});

Tensor output = TensorFactory::Create2D(arena, 1, network->GetLayer(1).weights.shape[1]);
network->forward(input, output, arena);
```

See [cpp-api.md](cpp-api.md) for the full C++ reference.

## Model file bundles

Each model is three files sharing a base name. See [MODEL_FORMAT.md](MODEL_FORMAT.md) for the full JSON schema, weight byte layout, and supported activations.

| File | Purpose |
|------|---------|
| `model.json` | Architecture (layers, activations, input shape) |
| `model.bin` | Raw float32 weights (row-major, layer order) |
| `model.vectors.json` | Regression test cases (optional) |

Example workflow:

1. Edit `model.json`
2. Export or generate `model.bin` (see `tools/write_hand_models.py` for hand models)
3. Add cases to `model.vectors.json` — see [VECTORS_TESTS.md](VECTORS_TESTS.md)
4. Register the vectors file in `src/test.cpp` if it is a new bundle
5. Run `make test`

## Project layout

```
netkit/
├── include/          Headers (C++ + netkit.h)
├── src/              C++26 implementation
├── examples/
│   ├── infer_cpp.cpp # C++26 usage demo
│   └── infer_c.c     # C23 usage demo
├── tests/
│   └── test_c_api.c  # C23 API regression tests
├── models/           JSON + bin + vectors test bundles
├── tools/            Python helpers for weight generation
└── docs/             Guides and API reference
```

## Next steps

- Read [API.md](API.md) for an overview of both APIs
- Read [CLI.md](CLI.md) for `test`, `run`, and `inspect`
- Add a regression case — [VECTORS_TESTS.md](VECTORS_TESTS.md)
- Use `./netkit inspect` to size your arena before deploying to embedded targets
