# netkit — Neural Network Kit

netkit is a C++26 neural network kit for on-device inference on MCUs and MPUs. It is developed and validated on the desktop, then deployed to embedded targets. Companion to [memkit](https://github.com/jameslavrenz/memkit) for memory management.

Models are loaded from JSON architecture files and companion float32 `.bin` weight files.

## Documentation

| Guide | Description |
|-------|-------------|
| **[Getting Started](docs/GETTING_STARTED.md)** | Build, test, and first inference in minutes |
| **[API Overview](docs/API.md)** | C vs C++ APIs, linking, memory model |
| **[CLI Reference](docs/CLI.md)** | `test`, `run`, and `inspect` commands |
| **[Model File Format](docs/MODEL_FORMAT.md)** | JSON architecture + float32 `.bin` weights |
| **[Vectors Tests](docs/VECTORS_TESTS.md)** | Declarative regression test files |
| **[C API Reference](docs/c-api.md)** | `netkit.h` (C23) |
| **[C++ API Reference](docs/cpp-api.md)** | Headers in `include/` (C++26) |
| **[API Parity Policy](docs/API_PARITY.md)** | C ↔ C++ symbol map and contribution rules |
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
- **MLP & CNN** — High-level network abstractions with JSON + `.bin` loading
- **Arena allocator** — Bump-pointer memory management (no heap in layer paths)
- **Vectors test runner** — Declarative regression tests in `*.vectors.json`
- **Float32 inference** — all tensors, weights, and math use IEEE-754 single precision (`float`)

## Quick start

```bash
make              # build netkit CLI + libnetkit.a
make test         # C++ API tests + C API tests
./netkit run models/test_mlp.json --input 1,2
make example-cpp    # C++26 usage demo
make example-c      # C23 usage demo
```

See [Getting Started](docs/GETTING_STARTED.md) for full details.

## CLI

Full reference: [docs/CLI.md](docs/CLI.md)

```bash
./netkit test                              # C++ API regression suite
./netkit run models/test_mlp.json --input 1,2
./netkit inspect models/test_mlp.json      # architecture + arena sizing
```

## Examples

| Demo | Language | Build | Run |
|------|----------|-------|-----|
| `examples/infer_cpp.cpp` | C++26 | `make example-cpp` | `./examples/infer_cpp models/test_mlp.json 1 2` |
| `examples/infer_c.c` | C23 | `make example-c` | `./examples/infer_c models/test_mlp.json 1 2` |

Both load a model from JSON + `.bin` and print input/output tensors. See [Getting Started](docs/GETTING_STARTED.md) for minimal code snippets and linking.

## Project structure

```
netkit/
├── include/
│   ├── netkit.h            # C23 public API
│   ├── arena.hpp           # Memory management
│   ├── tensor.hpp          # Tensor definitions
│   ├── mlp.hpp / cnn.hpp   # Network abstractions
│   ├── model_loader.hpp    # JSON + .bin loader
│   └── ...
├── src/                    # C++26 implementation
├── examples/
│   ├── infer_cpp.cpp       # C++26 usage example
│   └── infer_c.c           # C23 usage example
├── tests/
│   └── test_c_api.c        # C23 API regression tests
├── models/                 # JSON + bin + vectors bundles
├── tools/
│   └── write_hand_models.py
└── docs/                   # Guides and API reference
    ├── GETTING_STARTED.md
    ├── API.md
    ├── CLI.md
    ├── MODEL_FORMAT.md
    ├── VECTORS_TESTS.md
    ├── c-api.md / cpp-api.md
    └── API_PARITY.md
```

## Model file bundles

| File | Purpose |
|------|---------|
| `model.json` | Architecture (layers, activations, input shape) |
| `model.bin` | Raw float32 weights in layer order |
| `model.vectors.json` | Regression test cases (optional) |

Full schema, weight layout, and activations: [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md).  
Regression tests: [docs/VECTORS_TESTS.md](docs/VECTORS_TESTS.md).

## Building

### Requirements

- C++26 compiler (clang++ 17+, g++ 14+)
- C23 compiler for C examples (clang 17+, gcc 14+)
- Make

### Targets

```bash
make              # netkit CLI + libnetkit.a
make lib          # libnetkit.a only
make test         # C++ API tests + C API tests
make test-cpp     # C++ API regression only
make test-c       # C API regression only
make example-cpp  # C++26 usage demo
make example-c    # C23 usage demo
make clean
make rebuild
```

## Testing

```bash
make test       # C++ API tests, then C API tests
make test-cpp   # ./netkit test  (C++26 vectors regression)
make test-c     # ./tests/test_c_api  (C23 API regression)
```

| Suite | Language | Entry point |
|-------|----------|-------------|
| C++ API | C++26 | `src/test.cpp` via `./netkit test` |
| C API | C23 | `tests/test_c_api.c` |

Both suites cover the same eight vector models plus API-specific smoke checks.

## Design principles

- **Lightweight** — Standard C/C++ only, no external dependencies
- **Memory-conscious** — Arena allocator throughout
- **Single-threaded** — Sequential forward pass
- **Inference-only** — No training

## Roadmap

- Pooling (max, average) and conv padding
- Batch normalization
- Quantization (int8, uint8)
- Python model exporter

## License

[Add license information here]

## Contributing

- C++ sources: C++26
- C sources and `netkit.h`: C23
- All tests must pass (`make test`)
- Update docs when changing public API
- **New C++ public API requires a matching C entry in `netkit.h`** — see [API_PARITY.md](docs/API_PARITY.md)
