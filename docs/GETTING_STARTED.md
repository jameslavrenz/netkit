# Getting Started

Welcome to netkit. This guide takes you from clone to your first inference in a few minutes.

**New here?** Read [PHILOSOPHY.md](PHILOSOPHY.md) for the big picture (Phase 1 interpreter runtime, Phase 2 packager optimizations, memory model).

**Related docs:** [CLI](CLI.md) · [Build targets](BUILD_TARGETS.md) · [Arena](ARENA.md) · [C API](c-api.md) · [C++ API](cpp-api.md) · [Testing](TESTING.md)

---

## What you need

| Component | Requirement |
|-----------|-------------|
| Compiler (engine) | **C++26** — clang++ 17+, g++ 14+ |
| Compiler (C API) | **C23** — clang 17+ or gcc 14+ |
| Build | GNU **Make** (primary); **CMake** 3.16+ optional |
| Python packager (optional) | Python 3 + numpy + onnx — `pip install -e python`; training scripts also need `pip install -e "python[train]"` (PyTorch) |

Inference is **float32-only** today. float16, int16, int8, and int4 are on the roadmap — [DATATYPES.md](DATATYPES.md).

---

## 1. Clone and build (desktop)

```bash
git clone https://github.com/jameslavrenz/netkit.git
cd netkit
make              # NETKIT_TARGET=cpu (default): netkit CLI + libnetkit.a
```

You get:

- **`./netkit`** — desktop CLI (`test`, `run`, `inspect`)
- **`libnetkit.a`** — static library (C++ engine + C API bridge)

Verify:

```bash
make test         # 73 embedded .nk cases + Python ONNX parity + AOT tests (make test-python)
```

---

## 2. Run inference from the command line

Full reference: [CLI.md](CLI.md).

```bash
./netkit help

# Small MLP: 2 inputs → 2 outputs
./netkit run models/test_mlp.nk --input 1,2

# Small CNN: 4×4×1 input (16 values)
./netkit run models/cnn_4x4_single.nk --input=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

# Inspect architecture (add --full for arena sizing)
./netkit inspect models/mnist_cnn.nk
./netkit inspect models/mnist_cnn.nk --full
```

| Command | Purpose |
|---------|---------|
| `netkit test` | Full regression suite (same as `make test-cpp`) |
| `netkit run <model.nk> --input <values>` | One forward pass |
| `netkit inspect <model.nk> [--full]` | Network summary; `--full` reports arena usage |

The CLI is **CPU-only** (`NETKIT_TARGET=cpu`). It uses **heap-backed arenas** sized per model (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN) so MNIST works without manual buffer sizing.

---

## 3. Build flags and memory defaults

Select target with **`NETKIT_TARGET`**:

| Target | Command | What you get |
|--------|---------|--------------|
| **CPU** (default) | `make` | CLI + full library + tests |
| **MCU** | `make NETKIT_TARGET=mcu lib` | Lean runtime only |
| **MPU** | `make NETKIT_TARGET=mpu lib` | Lean runtime only |

### Arena defaults

| Target | `NK_ARENA_DEFAULT_CAPACITY` | Arena backing |
|--------|------------------------------|---------------|
| CPU | **4 MiB** | **Heap** (default); `NETKIT_GLOBAL_ARENA=1` for static buffer |
| MCU | **64 KiB** | Your static/global buffer; `NETKIT_HEAP_ARENA=1` for optional heap API |
| MPU | **128 KiB** | Same as MCU |

```bash
make                                    # CPU, heap default
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all   # CPU, static arena
make NETKIT_TARGET=mcu lib              # firmware, 64 KiB constant
make NETKIT_TARGET=mpu lib              # firmware, 128 KiB constant
make NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib     # MPU + heap helpers
```

Macros are defined in [`include/netkit_config.h`](../include/netkit_config.h). Full tables: [BUILD_TARGETS.md](BUILD_TARGETS.md).

### Target architecture (`NETKIT_ARCH`)

Leave **`NETKIT_ARCH` unset** for native desktop builds. Set it when cross-compiling firmware so CMSIS gets the right `ARM_MATH_*` defines:

```bash
# Desktop (default) — no NETKIT_ARCH
make

# Cortex-M4 firmware library with CMSIS backends
make cmsis-init
make NETKIT_ARCH=CM4 NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib

# Cortex-M33 (adds __DSP_PRESENT=1)
make NETKIT_ARCH=M33 NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib
```

| Core | `NETKIT_ARCH` | Extra CMSIS flags |
|------|---------------|-------------------|
| Cortex-M4 | `CM4` | `ARM_MATH_LOOPUNROLL` |
| Cortex-M33 | `M33` | `__DSP_PRESENT=1`, loop unroll |
| Cortex-M55/M85 | `M55` / `M85` | `ARM_MATH_MVEF`, `ARM_MATH_MVEI`, loop unroll |
| AArch64 Neon | `NEON` | loop unroll |

See the full core table in [BUILD_TARGETS.md](BUILD_TARGETS.md#target-architecture-netkit_arch).

### Optional CMSIS backends

Fetch once, then enable at compile time (no runtime switching):

```bash
make cmsis-init
make NETKIT_CMSIS_DSP=1 test-cpp   # desktop / MPU vector ops
make NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 lib   # MCU CMSIS-NN (firmware)
```

### Embedded smoke (MCU/MPU bring-up)

Before flashing firmware, validate lean runtime linking on the host:

```bash
make cmsis-init
make test-embedded-smoke-matrix   # MCU CM4/M33 + CMSIS-NN; MPU DSP-only profiles
```

Single profile:

```bash
make NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 embedded-smoke
./tests/embedded_smoke
```

The matrix sets `NETKIT_HOST_SMOKE=1` so CMSIS-DSP uses the portable host path without CMSIS-Core headers. On hardware, omit `NETKIT_HOST_SMOKE` and use your toolchain `-mcpu` flags. Details: [TESTING.md](TESTING.md#embedded-smoke-mcupu).

### CMake alternative

```bash
cmake -B cmake-build
cmake --build cmake-build
./cmake-build/netkit test
```

Use `-DNETKIT_ARCH=CM4`, `-DNETKIT_TARGET=mcu`, `-DNETKIT_CMSIS_NN=ON` for MCU firmware. Desktop CMake builds enable CMSIS-DSP by default; CMSIS-NN requires MCU + Cortex-M arch.

### Size a buffer for your model

Arena size is **not** in the `.nk` file. On desktop:

```bash
./netkit inspect models/your_model.nk --full
```

Note **arena bytes after forward**, add ~25–50% headroom, then declare that size in firmware. See [ARENA.md](ARENA.md).

---

## 4. Convert ONNX to `.nk`

```bash
pip install -e python
python -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk
make export-nk    # regenerate all bundled models + embedded tests
```

Format spec: [NK_FORMAT.md](NK_FORMAT.md). Python details: [python/README.md](../python/README.md).

---

## 5. AOT compile (embed `.nk` in firmware)

After you have a validated `.nk`, the Python packager can embed it as a static byte array and generate load/run wrappers for the runtime:

```bash
# Default: C++26 (.hpp + .cpp)
python -m netkit aot models/test_mlp.nk -o build/aot

# C23 (.h + .c)
python -m netkit aot models/test_mlp.nk -o build/aot --language c

# Optional smoke main (compile with -DNETKIT_AOT_MAIN)
python -m netkit aot models/test_mlp.nk -o build/aot --main

# Optional graph optimizations (fewer ops at runtime; verified against original .nk)
python -m netkit aot models/cnn_extended_ops.nk -o build/aot --optimize
```

Typical pipeline (single build session):

```text
model.onnx  →  convert  →  model.nk  →  aot  →  model_aot.{hpp,cpp}
                              ↓
                         make test-python / ./netkit test
```

At runtime the generated code calls `NkLoader::LoadMLPFromBuffer` / `LoadCNNFromBuffer` (C++) or `nk_model_load_memory` (C) — same `.nk` format, loaded from flash/RAM instead of a filesystem path.

With `--optimize`, the packager applies stable graph passes before embedding (conv+BN fusion, BN folded into the following dense head, consecutive linear dense merge, identity BN removal). Each pass is checked against the original model numerically before the optimized `.nk` is emitted.

Link generated sources with `libnetkit.a`:

```bash
make lib
clang++ -std=c++26 -Iinclude -c build/aot/test_mlp_aot.cpp -o test_mlp_aot.o
clang++ -std=c++26 -Iinclude -o my_app my_app.cpp test_mlp_aot.o libnetkit.a
```

Tests: `make test-python` includes `python/tests/test_aot_compile.py` (compile + run vs reference). See [TESTING.md](TESTING.md).

---

## 6. Integrate in your application

### C API (C23) — typical for firmware

```bash
make example-c
./examples/infer_c models/test_mlp.nk 1 2
```

Minimal pattern ([`examples/infer_c.c`](../examples/infer_c.c)):

```c
#include "netkit.h"

alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
nk_arena_t arena;
nk_model_t model;

#if defined(NETKIT_ARENA_HEAP)
nk_arena_init_heap(&arena, NK_ARENA_DEFAULT_CAPACITY);
#else
nk_arena_init(&arena, memory, sizeof(memory));
#endif

nk_model_load("models/test_mlp.nk", &arena, &model);

float input[] = {1.0f, 2.0f};
float output[2];
uint32_t output_count = 0;
nk_model_run(&model, &arena, input, 2, output, 2, &output_count);

#if defined(NETKIT_ARENA_HEAP)
nk_arena_destroy_heap(&arena);
#endif
```

Link with a C++ driver (library contains C++ objects):

```bash
clang -std=c23 -Iinclude -c my_app.c -o my_app.o
clang++ -std=c++26 -o my_app my_app.o libnetkit.a
```

Full reference: [c-api.md](c-api.md).

### C++ API (C++26)

```bash
make example-cpp
./examples/infer_cpp models/test_mlp.nk 1 2
```

Uses `NkLoader::LoadMLP`, `TensorFactory`, and `Arena` — see [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp) and [cpp-api.md](cpp-api.md).

Helper for target-default arena setup: `include/arena_util.hpp` (`ArenaUtil::Init`, `ArenaUtil::Scoped`).

---

## 7. Project layout

```
netkit/
├── include/           netkit.h (C) + *.hpp (C++)
├── src/               C++26 engine
├── python/netkit/     ONNX → .nk packager + AOT compiler (Phase 2 optimizations land here)
├── examples/          infer_c.c, infer_cpp.cpp
├── tests/             test_c_api.c, embedded_smoke.c
├── models/            bundled .nk + .onnx
├── tools/             export scripts, embedded smoke, local FVP/cross-compile helpers
└── docs/              guides (start with this file)
```

---

## 7. Common workflows

| Goal | Steps |
|------|-------|
| Validate a change | `make test` |
| Try a model quickly | `./netkit run model.nk --input ...` |
| Size firmware RAM | `./netkit inspect model.nk --full` |
| Ship on MCU | `make NETKIT_TARGET=mcu lib`, link into firmware, static arena |
| Smoke MCU/MPU + CMSIS on host | `make test-embedded-smoke-matrix` |
| Cross-compile CM4 lib locally | `./tools/compile_cm4_cross.sh` (requires `gcc-arm-none-eabi`) |
| Optional FVP cycle timing | `make bench-hand-fvp` (local; see [TESTING.md](TESTING.md)) |
| Add regression case | Edit `python/netkit/regression_data.py`, `make embed-tests`, register in `src/test.cpp` |

---

## 8. Next steps

| Topic | Document |
|-------|----------|
| Philosophy & roadmap | [PHILOSOPHY.md](PHILOSOPHY.md) |
| Build flags (CPU/MCU/MPU) | [BUILD_TARGETS.md](BUILD_TARGETS.md) |
| CLI commands | [CLI.md](CLI.md) |
| Arena sizing & alignment | [ARENA.md](ARENA.md) |
| C / C++ API reference | [c-api.md](c-api.md), [cpp-api.md](cpp-api.md) |
| Regression & CI | [TESTING.md](TESTING.md) |
| MNIST regression tests | [MNIST.md](MNIST.md), [MNIST_CNN.md](MNIST_CNN.md) |
| Planned dtypes | [DATATYPES.md](DATATYPES.md) |
