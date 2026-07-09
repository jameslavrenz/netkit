# Getting Started

Welcome to netkit — a **multi-modal inference engine** (voice, image, vision) with an **embedded-first** design for **MCUs, MPUs, and NPUs**, implemented in **C++26** with a **C23** API. The project is **under active development**: **float32** and **int8** inference work today; **Kalman estimation and tracking** are on the roadmap.

**Two ways to run inference:** load a `.nk` and execute through the **`NkOpsResolver` interpreter** (flexible — swap models, use the CLI), or **compile for maximum speed** with `python -m netkit aot` (embed the model in flash, apply packager graph optimizations, trim linked ops, optional CMSIS kernels). Both paths share the same kernels — see [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

This guide takes you from clone to your first inference in a few minutes.

**New here?** Read [PHILOSOPHY.md](PHILOSOPHY.md) for the big picture (interpreter vs compiled deployment, Phase 1 runtime, Phase 2 packager optimizations, memory model).

**Related docs:** [CLI](CLI.md) · [Build targets](BUILD_TARGETS.md) · [Arena](ARENA.md) · [C API](c-api.md) · [C++ API](cpp-api.md) · [Testing](TESTING.md)

---

## What you need

| Component | Requirement |
|-----------|-------------|
| Compiler (engine) | **C++26** — clang++ 17+, g++ 14+ |
| Compiler (C API) | **C23** — clang 17+ or gcc 14+ |
| Build | GNU **Make** (primary); **CMake** 3.16+ optional |
| Python packager (optional) | Python 3 + numpy + onnx — `pip install -e python`; training/backbone pack also need `pip install -e "python[train]"` (torch + timm) |

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
make test         # default: C++/C (88) + fast Python (AOT, unit tests)
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

The CLI is **CPU-only** (`NETKIT_TARGET=cpu`). It uses a **heap-backed arena** (`NK_ARENA_DEFAULT_CAPACITY` = **64 MiB** by default). Override with `./netkit --arena <size>`.

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
| CPU | **64 MiB** | **Heap** (default); `NETKIT_GLOBAL_ARENA=1` for static buffer |
| MCU | **64 KiB** | Your static/global buffer; `NETKIT_HEAP_ARENA=1` for optional heap API |
| MPU | **64 MiB** | Same as MCU (caller-owned buffer; size with inspect) |

```bash
make                                    # CPU, heap default
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all   # CPU, static arena
make NETKIT_TARGET=mcu lib              # firmware, 64 KiB constant
make NETKIT_TARGET=mpu lib              # firmware, 64 MiB constant
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

`make cmsis-init` fetches **CMSIS-Core** (device headers for MCU cross-builds), **CMSIS-NN**, and **CMSIS-DSP** as git submodules. Backends are **opt-in** at compile time (`NETKIT_CMSIS_*=1`) — not inferred from `NETKIT_ARCH` alone.

**Profile defaults** after `cmsis-init`:

| `NETKIT_TARGET` | CMSIS-DSP | CMSIS-NN |
|-----------------|-----------|----------|
| `cpu` | on | off |
| `mcu` | on | on (needs Cortex-M `NETKIT_ARCH`) |
| `mpu` | on | off |

```bash
make cmsis-init
make test-cpp                                    # cpu: CMSIS-DSP on by default
make NETKIT_TARGET=mcu NETKIT_ARCH=CM4 lib       # mcu firmware: DSP + NN
make NETKIT_CMSIS_DSP=0 NETKIT_CMSIS_NN=0 test   # reference kernels only (CI)
```

### Embedded smoke (MCU/MPU bring-up)

Before flashing firmware, validate lean runtime linking on the host. The smoke binary parses and runs **`test_mlp.nk`** and **`cnn_4x4_single.nk`** across seven MCU/MPU + CMSIS profiles.

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

Use `-DNETKIT_ARCH=CM4`, `-DNETKIT_TARGET=mcu` for MCU firmware (CMake defaults: CMSIS-DSP and CMSIS-NN on for `mcu`, DSP-only for `cpu`/`mpu`). Override with `-DNETKIT_CMSIS_DSP=OFF` / `-DNETKIT_CMSIS_NN=OFF`.

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

## 5. Embed `.nk` in firmware (`netkit aot`)

The Python **`netkit aot`** command **packages** a `.nk` into C/C++ sources. That packaging step is required for both interpreter and lowered firmware — it is not the same as “compiled / no interpreter.” See [PHILOSOPHY.md — embed vs lowered](PHILOSOPHY.md#terminology-embed-vs-lowered).

| Goal | Command |
|------|---------|
| **Interpreter embed** (TFLM-fair, swap graph without re-lowering) | `python -m netkit aot model.nk -o out --no-lower` |
| **Lowered / compiled** (C++ default — static kernel chain) | `python -m netkit aot model.nk -o out` |

After you have a validated `.nk`, embed it as a static byte array and generate load/run wrappers:

```bash
# Default: C++26 lowered static Kernels:: chain (.hpp + .cpp)
python -m netkit aot models/test_mlp.nk -o build/aot

# Embed .nk + interpreter loader instead of lowered kernels
python -m netkit aot models/test_mlp.nk -o build/aot --no-lower

# C23 (.h + .c) — interpreter path only (no lowered emitter)
python -m netkit aot models/test_mlp.nk -o build/aot --language c

# MCU: flash-backed coefs (always), size arena without weight copy
python -m netkit aot models/mlp_hand.nk -o build/aot --target mcu --arena-headroom 15

# Optional smoke main (compile with -DNETKIT_AOT_MAIN)
python -m netkit aot models/test_mlp.nk -o build/aot --main

# Optional graph optimizations (fewer ops at runtime; verified against original .nk)
python -m netkit aot models/cnn_extended_ops.nk -o build/aot --optimize
```

**Lowered C++ AOT** (default for `--language cpp`) emits a static `Kernels::` call chain — no runtime `.nk` loader. **C AOT** always embeds the `.nk` blob and calls `nk_model_load_memory`. See [boards/nucleo-f446re/README.md](../boards/nucleo-f446re/README.md) for an MCU firmware example (MNIST MLP, flash weights, CMSIS-DSP).

Generated headers expose measured arena usage for static buffer allocation:

- **C++ lowered** — `kArenaBytesRecommended`, `kLowered = true`, `Model::load` / `Model::forward`
- **C++ / C interpreter** — `kArenaBytesAfterLoad`, `kArenaBytesAfterForward`, `kArenaBytesRecommended`, plus `InitArena()` / `{symbol}_aot_init_arena()`

The `.nk` blob is placed in flash-friendly `.rodata` by default (GCC `__attribute__((section(".rodata")))`). Pass `--no-flash-section` to omit the attribute.

Typical MCU bring-up:

```c
alignas(max_align_t) static unsigned char arena_mem[MLP_HAND_AOT_ARENA_BYTES_RECOMMENDED];
nk_arena_t arena;
mlp_hand_aot_init_arena(&arena, arena_mem, sizeof(arena_mem));
nk_model_t model;
mlp_hand_aot_load(&arena, &model);
```

Typical pipeline (single build session):

```text
model.onnx  →  convert  →  model.nk  →  aot  →  model_aot.{hpp,cpp}
                              ↓
                         make test / ./netkit test
```

At runtime:

- **Lowered C++ AOT** — static `Kernels::` chain; coefs in `.rodata`
- **Interpreter AOT** — `NkLoader::LoadMLPFromBuffer` / `LoadCNNFromBuffer` (C++) or `nk_model_load_memory` (C) on the embedded `.nk` blob

With `--optimize`, the packager applies stable graph passes before embedding (conv+BN fusion, BN folded into the following dense head, consecutive linear dense merge, identity BN removal). Each pass is checked against the original model numerically before the optimized `.nk` is emitted.

Link generated sources with `libnetkit.a`:

```bash
make lib
clang++ -std=c++26 -Iinclude -c build/aot/test_mlp_aot.cpp -o test_mlp_aot.o
clang++ -std=c++26 -Iinclude -o my_app my_app.cpp test_mlp_aot.o libnetkit.a
```

Tests: default `make test` / `make test-python` includes `python/tests/test_aot_compile.py` (compile + run vs reference). Full ONNX parity: `make test-full`. See [TESTING.md](TESTING.md).

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
├── tools/             export scripts, run_embedded_smoke.sh
└── docs/              guides (start with this file)
```

---

## 7. Common workflows

| Goal | Steps |
|------|-------|
| Validate before push | `make cmsis-init && make test` |
| Full ONNX/backbone regression | `make test-full` |
| Run GitHub Actions CI (fast) | Automatic on push/PR; or `gh workflow run ci.yml` |
| Run full regression in CI | `gh workflow run test-full.yml` (manual only) |
| Try a model quickly | `./netkit run model.nk --input ...` |
| Size firmware RAM | `./netkit inspect model.nk --full` |
| Ship on MCU | `make NETKIT_TARGET=mcu lib`, link into firmware, static arena |
| Smoke MCU/MPU + CMSIS on host | `make test-embedded-smoke-matrix` |
| Add regression case | Edit `HAND_CASE_INPUTS` in `python/netkit/regression_data.py`, `make embed-tests`, register in `src/test.cpp` |

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
