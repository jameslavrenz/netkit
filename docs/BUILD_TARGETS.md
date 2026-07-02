# Build Targets

netkit builds for three deployment profiles. Select one with **`NETKIT_TARGET`** when invoking Make or CMake, or define the matching `-D` flag in your embedded toolchain.

| Target | Makefile / CMake | Role | What's included |
|--------|------------------|------|-----------------|
| **CPU** | `NETKIT_TARGET=cpu` (default) | Desktop dev, debug, CI | Lean runtime **plus** CLI, embedded regression tests, future debug/analysis tooling |
| **MCU** | `NETKIT_TARGET=mcu` | Microcontroller firmware | Lean runtime only — `.nk` load, arena, tensors, kernel ops, inference |
| **MPU** | `NETKIT_TARGET=mpu` | Microprocessor / RTOS | Same lean runtime as MCU |

## Build systems

| System | Primary? | Notes |
|--------|----------|-------|
| **GNU Make** | Yes | `make`, `make test`, CI default |
| **CMake** | Optional | `cmake -B cmake-build && cmake --build cmake-build` — same flags, auto-detects desktop vs embedded via `NETKIT_ARCH` |

Both use the same optional CMSIS backends and `NETKIT_ARCH` → `ARM_MATH_*` mapping (`third_party/netkit_arch.mk`, `cmake/netkit_arch.cmake`).

## Target architecture (`NETKIT_ARCH`)

**If `NETKIT_ARCH` is unset (empty), the build is a native desktop / CPU application** — no core-specific `ARM_MATH_*` defines. Set `NETKIT_ARCH` when cross-compiling firmware for a specific Arm core.

| `NETKIT_ARCH` | CMSIS-DSP flag(s) | Profile | CMSIS-NN |
|---------------|-------------------|---------|----------|
| *(unset)* | `__GNUC_PYTHON__` (desktop host only) | CPU | off (reference NN) |
| `CM0`, `M0` | `ARM_MATH_CM0` | MCU | on |
| `CM0PLUS`, `M0PLUS` | `ARM_MATH_CM0PLUS` | MCU | on |
| `CM3`, `M3` | `ARM_MATH_CM3` | MCU | on |
| `CM4`, `M4` | `ARM_MATH_CM4` | MCU | on |
| `CM7`, `M7` | `ARM_MATH_CM7` | MCU | on |
| `M23`, `CM23` | `ARM_MATH_ARMV8MBL` | MCU | on |
| `M33`, `CM33` | `ARM_MATH_ARMV8MML`, `__DSP_PRESENT=1` | MCU | on |
| `M55`, `CM55` | `ARM_MATH_M55`, `ARM_MATH_MVEF`, `ARM_MATH_MVEI` | MCU | on |
| `M85`, `CM85` | `ARM_MATH_M85`, `ARM_MATH_MVEF`, `ARM_MATH_MVEI` | MCU | on |
| `A32`, `MPU` | `ARM_MATH_A32` | MPU | off (reference NN) |
| `NEON`, `A64` | `ARM_MATH_NEON` | MPU | off (reference NN) |

Aliases like `Cortex-M4` normalize to `CM4`. CMake also sets `NETKIT_TARGET` from the arch (`cpu` / `mcu` / `mpu`); with Make you pass `NETKIT_TARGET` explicitly for firmware.

### Additional CMSIS preprocessor flags (auto-applied)

| Flag | When | Purpose |
|------|------|---------|
| `ARM_MATH_LOOPUNROLL` | All CMSIS-DSP builds (desktop + embedded) | CMSIS-DSP 1.10+ defaults to no unroll; this enables 4× loop unroll for faster inference |
| `__DSP_PRESENT=1` | `NETKIT_ARCH=M33` | Required for Armv8-M DSP extension; avoids scalar fallbacks |
| `ARM_MATH_MVEF` / `ARM_MATH_MVEI` | `NETKIT_ARCH=M55`, `M85` | Helium vector extensions (complements toolchain `-mcpu=cortex-m55`) |
| `HOST` / `__GNUC_PYTHON__` | Desktop only | CMSIS-DSP portable host path (no CMSIS-Core device headers) |

MCU builds also add `-Ithird_party/CMSIS-Core/Include` when that directory exists.

## Arena backing defaults

| Target | Default arena | Override flag |
|--------|---------------|---------------|
| **CPU** | **Heap** (`malloc` backing via `nk_arena_init_heap` / `Arena::init_heap`) | `NETKIT_GLOBAL_ARENA=1` → static/global buffer only |
| **MCU** | **Global/static** (`nk_arena_init` with your buffer) | `NETKIT_HEAP_ARENA=1` → also compile heap helpers |
| **MPU** | **Global/static** (same as MCU) | `NETKIT_HEAP_ARENA=1` → also compile heap helpers |

Compile-time macros (from `include/netkit_config.h`):

| Macro | Meaning |
|-------|---------|
| `NETKIT_TARGET_CPU` | Desktop / CPU build |
| `NETKIT_TARGET_MCU` | MCU build |
| `NETKIT_TARGET_MPU` | MPU build |
| `NETKIT_DESKTOP` | CPU only — CLI, regression, debug tooling |
| `NETKIT_ARENA_HEAP` | Heap arena API compiled in (CPU default; MCU/MPU when opted in) |
| `NETKIT_GLOBAL_ARENA` | CPU only — force global/static arena instead of heap default |
| `NETKIT_USE_CMSIS_NN` | CMSIS-NN backends enabled (see CMSIS section) |
| `NETKIT_USE_CMSIS_DSP` | CMSIS-DSP backends enabled (see CMSIS section) |

Default arena constant (`NK_ARENA_DEFAULT_CAPACITY` / `Arena::kDefaultCapacity`):

| Target | Default |
|--------|---------|
| **CPU** | **4 MiB** (MNIST CNN-capable; examples and C API smoke tests) |
| **MCU** | **64 KiB** |
| **MPU** | **128 KiB** |

CLI/regression on CPU still use **model-sized heap** via `ArenaUtil::CapacityForInputElements` (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN).

## Quick commands (Make)

```bash
# Desktop (default) — CLI + tests + libnetkit.a, heap arena default
make
make test
make build-all

# CPU with static/global arena (firmware-style backing on desktop)
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all

# Lean runtime libraries for firmware
make NETKIT_TARGET=mcu lib
make NETKIT_TARGET=mpu lib

# MCU firmware with CMSIS + core flags
make NETKIT_ARCH=CM4 NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib
make NETKIT_ARCH=M33 NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib

# MCU/MPU with optional heap arena API
make NETKIT_TARGET=mcu NETKIT_HEAP_ARENA=1 lib
make NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib

# Convenience aliases
make cpu              # NETKIT_TARGET=cpu (heap default)
make cpu-global       # NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1
make mcu              # NETKIT_TARGET=mcu lib
make mcu-heap         # NETKIT_TARGET=mcu NETKIT_HEAP_ARENA=1 lib
make mpu              # NETKIT_TARGET=mpu lib
make mpu-heap         # NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib
make cmsis-init       # fetch CMSIS-NN + CMSIS-DSP
```

## Quick commands (CMake)

```bash
# Desktop — CMSIS-DSP on by default, reference NN, loop unroll enabled
cmake -B cmake-build
cmake --build cmake-build
./cmake-build/netkit test

# Desktop — force CMSIS-NN for parity testing
cmake -B cmake-build -DNETKIT_CMSIS_NN=ON -DNETKIT_CMSIS_DSP=ON
cmake --build cmake-build

# Embedded firmware (use with your toolchain file)
cmake -B build-firmware -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_ARCH=CM7
cmake --build build-firmware

cmake -B build-m55 -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_ARCH=M55
```

CMake cache options mirror Make: `NETKIT_TARGET`, `NETKIT_ARCH`, `NETKIT_CMSIS_NN`, `NETKIT_CMSIS_DSP`, `NETKIT_GLOBAL_ARENA`, `NETKIT_HEAP_ARENA`.

## CPU (desktop)

Use for local development, CI, and the **`netkit` CLI**.

**Build outputs:**

- `netkit` — CLI (`test`, `run`, `inspect`)
- `libnetkit.a` — runtime + desktop extras
- `examples/infer_cpp`, `examples/infer_c`, `tests/test_c_api`

**Desktop-only** (guarded by `NETKIT_DESKTOP`):

- `Cli::Run` / `nk_cli_run`
- `run_all_tests` / `nk_run_all_tests`
- Future tensor analysis / debug tooling

**Arena:** CLI and regression use **heap** with model-sized buffers (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN). Examples and C API smoke tests use **`NK_ARENA_DEFAULT_CAPACITY` (4 MiB)**. Build with `NETKIT_GLOBAL_ARENA=1` to use a static buffer instead of heap.

See [CLI.md](CLI.md).

## MCU (lean runtime)

Inference-only library for microcontrollers. **No CLI, no regression runner.**

Default arena: caller-owned static or global buffer sized with **`NK_ARENA_DEFAULT_CAPACITY` (64 KiB)**:

```c
alignas(max_align_t) static unsigned char arena_mem[65536];
nk_arena_t arena;
nk_arena_init(&arena, arena_mem, sizeof(arena_mem));
```

Optional heap arena when built with `NETKIT_HEAP_ARENA=1` — **one** `malloc` at startup, **never** freed:

```c
nk_arena_init_heap(&arena, capacity);
/* ... entire firmware lifetime ... */
/* nk_arena_destroy_heap() is a no-op on MCU/MPU */
```

**Build output:** `libnetkit.a` only.

## MPU (lean runtime)

Same lean runtime as MCU. Default static arena constant is **`NK_ARENA_DEFAULT_CAPACITY` (128 KiB)** — use a caller-owned buffer of at least that size for small models, or size up with `nk_inspect_model()` for larger graphs.

## Source split

| Component | CPU | MCU / MPU |
|-----------|:---:|:---------:|
| Arena, tensor, ops, MLP, CNN | ✓ | ✓ |
| `.nk` loader (`nk_loader`, `nk_format`) | ✓ | ✓ |
| C API bridge (`netkit_api.cpp`) | ✓ | ✓ |
| CLI (`cli.cpp`, `main.cpp`) | ✓ | — |
| Regression (`test.cpp`, `nk_regression.cpp`) | ✓ | — |

## Cross-compilation

The Makefile uses host `clang`/`clang++` for desktop builds. For firmware:

1. Set **`NETKIT_ARCH`** to your core (e.g. `CM4`, `M33`, `M55`, `NEON`).
2. Build with `NETKIT_TARGET=mcu` or `mpu` and link `libnetkit.a` into your board project, **or**
3. Use CMake with a **`CMAKE_TOOLCHAIN_FILE`** and `-DNETKIT_ARCH=...`, **or**
4. Add runtime `.cpp` sources to your board build with `-DNETKIT_TARGET_MCU=1` (or `MPU`) and `-std=c++26 -Iinclude`.

Your firmware toolchain must still pass the appropriate `-mcpu` / `-march` flags; netkit's `NETKIT_ARCH` adds the matching CMSIS `ARM_MATH_*` preprocessor defines.

## CMSIS backends

Optional compile-time kernel backends (Apache-2.0). Fetch once with `make cmsis-init`.

### CMSIS-NN

[ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) accelerates **conv2d**, **max-pool**, **fully-connected**, **activations** (ReLU, sigmoid, tanh, leaky ReLU, ReLU6, softmax), and **elementwise add** when enabled.

### CMSIS-DSP

[ARM CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) accelerates **Ops** primitives: elementwise add/mul, scale, clip (ReLU/ReLU6 fallback), matrix multiply, and fully-connected (via `arm_mat_vec_mult_f32`).

Both backends are **compile-time only** — one binary, one backend set; no runtime switching. Desktop builds use CMSIS portable scalar paths for regression; embedded builds use core-specific `ARM_MATH_*` flags from `NETKIT_ARCH`.

### Make flags

```bash
make cmsis-init

# CMSIS-NN only
make NETKIT_CMSIS_NN=1 test-cpp

# CMSIS-DSP only (loop unroll enabled automatically)
make NETKIT_CMSIS_DSP=1 test-cpp

# Both backends
make NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 test-cpp

# Firmware library
make NETKIT_ARCH=CM4 NETKIT_TARGET=mcu NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib
```

| Makefile flag | Macro | Effect |
|---------------|-------|--------|
| `NETKIT_CMSIS_NN=1` | `NETKIT_USE_CMSIS_NN` | Conv2d, pool, FC, activations, softmax, NN elementwise add |
| `NETKIT_CMSIS_DSP=1` | `NETKIT_USE_CMSIS_DSP` | Ops add/mul/scale/clip/matmul; FC fallback; `ARM_MATH_LOOPUNROLL` |
| `NETKIT_ARCH=<core>` | `ARM_MATH_*` (see table above) | Core-specific CMSIS-DSP/NN tuning |

Dense weights use CMSIS-NN `[out, in]` layout via `FullyConnected` (same as PyTorch `nn.Linear`).

Float32 CMSIS-NN support is **experimental** upstream. Helium (MVE) and Neon targets get optimized kernels when `NETKIT_ARCH` and the toolchain flags align.

### Dispatch priority (typical forward pass)

```
Conv/FC (ReLU/ReLU6)  → fused in CMSIS-NN when that path succeeds
Other activations     → CMSIS-NN arm_nn_activation_f32 / arm_softmax_f32
ReLU/ReLU6 fallback   → CMSIS-DSP arm_clip_f32 → reference
Bias add (ref FC)     → CMSIS-NN add → CMSIS-DSP add → reference
```

## Testing

Full regression (`make test` — 69 embedded C++ cases + 49 Python ONNX parity) requires **`NETKIT_TARGET=cpu`**. Validate on desktop first, then run device smoke tests with the lean MCU/MPU library.

```bash
make cmsis-init
make NETKIT_CMSIS_NN=1 test-cpp
make NETKIT_CMSIS_DSP=1 test-cpp
make NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 test-cpp
```

See [TESTING.md](TESTING.md) and [ARENA.md](ARENA.md).

Related: [PHILOSOPHY.md](PHILOSOPHY.md) · [GETTING_STARTED.md](GETTING_STARTED.md)
