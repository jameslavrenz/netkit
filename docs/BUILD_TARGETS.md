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
| **GNU Make** | Yes | `make`, `make test` (primary; local + manual CI) |
| **CMake** | Optional | `cmake -B cmake-build && cmake --build cmake-build` — same flags, auto-detects desktop vs embedded via `NETKIT_ARCH` |

Both use the same optional CMSIS backends and `NETKIT_ARCH` → `ARM_MATH_*` mapping (`third_party/netkit_arch.mk`, `cmake/netkit_arch.cmake`).

## Target architecture (`NETKIT_ARCH`)

**If `NETKIT_ARCH` is unset (empty), the build is a native desktop / CPU application** — no core-specific `ARM_MATH_*` defines. Set `NETKIT_ARCH` when cross-compiling firmware for a specific Arm core.

| `NETKIT_ARCH` | CMSIS-DSP flag(s) | Profile | CMSIS-NN |
|---------------|-------------------|---------|----------|
| *(unset)* | `__GNUC_PYTHON__` (desktop host only) | CPU | off (flag ignored) |
| `CM0`, `M0` | `ARM_MATH_CM0` | MCU | when `NETKIT_CMSIS_NN=1` |
| `CM0PLUS`, `M0PLUS` | `ARM_MATH_CM0PLUS` | MCU | when `NETKIT_CMSIS_NN=1` |
| `CM3`, `M3` | `ARM_MATH_CM3` | MCU | when `NETKIT_CMSIS_NN=1` |
| `CM4`, `M4` | `ARM_MATH_CM4` | MCU | when `NETKIT_CMSIS_NN=1` |
| `CM7`, `M7` | `ARM_MATH_CM7` | MCU | when `NETKIT_CMSIS_NN=1` |
| `M23`, `CM23` | `ARM_MATH_ARMV8MBL` | MCU | when `NETKIT_CMSIS_NN=1` |
| `M33`, `CM33` | `ARM_MATH_ARMV8MML`, `__DSP_PRESENT=1` | MCU | when `NETKIT_CMSIS_NN=1` |
| `M55`, `CM55` | `ARM_MATH_M55`, `ARM_MATH_MVEF`, `ARM_MATH_MVEI` | MCU | when `NETKIT_CMSIS_NN=1` |
| `M85`, `CM85` | `ARM_MATH_M85`, `ARM_MATH_MVEF`, `ARM_MATH_MVEI` | MCU | when `NETKIT_CMSIS_NN=1` |
| `A32`, `MPU` | `ARM_MATH_A32` | MPU | off (flag ignored) |
| `NEON`, `A64` | `ARM_MATH_NEON` | MPU | off (flag ignored) |

Aliases like `Cortex-M4` normalize to `CM4`. CMake also sets `NETKIT_TARGET` from the arch (`cpu` / `mcu` / `mpu`); with Make you pass `NETKIT_TARGET` explicitly for firmware.

### Additional CMSIS preprocessor flags (auto-applied)

| Flag | When | Purpose |
|------|------|---------|
| `ARM_MATH_LOOPUNROLL` | All CMSIS-DSP builds (desktop + embedded) | CMSIS-DSP 1.10+ defaults to no unroll; this enables 4× loop unroll for faster inference |
| `__DSP_PRESENT=1` | `NETKIT_ARCH=M33` | Required for Armv8-M DSP extension; avoids scalar fallbacks |
| `ARM_MATH_MVEF` / `ARM_MATH_MVEI` | `NETKIT_ARCH=M55`, `M85` | Helium vector extensions (complements toolchain `-mcpu=cortex-m55`) |
| `HOST` / `__GNUC_PYTHON__` | Desktop only, or `NETKIT_HOST_SMOKE=1` on host MCU/MPU smoke | CMSIS-DSP portable host path (no CMSIS-Core device headers) |

MCU builds also add `-Ithird_party/CMSIS-Core/CMSIS/Core/Include` when that directory exists.

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
| `NETKIT_WEIGHTS_IN_RAM` | `1` — copy weight/bias payload into arena at buffer/AOT load (default **CPU/MPU**); `0` — coefs stay in `.nk` flash blob (default **MCU**) |
| `NETKIT_USE_CMSIS_NN` | CMSIS-NN backends enabled (see CMSIS section) |
| `NETKIT_USE_CMSIS_DSP` | CMSIS-DSP backends enabled (see CMSIS section) |
| `NETKIT_IM2COL` | float Conv2D strategy (single tri-state knob): `0` = direct loops only, `1` = partial im2col, `2` = full im2col + GEMM. Default **`0` (direct) on all targets** (cpu/mcu/mpu) — direct convolution with the multi-accumulator dot is fastest for the small models we target. Leave unset for the default, or opt into `1`/`2` per workload. Int8 quantized inference uses CMSIS-NN, not float im2col. |
| `NETKIT_LOOP_UNROLL` | `1` — **experimental** 4× manual loop unroll in **netkit reference kernels** only (default **0**). Increases `.text` size; can exceed flash on small MCUs. Most likely worth considering on **MPU** targets with flash headroom — avoid on tight MCUs. Does not affect CMSIS (`ARM_MATH_LOOPUNROLL` is separate). |
| `NETKIT_DW_ROW_ACCUM` | Depthwise conv cross-row accumulator strategy (default **1**). `1` = round-robin kernel rows across 4 independent accumulators (breaks the cross-row serial dependency; +~144 B). `0` = single serial cross-row accumulator. Both keep the 4-accumulator inner tap reduction (`dot_strided`). Benchmarks show no measurable difference on out-of-order hosts; the break can help in-order MCUs with tall (5×5) kernels. Defined in `src/conv_depthwise_kernel.cpp`. |
| `NETKIT_HOST_SMOKE` | Host MCU/MPU smoke only — adds `__GNUC_PYTHON__` for CMSIS without CMSIS-Core |

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

# Override weight load policy (buffer/AOT path; file load always copies)
make NETKIT_TARGET=mcu NETKIT_WEIGHTS_IN_RAM=1 lib   # copy coefs to SRAM when RAM fits (faster inference)
make NETKIT_TARGET=cpu NETKIT_WEIGHTS_IN_RAM=0 lib     # test flash-backed load on desktop

# Convenience aliases
make cpu              # NETKIT_TARGET=cpu (heap default)
make cpu-global       # NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1
make mcu              # NETKIT_TARGET=mcu lib
make mcu-heap         # NETKIT_TARGET=mcu NETKIT_HEAP_ARENA=1 lib
make mpu              # NETKIT_TARGET=mpu lib
make mpu-heap         # NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib
make cmsis-init       # fetch CMSIS-Core + CMSIS-NN + CMSIS-DSP
make embedded-smoke   # lean MCU/MPU smoke binary (test_mlp, cnn_4x4_single)
make test-embedded-smoke-matrix   # 7-profile host smoke (see TESTING.md)
```

## Quick commands (CMake)

```bash
# Desktop — CMSIS-DSP on by default (cpu profile), CMSIS-NN off
cmake -B cmake-build -DNETKIT_TARGET=cpu
cmake --build cmake-build
./cmake-build/netkit test

# MCU firmware (use with your toolchain file)
cmake -B build-firmware -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_TARGET=mcu -DNETKIT_ARCH=CM7 -DNETKIT_CMSIS_NN=ON
cmake --build build-firmware

cmake -B build-m55 -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_TARGET=mcu -DNETKIT_ARCH=M55 -DNETKIT_CMSIS_NN=ON
```

CMake cache options mirror Make: `NETKIT_TARGET`, `NETKIT_ARCH`, `NETKIT_CMSIS_NN`, `NETKIT_CMSIS_DSP`, `NETKIT_GLOBAL_ARENA`, `NETKIT_HEAP_ARENA`, `NETKIT_WEIGHTS_IN_RAM`.

## CPU (desktop)

Use for local development and the **`netkit` CLI**.

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

Optional compile-time kernel backends (Apache-2.0). Fetch once with `make cmsis-init` (CMSIS-Core headers + optional NN/DSP libraries).

### CMSIS-Core

[ARM CMSIS 6](https://github.com/ARM-software/CMSIS_6) **Core(M)** headers are required for **on-device** MCU firmware when `NETKIT_ARCH` is set and CMSIS backends are enabled. Submodule path: `third_party/CMSIS-Core` → `CMSIS/Core/Include`. Not needed for host `make test` or `NETKIT_HOST_SMOKE=1` smoke builds.

### CMSIS-NN

[ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) accelerates layer kernels when **`NETKIT_CMSIS_NN=1`**, **`NETKIT_TARGET=mcu`**, and **`NETKIT_ARCH`** is Cortex-M (CM4, M33, …). Includes **depthwise conv** (linked from `arm_depthwise_conv_f32.c` and support kernels). On **cpu** or **mpu**, the flag is **ignored** (Make warning) and reference kernels run.

### CMSIS-DSP

[ARM CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) accelerates vector/matrix ops: elementwise add/mul, scale, clip (ReLU/ReLU6 fallback), `MatMul`, fully-connected / batch-norm fallbacks, **LayerNorm2d**, and **GRN** on **desktop, MPU, and MCU** (vector role when both NN and DSP are enabled).

When enabled, **`cmsis_dsp_util`** (`include/cmsis_dsp_util.hpp`) routes contiguous f32/q7 copy, argmax, dot-product, mul/add/scale through CMSIS-DSP in float conv/im2col paths, reference fallbacks, and MCU benchmark firmware. See [KERNELS.md](KERNELS.md).

**MCU firmware:** stage inputs in SRAM before timed inference (flash-resident test vectors + conv from XIP is slower than TFLM’s tensor-arena copy). Both NUCLEO board `main.cpp` files use `g_input_staging` + `CmsisDspUtil::CopyInt8` / `CopyF32`.

Both backends are **compile-time only** — one binary, one backend set; no runtime switching.

**NN vs DSP:** On **MCU with both flags**, CMSIS-NN owns layer kernels — CMSIS-DSP does not substitute. On **desktop and MPU**, use CMSIS-DSP (CMSIS-NN is not linked).

### Make flags

CMSIS backends are **not** inferred from `NETKIT_ARCH` alone — use `NETKIT_CMSIS_*=1` or the **profile defaults** below. Override with `make NETKIT_CMSIS_DSP=0` for a reference-kernel build (the CMake smoke step in CI builds this way).

**MCU board firmware:** `boards/nucleo-f446re-cnn-int8/Makefile` **overrides** `NETKIT_CMSIS_DSP` and `NETKIT_CMSIS_NN` to `1` so a host env that leaves CMSIS-NN off cannot link CMSIS stub kernels. Pass `NETKIT_CPPFLAGS` on the LTO link line so CMSIS macros match compile units.

| `NETKIT_TARGET` | Default `NETKIT_CMSIS_DSP` | Default `NETKIT_CMSIS_NN` | Default `NETKIT_XNNPACK` |
|-----------------|----------------------------|---------------------------|--------------------------|
| `cpu` | 1 | 0 | 1 (requires `./tools/fetch_xnnpack.sh`) |
| `mcu` | 1 | 1 (effective only with Cortex-M `NETKIT_ARCH`) | 0 |
| `mpu` | 1 | 0 | 1 (requires `./tools/fetch_xnnpack.sh`) |

```bash
make cmsis-init
make xnnpack-init                # once, for cpu/mpu XNNPACK LayerFast

# Profile defaults (no extra flags needed after fetch)
make test-cpp                    # cpu: CMSIS-DSP + XNNPACK
make NETKIT_TARGET=mcu lib       # mcu: CMSIS-DSP + CMSIS-NN (set NETKIT_ARCH=CM4 for NN)
make NETKIT_TARGET=mpu lib       # mpu: CMSIS-DSP + XNNPACK

# Explicit off (reference kernels / CMSIS-DSP only)
make NETKIT_CMSIS_DSP=0 NETKIT_CMSIS_NN=0 NETKIT_XNNPACK=0 rebuild test

# Reference-kernel 4× loop unroll — experimental (MPU only if at all; increases .text)
make NETKIT_LOOP_UNROLL=1 test-cpp
```

| Makefile flag | Macro | Effect |
|---------------|-------|--------|
| `NETKIT_CMSIS_NN=1` | `NETKIT_USE_CMSIS_NN` | MCU + Cortex-M `NETKIT_ARCH` only (ignored with warning on cpu/mpu) |
| `NETKIT_CMSIS_DSP=1` | `NETKIT_USE_CMSIS_DSP` | Ops add/mul/scale/clip/matmul; FC/batch-norm on desktop/MPU; `ARM_MATH_LOOPUNROLL` |
| `NETKIT_XNNPACK=1` | `NETKIT_USE_XNNPACK` | cpu/mpu float32 LayerFast + int8 qs8 (conv/depthwise/pool/FC); ignored on mcu; soft-falls back if not fetched |
| `NETKIT_LOOP_UNROLL=1` | `NETKIT_LOOP_UNROLL=1` | **Experimental.** 4× manual unroll in reference kernels only (default 0). Larger `.text`; verify flash budget. Most likely an **MPU** consideration — avoid on flash-limited MCUs. |
| `NETKIT_ARCH=<core>` | `ARM_MATH_*` (see table above) | Core-specific CMSIS-DSP/NN tuning |

Dense weights use CMSIS-NN `[out, in]` layout via `Kernels::FullyConnectedWithBias` (same as PyTorch `nn.Linear`).

Float32 CMSIS-NN support is **experimental** upstream. Helium (MVE) and Neon targets get optimized kernels when `NETKIT_ARCH` and the toolchain flags align.

### XNNPACK

[Google XNNPACK](https://github.com/google/XNNPACK) (BSD-3) accelerates float32 and int8 (qs8) kernels on **cpu** and **mpu** when `NETKIT_XNNPACK=1` (the default for those profiles). It is the host/MPU analogue of CMSIS-NN on MCU:

- **float32:** `XnnpackKernel` is `LayerFast` in `active_kernel.hpp` (conv, depthwise, max/avg pool, FC with ReLU/ReLU6 clamp).
- **int8:** `XnnpackQuant` is tried first in the quantized plan / `QuantOps` path (qs8 conv, depthwise, max pool, FC), then CMSIS-NN if enabled, then reference.

MCU builds default to `NETKIT_XNNPACK=0` and ignore the flag if forced on.

```bash
./tools/fetch_xnnpack.sh   # or: make xnnpack-init
make NETKIT_XNNPACK=1 lib  # default on cpu/mpu once fetched
make NETKIT_XNNPACK=0 lib  # force reference/CMSIS-DSP LayerFast path
```

If headers/libs are missing, Make prints a warning and builds without XNNPACK (reference LayerFast). CI forces `NETKIT_XNNPACK=0`.

### Kernel dispatch (CRTP)

Backends are composed at **compile time** via `active_kernel.hpp` — there is no runtime backend switch. Layer and ops code call `Kernels::Op(...)`; `ComposedKernel` tries CMSIS/XNNPACK `Try*` methods and falls back to `ReferenceKernel`.

### Layer dispatch (OpsResolver)

CNN graph execution uses an **MCU-safe op registry** (`include/ops_resolver.hpp`) — the **interpreter path**: function pointers and fixed static storage, no virtuals, heap, or `std::vector`. For production speed on a fixed graph, pair AOT embed + packager optimizations with a trimmed `NkOpList` — [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

**C++26 constinit resolver tables**: `NkOpList<Ops...>` builds the lookup table at compile time into `constinit` static storage (no dynamic static initialization).

```cpp
#include "layer_op_registry.hpp"

// Only conv + dense — linker can drop unused operator bodies
cnn.SetOpsResolver(NkOpList<NkConv2DOpDescriptor, NkDenseOpDescriptor>::View());
```

`GetDefaultOpsResolver()` returns `NkAllLayerOps::View()` when all `src/layer_ops/*.cpp` units are linked.

### Trimmed firmware (link only the ops you need)

Each layer kind lives in its own translation unit under `src/layer_ops/` with a matching descriptor in `include/layer_ops/`. Link only the `.cpp` files your graph uses, then point the network at a compile-time resolver:

```cpp
#include "layer_ops/nk_conv2d_op.hpp"
#include "layer_ops/nk_max_pool2d_op.hpp"
#include "layer_ops/nk_flatten_op.hpp"
#include "layer_ops/nk_dense_op.hpp"

using TrimOps = NkOpList<NkConv2DOpDescriptor, NkMaxPool2DOpDescriptor,
                         NkFlattenOpDescriptor, NkDenseOpDescriptor>;
cnn.SetOpsResolver(TrimOps::View());
```

Link: `nk_op_conv2d.cpp`, `nk_op_max_pool2d.cpp`, `nk_op_flatten.cpp`, `nk_op_dense.cpp`, `ops_resolver.cpp` (omit `ops_resolver_default.cpp`, avg-pool, and batch-norm TUs).

Verify on desktop:

```bash
make trim-lib              # libnetkit_trim.a — conv/max-pool/flatten/dense only
make check-trim-lib        # asserts avg-pool + batch-norm bodies are absent
```

See `examples/trim_firmware.cpp` for a minimal recipe.

Full architecture: [KERNELS.md](KERNELS.md).

**Typical forward pass (MCU + CMSIS-NN + CMSIS-DSP):**

| Op family | Primary backend |
|-----------|-----------------|
| Conv / depthwise conv / pool / batch norm / FC / NN activations / GELU / softmax | CMSIS-NN (`LayerFast`) |
| Elementwise mul / matmul / add / scale / ReLU6 clip / LayerNorm2d / GRN | CMSIS-DSP (`VectorFast`) |
| Fused blocks (ResNet / MobileNet / ConvNeXt internals) | Same via `fused_kernel_ops.hpp` |
| Any `Try*` miss | Reference |

## Testing

Default regression (`make test`) and full suite (`make test-full`) require **`NETKIT_TARGET=cpu`**. CMSIS backends are validated locally via host smoke (`make test-embedded-smoke-matrix` with `NETKIT_HOST_SMOKE=1`), which exercises `test_mlp` and `cnn_4x4_single` on seven MCU/MPU profiles.

```bash
make cmsis-init
make NETKIT_CMSIS_DSP=1 test-cpp
make test-embedded-smoke-matrix          # host MCU/MPU + CMSIS profiles
make NETKIT_HOST_SMOKE=1 NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 lib
```

See [TESTING.md](TESTING.md) and [ARENA.md](ARENA.md).

Related: [PHILOSOPHY.md](PHILOSOPHY.md) · [GETTING_STARTED.md](GETTING_STARTED.md)
