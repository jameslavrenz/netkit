# Build Targets

netkit builds for ISA-qualified deployment profiles. Select one with **`NETKIT_TARGET`** when invoking Make or CMake, or define the matching `-D` flag in your embedded toolchain.

| Target | Makefile / CMake | Role | Default backends |
|--------|------------------|------|------------------|
| **CPU** | `NETKIT_TARGET=cpu` (default) | Desktop dev, debug, CI | XNNPACK on (any host ISA); CMSIS-NN off; **no MLAS** (not needed — see [STATUS.md](STATUS.md#host-three-way-suite-netkit-vs-tf-lite-vs-onnx-runtime)) |
| **MCU_ARM** | `NETKIT_TARGET=mcu_arm` | Arm microcontroller firmware | CMSIS-NN (int8 production); XNNPACK forbidden |
| **MPU_ARM** | `NETKIT_TARGET=mpu_arm` | Arm microprocessor / RTOS | XNNPACK |
| **MCU_RISC** | `NETKIT_TARGET=mcu_risc` | RISC-V MCU | generic kernels only (fast; CMSIS + XNNPACK forbidden) |
| **MPU_RISC** | `NETKIT_TARGET=mpu_risc` | RISC-V MPU | XNNPACK on (strong RISC-V MPU support); CMSIS-NN forbidden |
| **MCU_ESP** | `NETKIT_TARGET=mcu_esp` | Espressif MCU (ESP32 / S3 / C3 / C6 / P4) | ESP-NN on (int8 production); float32 reference; XNNPACK forbidden |

Legacy `NETKIT_TARGET=mcu` / `mpu` are **rejected** — use `mcu_arm` / `mpu_arm`.

**Maturity:** float32 + int8 are **complete** on **cpu**, **Arm MCU/MPU**, and **RISC MCU/MPU**. **MCU_ESP** uses [ESP-NN](https://github.com/espressif/esp-nn) for int8 (CMSIS-style); ESP-NN has no float API, so float32 uses reference. RISC MCU uses fast generic kernels today — [STATUS.md](STATUS.md).

Derived macros: `NETKIT_CLASS_MCU` / `NETKIT_CLASS_MPU` (firmware class), `NETKIT_ISA_ARM` / `NETKIT_ISA_RISC` / `NETKIT_ISA_ESP` (backend family).

## Build systems

| System | Primary? | Notes |
|--------|----------|-------|
| **GNU Make** | Yes | `make`, `make test` (primary; local + manual CI) |
| **CMake** | Optional | `cmake -B cmake-build && cmake --build cmake-build` — same flags, auto-detects desktop vs embedded via `NETKIT_ARCH` |

Both use the same optional backends and `NETKIT_ARCH` mapping (`third_party/netkit_arch.mk`, `cmake/netkit_arch.cmake`) — Arm `ARM_MATH_*` for CMSIS-NN, Espressif `CONFIG_IDF_TARGET_*` for ESP-NN.

## Target architecture (`NETKIT_ARCH`)

**If `NETKIT_ARCH` is unset (empty), the build is a native desktop / CPU application** — no core-specific `ARM_MATH_*` / ESP chip defines. Set `NETKIT_ARCH` when cross-compiling firmware for a specific Arm or Espressif core.

| `NETKIT_ARCH` | Flag(s) | Profile | Accelerated NN |
|---------------|---------|---------|----------------|
| *(unset)* | `__GNUC_PYTHON__` (desktop host only) | CPU | CMSIS / ESP off |
| `CM0`, `M0` | `ARM_MATH_CM0` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `CM0PLUS`, `M0PLUS` | `ARM_MATH_CM0PLUS` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `CM3`, `M3` | `ARM_MATH_CM3` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `CM4`, `M4` | `ARM_MATH_CM4` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `CM7`, `M7` | `ARM_MATH_CM7` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `M23`, `CM23` | `ARM_MATH_ARMV8MBL` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `M33`, `CM33` | `ARM_MATH_ARMV8MML`, `__DSP_PRESENT=1` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `M55`, `CM55` | `ARM_MATH_M55`, `ARM_MATH_MVEF`, `ARM_MATH_MVEI` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `M85`, `CM85` | `ARM_MATH_M85`, `ARM_MATH_MVEF`, `ARM_MATH_MVEI` | MCU Arm | CMSIS-NN when `NETKIT_CMSIS_NN=1` |
| `A32`, `MPU` | `ARM_MATH_A32` | MPU Arm | CMSIS off |
| `NEON`, `A64` | `ARM_MATH_NEON` | MPU Arm | CMSIS off |
| `ESP32` | `CONFIG_IDF_TARGET_ESP32` | MCU Espressif | ESP-NN when `NETKIT_ESP_NN=1` |
| `ESP32S3` | `CONFIG_IDF_TARGET_ESP32S3` | MCU Espressif | ESP-NN when `NETKIT_ESP_NN=1` |
| `ESP32C3` | `CONFIG_IDF_TARGET_ESP32C3` | MCU Espressif | ESP-NN when `NETKIT_ESP_NN=1` |
| `ESP32C6` | `CONFIG_IDF_TARGET_ESP32C6` | MCU Espressif | ESP-NN when `NETKIT_ESP_NN=1` |
| `ESP32P4` | `CONFIG_IDF_TARGET_ESP32P4` | MCU Espressif | ESP-NN when `NETKIT_ESP_NN=1` |

Aliases like `Cortex-M4` normalize to `CM4`; `ESP32-S3` / `ESP32_S3` normalize to `ESP32S3`. CMake also sets `NETKIT_TARGET` from the arch (`cpu` / `mcu_arm` / `mpu_arm` / `mcu_esp`); with Make you pass `NETKIT_TARGET` explicitly for firmware.

### Additional CMSIS preprocessor flags (auto-applied)

| Flag | When | Purpose |
|------|------|---------|
| `ARM_MATH_LOOPUNROLL` | CMSIS-NN embedded builds | Enables 4× loop unroll in CMSIS-NN kernels |
| `__DSP_PRESENT=1` | `NETKIT_ARCH=M33` | Required for Armv8-M DSP extension; avoids scalar fallbacks |
| `ARM_MATH_MVEF` / `ARM_MATH_MVEI` | `NETKIT_ARCH=M55`, `M85` | Helium vector extensions (complements toolchain `-mcpu=cortex-m55`) |
| `HOST` / `__GNUC_PYTHON__` | Desktop only, or `NETKIT_HOST_SMOKE=1` on host MCU/MPU smoke | CMSIS-NN portable host path (no CMSIS-Core device headers) |

MCU builds also add `-Ithird_party/CMSIS-Core/CMSIS/Core/Include` when that directory exists.

## Arena backing defaults

| Target | Default arena | Override flag |
|--------|---------------|---------------|
| **CPU** | **Heap** (`malloc` backing via `nk_arena_init_heap` / `Arena::init_heap`) | `NETKIT_GLOBAL_ARENA=1` → static/global buffer only |
| **MCU** | **Global/static only** (`nk_arena_init` with your buffer). **No heap ever** — `malloc` / `new` / `delete` / `free` are forbidden; weights stay in the flash `.nk` image | — |
| **MPU** | **Global/static** (same as MCU default) | `NETKIT_HEAP_ARENA=1` → also compile heap helpers |

Compile-time macros (from `include/netkit_config.h`):

| Macro | Meaning |
|-------|---------|
| `NETKIT_TARGET_CPU` | Desktop / CPU build |
| `NETKIT_TARGET_MCU_ARM` | Arm MCU build |
| `NETKIT_TARGET_MPU_ARM` | Arm MPU build |
| `NETKIT_TARGET_MCU_RISC` | RISC-V MCU (generic kernels; CMSIS + XNNPACK forbidden) |
| `NETKIT_TARGET_MPU_RISC` | RISC-V MPU (XNNPACK default; CMSIS forbidden) |
| `NETKIT_TARGET_MCU_ESP` | Espressif MCU (ESP-NN int8 production) |
| `NETKIT_CLASS_MCU` / `NETKIT_CLASS_MPU` | Firmware class (arena / lean API) |
| `NETKIT_ISA_ARM` / `NETKIT_ISA_RISC` / `NETKIT_ISA_ESP` | Instruction-set family (backend policy) |
| `NETKIT_DESKTOP` | CPU only — CLI, regression, debug tooling |
| `NETKIT_ARENA_HEAP` | Heap arena API compiled in (CPU default; MPU when opted in; **never MCU**) |
| `NETKIT_GLOBAL_ARENA` | CPU only — force global/static arena instead of heap default |
| `NETKIT_MCU_ACCEL_ONLY` / `NETKIT_MCU_CMSIS_ONLY` | MCU class + `REFERENCE_QUANT_LOOPS=0` — QuantOps reference loops omitted (flash); applies to CMSIS-NN **and** ESP-NN production |
| `NETKIT_DISABLE_IOSTREAM` | Default on MCU — keeps iostream out of flash |
| `NETKIT_USE_CMSIS_NN` | CMSIS-NN backends enabled (see CMSIS section) |
| `NETKIT_USE_ESP_NN` | ESP-NN backends enabled (see ESP-NN section) |
| `NETKIT_IM2COL` | Conv2D strategy for **float reference** and **int8 QuantOps** only: `0` = direct loops, `1` = partial im2col, `2` = full im2col + GEMM. **Default `0` on cpu / MCU / MPU.** CMSIS-NN, ESP-NN, and XNNPACK ignore this knob. Prefer leaving `0`; at most try `1` on MCU or on MPU/cpu when XNNPACK is off (small bump possible). Avoid `2` unless profiling shows a clear win. See guidance below. |
| `NETKIT_LOOP_UNROLL` | `1` — **experimental** 4× manual loop unroll in **netkit reference kernels** only (default **0**). Increases `.text` size; can exceed flash on small MCUs. Does not affect CMSIS (`ARM_MATH_LOOPUNROLL` is separate). |
| `NETKIT_DW_ROW_ACCUM` | Depthwise conv cross-row accumulator strategy (default **1**). See `src/conv_depthwise_kernel.cpp`. |
| `NETKIT_HOST_SMOKE` | Host MCU/MPU smoke only — `__GNUC_PYTHON__` for CMSIS-NN without CMSIS-Core; clears ESP chip asm flags so ESP-NN builds ANSI-only on host |
| `NETKIT_USE_MMAP` | File mmap for path-based `.nk` load (`NETKIT_MMAP=0\|1`). POSIX on macOS/Linux; Win32 on Windows. Default **1** on cpu + any MPU; **forbidden** on MCU |

### `NETKIT_IM2COL` guidance

im2col is primarily an **MCU / reference-path** Conv2D optimization. It does **not** affect CMSIS-NN, ESP-NN, or XNNPACK LayerFast convolutions.

| Value | Meaning | Recommendation |
|-------|---------|----------------|
| **`0` (default)** | Direct nested loops | **Safest — leave this everywhere** (cpu, MPU, MCU) |
| **`1`** | Partial im2col | Optional small speedup on **MCU**, or on **MPU/cpu when `NETKIT_XNNPACK=0`**. Host A/B saw modest gains on some float CNN reference runs; not worth it when XNNPACK is on |
| **`2`** | Full im2col + GEMM | Rarely useful; higher scratch / code size. Do not enable without profiling |

```bash
# Default (recommended)
make NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 lib

# Optional MCU / reference-path experiment
make NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 NETKIT_IM2COL=1 lib
make NETKIT_XNNPACK=0 NETKIT_IM2COL=1 lib   # MPU/cpu reference only
```

Default arena constant (`NK_ARENA_DEFAULT_CAPACITY` / `Arena::kDefaultCapacity`):

| Target | Default | Override |
|--------|---------|----------|
| **MCU** | **64 KiB** | `-DNK_ARENA_DEFAULT_CAPACITY=<bytes>`, or Make/CMake `NETKIT_ARENA_CAPACITY=<bytes>` / `NETKIT_ARENA_KB=<KiB>` |
| **CPU / MPU** | **64 MiB** | same |

```bash
make NETKIT_TARGET=mcu_arm NETKIT_ARENA_KB=116 lib
make NETKIT_ARENA_CAPACITY=134217728 lib   # 128 MiB on CPU
c++ ... -DNK_ARENA_DEFAULT_CAPACITY=131072 ...
```

Weights always stay in the `.nk` blob. **Preferred on MCU and RTOS/bare-metal MPU:** flash/XIP or `Load*FromBuffer`. **File mmap** (`NETKIT_MMAP` / `NETKIT_USE_MMAP`): default **on** for cpu and any MPU (POSIX on macOS/Linux; Win32 `CreateFileMapping` / `MapViewOfFile` on Windows); **forbidden** on MCU. Opt out on no-OS MPU with `NETKIT_MMAP=0` (falls back to `fread` into the arena, or prefer buffer/flash). The bump arena holds activations and structs. CLI override: `./netkit --arena <size> run|inspect …`.

## Quick commands (Make)

```bash
# Desktop (default) — CLI + tests + libnetkit.a, heap arena default
make
make test
make build-all

# CPU with static/global arena (firmware-style backing on desktop)
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all

# Lean runtime libraries for firmware
make NETKIT_TARGET=mcu_arm lib
make NETKIT_TARGET=mpu_arm lib
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 lib

# MCU firmware with CMSIS + core flags
make NETKIT_ARCH=CM4 NETKIT_TARGET=mcu_arm NETKIT_CMSIS_NN=1 lib
make NETKIT_ARCH=M33 NETKIT_TARGET=mcu_arm NETKIT_CMSIS_NN=1 lib

# Espressif MCU with ESP-NN (int8 production; float32 reference)
make esp-nn-init
make NETKIT_ARCH=ESP32S3 NETKIT_TARGET=mcu_esp lib
make NETKIT_ARCH=ESP32C6 NETKIT_TARGET=mcu_esp NETKIT_HOST_SMOKE=1 lib   # host ANSI

# MPU with optional heap arena API (MCU forbids heap)
make NETKIT_TARGET=mpu_arm NETKIT_HEAP_ARENA=1 lib

# Convenience aliases
make cpu              # NETKIT_TARGET=cpu (heap default)
make cpu-global       # NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1
make mcu-arm              # NETKIT_TARGET=mcu_arm lib
make mpu-arm              # NETKIT_TARGET=mpu_arm lib
make mpu-arm-heap         # NETKIT_TARGET=mpu_arm NETKIT_HEAP_ARENA=1 lib
make mcu-risc             # NETKIT_TARGET=mcu_risc lib (generic kernels)
make mpu-risc             # NETKIT_TARGET=mpu_risc lib (XNNPACK when fetched)
make mcu-esp              # NETKIT_TARGET=mcu_esp lib (needs NETKIT_ARCH=ESP32*)
make cmsis-init       # fetch CMSIS-Core + CMSIS-NN
make esp-nn-init      # fetch ESP-NN (Espressif)
make embedded-smoke   # lean MCU/MPU smoke binary (test_mlp, cnn_4x4_single)
make test-embedded-smoke-matrix   # host smoke matrix (see TESTING.md)
```

## Quick commands (CMake)

```bash
# Desktop — XNNPACK on by default (cpu profile), CMSIS-NN off
cmake -B cmake-build -DNETKIT_TARGET=cpu
cmake --build cmake-build
./cmake-build/netkit test

# MCU firmware (use with your toolchain file)
cmake -B build-firmware -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_TARGET=mcu_arm -DNETKIT_ARCH=CM7 -DNETKIT_CMSIS_NN=ON
cmake --build build-firmware

cmake -B build-m55 -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_TARGET=mcu_arm -DNETKIT_ARCH=M55 -DNETKIT_CMSIS_NN=ON

# Espressif MCU (ESP-NN)
cmake -B build-esp -DCMAKE_TOOLCHAIN_FILE=... -DNETKIT_TARGET=mcu_esp -DNETKIT_ARCH=ESP32S3 -DNETKIT_ESP_NN=ON
```

CMake cache options mirror Make: `NETKIT_TARGET`, `NETKIT_ARCH`, `NETKIT_CMSIS_NN`, `NETKIT_ESP_NN`, `NETKIT_XNNPACK`, `NETKIT_GLOBAL_ARENA`, `NETKIT_HEAP_ARENA`.

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

**Arena:** CLI and regression use **heap** with **`NK_ARENA_DEFAULT_CAPACITY` (64 MiB)** on CPU. Override with `./netkit --arena <size>`. Build with `NETKIT_GLOBAL_ARENA=1` to use a static buffer instead of heap.

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

Same lean runtime as MCU. Default static arena constant is **`NK_ARENA_DEFAULT_CAPACITY` (64 MiB)** — MPU firmware typically uses a caller-owned buffer sized with `nk_inspect_model()` rather than the full default.

**OS is orthogonal to the MPU target.** Many Cortex-A boards run FreeRTOS, Zephyr, or bare metal (no `mmap`). Defaults assume a VM-capable OS; opt out when you do not have one:

| MPU deployment | Weight load |
|----------------|-------------|
| Embedded Linux / POSIX OS / Windows | Default `NETKIT_MMAP=1` — file mmap (POSIX or Win32; same as cpu). |
| RTOS / bare metal | `make NETKIT_TARGET=mpu_arm NETKIT_MMAP=0 lib` (or buffer/flash / `Load*FromBuffer`). |

MCU builds **cannot** enable mmap (`NETKIT_MMAP=1` is forced off / compile error). Do not assume every MPU build has a virtual-memory OS.

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

1. Set **`NETKIT_ARCH`** to your core (e.g. `CM4`, `M33`, `M55`, `NEON`, `ESP32S3`, `ESP32C6`).
2. Build with `NETKIT_TARGET=mcu_arm` / `mpu_arm` / `mcu_risc` / `mpu_risc` / `mcu_esp` and link `libnetkit.a` into your board project, **or**
3. Use CMake with a **`CMAKE_TOOLCHAIN_FILE`** and `-DNETKIT_ARCH=...`, **or**
4. Add runtime `.cpp` sources to your board build with `-DNETKIT_TARGET_MCU_ARM=1` (or `MPU_ARM` / RISC / `MCU_ESP` macros) and `-std=c++26 -Iinclude`.

Your firmware toolchain must still pass the appropriate `-mcpu` / `-march` flags; netkit's `NETKIT_ARCH` adds matching CMSIS `ARM_MATH_*` or Espressif `CONFIG_IDF_TARGET_*` preprocessor defines.

## CMSIS backends

Optional compile-time kernel backends (Apache-2.0). Fetch once with `make cmsis-init` (CMSIS-Core headers + CMSIS-NN).

**CMSIS-DSP is not used or linked.** Portable helpers live in `netkit_util` (`NetkitUtil::`).

Backend story: **reference** + **XNNPACK** (cpu / MPU) + **CMSIS-NN** (Arm MCU int8) + **ESP-NN** (Espressif MCU int8).

### CMSIS-Core

[ARM CMSIS 6](https://github.com/ARM-software/CMSIS_6) **Core(M)** headers are required for **on-device** MCU firmware when `NETKIT_ARCH` is set and CMSIS-NN is enabled. Submodule path: `third_party/CMSIS-Core` → `CMSIS/Core/Include`. Not needed for host `make test` or `NETKIT_HOST_SMOKE=1` smoke builds.

### CMSIS-NN

[ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) accelerates layer kernels when **`NETKIT_CMSIS_NN=1`**, **`NETKIT_TARGET=mcu_arm`**, and **`NETKIT_ARCH`** is Cortex-M (CM4, M33, …). Includes **depthwise conv** (linked from `arm_depthwise_conv_f32.c` and support kernels). On **cpu** / **mpu_arm** the flag is **ignored** (Make warning) and reference / XNNPACK kernels run. On **RISC** targets CMSIS-NN is **forbidden** (forced off).

**Production MCU path** is **int8 + CMSIS-NN**. Float32 on MCU is still supported via portable/reference kernels, but there is **no plan** for an optimized float32 MCU build (CMSIS-DSP float helpers are intentionally not built-in). Use `NETKIT_TARGET=mcu_arm` with reference float kernels when you need float32 on-device.

**MCU firmware:** stage inputs in SRAM before timed inference. NUCLEO board `main.cpp` files use `g_input_staging` + `NetkitUtil::CopyInt8` / `CopyF32`.

Backends are **compile-time only** — one binary, one backend set; no runtime switching.

### Make flags

CMSIS-NN is **not** inferred from `NETKIT_ARCH` alone — use `NETKIT_CMSIS_NN=1` or the **profile defaults** below.

**MCU board firmware:** `boards/nucleo-f446re-*-int8/Makefile` **overrides** `NETKIT_CMSIS_NN` to `1` (unless reference quant loops). Pass `NETKIT_CPPFLAGS` on the LTO link line so CMSIS macros match compile units.

| `NETKIT_TARGET` | Default `NETKIT_CMSIS_NN` | Default `NETKIT_ESP_NN` | Default `NETKIT_XNNPACK` |
|-----------------|--------------------------|------------------------|--------------------------|
| `cpu` | 0 | 0 | 1 (requires `./tools/fetch_xnnpack.sh`) |
| `mcu_arm` | 1 (effective only with Cortex-M `NETKIT_ARCH`) | 0 (forbidden) | 0 (forbidden) |
| `mpu_arm` | 0 | 0 | 1 (requires `./tools/fetch_xnnpack.sh`) |
| `mcu_risc` | 0 (forbidden) | 0 (forbidden) | 0 (forbidden) |
| `mpu_risc` | 0 (forbidden) | 0 (forbidden) | 1 (requires `./tools/fetch_xnnpack.sh`) |
| `mcu_esp` | 0 (forbidden) | 1 (effective with `NETKIT_ARCH=ESP32*`) | 0 (forbidden) |

```bash
make cmsis-init
make esp-nn-init                 # once, for mcu_esp ESP-NN
make xnnpack-init                # once, for cpu / MPU XNNPACK LayerFast

# Profile defaults (no extra flags needed after fetch)
make test-cpp                    # cpu: XNNPACK (CMSIS-NN / ESP-NN off)
make NETKIT_TARGET=mcu_arm lib   # mcu_arm: CMSIS-NN (set NETKIT_ARCH=CM4 for NN)
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 lib   # mcu_esp: ESP-NN
make NETKIT_TARGET=mpu_arm lib   # mpu_arm: XNNPACK
make NETKIT_TARGET=mpu_risc lib  # mpu_risc: XNNPACK on (Arm CMSIS-NN off)

# Explicit off (reference kernels)
make NETKIT_CMSIS_NN=0 NETKIT_XNNPACK=0 rebuild test

# Reference-kernel 4× loop unroll — experimental (increases .text)
make NETKIT_LOOP_UNROLL=1 test-cpp
```

| Makefile flag | Macro | Effect |
|---------------|-------|--------|
| `NETKIT_CMSIS_NN=1` | `NETKIT_USE_CMSIS_NN` | `mcu_arm` + Cortex-M `NETKIT_ARCH` only |
| `NETKIT_ESP_NN=1` | `NETKIT_USE_ESP_NN` | `mcu_esp` + `NETKIT_ARCH=ESP32*` only |
| `NETKIT_XNNPACK=1` | `NETKIT_USE_XNNPACK` | cpu + any MPU float32/int8 LayerFast; **forbidden on MCU** |
| `NETKIT_LOOP_UNROLL=1` | `NETKIT_LOOP_UNROLL=1` | **Experimental.** 4× manual unroll in reference kernels only (default 0); see [GENERIC_KERNELS.md](GENERIC_KERNELS.md) |
| `NETKIT_ARCH=<core>` | `ARM_MATH_*` or `CONFIG_IDF_TARGET_*` (see table above) | Core-specific CMSIS-NN / ESP-NN tuning |

Dense weights use CMSIS-NN `[out, in]` layout via `Kernels::FullyConnectedWithBias` (same as PyTorch `nn.Linear`).

Float32 CMSIS-NN support is **experimental** upstream. Helium (MVE) and Neon targets get optimized kernels when `NETKIT_ARCH` and the toolchain flags align.

### XNNPACK

[Google XNNPACK](https://github.com/google/XNNPACK) (BSD-3) accelerates float32 and int8 (qs8) kernels on **cpu** and **any MPU** (`mpu_arm`, `mpu_risc`, …) when `NETKIT_XNNPACK=1` (the default for those profiles). It is portable across host ISAs (x86, Arm, …); netkit still **forbids** it on MCU class targets (flash/RAM), where CMSIS-NN / ESP-NN / reference kernels apply instead.

- **float32:** `XnnpackKernel` is `LayerFast` in `active_kernel.hpp` (conv, depthwise, max/avg pool, FC with ReLU/ReLU6 clamp).
- **int8:** `XnnpackQuant` is tried first in the quantized plan / `QuantOps` path (qs8 conv, depthwise, max pool, FC), then ESP-NN / CMSIS-NN if enabled, then reference.

### ESP-NN

[Espressif ESP-NN](https://github.com/espressif/esp-nn) (Apache-2.0) accelerates **int8** kernels when **`NETKIT_ESP_NN=1`**, **`NETKIT_TARGET=mcu_esp`**, and **`NETKIT_ARCH`** is an Espressif chip (`ESP32`, `ESP32S3`, `ESP32C3`, `ESP32C6`, `ESP32P4`). Fetch once with `make esp-nn-init` (`./tools/fetch_esp_nn.sh`).

**ESP-NN has no float32 API.** `EspNnKernel` occupies the same LayerFast slot as CMSIS float LayerFast, but every float `Try*` returns false so `ComposedKernel` falls through to `ReferenceKernel`. Production Espressif path is **int8 + ESP-NN** (`EspNnQuant`), wired like CMSIS-NN (`Try*` + shared quant plan types).

On **cpu** / Arm / RISC targets the flag is **forced off**. Host smoke uses `NETKIT_HOST_SMOKE=1` to build ANSI-only ESP-NN (no chip assembly).

```bash
make esp-nn-init
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 lib
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C6 NETKIT_HOST_SMOKE=1 lib
```

C and C++ firmware both call `nk_model_load` / `nk_model_run_int8` (or C++ `Load` / `forward`); backend choice is compile-time only — [API_PARITY.md](API_PARITY.md).

MCU builds force `NETKIT_XNNPACK=0`; enabling it is a Make override to off / compile error.

```bash
./tools/fetch_xnnpack.sh   # or: make xnnpack-init
make NETKIT_XNNPACK=1 lib  # default on cpu + MPU once fetched
make NETKIT_XNNPACK=0 lib  # force reference LayerFast path
```

If headers/libs are missing, Make prints a warning and builds without XNNPACK (reference LayerFast). CI forces `NETKIT_XNNPACK=0`.

### Kernel dispatch (CRTP)

Backends are composed at **compile time** via `active_kernel.hpp` — there is no runtime backend switch. Layer and ops code call `Kernels::Op(...)`; `ComposedKernel` tries CMSIS-NN / ESP-NN / XNNPACK `Try*` methods and falls back to `ReferenceKernel`.

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

**Typical forward pass (MCU + CMSIS-NN):**

| Op family | Primary backend |
|-----------|-----------------|
| Conv / depthwise conv / pool / batch norm / FC / NN activations / GELU / softmax | CMSIS-NN (`LayerFast`) |
| Elementwise mul / matmul / add / scale / ReLU6 clip / LayerNorm2d / GRN | Reference / `NetkitUtil` |
| Fused blocks (ResNet / MobileNet / ConvNeXt internals) | Same via `fused_kernel_ops.hpp` |
| Any `Try*` miss | Reference |

## Testing

Default regression (`make test`) and full suite (`make test-full`) require **`NETKIT_TARGET=cpu`**. CMSIS / ESP-NN / RISC backends are validated locally via host smoke (`make test-embedded-smoke-matrix` with `NETKIT_HOST_SMOKE=1`), which exercises `test_mlp` and `cnn_4x4_single` on Arm + RISC + Espressif MCU/MPU profiles.

```bash
make cmsis-init
make esp-nn-init
make test-cpp
make test-embedded-smoke-matrix          # host MCU/MPU + CMSIS + ESP-NN profiles
make NETKIT_HOST_SMOKE=1 NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 lib
make NETKIT_HOST_SMOKE=1 NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C6 NETKIT_ESP_NN=1 lib
```

See [TESTING.md](TESTING.md) and [ARENA.md](ARENA.md).

Related: [PHILOSOPHY.md](PHILOSOPHY.md) · [GETTING_STARTED.md](GETTING_STARTED.md)
