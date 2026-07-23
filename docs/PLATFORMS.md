# Platform configuration

How to configure netkit for every **supported deployment target**. Companion to
[BUILD_TARGETS.md](BUILD_TARGETS.md) (flags / tables), [STATUS.md](STATUS.md)
(maturity), and [GETTING_STARTED.md](GETTING_STARTED.md) (first build).

**C and C++ share the same build flags** (`include/netkit_config.h`). Callers use
`nk_*` (C23) or C++ headers; backends are selected at **compile time** — there is
no runtime “switch to CMSIS/ESP/NMSIS” API. See [API_PARITY.md](API_PARITY.md).

| Target | Role | Production kernels | Arena default |
|--------|------|--------------------|---------------|
| [`cpu`](#cpu-desktop--ci) | Desktop / CI | XNNPACK (default) | 64 MiB heap |
| [`mcu_arm`](#mcu_arm--arm-cortex-m) | Arm Cortex-M firmware | CMSIS-NN int8 | 64 KiB static |
| [`mpu_arm`](#mpu_arm--arm-mpu) | Arm MPU / RTOS | XNNPACK | 64 MiB static |
| [`mcu_risc`](#mcu_risc--risc-v-mcu) | Non-Espressif RISC-V MCU (Nuclei / RV32) | NMSIS-NN int8 | 64 KiB static |
| [`mpu_risc`](#mpu_risc--risc-v-mpu) | RISC-V MPU | XNNPACK | 64 MiB static |
| [`mcu_esp`](#mcu_esp--espressif-mcu) | Espressif MCU (Xtensa **and** RISC-V) | ESP-NN int8 | 64 KiB static |

Fetch once as needed: `make cmsis-init` · `make esp-nn-init` · `make nmsis-init` ·
`make xnnpack-init`. Licenses: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

### Target ≠ CPU ISA

`NETKIT_TARGET` selects the **vendor runtime + NN backend**, not the instruction-set
alone. Several Espressif chips are RISC-V (C3 / C6 / P4) but still use **`mcu_esp`**
because they all run **ESP-NN** (and ESP-IDF). Do **not** pick `mcu_risc` for those.

| Silicon | Example chips | Correct target | Backend |
|---------|---------------|----------------|---------|
| Espressif (Xtensa) | ESP32, ESP32-S3 | `mcu_esp` + `NETKIT_ARCH=ESP32` / `ESP32S3` | ESP-NN |
| Espressif (RISC-V) | ESP32-C3, C6, P4 | `mcu_esp` + `NETKIT_ARCH=ESP32C3` / `ESP32C6` / `ESP32P4` | ESP-NN |
| Non-Espressif RISC-V MCU | Nuclei N300, generic RV32 | `mcu_risc` + `NETKIT_ARCH=N300` / `RV32IMAC` / … | NMSIS-NN |

Multiple Espressif boards (C3, S3, P4, …) share **`mcu_esp`**; each board tree sets its
own `NETKIT_ARCH` (and IDF/PlatformIO board id). Same pattern as several NUCLEO
firmwares under one `mcu_arm` + `CM4`.

---

## Shared rules

| Topic | MCU (`mcu_*`) | MPU (`mpu_*`) | CPU |
|-------|---------------|---------------|-----|
| Heap / `malloc` | **Forbidden** | Optional (`NETKIT_HEAP_ARENA=1`) | Default arena backing |
| mmap `.nk` load | **Forbidden** | On by default (`NETKIT_MMAP=0` to opt out) | On by default |
| XNNPACK | **Forbidden** | Default **on** | Default **on** |
| CLI (`./netkit`) | No | No | Yes |
| Host bring-up | `NETKIT_HOST_SMOKE=1` + `embedded-smoke` | Same | `make test` |

**API (same on every target):**

```c
/* C23 — firmware / FFI */
nk_arena_init(&arena, buf, sizeof(buf));
nk_model_load_memory(blob, len, &arena, &model);   /* or nk_model_load on cpu/MPU */
nk_model_run_int8(&model, &arena, in, n, out, cap, &out_n);  /* or nk_model_run for float32 */
```

```cpp
/* C++26 — same engine */
Arena arena; arena.init(buf, sizeof(buf));
/* NkLoader::Load* / Load*FromBuffer → forward / forward_quantized */
```

Full references: [c-api.md](c-api.md), [cpp-api.md](cpp-api.md).

---

## `cpu` (desktop / CI)

| Knob | Value |
|------|-------|
| `NETKIT_TARGET` | `cpu` (default) |
| `NETKIT_ARCH` | unset |
| Backends | XNNPACK **on**; CMSIS / ESP / NMSIS **off** |
| Fetch | `make xnnpack-init` (once) |

```bash
make xnnpack-init
make                          # CLI + libnetkit.a
make test                     # C++ + C regression
# Reference-only (CI-style):
make NETKIT_XNNPACK=0 rebuild test
```

**CMake:** `cmake -B cmake-build && cmake --build cmake-build`

---

## `mcu_arm` — Arm Cortex-M

| Knob | Value |
|------|-------|
| `NETKIT_TARGET` | `mcu_arm` |
| `NETKIT_ARCH` | `CM4`, `M33`, `M55`, … (see [BUILD_TARGETS](BUILD_TARGETS.md#target-architecture-netkit_arch)) |
| Backends | CMSIS-NN **on** (int8 production); float32 via reference |
| Fetch | `make cmsis-init` |

```bash
make cmsis-init
# Host smoke (portable CMSIS path, no device Core headers):
make NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 NETKIT_HOST_SMOKE=1 lib embedded-smoke
# Cross-compile firmware (omit HOST_SMOKE; add toolchain flags):
make NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 lib
```

**CMake:** `-DNETKIT_TARGET=mcu_arm -DNETKIT_ARCH=CM4 -DNETKIT_CMSIS_NN=ON`

**On-device:** NUCLEO-F446RE boards — setup index [boards/README.md](../boards/README.md#stm32-nucleo-f446re) (int8 peers vs TFLM / microTVM). Espressif: [XIAO ESP32C3](../boards/README.md#seeed-studio-xiao-esp32c3) (int8 peers vs TFLM / ESP-NN). Stage int8 inputs in SRAM before timed inference.

**Override:** `NETKIT_CMSIS_NN=0` → reference QuantOps / float reference only.

---

## `mpu_arm` — Arm MPU

| Knob | Value |
|------|-------|
| `NETKIT_TARGET` | `mpu_arm` |
| `NETKIT_ARCH` | optional (`A32`, `NEON`, …) |
| Backends | XNNPACK **on**; CMSIS-NN **off** |
| Fetch | `make xnnpack-init` |

```bash
make xnnpack-init
make NETKIT_TARGET=mpu_arm lib
make NETKIT_TARGET=mpu_arm NETKIT_HOST_SMOKE=1 embedded-smoke
# Optional heap arena helpers:
make NETKIT_TARGET=mpu_arm NETKIT_HEAP_ARENA=1 lib
```

**CMake:** `-DNETKIT_TARGET=mpu_arm -DNETKIT_XNNPACK=ON`

**On-device example:** [boards/pi-zero-2w/README.md](../boards/pi-zero-2w/README.md) (cross-build + SSH A/B); index [boards/README.md](../boards/README.md#raspberry-pi-zero-2-w).

---

## `mcu_risc` — RISC-V MCU

**Scope:** Nuclei and other **non-Espressif** RISC-V MCUs (NMSIS-NN). Espressif
RISC-V parts (ESP32-C3 / C6 / P4) use [`mcu_esp`](#mcu_esp--espressif-mcu) instead —
see [Target ≠ CPU ISA](#target--cpu-isa).

| Knob | Value |
|------|-------|
| `NETKIT_TARGET` | `mcu_risc` |
| `NETKIT_ARCH` | `N300`, `N600`, `NX900`, `RV32IMAC`, … |
| Backends | NMSIS-NN **on** (int8 production); float32 via reference (no NMSIS float API) |
| Fetch | `make nmsis-init` |

```bash
make nmsis-init
# Host smoke (portable NMSIS path + nmsis_host_compat.h):
make NETKIT_TARGET=mcu_risc NETKIT_ARCH=N300 NETKIT_HOST_SMOKE=1 lib embedded-smoke
make NETKIT_TARGET=mcu_risc NETKIT_ARCH=RV32IMAC NETKIT_HOST_SMOKE=1 lib
# Cross-compile (omit HOST_SMOKE; use your RISC-V toolchain):
make NETKIT_TARGET=mcu_risc NETKIT_ARCH=N300 lib
```

**CMake:** `-DNETKIT_TARGET=mcu_risc -DNETKIT_ARCH=N300 -DNETKIT_NMSIS_NN=ON`

**Maturity:** float32 + int8 runtime and host smoke are **done**. On-device peer boards TBD — [STATUS.md](STATUS.md). Same C `nk_*` load/run as Arm MCU.

**Override:** `NETKIT_NMSIS_NN=0` → generic / QuantOps only.

NMSIS-NN is the **CMSIS-NN API twin** for RISC-V (`riscv_*` / `nmsis_nn_*`); wiring matches Arm (`NmsisNnQuant` + shared quant plan).

---

## `mpu_risc` — RISC-V MPU

| Knob | Value |
|------|-------|
| `NETKIT_TARGET` | `mpu_risc` |
| `NETKIT_ARCH` | optional |
| Backends | XNNPACK **on**; CMSIS / ESP / NMSIS **forbidden** |
| Fetch | `make xnnpack-init` |

```bash
make xnnpack-init
make NETKIT_TARGET=mpu_risc lib
make NETKIT_TARGET=mpu_risc NETKIT_HOST_SMOKE=1 embedded-smoke
```

**CMake:** `-DNETKIT_TARGET=mpu_risc -DNETKIT_XNNPACK=ON`

XNNPACK covers RISC-V MPU-class cores the same way as Arm MPU / cpu. No board README yet — use the library API + your BSP.

---

## `mcu_esp` — Espressif MCU

**All Espressif MCUs use this one target** — including RISC-V silicon (C3 / C6 / P4)
and Xtensa (ESP32 / S3). They share **ESP-NN**; distinguish chips with
`NETKIT_ARCH` and a per-board firmware tree. Do **not** use `mcu_risc` / NMSIS-NN
for ESP32* parts — see [Target ≠ CPU ISA](#target--cpu-isa).

| Knob | Value |
|------|-------|
| `NETKIT_TARGET` | `mcu_esp` |
| `NETKIT_ARCH` | `ESP32`, `ESP32S3`, `ESP32C3`, `ESP32C6`, `ESP32P4` |
| Backends | ESP-NN **on** (int8 production); float32 via reference (no ESP float API) |
| Fetch | `make esp-nn-init` |

```bash
make esp-nn-init
# Host ANSI smoke (no Xtensa / chip asm):
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C6 NETKIT_HOST_SMOKE=1 lib embedded-smoke
# Device / IDF build (omit HOST_SMOKE; set chip ARCH):
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 lib
# Seeed XIAO ESP32C3 board firmware (PlatformIO ESP-IDF):
make -C boards/xiao-esp32c3-cnn-int8
PORT=/dev/cu.usbmodem* ./boards/xiao-esp32c3/scripts/run_esp_int8_ab.sh
```

**CMake:** `-DNETKIT_TARGET=mcu_esp -DNETKIT_ARCH=ESP32S3 -DNETKIT_ESP_NN=ON`

**Boards:** [boards/xiao-esp32c3/](../boards/xiao-esp32c3/README.md) — MLP / CNN / DS-CNN int8 (netkit + TFLM) on Seeed XIAO ESP32C3 (RISC-V silicon, **ESP-NN** profile).

**Maturity:** float32 + int8 runtime and host ANSI smoke are **done**. On-device peer A/B vs TFLM (CNN / DS-CNN int8) **done** — [STATUS.md](STATUS.md#mcu-seeed-xiao-esp32c3). Same C `nk_*` load/run as Arm MCU. Note: C3 has **no FPU** (soft-float); production peers are int8.

**Override:** `NETKIT_ESP_NN=0` → reference only.

---

## Matrix smoke (all profiles)

Before flashing, validate lean linking on the host:

```bash
make cmsis-init esp-nn-init nmsis-init
make test-embedded-smoke-matrix
```

Runs `tests/embedded_smoke` against `test_mlp.nk` and `cnn_4x4_single.nk` for
Arm / RISC / Espressif MCU+MPU profiles (including CMSIS / ESP-NN / NMSIS-NN).
Details: [TESTING.md](TESTING.md#embedded-smoke-mcupu).

---

## Quick flag cheat sheet

| Flag | Meaning |
|------|---------|
| `NETKIT_TARGET` | `cpu` \| `mcu_arm` \| `mpu_arm` \| `mcu_risc` \| `mpu_risc` \| `mcu_esp` |
| `NETKIT_ARCH` | Core tag — required for MCU NN accel effectiveness |
| `NETKIT_CMSIS_NN` / `NETKIT_ESP_NN` / `NETKIT_NMSIS_NN` / `NETKIT_XNNPACK` | Backend on/off (profile defaults apply) |
| `NETKIT_HOST_SMOKE=1` | Host-compile MCU/MPU smoke without device headers / chip asm |
| `NETKIT_GLOBAL_ARENA=1` | CPU: force static arena |
| `NETKIT_HEAP_ARENA=1` | MPU: compile heap arena helpers |
| `NETKIT_MMAP=0` | Disable file mmap (MPU/cpu) |

Full tables: [BUILD_TARGETS.md](BUILD_TARGETS.md). Kernels: [KERNELS.md](KERNELS.md).
