# Third-party dependencies

**Licenses / attribution:** see repo-root
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and vendored texts in
[`licenses/`](licenses/) (CMSIS, ESP-NN, NMSIS, XNNPACK, ONNX/ORT, NumPy,
LiteRT, TFLM, **Apache TVM / microTVM**, …). netkit is MIT; each dependency
keeps its own license. Refresh runtime copies with
`../tools/sync_third_party_licenses.sh` (also run by `tools/fetch_*.sh`).
**CMSIS-DSP is not used.**

## CMSIS-Core (MCU firmware)

[ARM CMSIS 6](https://github.com/ARM-software/CMSIS_6) provides **CMSIS-Core** device headers (`core_cm4.h`, `cmsis_compiler.h`, …) used when cross-compiling Cortex-M firmware with `NETKIT_ARCH` set.

When `NETKIT_ARCH` is set and `third_party/CMSIS-Core/CMSIS/Core/Include` exists, Make and CMake add `-I` for that path. Host desktop builds and `NETKIT_HOST_SMOKE=1` embedded smoke **do not** require CMSIS-Core (CMSIS-NN host smoke uses the portable `__GNUC_PYTHON__` path).

## CMSIS-NN (optional)

[ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) is Apache-2.0 licensed and provides
optimized neural-network kernels for Arm Cortex-M (with portable scalar fallbacks for host builds).

When enabled on **`NETKIT_TARGET=mcu_arm`** with a **Cortex-M `NETKIT_ARCH`**, netkit uses CMSIS-NN for conv, pool, batch norm, FC, and activations. On **cpu** or **mpu_arm**, `NETKIT_CMSIS_NN=1` is ignored (build warning) and reference / XNNPACK kernels are used. On **RISC** targets CMSIS-NN is **forbidden**.

- Conv2d (with symmetric padding), max-pool, avg-pool, batch norm, fully-connected (dense weights in `[out, in]` layout)
- Activations (ReLU, sigmoid, tanh, leaky ReLU, ReLU6) and softmax
- Elementwise add (bias / residual paths)
- ReLU/ReLU6 fusion inside conv and FC when the CMSIS path succeeds

## CMSIS-DSP (not used)

netkit does **not** use or link [ARM CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP).
Portable helpers live in `netkit_util` (`NetkitUtil::`). Float32 on MCU uses reference kernels only;
there is no plan for an optimized float32 MCU build via CMSIS-DSP.

## ESP-NN (optional)

[Espressif ESP-NN](https://github.com/espressif/esp-nn) is Apache-2.0 and provides optimized **int8**
kernels for Espressif chips (ESP32-S3 / ESP32-P4 assembly; ESP32 / C3 / C6 generic opt).

When enabled on **`NETKIT_TARGET=mcu_esp`** with `NETKIT_ARCH=ESP32|ESP32S3|ESP32C3|ESP32C6|ESP32P4`,
netkit uses ESP-NN for quantized conv, depthwise, pool, FC, softmax, and elementwise add
(`EspNnQuant`, same Try* shape as CMSIS-NN). **ESP-NN has no float32 API** — float LayerFast
falls through to reference kernels (`EspNnKernel` Try* always miss).

```bash
make esp-nn-init
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32S3 lib
# Host ANSI smoke (no Xtensa/RISC-V asm):
make NETKIT_TARGET=mcu_esp NETKIT_ARCH=ESP32C6 NETKIT_HOST_SMOKE=1 lib
```

## NMSIS-NN (optional)

[Nuclei NMSIS](https://github.com/Nuclei-Software/NMSIS) (Apache-2.0) includes **NMSIS-NN**, a
CMSIS-NN API twin for RISC-V (`riscv_*` / `nmsis_nn_*`).

When enabled on **`NETKIT_TARGET=mcu_risc`** with `NETKIT_ARCH=N300|N600|NX900|RV32IMAC|…`,
netkit uses NMSIS-NN for quantized conv, depthwise, pool, FC, softmax, and elementwise add
(`NmsisNnQuant`, same Try* shape as CMSIS-NN). **NMSIS-NN has no float32 API** — float LayerFast
falls through to reference kernels (`NmsisNnKernel` Try* always miss).

```bash
make nmsis-init
make NETKIT_TARGET=mcu_risc NETKIT_ARCH=N300 lib
# Host smoke (portable path + nmsis_host_compat.h):
make NETKIT_TARGET=mcu_risc NETKIT_ARCH=RV32IMAC NETKIT_HOST_SMOKE=1 lib
```

## XNNPACK (optional)

[Google XNNPACK](https://github.com/google/XNNPACK) is BSD-3 licensed and provides highly optimized neural-network operators for desktop/server CPUs and Cortex-A (x86 AVX, Arm NEON, WASM SIMD, …).

When enabled (`NETKIT_XNNPACK=1`, default on **cpu** and **any MPU**), netkit uses XNNPACK for:

- **float32** `LayerFast`: conv2d, depthwise conv, max/avg pool, fully-connected (ReLU/ReLU6 clamp)
- **int8 (qs8)**: same ops via `XnnpackQuant` in the quantized plan / `QuantOps` path (tried before CMSIS-NN / reference)

**MLAS is not needed for netkit** (host production is XNNPACK; see [docs/STATUS.md](../docs/STATUS.md#host-three-way-suite-netkit-vs-tf-lite-vs-onnx-runtime)).

MCU builds force `NETKIT_XNNPACK=0`. Forcing `=1` on MCU is rejected (Make override / compile error).

## Fetching

```bash
make cmsis-init
make esp-nn-init
make nmsis-init
make xnnpack-init
# or individually:
./tools/fetch_cmsis_core.sh
./tools/fetch_cmsis_nn.sh
./tools/fetch_esp_nn.sh
./tools/fetch_nmsis.sh
./tools/fetch_xnnpack.sh
```

CMSIS-Core and CMSIS-NN are **git submodule pins**. ESP-NN, NMSIS, and XNNPACK
are fetched into gitignored trees (`third_party/ESP-NN/`, `third_party/NMSIS/`,
`third_party/XNNPACK/`); XNNPACK is CMake-built with a stable
`netkit_lib/libXNNPACK.a` for linking.

**Peer-bench pin:** `tools/fetch_xnnpack.sh` defaults to the same XNNPACK commit LiteRT embeds (`ai_edge_litert` 2.1.6 → TF `b8a17154` → `c2e81f01…`). Override with `NETKIT_XNNPACK_PIN=<sha>` when bumping the LiteRT wheel.

## Architecture flags (`NETKIT_ARCH`)

`third_party/netkit_arch.mk` (Make) and `cmake/netkit_arch.cmake` (CMake) map `NETKIT_ARCH`
to CMSIS `ARM_MATH_*` preprocessor defines — e.g. `CM4` → `ARM_MATH_CM4`, `M33` → `ARM_MATH_ARMV8MML` + `__DSP_PRESENT=1`, `M55` → Helium (`ARM_MATH_MVEF`, `ARM_MATH_MVEI`).

Leave `NETKIT_ARCH` unset for native desktop builds (`__GNUC_PYTHON__` host path).

## Build integration

| File | Role |
|------|------|
| `third_party/cmsis_nn.mk` | Minimal CMSIS-NN source set linked into `libnetkit.a` |
| `third_party/esp_nn.mk` | ESP-NN ANSI / chip sources for `mcu_esp` |
| `third_party/nmsis_nn.mk` | NMSIS-NN s8 sources for `mcu_risc` |
| `third_party/nmsis_host_compat.h` | Host-smoke `__SSAT` / `__RESTRICT` / `__CLZ` for NMSIS |
| `third_party/xnnpack.mk` | XNNPACK link flags for cpu / MPU |
| `third_party/netkit_arch.mk` | `NETKIT_ARCH` → Arm / ESP / RISC-NN flags (Make) |
| `cmake/netkit_cmsis.cmake` | CMSIS object libraries + target flags (CMake) |
| `cmake/netkit_esp_nn.cmake` | ESP-NN object library + target flags (CMake) |
| `cmake/netkit_nmsis_nn.cmake` | NMSIS-NN object library + target flags (CMake) |
| `cmake/netkit_xnnpack.cmake` | XNNPACK fetch/link (CMake) |
| `cmake/netkit_arch.cmake` | `NETKIT_ARCH` resolver (CMake) |

See [docs/BUILD_TARGETS.md](../docs/BUILD_TARGETS.md#cmsis-backends) and [docs/STATUS.md](../docs/STATUS.md).

## Host embedded smoke

`make test-embedded-smoke-matrix` (or `./tools/run_embedded_smoke.sh`) rebuilds `libnetkit.a` for Arm + RISC + Espressif MCU/MPU profiles and runs `tests/embedded_smoke` against `test_mlp.nk` and `cnn_4x4_single.nk`. CMSIS / NMSIS profiles pass `NETKIT_HOST_SMOKE=1` so object code compiles on Linux/macOS hosts without device Core headers.
