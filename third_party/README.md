# Third-party dependencies

## CMSIS-NN (optional)

[ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) is Apache-2.0 licensed and provides
optimized neural-network kernels for Arm Cortex-M (with portable scalar fallbacks for host builds).

When enabled (`NETKIT_CMSIS_NN=1` / `-DNETKIT_CMSIS_NN=ON`), netkit uses CMSIS-NN for:

- Conv2d, max-pool, fully-connected (dense weights in `[out, in]` layout)
- Activations (ReLU, sigmoid, tanh, leaky ReLU, ReLU6) and softmax
- Elementwise add (bias / residual paths)
- ReLU/ReLU6 fusion inside conv and FC when the CMSIS path succeeds

## CMSIS-DSP (optional)

[ARM CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) is Apache-2.0 licensed and provides
optimized digital signal processing kernels (vector math, matrix multiply) for Arm Cortex-M/A.

When enabled (`NETKIT_CMSIS_DSP=1` / `-DNETKIT_CMSIS_DSP=ON`), netkit uses CMSIS-DSP for:

- Ops elementwise add/mul, scale, clip (ReLU/ReLU6 fallback), matrix multiply
- Fully-connected fallback via `arm_mat_vec_mult_f32`
- `ARM_MATH_LOOPUNROLL` on all CMSIS-DSP builds (desktop and embedded)

## Fetching

```bash
make cmsis-init
# or individually:
./tools/fetch_cmsis_nn.sh
./tools/fetch_cmsis_dsp.sh
# or: git submodule update --init third_party/CMSIS-NN third_party/CMSIS-DSP
```

## Architecture flags (`NETKIT_ARCH`)

`third_party/netkit_arch.mk` (Make) and `cmake/netkit_arch.cmake` (CMake) map `NETKIT_ARCH`
to CMSIS `ARM_MATH_*` preprocessor defines — e.g. `CM4` → `ARM_MATH_CM4`, `M33` → `ARM_MATH_ARMV8MML` + `__DSP_PRESENT=1`, `M55` → Helium (`ARM_MATH_MVEF`, `ARM_MATH_MVEI`).

Leave `NETKIT_ARCH` unset for native desktop builds (`__GNUC_PYTHON__` host path).

## Build integration

| File | Role |
|------|------|
| `third_party/cmsis_nn.mk` | Minimal CMSIS-NN source set linked into `libnetkit.a` |
| `third_party/cmsis_dsp.mk` | Minimal CMSIS-DSP source set |
| `third_party/netkit_arch.mk` | `NETKIT_ARCH` → `ARM_MATH_*` flags (Make) |
| `cmake/netkit_cmsis.cmake` | CMSIS object libraries + target flags (CMake) |
| `cmake/netkit_arch.cmake` | `NETKIT_ARCH` resolver (CMake) |

See [docs/BUILD_TARGETS.md](../docs/BUILD_TARGETS.md#cmsis-backends).

## Host embedded smoke

`make test-embedded-smoke-matrix` (or `./tools/run_embedded_smoke.sh`) rebuilds `libnetkit.a` for seven MCU/MPU profiles and runs `tests/embedded_smoke`. CMSIS profiles pass `NETKIT_HOST_SMOKE=1` so object code compiles on Linux/macOS hosts without CMSIS-Core.
