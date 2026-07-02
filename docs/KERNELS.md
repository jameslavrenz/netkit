# Kernel backends (CRTP)

netkit routes all numeric work through a **compile-time kernel facade**. Layer code (`ops`, `conv2d`, `cnn`, `mlp`) calls `Kernels::Op(...)` only — no `#if CMSIS` and no runtime backend switching in those paths.

## Architecture

```
ops.cpp / conv2d.cpp / cnn.cpp / mlp.cpp
        │
        ▼
  active_kernel.hpp          ← picks backend alias at compile time
        │
        ▼
  ComposedKernel<…>          ← kernel_dispatch.hpp (Try* + fallback)
        │
   ┌────┴────┐
   ▼         ▼
VectorFast  LayerFast        ← CmsisDspKernel, CmsisNnKernel, or ReferenceKernel
   │         │
   └────┬────┘
        ▼
  ReferenceKernel            ← always linked; fallback for any Try* miss
```

| Header | Role |
|--------|------|
| `kernel_crtp.hpp` | `KernelBase<Derived>` — static `Kernels::Mul` → `Derived::MulImpl` |
| `kernel_activation.hpp` | `NetkitKernelActivation` enum for fused conv/FC activations |
| `reference_kernel.hpp` / `reference_kernel.cpp` | Portable float32 implementations |
| `cmsis_dsp_kernel.hpp` / `cmsis_dsp_backend.cpp` | CMSIS-DSP `Try*` for vector ops (add, mul, matmul, clip) |
| `cmsis_nn_kernel.hpp` / `cmsis_nn_backend.cpp` | CMSIS-NN `Try*` for layer ops (conv, pool, FC, batch norm, activations) |
| `kernel_dispatch.hpp` | `ComposedKernel<VectorFast, LayerFast>` and `Try*` helpers |
| `active_kernel.hpp` | `using Kernels = …` alias for the current build |
| `activation_followup.hpp` | Shared post-kernel activation when not fused in CMSIS-NN |

Backend `.cpp` files **must** include `netkit_config.h` so `NETKIT_CMSIS_NN_ALLOWED` and related macros match the active build profile.

## Active backend aliases

Selected in `active_kernel.hpp` from `NETKIT_USE_CMSIS_NN`, `NETKIT_USE_CMSIS_DSP`, and `NETKIT_CMSIS_NN_ALLOWED`:

| Build profile | `Kernels` alias |
|---------------|-----------------|
| Reference only | `ReferenceKernel` |
| CMSIS-DSP only (desktop / MPU, or MCU without NN) | `ComposedKernel<CmsisDspKernel, ReferenceKernel>` |
| CMSIS-NN only (MCU + Cortex-M) | `ComposedKernel<ReferenceKernel, CmsisNnKernel>` |
| CMSIS-NN + CMSIS-DSP (MCU firmware) | `ComposedKernel<CmsisDspKernel, CmsisNnKernel>` |

### Role split in `ComposedKernel<VectorFast, LayerFast>`

| Role | Typical backend | Ops |
|------|-----------------|-----|
| **VectorFast** | CMSIS-DSP when enabled | `Mul`, `MatMul`, `MulScalar`, `ReLU6` clip fallback |
| **LayerFast** | CMSIS-NN when enabled | `Conv2d`, pool, batch norm, FC, NN activations, softmax |
| **Reference** | Always | Fallback when `Try*` returns false or backend is `ReferenceKernel` |

On **MCU with both CMSIS flags**, CMSIS-NN owns layer kernels; CMSIS-DSP accelerates vector ops only — they do not compete for the same op.

## Dispatch pattern

Each fast backend exposes static `Try*` methods that return `bool`:

- `true` — backend handled the op
- `false` — fall through to reference (or the other backend role)

Example from `ComposedKernel`:

```cpp
static void MaxPool2dForwardImpl(const Tensor& input, int pool_size, int stride,
                                 int pad_h, int pad_w, Tensor& output)
{
    if constexpr (!IsReferenceKernel<LayerFast>)
    {
        if (!LayerFast::TryMaxPool2dForward(input, pool_size, stride, pad_h, pad_w,
                                              NetkitKernelActivation::None, output))
            ReferenceKernel::MaxPool2dForwardImpl(input, pool_size, stride, pad_h, pad_w, output);
    }
    else
        ReferenceKernel::MaxPool2dForwardImpl(input, pool_size, stride, pad_h, pad_w, output);
}
```

No virtual functions; the compiler inlines the selected path for each translation unit.

## Adding a new kernel op

1. Add `OpImpl` to `ReferenceKernel` and declare on `KernelBase`.
2. Add `TryOp` to CMSIS backends if applicable (guard with `#if NETKIT_USE_CMSIS_*` in `.cpp`).
3. Wire `ComposedKernel::OpImpl` via a `Try*` helper in `kernel_dispatch.hpp`.
4. Call `Kernels::Op(...)` from layer code — not from a new `#if` branch.

## Related docs

- [BUILD_TARGETS.md](BUILD_TARGETS.md) — Make/CMake flags for CMSIS backends
- [PHILOSOPHY.md](PHILOSOPHY.md) — Phase 1 interpreter vs Phase 2 packager optimizations
