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
| `reference_kernel.hpp` / `reference_kernel.cpp` | Portable float32 implementations (`NETKIT_LOOP_UNROLL` optional 4× unroll, experimental) |
| `netkit_loop_unroll.hpp` | Compile-time loop unroll helpers for reference kernels only |
| `cmsis_dsp_kernel.hpp` / `cmsis_dsp_backend.cpp` | CMSIS-DSP `Try*` for vector ops (add, mul, matmul, clip, batch-norm fallback, LayerNorm2d, GRN) |
| `cmsis_nn_kernel.hpp` / `cmsis_nn_backend.cpp` | CMSIS-NN `Try*` for layer ops (conv, depthwise conv, pool, FC, batch norm, activations, GELU, softmax) |
| `kernel_dispatch.hpp` | `ComposedKernel<VectorFast, LayerFast>` and `Try*` helpers |
| `active_kernel.hpp` | `using Kernels = …` alias for the current build |
| `fused_kernel_ops.hpp` | Inline helpers for fused blocks (BN, ReLU, MatAdd, FC 1×1, GELU, GRN) → `Kernels::` |
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
| **VectorFast** | CMSIS-DSP when enabled | `Mul`, `MatMul`, `MulScalar`, `MatAdd`/`MatAddND`, `ReLU6` clip fallback, `BatchNorm2d` (desktop fallback), `LayerNorm2d`, `Grn2dForward` |
| **LayerFast** | CMSIS-NN when enabled | `Conv2d`, depthwise conv, pool, batch norm, FC, NN activations, `Gelu`, softmax |
| **Reference** | Always | Fallback when `Try*` returns false or backend is `ReferenceKernel` |

**GEMM:** there is no separate `Gemm` symbol. General matrix multiply is `Kernels::MatMul` (`Ops::MatMul`); linear layers use `Kernels::FullyConnectedWithBias` (internally matmul + bias). ONNX `Gemm` is lowered to packed dense weights at export time.

On **MCU with both CMSIS flags**, CMSIS-NN owns layer kernels; CMSIS-DSP accelerates vector ops and ops without NN float APIs (LayerNorm, GRN). They do not compete for the same op.

### Float32 op coverage

| Op | Reference | CMSIS-NN | CMSIS-DSP |
|----|-----------|----------|-----------|
| Conv2D, depthwise conv, pool, batch norm, FC, activations, softmax | ✓ | ✓ (`Try*` + fallback) | FC/BN fallback on some builds |
| MatMul, elementwise mul/add/scale | ✓ | add (elementwise) | ✓ |
| LayerNorm2d | ✓ | — | ✓ |
| GELU | ✓ | ✓ (tanh on inner polynomial) | — (falls back to reference) |
| GRN | ✓ | — | ✓ (mean + vector mul/add per pixel) |
| Residual / skip merge | ✓ | add (`MatAddND`) | add |

**Depthwise conv** is **2D-only** in the API (`DepthwiseConv2D`, NHWC `[H,W,C]`, weights `[C,Kh,Kw]`). **1D** along time/height is expressed as a degenerate 2D kernel (e.g. `kernel_h=5`, `kernel_w=1` on input `[T,1,C]`). See [NK_FORMAT.md](NK_FORMAT.md) and `python/README.md`.

**Fused blocks** (ResNet BasicBlock, MobileNetV4 UIB, ConvNeXt V2) route internal BN, ReLU, FC, LayerNorm, GELU, GRN, and residual adds through `fused_kernel_ops.hpp` → `Kernels::`, so CMSIS applies when enabled. Composite blocks do not introduce separate CMSIS entry points — they delegate to the same `Try*` paths as primitives. When CMSIS-NN rejects a case (e.g. depthwise conv with asymmetric padding), the reference kernel handles it automatically.

**Float im2col conv** (reference path when CMSIS-NN is off or rejects a case): partial and full im2col use `Kernels::MatMulImpl` and `arm_dot_prod_f32` when `NETKIT_USE_CMSIS_DSP=1`, not hard-coded reference matmul/dot loops. Asymmetric-padding conv fallbacks go through `Conv2dDispatchForward` so specialized kernels remain in the dispatch chain.

**Float32 path:** when `NETKIT_USE_CMSIS_DSP=1`, `cmsis_dsp_util.cpp` provides contiguous f32/q7 helpers (`arm_copy_f32`, `arm_max_f32`, `arm_dot_prod_f32`, `arm_add_f32`, `arm_mult_f32`, `arm_scale_f32`) used by reference fallbacks, im2col/direct conv kernels, and board firmware staging/argmax. Asymmetric pool layers route through `Kernels::MaxPool2dForwardPadded` / `AvgPool2dForwardPadded` instead of bypassing to `ReferenceKernel`.

**Hot dot product is header-inline (code-size note).** `CmsisDspUtil::DotProductF32` (in `cmsis_dsp_util.hpp`) is `inline` so the FC/conv reduction loops inline at `-O2` without LTO. It resolves at compile time:

- **`NETKIT_USE_CMSIS_DSP=1`** → out-of-line call to `arm_dot_prod_f32` (via the `DotProductF32Cmsis` shim). Callers (`conv1x1`, `conv_direct`, `im2col_partial`, dense FC) stay small.
- **CMSIS-DSP off (pure reference)** → the 4-accumulator `NetkitLoopUnroll::dot_contiguous` is inlined into **every** caller. This is the speed win (no cross-TU call, no LTO) but it grows `.text`. Measured at `-O2` (Cortex-M4): `conv_direct` ~4.0 KB → ~10.3 KB (**+6.2 KB**), with smaller bumps in `reference_kernel`/`conv1x1`/`im2col_partial` (~+9.4 KB total across the float conv/FC kernels). CMSIS-DSP builds see almost none of this (the dot is a call), and firmwares that do not link the float conv path (e.g. FC/AOT MLP, int8 CMSIS-NN CNN) pay nothing after `--gc-sections`.

  **Consideration — reclaiming the ~6 KB:** the only config that actually pays it is a **float CNN built with CMSIS-DSP disabled** (e.g. a flash-constrained MCU running float conv on reference kernels). If you hit that, make the reference `DotProductF32` *out-of-line* (a single shared definition in `cmsis_dsp_util.cpp` instead of the header-inline `dot_contiguous`) so `conv_direct`/`conv1x1` call it once rather than inlining per output. That recovers most of the size at a small speed cost **only in that reference-only config** — CMSIS builds are unaffected because they already call the shim. Left inlined by default because netkit prioritizes speed and no shipped firmware currently pays the size.

**Int8 quant path:** layer compute is CMSIS-NN (`arm_convolve_wrapper_s8`, pool, FC, softmax). When `NETKIT_USE_CMSIS_DSP=1`, the same util module uses CMSIS-DSP for int8 copy/argmax (`arm_copy_q7`, `arm_max_q7`). MCU firmware should stage int8 inputs in SRAM before the first conv (TFLM copies into the tensor arena; netkit benchmark firmware uses `g_input_staging` in `main.cpp`) so the conv kernel reads activations from SRAM, not flash-resident test vectors.

### CMSIS kernel workspace

On **CMSIS-NN** builds, CNN `InitActivationBuffers` walks the layer graph and sizes one **shared arena buffer** to the maximum CMSIS scratch requirement (conv, depthwise conv, GELU). `CNNNetwork::forward` activates this buffer for the duration of the pass; `CmsisNnKernel` binds it via `BindCmsisWorkspace` instead of stack `alloca`. If the workspace is missing or too small, `Try*` returns false and the reference kernel runs.

Sizing is included in `./netkit inspect --full` arena high-water. The CLI also prints **kernel workspace** when non-zero (CNN scratch); `nk_inspect_model()` reports arena peaks and `flash_payload_bytes` but not per-layer workspace separately.

### Arena sizing for composite models

Fused blocks increase per-layer scratch (ConvNeXt V2 GRN norms, UIB ping-pong paths) but **ping-pong activation buffers** still dominate peak memory. Size firmware arenas from **`./netkit inspect models/your_model.nk --full`** or `nk_inspect_model()`: use **arena bytes after forward** plus 1.5–2× headroom. Weights stay flash/blob-backed — use `flash_payload_bytes` from inspect for flash budget, not SRAM. Composite backbones (`resnet18.nk`, `mobilenetv4_small.nk`, `convnextv2_atto.nk`) typically need **multi‑MiB** CPU heap arenas; see [ARENA.md](ARENA.md#choosing-arena-size).

## Dispatch pattern

`kernel_dispatch.hpp` uses **C++20 concepts** (`NkAcceleratedKernel`) and `if constexpr` to pick CMSIS `Try*` paths at compile time — no runtime backend switching, no virtual functions:

```cpp
template<typename T>
concept NkAcceleratedKernel = !std::same_as<T, ReferenceKernel>;

if constexpr (NkAcceleratedKernel<LayerFast>) {
    if (!LayerFast::TryConv2dForward(...)) { /* reference fallback */ }
}
```

Layer code calls `Kernels::Op(...)` only — not raw `ReferenceKernel::` (except inside kernel dispatch fallbacks).

Each fast backend exposes static `Try*` methods that return `bool`:

- `true` — backend handled the op
- `false` — fall through to reference (or the other backend role)

Example from `ComposedKernel`:

```cpp
static void MaxPool2dForwardImpl(const Tensor& input, int pool_size, int stride,
                                 int pad_h, int pad_w, Tensor& output)
{
    if constexpr (NkAcceleratedKernel<LayerFast>)
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

## Reference-kernel loop unroll (`NETKIT_LOOP_UNROLL`) — **experimental**

Optional **4× manual loop unroll** for netkit reference kernels only. **Off by default** (`NETKIT_LOOP_UNROLL=0`). Independent of CMSIS-DSP `ARM_MATH_LOOPUNROLL` — CMSIS translation units never receive this flag.

> **Experimental:** Duplicating loop bodies increases **`.text` / flash size**. On tight MCUs, enabling this can push the firmware image over available program memory even when RAM (arena) is sized correctly. Measure `.text` before shipping; prefer CMSIS backends on production firmware unless you have flash headroom.
>
> **Where it might help:** In practice this is most likely something to consider on an **MPU** — more program memory than a typical MCU, and reference kernels are often the primary path when CMSIS-NN is unavailable. **Avoid on flash-constrained MCUs**; on desktop CPU, CMSIS-DSP or default reference builds are usually sufficient.

```bash
make NETKIT_LOOP_UNROLL=1 lib
cmake -DNETKIT_LOOP_UNROLL=ON ...
```

When enabled, hot loops in `reference_kernel.cpp` use helpers from `include/netkit_loop_unroll.hpp` (element-wise ops, matmul/FC dot products, activations). CMSIS backends (`cmsis_*_backend.cpp`) are unchanged. If CMSIS handles an op, reference unroll does not apply to that path.

**Tradeoff:** larger compiled code for potentially faster reference fallback and reference-only builds. Not recommended for flash-constrained MCU images without verifying the link map.

## Cache-friendly reference kernels (default build)

Without enabling `NETKIT_LOOP_UNROLL`, reference spatial ops follow a few low-cost conventions for NHWC tensors. **More reference-kernel optimizations are planned** (same goals: better cache/line use and fewer branches, without a separate code-size toggle unless noted).

| Pattern | Where | Why |
|---------|-------|-----|
| **Spatial → channel loop order** (`oh`, `ow`, then `c` / `oc`) | Conv2D, depthwise conv, max/avg pool | Writes `out[(oh·W+ow)·C + c]` sequentially in `c` — better line/cache use than channel-outer loops |
| **Channel-inner reduction** | Conv2D inner `ic` loop | At a fixed `(ih, iw)`, NHWC channels are contiguous — dot products walk memory linearly |
| **Row-level padding skip** | Conv / pool kernel loops | When `ih` is out of bounds, skip the entire `kw` row instead of branching per tap |
| **Inlined NHWC indexing** | Hot conv/pool paths | `(h·W+w)·C+c` computed inline instead of repeated helper calls |
| **Contiguous 2D fast path** | `MatAddImpl` | Row-major tensors (`stride[1]==1`) use one linear pass (e.g. FC bias add) |

No extra code-size toggle — these apply whenever reference kernels run (including CMSIS fallback). Runtime inference has **no `while` loops** on the hot path; padding uses structured `for` loops with early row skip, not unbounded control flow.

### Reference Conv2D lowering (`conv_dispatch.cpp`)

When CMSIS-NN is off or `TryConv2dForward` returns false, `Conv2dDispatchForward` applies a **compiler-style lowering policy** (not user-configurable). Kernel modules:

| Module | Role |
|--------|------|
| `conv_dispatch.cpp` | Policy selection + orchestration |
| `conv1x1_kernel.cpp` | 1×1 stride-1 direct dot product (mandatory for 1×1) |
| `conv_depthwise_kernel.cpp` | Depthwise direct loops (always manual) |
| `conv_direct_kernel.cpp` | 3×3 specialist, input-stationary, padded/unpadded spatial |
| `im2col_partial.cpp` | Hybrid: one patch per output pixel + per-filter dots |
| `im2col_full.cpp` | Full im2col matrix + `MatMul` GEMM |

**Hard rules:**

| Case | Policy |
|------|--------|
| **1×1, stride 1** | Always **direct** — never im2col |
| **Depthwise** | Always **direct** — never im2col |
| **3×3, stride 1** | Direct preferred; **partial im2col** (`NETKIT_IM2COL≥1`) when patch volume ≥ 2048. Full im2col only when `NETKIT_IM2COL=2` and patch×spatial ≥ 32768 |
| **≥ 5×5 or large generic** | Direct or **partial im2col** (`NETKIT_IM2COL≥1`) when volume ≥ 2048. Full im2col only when `NETKIT_IM2COL=2` |

**Weight repack at load:** `CNNNetwork::InitActivationBuffers` calls `RepackConv2dWeights` for each conv layer, allocating `[kh, kw, in, out]` (**HWIO**) beside the stored `[out, kh, kw, in]` (**OIHW**) blob. Repack is one-time arena cost; inference reads `weights_hwio` on the input-stationary direct path.

**Workspace:** `Conv2dWorkspaceBytes` sizes scratch from the selected policy; `CNNNetwork` / `NkConv2DOp` planning takes the max across layers. Workspace is arena-backed on the interpreter path (no heap allocation at inference time).

Padding uses inclusive input bounds (`ih ∈ [0, in_h)`, `iw ∈ [0, in_w)`) consistent with the spatial reference kernel.

**Int8 quantized Conv2D** does not use this float im2col path — `forward_quantized()` routes through CMSIS-NN (`CmsisQuantPlan`) on supported MCU builds.

## Adding a new kernel op

1. Add `OpImpl` to `ReferenceKernel` and declare on `KernelBase`.
2. Add `TryOp` to CMSIS backends if applicable (guard with `#if NETKIT_USE_CMSIS_*` in `.cpp`).
3. Wire `ComposedKernel::OpImpl` via a `Try*` helper in `kernel_dispatch.hpp`.
4. Call `Kernels::Op(...)` from layer code — not from a new `#if` branch.

## Layer dispatch (OpsResolver)

CNN forward uses a **static function-pointer registry** (`ops_resolver.hpp`) — the core of the **interpreter path**. Layers are looked up at runtime via `NkOpsResolver::Find(opcode)`; no virtuals, heap, or `std::vector`. For maximum speed on a fixed graph, combine AOT embed + packager `--optimize` with a trimmed `NkOpList` — see [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

### C++26 compile-time resolver tables

Each layer op is a descriptor struct with `static constexpr NkLayerOpRegistration kRegistration`. `NkOpList<Ops...>` folds descriptors into a **`constinit`** registration array and resolver view — built at compile time with no dynamic static initialization (MCU-safe):

```cpp
struct NkConv2DOpDescriptor {
    static constexpr NkLayerOpRegistration kRegistration = { ... };
};

using MyOps = NkOpList<NkConv2DOpDescriptor, NkDenseOpDescriptor>;
cnn.SetOpsResolver(MyOps::View());  // reference to constinit static storage
```

Descriptors live in `layer_op_registry.hpp`; implementations in `src/layer_ops/*.cpp`. `NkAllLayerOps` is the full six-op table used by `GetDefaultOpsResolver()`.

### Per-op translation units (firmware DCE)

| Descriptor header | Implementation |
|-------------------|----------------|
| `layer_ops/nk_conv2d_op.hpp` | `src/layer_ops/nk_op_conv2d.cpp` |
| `layer_ops/nk_max_pool2d_op.hpp` | `src/layer_ops/nk_op_max_pool2d.cpp` |
| `layer_ops/nk_avg_pool2d_op.hpp` | `src/layer_ops/nk_op_avg_pool2d.cpp` |
| `layer_ops/nk_batch_norm2d_op.hpp` | `src/layer_ops/nk_op_batch_norm2d.cpp` |
| `layer_ops/nk_flatten_op.hpp` | `src/layer_ops/nk_op_flatten.cpp` |
| `layer_ops/nk_dense_op.hpp` | `src/layer_ops/nk_op_dense.cpp` |

Link only the `.cpp` files matching your `NkOpList<...>` — unused `NkEval*` bodies are dropped by the linker.

```
CNNNetwork::forward
        │
        ▼
  NkOpsResolver::Find(opcode)
        │
        ▼
  prepare_output / eval  →  Kernels::…
```

## Related docs

- [BUILD_TARGETS.md](BUILD_TARGETS.md) — Make/CMake flags for CMSIS backends
- [PHILOSOPHY.md](PHILOSOPHY.md) — interpreter vs compiled deployment; Phase 1 vs Phase 2 packager optimizations
