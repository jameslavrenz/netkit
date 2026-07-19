# Kernel backends (CRTP)

netkit routes all numeric work through a **compile-time kernel facade**. Layer code (`ops`, `conv2d`, `cnn`, `mlp`) calls `Kernels::Op(...)` only — no `#if CMSIS` and no runtime backend switching in those paths.

How the **generic / reference** path is tuned for 32-bit+ devices (multi-accumulator dots, NHWC loops, im2col, int8 MAC+requant, portability limits): [GENERIC_KERNELS.md](GENERIC_KERNELS.md).

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
VectorFast  LayerFast        ← ReferenceKernel, CmsisNnKernel, EspNnKernel, NmsisNnKernel, or XnnpackKernel
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
| `netkit_util.hpp` / `netkit_util.cpp` | Portable vector helpers (`NetkitUtil::`; CMSIS-DSP is not used) |
| `cmsis_nn_kernel.hpp` / `cmsis_nn_backend.cpp` | CMSIS-NN `Try*` for layer ops (conv, depthwise conv, pool, FC, batch norm, activations, GELU, softmax) |
| `esp_nn_kernel.hpp` / `esp_nn_backend.cpp` | ESP-NN float LayerFast slot (all `Try*` miss → reference; int8 via `EspNnQuant`) |
| `esp_nn_quant.hpp` / `esp_nn_quant_backend.cpp` | ESP-NN int8 `Try*` (conv, depthwise, pool, FC, softmax, add) |
| `nmsis_nn_kernel.hpp` / `nmsis_nn_backend.cpp` | NMSIS-NN float LayerFast slot (all `Try*` miss → reference; int8 via `NmsisNnQuant`) |
| `nmsis_nn_quant.hpp` / `nmsis_nn_quant_backend.cpp` | NMSIS-NN int8 `Try*` (CMSIS-NN twin: `riscv_*` / `nmsis_nn_*`) |
| `kernel_dispatch.hpp` | `ComposedKernel<VectorFast, LayerFast>` and `Try*` helpers |
| `active_kernel.hpp` | `using Kernels = …` alias for the current build |
| `fused_kernel_ops.hpp` | Inline helpers for fused blocks (BN, ReLU, MatAdd, FC 1×1, GELU, GRN) → `Kernels::` |
| `activation_followup.hpp` | Shared post-kernel activation when not fused in CMSIS-NN |

Backend `.cpp` files **must** include `netkit_config.h` so `NETKIT_CMSIS_NN_ALLOWED` / `NETKIT_ESP_NN_ALLOWED` / `NETKIT_NMSIS_NN_ALLOWED` and related macros match the active build profile.

## Active backend aliases

Selected in `active_kernel.hpp` from `NETKIT_USE_CMSIS_NN`, `NETKIT_USE_ESP_NN`, `NETKIT_USE_NMSIS_NN`, `NETKIT_USE_XNNPACK`, and allow macros:

| Build profile | `Kernels` alias |
|---------------|-----------------|
| Reference only (float32 MCU; accel off) | `ReferenceKernel` — portable generic kernels |
| CMSIS-NN (Arm MCU + Cortex-M, production int8) | `ComposedKernel<ReferenceKernel, CmsisNnKernel>` |
| NMSIS-NN (RISC-V MCU, production int8) | `ComposedKernel<ReferenceKernel, NmsisNnKernel>` (float `Try*` always miss) |
| ESP-NN (Espressif MCU, production int8) | `ComposedKernel<ReferenceKernel, EspNnKernel>` (float `Try*` always miss) |
| XNNPACK LayerFast (`cpu` / any MPU) | `XnnpackKernel` |

**CMSIS-DSP is not used.** CMSIS-NN is **forbidden** on RISC and ESP targets. ESP-NN is **forbidden** outside `mcu_esp`. NMSIS-NN is **forbidden** outside `mcu_risc`. XNNPACK is **forbidden** on all MCU targets.

### Role split in `ComposedKernel<VectorFast, LayerFast>`

| Role | Typical backend | Ops |
|------|-----------------|-----|
| **VectorFast** | Reference (portable) | `Mul`, `MatMul`, `MulScalar`, `MatAdd`/`MatAddND`, clip, LayerNorm, GRN |
| **LayerFast** | CMSIS-NN, ESP-NN / NMSIS-NN slot, or XNNPACK when enabled | `Conv2d`, depthwise conv, pool, batch norm, FC, NN activations, softmax |
| **Reference** | Always | Fallback when `Try*` returns false or backend is `ReferenceKernel` |

**GEMM:** there is no separate `Gemm` symbol. General matrix multiply is `Kernels::MatMul` (`Ops::MatMul`); linear layers use `Kernels::FullyConnectedWithBias` (internally matmul + bias). ONNX `Gemm` is lowered to packed dense weights at export time.

On **MCU with CMSIS-NN**, CMSIS-NN owns layer kernels; vector ops and ops without NN float APIs use reference / `NetkitUtil`. On **MCU with ESP-NN / NMSIS-NN**, int8 uses `EspNnQuant` / `NmsisNnQuant`; float LayerFast always falls to reference.

### Float32 op coverage

| Op | Reference | CMSIS-NN | ESP-NN | NMSIS-NN | XNNPACK |
|----|-----------|----------|--------|----------|---------|
| Conv2D, depthwise conv, pool, batch norm, FC, activations, softmax | ✓ | ✓ (`Try*` + fallback) | — (slot miss → reference) | — (slot miss → reference) | ✓ (cpu/MPU) |
| MatMul, elementwise mul/add/scale | ✓ | add (elementwise) | — | — | — |
| LayerNorm2d | ✓ | — | — | — | — |
| GELU | ✓ | ✓ (tanh on inner polynomial) | — | — | — |
| GRN | ✓ | — | — | — | — |
| Residual / skip merge | ✓ | add (`MatAddND`) | — | — | — |

**Float32 on MCU:** supported via reference kernels (CMSIS may accelerate some float LayerFast ops on Arm; ESP-NN / NMSIS-NN have no float API). Production MCU paths are **int8 + CMSIS-NN** (Arm), **int8 + ESP-NN** (Espressif), or **int8 + NMSIS-NN** (RISC-V).

**Host CPU:** production accel is XNNPACK. With XNNPACK ON, TF Lite’s optimized builtins and ORT **MLAS** are moot — netkit ≈ TF Lite there. **MLAS is not needed for netkit** (ORT OFF keeps MLAS on; that is not a slow-reference peer). See [STATUS.md](STATUS.md#host-three-way-suite-netkit-vs-tf-lite-vs-onnx-runtime).

**Depthwise conv** is **2D-only** in the API (`DepthwiseConv2D`, NHWC `[H,W,C]`, weights `[C,Kh,Kw]`). **1D** along time/height is expressed as a degenerate 2D kernel (e.g. `kernel_h=5`, `kernel_w=1` on input `[T,1,C]`). See [NK_FORMAT.md](NK_FORMAT.md) and `python/README.md`.

**Fused blocks** (ResNet BasicBlock, MobileNetV4 UIB, ConvNeXt V2) route internal BN, ReLU, FC, LayerNorm, GELU, GRN, and residual adds through `fused_kernel_ops.hpp` → `Kernels::`, so CMSIS-NN / XNNPACK apply when enabled. Composite blocks do not introduce separate CMSIS entry points — they delegate to the same `Try*` paths as primitives. When CMSIS-NN rejects a case (e.g. depthwise conv with asymmetric padding), the reference kernel handles it automatically.

**Float im2col conv** (reference path when CMSIS-NN is off or rejects a case): partial and full im2col use `Kernels::MatMulImpl`. Hot contiguous dots use the inlined 4-accumulator path. Asymmetric-padding conv fallbacks go through `Conv2dDispatchForward` so specialized kernels remain in the dispatch chain.

**Portable helpers:** `netkit_util.cpp` (`NetkitUtil::`) provides contiguous f32/int8 copy, argmax, and related helpers used by reference fallbacks and board firmware staging. Asymmetric pool layers route through `Kernels::MaxPool2dForwardPadded` / `AvgPool2dForwardPadded`.

**Int8 quant path:** layer compute is CMSIS-NN (`arm_convolve_wrapper_s8`, …) on Arm MCU, ESP-NN (`esp_nn_*`) on Espressif MCU, or NMSIS-NN (`riscv_convolve_wrapper_s8`, …) on RISC-V MCU — all tried ahead of QuantOps reference in the shared plan path. MCU firmware should stage int8 inputs in SRAM before the first conv (TFLM copies into the tensor arena; netkit benchmark firmware uses `g_input_staging` in `main.cpp`) so the conv kernel reads activations from SRAM, not flash-resident test vectors.

### Kernel workspace (CMSIS-NN / NMSIS-NN)

On **CMSIS-NN** or **NMSIS-NN** builds, CNN `InitActivationBuffers` walks the layer graph and sizes one **shared arena buffer** to the maximum scratch requirement (`arm_*` / `riscv_*` get_buffer_size for conv, depthwise conv, GELU). `CNNNetwork::forward` activates this buffer for the duration of the pass; the LayerFast backend binds it instead of stack `alloca`. If the workspace is missing or too small, `Try*` returns false and the reference kernel runs. **ESP-NN** does not use this shared workspace path.

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

Optional **4× manual loop unroll** for netkit reference kernels only. **Off by default** (`NETKIT_LOOP_UNROLL=0`). Independent of CMSIS-NN `ARM_MATH_LOOPUNROLL` — CMSIS translation units never receive this flag.

> **Experimental:** Duplicating loop bodies increases **`.text` / flash size**. On tight MCUs, enabling this can push the firmware image over available program memory even when RAM (arena) is sized correctly. Measure `.text` before shipping; prefer CMSIS backends on production firmware unless you have flash headroom.
>
> **Where it might help:** In practice this is most likely something to consider on an **MPU** — more program memory than a typical MCU, and reference kernels are often the primary path when CMSIS-NN is unavailable. **Avoid on flash-constrained MCUs**; on desktop CPU, XNNPACK or default reference builds are usually sufficient.

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
| `im2col_partial.cpp` | Hybrid: one patch per output pixel + per-filter dots (float) |
| `im2col_full.cpp` | Full im2col matrix + `MatMul` GEMM (float) |
| `im2col_quant.cpp` | Int8 QuantOps partial / full im2col + int32 MAC + requant |

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

**Int8 QuantOps Conv2D** uses the same `NETKIT_IM2COL` policy via `im2col_quant.cpp` when CMSIS / ESP / NMSIS / XNNPACK do not handle the op. Depthwise stays direct. If full-matrix scratch does not fit the arena, QuantOps degrades to partial then direct.

**Product guidance:** default **`NETKIT_IM2COL=0`** on all targets. im2col is mainly for MCU / reference Conv2D; CMSIS-NN / ESP-NN / NMSIS-NN / XNNPACK ignore it. On MPU/cpu with XNNPACK off, `NETKIT_IM2COL=1` can give a small float CNN bump — at most try `1`; safest is to leave `0`. See [BUILD_TARGETS.md](BUILD_TARGETS.md#netkit_im2col-guidance).

## Adding a new kernel op

1. Add `OpImpl` to `ReferenceKernel` and declare on `KernelBase`.
2. Add `TryOp` to accel backends if applicable (CMSIS-NN / ESP-NN / NMSIS-NN / XNNPACK; guard with the matching `NETKIT_USE_*` in `.cpp`).
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

- [BUILD_TARGETS.md](BUILD_TARGETS.md) — Make/CMake flags for CMSIS / ESP / NMSIS / XNNPACK
- [PLATFORMS.md](PLATFORMS.md) — Per-device configuration cookbooks
- [PHILOSOPHY.md](PHILOSOPHY.md) — interpreter vs compiled deployment; Phase 1 vs Phase 2 packager optimizations
