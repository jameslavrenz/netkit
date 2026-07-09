# netkit Philosophy

netkit is a **multi-modal inference engine** (voice, image, vision) with an **embedded-first** design optimized for **MCUs, MPUs, and NPUs**. Models ship as single **`.nk`** files. You develop and validate on the desktop (**CPU** build), then link the lean runtime into firmware (**MCU** / **MPU** builds). The engine is written in **C++26** (modern patterns, type-safe primary API) with a **C23** mirror for C-only firmware.

**Status:** Active development. **Float32** and **int8** inference work today ([DATATYPES.md](DATATYPES.md)); float16, int16, and int4 are on the roadmap. **Kalman estimation and tracking** are planned backend capabilities alongside neural inference.

Companion project: [memkit](https://github.com/jameslavrenz/memkit) for general-purpose embedded memory management. netkit owns the **inference arena and tensor lifecycle** inside a caller-provided buffer.

## Deployment modes: interpreter or compiled

netkit supports two ways to run the same forward engine — pick based on whether you need **flexibility** or **maximum on-device performance**.

### Interpreter — `NkOpsResolver` + `.nk` load

The default runtime is an **interpreter-style forward executor**:

1. Load a `.nk` file from disk (`LoadMLP` / `LoadCNN`, `nk_model_load`) or from a memory buffer (`LoadMLPFromBuffer` / `LoadCNNFromBuffer`, `nk_model_load_memory`).
2. Walk the layer list at inference time.
3. Dispatch each layer through **`NkOpsResolver`** — a static function-pointer registry (TFLite `MicroMutableOpResolver` style): no virtuals, heap, or `std::vector` ([KERNELS.md](KERNELS.md)).

**Best for:** desktop development, CLI regression, swapping models without reflashing, prototyping, and any path where the graph may change at runtime.

**Firmware trimming:** register only the op handlers your graph needs with compile-time `NkOpList<Ops...>::View()` and link matching `src/layer_ops/*.cpp` units — the linker drops unused eval bodies.

```cpp
using TrimOps = NkOpList<NkConv2DOpDescriptor, NkDenseOpDescriptor>;
cnn.SetOpsResolver(TrimOps::View());
```

### Compiled — AOT embed + packager optimizations

For production firmware with a **fixed model**, compile as much work as possible **before** inference:

| Step | Tool | Effect |
|------|------|--------|
| ONNX → `.nk` | `python -m netkit convert` | Serialize graph + float32 weights |
| Graph optimize | `convert` (default) or `aot --optimize` | BN fold, conv+BN fusion, dense merge, composite block fuse — **fewer layer dispatches** at runtime; each pass verified numerically |
| AOT embed | `python -m netkit aot` | Bake `.nk` into flash `.rodata`; emit arena sizing constants and thin load/run wrappers |
| Lean link | `NkOpList` + trimmed `libnetkit.a` | Only op TUs and kernels your model uses |
| Kernel backends | `NETKIT_CMSIS_NN` / `NETKIT_CMSIS_DSP` | Hardware-accelerated matmul, conv, pool, FC where available |

**Best for:** shipping firmware — minimum RAM, predictable latency, no filesystem, coefs in flash.

Both modes call the **same kernels** (`Kernels::MatMul`, `Conv2D`, …). The compiled path moves graph rewriting and model embedding to **build time** so inference does less dispatch and allocation work. Phase 2 expands packager-side compilation (layout, quantization, target profiles) — see below.

### Terminology: embed vs lowered

The CLI command is `python -m netkit aot`, but that name covers **two different outputs**. Do not read “AOT” as always meaning “no interpreter.”

| Term | CLI | On-device runtime | Comparable to |
|------|-----|-------------------|---------------|
| **Interpreter embed** | `aot … --no-lower` | `.nk` blob in flash → `NkLoader` → `NkOpsResolver` per layer | TFLM: `.tflite` blob → `MicroInterpreter` |
| **Lowered / compiled** | `aot` (C++ default) | Static `Kernels::` or `CmsisQuantPlan` call chain; explicit weights; **no loader** | Custom fused firmware (not TFLM) |

Generated files are still named `*_aot.{hpp,cpp}` for historical reasons. Makefile targets use **`export-embed`** for the packaging step (both modes). MCU **benchmark** firmware defaults to **interpreter embed** for apples-to-apples host and on-device comparisons with TFLM.

Pipeline comparison:

```text
Interpreter embed:  model.nk  ──aot --no-lower──►  flash blob  ──load──►  NkOpsResolver  ──►  Kernels
Lowered / compiled: model.onnx  ──convert/optimize──►  model.nk  ──aot──►  static call chain  ──►  Kernels
```

Details: [GETTING_STARTED.md](GETTING_STARTED.md#5-aot-compile-embed-nk-in-firmware), [BUILD_TARGETS.md](BUILD_TARGETS.md#layer-dispatch-opsresolver), [python/README.md](../python/README.md).

## Two-phase roadmap

### Phase 1 — Interpreter runtime (today)

The C++ engine described above is an **interpreter-style forward executor** (see [Deployment modes](#deployment-modes-interpreter-or-compiled)):

1. Load a `.nk` file (architecture descriptor + float32 weights).
2. Walk the layer list at runtime (Dense, Conv2D, MaxPool2D, AvgPool2D, BatchNorm2d, Flatten, activations).
3. Execute kernel ops via the compile-time `Kernels` facade (`MatMul`, `Conv2D`, pool, activations) — reference implementations with optional CMSIS-NN / CMSIS-DSP backends ([KERNELS.md](KERNELS.md)).
4. Allocate weights and **ping-pong activation buffers** from a bump arena.

**Goals:** correctness, predictable memory, small firmware surface, dual C/C++ API, desktop CLI for debug and regression.

**Not in Phase 1:** training, autograd, dynamic shapes, automatic heap growth inside layer code, or aggressive operator fusion in the runtime.

The Python packager (`python/netkit/`) converts **ONNX → `.nk`** and can embed regression test cases. It is primarily a **serializer** today — it records the graph and weights the runtime will interpret.

### Phase 2 — Compiler optimizations (Python packager)

Phase 2 moves optimization **into the packager** so firmware stays lean:

| Optimization | Effect |
|------------------------|--------|
| Operator fusion (partial) | Conv/Gemm + ReLU, Sigmoid, Tanh, LeakyRelu, ReLU6, Softmax fused at ONNX export into `.nk` activation tags |
| Constant folding / layout | Precompute shapes, choose memory-friendly weight layout |
| Dead-code elimination | Strip unused nodes before `.nk` emission |
| Quantization-aware export | Emit int8/int16/float16 payloads with scale metadata |
| Target-specific profiles | Emit `.nk` tuned for MCU vs MPU buffer sizes |

The runtime may gain **specialized fast paths** for fused op tags, but the default mental model remains: **packager compiles, runtime interprets**.

See [DATATYPES.md](DATATYPES.md) for numeric roadmap.

### Phase 3 — Estimation and control (planned)

Beyond neural forward passes, netkit will add **Kalman estimation and tracking** to the backend — state estimation and sensor fusion alongside on-device inference. Details and APIs are TBD.

## Memory philosophy

- **One bump arena** per inference context — weights, network structs, ping-pong activations (interpreter embed).
- **Quant lowered** moves activations to **static compile-time buffers**; the arena is only a small scratch pool. See [ARENA.md](ARENA.md#quant-lowered-vs-interpreter-embed-on-mcu).
- **No hidden heap** in layer forward paths (MCU/MPU default).
- **Caller sizes the buffer** — not stored in the model file. Measure with `./netkit inspect --full` or `nk_inspect_model()`. On tight SRAM, firmware may fix a smaller arena than embed `kArenaBytesRecommended` (e.g. 64 KiB on STM32F446RE CNN int8).
- **Explicit alignment** on every allocation — odd weight sizes must not misalign following structs.

Default capacities by build target (`NK_ARENA_DEFAULT_CAPACITY`):

| Target | Default | Backing |
|--------|---------|---------|
| CPU | 64 MiB | Heap by default (`NETKIT_ARENA_HEAP`); optional static via `NETKIT_GLOBAL_ARENA=1` |
| MCU | 64 KiB | Caller-owned static/global buffer |
| MPU | 64 MiB | Caller-owned static/global buffer; optional heap via `NETKIT_HEAP_ARENA=1` |

Full details: [BUILD_TARGETS.md](BUILD_TARGETS.md), [ARENA.md](ARENA.md).

## Build targets

| Target | Role |
|--------|------|
| **CPU** | Desktop dev — CLI, embedded regression (88 cases), Python ONNX parity (82), AOT compile tests, embedded smoke orchestration on host (`test_mlp`, `cnn_4x4_single`) |
| **MCU** | Lean runtime — `.nk` load + inference only |
| **MPU** | Same lean runtime as MCU; slightly larger default static arena constant |
| **NPU** | Same lean runtime; offload paths and target-specific kernels TBD as NPUs are integrated |

Only **CPU** builds include `NETKIT_DESKTOP` APIs (`nk_cli_run`, `nk_run_all_tests`).

## Numeric precision

**Today:** float32 only for weights, activations, and math.

**Planned:** float16, int16, int8, int4 — see [DATATYPES.md](DATATYPES.md). Expect new `.nk` format versions and packager-side quantization, with runtime decode stubs or native kernels per type.

## Design principles

| Principle | Meaning |
|-----------|---------|
| Inference-only | Pre-trained weights; no training loop in firmware |
| Single-threaded | Sequential forward pass |
| Standard C/C++ | No external runtime dependencies in the engine |
| API parity | Stable C23 mirror of public C++26 surface ([API_PARITY.md](API_PARITY.md)) |
| Test in `.nk` | Optional embedded regression cases (`TCAS` section) for bring-up |

## Related docs

- [GETTING_STARTED.md](GETTING_STARTED.md) — first run for new users
- [BUILD_TARGETS.md](BUILD_TARGETS.md) — Makefile flags and macros
- [CLI.md](CLI.md) — desktop commands
- [API.md](API.md) — C vs C++ overview
