# netkit Philosophy

netkit is an **on-device inference kit** for MCUs and MPUs. Models ship as single **`.nk`** files. You develop and validate on the desktop (**CPU** build), then link the lean runtime into firmware (**MCU** / **MPU** builds).

Companion project: [memkit](https://github.com/jameslavrenz/memkit) for general-purpose embedded memory management. netkit owns the **inference arena and tensor lifecycle** inside a caller-provided buffer.

## Two-phase roadmap

### Phase 1 — Interpreter runtime (today)

The C++ engine is an **interpreter-style forward executor**:

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

## Memory philosophy

- **One bump arena** per inference context — weights, network structs, ping-pong activations.
- **No hidden heap** in layer forward paths (MCU/MPU default).
- **Caller sizes the buffer** — not stored in the model file. Measure with `./netkit inspect --full` or `nk_inspect_model()`.
- **Explicit alignment** on every allocation — odd weight sizes must not misalign following structs.

Default capacities by build target (`NK_ARENA_DEFAULT_CAPACITY`):

| Target | Default | Backing |
|--------|---------|---------|
| CPU | 4 MiB | Heap by default (`NETKIT_ARENA_HEAP`); optional static via `NETKIT_GLOBAL_ARENA=1` |
| MCU | 64 KiB | Caller-owned static/global buffer |
| MPU | 128 KiB | Caller-owned static/global buffer; optional heap via `NETKIT_HEAP_ARENA=1` |

Full details: [BUILD_TARGETS.md](BUILD_TARGETS.md), [ARENA.md](ARENA.md).

## Build targets

| Target | Role |
|--------|------|
| **CPU** | Desktop dev — CLI, embedded regression (81 cases), Python ONNX parity (77), AOT compile tests, embedded smoke orchestration on host |
| **MCU** | Lean runtime — `.nk` load + inference only |
| **MPU** | Same lean runtime as MCU; slightly larger default static arena constant |

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
