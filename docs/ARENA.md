# Arena Memory (Bump Allocator)

netkit uses a **single bump-pointer arena** for all inference-time allocation: network structs, weight blobs loaded from `.nk` files, and **two pre-sized ping-pong activation buffers** allocated at model load. Hidden layer outputs reuse those buffers during forward passes instead of allocating one tensor per layer. There is no per-object `free()` — memory is reclaimed in bulk with `reset()`.

## Why an arena?

Embedded and firmware targets benefit from:

- **Predictable memory** — one caller-provided buffer, no hidden heap use in layer code paths
- **Fast allocation** — pointer bump only (O(1) per alloc)
- **Simple lifetime** — reset between inferences or test cases instead of tracking individual frees

netkit ships a minimal ~86-line arena rather than linking an external allocator (see [API.md](API.md#memory-model) for the memkit comparison).

## How it works

```
base ──► [ used ........ | free ........................ ] ◄── capacity
         ▲
         offset (next alloc starts here)
```

1. **`init(memory, size)`** — bind a caller-owned byte buffer; offset = 0.
2. **`alloc(size, alignment)`** — if current offset is not aligned, skip padding bytes; carve `size` bytes; advance offset; return pointer (or `nullptr` on overflow).
3. **`reset()`** — offset = 0; all prior pointers are logically invalid.
4. **`remaining()`** — `capacity - offset`.

### Ping-pong activations

At **load time**, MLP and CNN networks scan layer output sizes and allocate **two** float32 buffers large enough for the biggest intermediate tensor. During `forward()`, layers alternate writing into those buffers (A → B → A → …). Peak activation memory is roughly **2 × largest layer output** instead of the sum of all layer outputs.

### Kernel workspace (CMSIS-NN)

When **CMSIS-NN** is enabled, CNN models also reserve a **single shared kernel workspace** in the same arena at load time. The size is the maximum `arm_*_get_buffer_size` over every conv, depthwise conv, and GELU in the graph (including convs inside fused blocks). During inference, CMSIS-NN conv/depthwise/GELU ops bind this buffer instead of using stack scratch — same idea as TensorFlow Lite Micro accounting op scratch inside the tensor arena.

On reference-only builds the workspace size is zero. `inspect --full` reports **kernel workspace** bytes separately when non-zero; those bytes are included in **after load** / **after forward** totals.

Weights and ping buffers are allocated together during load, so a forward pass does not grow the arena unless the caller allocates separate input/output tensors (e.g. CLI `run` or `nk_model_run`).

### Alignment

Weight blobs can have an odd float count, leaving the offset at 4 mod 8 on 64-bit platforms. Without padding, a following `MLPNetwork` or `CNNNetwork` struct would be misaligned for placement-new.

| Allocation | Typical alignment |
|------------|-------------------|
| float weights / tensor payload | `alignof(float)` (4) |
| Network structs, pointers | `alignof(T)` or `alignof(max_align_t)` (8 on 64-bit) |

The engine passes correct alignment at every internal call site. Direct API users must do the same.

### Backing buffer

Declare the buffer with platform max alignment:

```c
alignas(max_align_t) static unsigned char memory[65536];
```

```cpp
alignas(std::max_align_t) unsigned char buffer[65536];
```

## C++ API

```cpp
#include "arena.hpp"

Arena arena;
arena.init(buffer, sizeof(buffer));

void* weights = arena.alloc(weight_bytes, alignof(float));
void* net_mem = arena.alloc(sizeof(CNNNetwork), alignof(CNNNetwork));

arena.reset();  // reuse for next inference
```

Default capacity constant:

| Target | `NK_ARENA_DEFAULT_CAPACITY` / `Arena::kDefaultCapacity` |
|--------|-----------------------------------------------------------|
| MCU | **64 KiB** |
| CPU / MPU | **64 MiB** |

CLI/regression on CPU use the default heap capacity (`Arena::kDefaultCapacity`). Override with `./netkit --arena <size> …`.

### Heap-backed arena (CPU default; MCU/MPU optional)

When `NETKIT_ARENA_HEAP` is defined ( **CPU builds by default**, or MCU/MPU with `NETKIT_HEAP_ARENA=1` ), `init_heap()` performs **one** `malloc` for the backing buffer. All inference allocations are bump-pointer inside that buffer — no `realloc`, no per-tensor heap calls.

| Target | `init_heap` | `destroy_heap` / `ArenaUtil::Release` |
|--------|-------------|----------------------------------------|
| **CPU** | Once per session (CLI command or full test suite) | Frees backing memory when the session ends |
| **MCU / MPU** | Optional once at startup | **No-op** — heap backing is never freed |

Regression on CPU (`make test-cpp`) uses **one** heap arena for all 88 cases (`BeginRegressionArena` / `EndRegressionArena`), resetting the bump offset between cases instead of malloc/free per case.

See [BUILD_TARGETS.md](BUILD_TARGETS.md). Helper: `ArenaUtil::Init()` in `arena_util.hpp`.

## Choosing arena size

The arena size is **not** stored in the model file. **You** (or your test harness) provide a byte buffer large enough for that model.

### What consumes arena memory

| Allocation | When | Notes |
|------------|------|-------|
| Weight views (flash/mmap/blob) | Load | Bind into flash, mmap, or caller blob; coefs stay out of arena bump peaks (mmap owned by arena) |
| Network structs | Load | `MLPNetwork` / `CNNNetwork`, layer metadata |
| Ping-pong buffers | Load | **2 ×** largest intermediate activation (float32) |
| Kernel workspace | Load (CNN, CMSIS-NN builds) | **1 ×** max CMSIS conv/dw/GELU scratch across the graph |
| Input / output tensors | Caller | Optional — CLI and `nk_model_run` allocate these per run |

Ping-pong buffers are reserved at **load time**, so a forward pass does not grow the arena for hidden activations. Peak activation memory is roughly **2 × largest layer output**, not the sum of every layer.

### Weight storage (always flash/blob-backed)

Weights never copy into a separate RAM weight buffer. At load time, netkit **binds layer tensors to blob addresses**. Misaligned payloads return a load error (`SizeMismatch`).

| Target | File load (`LoadMLP` / `LoadCNN` / `nk_model_load`) | Buffer / AOT |
|--------|------------------------------------------------------|--------------|
| **CPU** (macOS, Linux; `NETKIT_MMAP=1` default) | POSIX **`mmap` (`MAP_PRIVATE`)**; arena owns mapping until `reset()` / `destroy_heap()`. Pages stay file-backed until a write (e.g. BN fold) copy-on-writes that page | Bind into caller-owned blob; `data` must outlive the network |
| **MPU** (default `NETKIT_MMAP=0`) | Same as MCU: `fread` into arena if you use a path API; prefer buffer/flash | Flash/XIP or embedded `.rodata`; bind views |
| **MPU** + embedded Linux (`NETKIT_MMAP=1`) | Same mmap path as CPU | Same as above |
| **MCU** | Copies the file into the arena (no mmap). Prefer buffer/AOT | Flash/XIP or embedded `.rodata`; bind views |

**Sizing firmware:** use `./netkit inspect --full` (or AOT constants). Arena peaks exclude weight/bias bytes when the blob is mmap'd or flash-backed; `flash_payload_bytes` reports the payload kept outside the bump arena. Size SRAM for **activations + structs + headroom** only.

See [NK_FORMAT.md](NK_FORMAT.md) and [BUILD_TARGETS.md](BUILD_TARGETS.md).

### How to pick a size

1. **Measure** — `./netkit inspect models/your_model.nk --full` or `nk_inspect_model()`. Use **arena bytes after forward** (includes load + ping buffers + a zero-input forward with caller I/O tensors). Weight/bias payload stays in flash — use `flash_payload_bytes` separately when budgeting flash, not SRAM.
2. **Add headroom** — typically **1.5–2×** measured high-water for batch or future changes.
3. **Declare static storage** — firmware usually uses a fixed `unsigned char memory[N]` sized from step 1–2.

```cpp
// Example: size from inspect, then deploy with margin
alignas(std::max_align_t) static unsigned char memory[3 * 1024 * 1024];  // 3 MiB
Arena arena;
arena.init(memory, sizeof(memory));
```

There is no automatic growth — if `alloc` fails, loaders return an arena overflow error.

### Where the repo uses which size

| Caller | Buffer size | Models |
|--------|-------------|--------|
| CLI `run` / `inspect` (CPU, heap) | **64 MiB** default; override with `--arena` | All |
| Examples, C API smoke (CPU) | **64 MiB** (`NK_ARENA_DEFAULT_CAPACITY`) | Includes MNIST CNN |
| Regression (`src/nk_regression.cpp`) | **64 MiB** heap (`Arena::kDefaultCapacity`) | All embedded cases |

MCU firmware typically declares a smaller static buffer (e.g. 64 KiB) sized from `inspect --full`.

### `reset()` and reload

`reset()` sets the arena offset to zero, **releases any mmap'd `.nk` file**, and **invalidates all pointers** (weights, network, ping buffers). To run again on the same buffer you must **reload the model**. The MNIST test suite calls `arena.reset()` then `NkLoader::LoadMLP` / `LoadCNN` per case for isolation.

## C API

```c
#include "netkit.h"

alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
nk_arena_t arena;
nk_arena_init(&arena, memory, sizeof(memory));

void* block = nk_arena_alloc(&arena, 1024, alignof(float));
nk_arena_reset(&arena);
```

| Function | C++ equivalent |
|----------|----------------|
| `nk_arena_init` | `Arena::init` |
| `nk_arena_alloc` | `Arena::alloc` |
| `nk_arena_reset` | `Arena::reset` |
| `nk_arena_capacity` | `Arena::capacity` |
| `nk_arena_used` | `Arena::offset` |
| `nk_arena_remaining` | `Arena::remaining` |

High-level loaders (`nk_model_load`, `nk_mlp_load`, `nk_cnn_load`) allocate from the arena you pass in. Size buffers with `nk_inspect_model()` or `./netkit inspect`.

## Sizing for deployment

1. Run `./netkit inspect models/your_model.nk --full` (or `nk_inspect_model`).
2. Note **arena bytes after forward** — add headroom (typically 1.5–2× for batch variance).
3. Use one arena per model context, or `reset()` between runs on the same buffer.

| Model | Approx. arena high-water | Test / CLI buffer |
|-------|--------------------------|-------------------|
| Hand test MLP/CNN | < 64 KiB | 64 KiB (default) |
| MNIST MLP | ~1–2 MiB measured | 2 MiB in tests |
| MNIST CNN | ~2–4 MiB measured | 4 MiB in tests |

Run `inspect --full` on your exact model and input shape for deployment numbers.

## Quant lowered vs interpreter embed on MCU

Two AOT deployment paths share the same `python -m netkit aot` packaging step but allocate memory differently. See [PHILOSOPHY.md](PHILOSOPHY.md#terminology-embed-vs-lowered).

| Aspect | Interpreter embed (`--no-lower`) | Quant lowered (default AOT for int8) |
|--------|----------------------------------|--------------------------------------|
| Runtime | `NkLoader` + `NkOpsResolver` walks `.nk` | Static `CmsisQuantPlan` call chain |
| Weights | Embedded `.nk` blob in flash (MCU default) | Quant params + tables in `.rodata` |
| Activations | **Bump arena** at load (`InitActivationBuffers`) | **Static BSS** ping-pong buffers sized at compile time |
| Arena role | Holds structs, ping-pong, optional weight copy | Tiny bump pool for composite-block scratch only |
| Typical MCU benchmark | Fair vs TFLM `MicroInterpreter` | Faster invoke; different memory layout |

### Ping-pong buffer sizing

**Interpreter:** at load, `InitActivationBuffers` allocates two activation tensors from the arena. Size is driven by the **largest intermediate** feature map (plus CMSIS kernel workspace when applicable).

**Quant lowered:** `aot_lower_quant.py` emits static `g_act_a[]` and `g_act_b[]`. Each layer reads from one buffer and writes to the other. Buffer sizes are **not** `2 × global_max`:

- **`odd_max`** — largest activation tensor at **odd** layer indices (write target for those steps).
- **`even_max`** — largest at **even** layer indices.

Only one buffer must hold the current write target; the other holds the previous layer output (often smaller). For MNIST CNN int8 lowered on NUCLEO-F446RE: `even_max` ≈ 21,632 B, `odd_max` ≈ 5,408 B, plus ~1,152 B CMSIS workspace — ~28 KiB static BSS total.

**Firmware takeaway:** do not size interpreter firmware from `kArenaBytesRecommended` alone on 128 KiB SRAM — declare an explicit static arena (e.g. **64 KiB** on `nucleo-f446re-cnn-int8`, verified 10/10) and confirm linker RAM. Lowered firmware instead budgets static activation arrays; inspect generated `*_aot.cpp` for `g_act_a` / `g_act_b` sizes.

## Related docs

- [DATATYPES.md](DATATYPES.md) — float32 weights and tensors today
- [c-api.md](c-api.md) — full C arena reference
- [cpp-api.md](cpp-api.md) — C++ arena reference
- [API.md](API.md#memory-model) — overview and memkit note
