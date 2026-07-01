# Arena Memory (Bump Allocator)

netkit uses a **single bump-pointer arena** for all inference-time allocation: network structs, weight blobs loaded from `.bin` files, and **two pre-sized ping-pong activation buffers** allocated at model load. Hidden layer outputs reuse those buffers during forward passes instead of allocating one tensor per layer. There is no per-object `free()` — memory is reclaimed in bulk with `reset()`.

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

Weights and ping buffers are allocated together during load, so a forward pass does not grow the arena unless the caller allocates separate input/output tensors (e.g. CLI `run` or `nk_model_run`).

### Alignment

Weight `.bin` files can have an odd float count, leaving the offset at 4 mod 8 on 64-bit platforms. Without padding, a following `MLPNetwork` or `CNNNetwork` struct would be misaligned for placement-new.

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

Default capacity constant: `Arena::kDefaultCapacity` (64 KiB) is a **convenience default**, not an engine limit. You always pass the actual buffer size to `init()`.

## Choosing arena size

The arena size is **not** stored in model JSON. The `.json` file describes architecture only; **you** (or your test harness) provide a byte buffer large enough for that model.

### What consumes arena memory

| Allocation | When | Notes |
|------------|------|-------|
| Weight `.bin` blob | Load | Size fixed by layer dimensions |
| Network structs | Load | `MLPNetwork` / `CNNNetwork`, layer metadata |
| Ping-pong buffers | Load | **2 ×** largest intermediate activation (float32) |
| Input / output tensors | Caller | Optional — CLI and `nk_model_run` allocate these per run |

Ping-pong buffers are reserved at **load time**, so a forward pass does not grow the arena for hidden activations. Peak activation memory is roughly **2 × largest layer output**, not the sum of every layer.

### How to pick a size

1. **Measure** — `./netkit inspect models/your_model.json --full` or `nk_inspect_model()`. Use **arena bytes after forward** (includes load + ping buffers + a zero-input forward with caller I/O tensors).
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
| CLI `run` / `inspect`, hand vector tests, examples, C API smoke tests | **64 KiB** (`Arena::kDefaultCapacity`) | Hand MLP/CNN only |
| MNIST MLP tests (`src/test_mnist.cpp`) | **2 MiB** | `mnist_mlp.json` |
| MNIST CNN tests (`src/test_mnist.cpp`) | **4 MiB** | `mnist_cnn.json` |

MNIST weights alone are ~400 KiB (MLP) or ~900 KiB (CNN); the test harness uses larger buffers than the default so load + ping buffers always fit. The CLI still uses 64 KiB — `./netkit inspect --full` on MNIST may report arena overflow unless you integrate with a larger buffer in your own app.

### `reset()` and reload

`reset()` sets the arena offset to zero and **invalidates all pointers** (weights, network, ping buffers). To run again on the same buffer you must **reload the model**. The MNIST test suite calls `arena.reset()` then `LoadMLP` / `LoadCNN` per case for isolation.

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

1. Run `./netkit inspect models/your_model.json --full` (or `nk_inspect_model`).
2. Note **arena bytes after forward** — add headroom (typically 1.5–2× for batch variance).
3. Use one arena per model context, or `reset()` between runs on the same buffer.

| Model | Approx. arena high-water | Test / CLI buffer |
|-------|--------------------------|-------------------|
| Hand test MLP/CNN | < 64 KiB | 64 KiB (default) |
| MNIST MLP | ~1–2 MiB measured | 2 MiB in tests |
| MNIST CNN | ~2–4 MiB measured | 4 MiB in tests |

Run `inspect --full` on your exact model and input shape for deployment numbers.

## Related docs

- [DATATYPES.md](DATATYPES.md) — float32 weights and tensors today
- [c-api.md](c-api.md) — full C arena reference
- [cpp-api.md](cpp-api.md) — C++ arena reference
- [API.md](API.md#memory-model) — overview and memkit note
