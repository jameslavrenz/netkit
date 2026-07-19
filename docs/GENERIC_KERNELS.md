# Generic (reference) kernels — 32-bit+ optimization

How netkit’s **portable reference kernels** are tuned for devices with a native
**32-bit (or wider) word**, and where those choices limit portability.

Companion docs: [KERNELS.md](KERNELS.md) (backend architecture),
[BUILD_TARGETS.md](BUILD_TARGETS.md) (flags), [STATUS.md](STATUS.md) (platform
maturity), [DATATYPES.md](DATATYPES.md) (float32 / int8).

---

## 1. What “generic” means here

netkit routes numeric work through a compile-time kernel facade
(`Kernels::…`). When CMSIS-NN, ESP-NN, and XNNPACK are off (or reject an op), execution
falls through to **`ReferenceKernel`** plus, for int8, **`QuantOps`** integer
loops. That stack is what this document calls the **generic** path.

It is **not** a vector-ISA microkernel library (no NEON / Helium / RVV / SSE in
these TUs). Speedups come from:

- algorithm / data-layout choices (NHWC traversal, specialists, im2col),
- instruction-level parallelism in scalar code (multi-accumulator reductions),
- integer-only int8 math with $\mathrm{int32}$ MACs and TFLite-style requant,
- optional code-size / workspace tradeoffs (`NETKIT_LOOP_UNROLL`,
  `NETKIT_IM2COL`).

**RISC MCU “fast generic”** is the **same** code as desktop reference — there
is no separate RISC kernel translation unit. Until a CMSIS-NN–class RISC MCU
library exists, `mcu_risc` ships on these kernels as the production path.

```text
Layer / QuantOps
        │
        ▼
  active_kernel.hpp  →  Kernels alias
        │
        ├─ mcu_arm + CMSIS-NN     → ComposedKernel<Reference, CmsisNn>
        ├─ cpu/MPU + XNNPACK      → ComposedKernel<Reference, Xnnpack>
        └─ else (incl. mcu_risc)  → ReferenceKernel (+ QuantOps for int8)
```

---

## 2. Target machine model (32-bit+)

The generic path assumes a machine that looks like a modern Cortex-M / RISC-V
MCU, Cortex-A MPU, or desktop CPU:

| Capability | Why the kernels need it |
|------------|-------------------------|
| Native **`int32_t` / `uint32_t`** | Loop extents, int8 MAC accumulators, quantized multipliers |
| Native **`float`** (IEEE-754 binary32) on float paths | FC / conv / pool / activations |
| Soft or hard **`int64_t`** | Int8 requant (`a·b` widen before rounding) |
| Pointers and `size_t` ≥ 32-bit | Arena buffers, NHWC indexing |
| Reasonable alignment | Arena `alignof(float)`; some int8 patches use `alignas(16)` |

Formally, extents and indices are almost always $\mathrm{uint32}$; an NHWC
activation at spatial $(h,w)$ and channel $c$ is addressed as

$$
\mathrm{idx}(h,w,c) = (h \cdot W + w)\cdot C + c .
$$

Products such as $H\cdot W\cdot C$ are assumed to fit in 32-bit indexing for
the model sizes netkit targets (MCU/MPU fixtures), not arbitrary host-scale
tensors.

### What is out of scope

| Constraint | Consequence |
|------------|-------------|
| **16-bit-only MCUs** (no native 32-bit ALU / float) | Not supported — MAC and float kernels are written for 32-bit types |
| **No `int64`** | Int8 requant breaks or needs a different implementation |
| **Strict-alignment cores with packed unaligned loads** | Kernels use typed `float*` / `int8_t*` arithmetic; they do not emulate unaligned access |
| **Big-endian hosts loading `.nk` as-is** | `.nk` is **little-endian**; load path must byte-swap multi-byte fields (kernels themselves are endian-agnostic once values are in registers) |
| **Vector ISA required for correctness** | None — generics are scalar C++; SIMD is left to XNNPACK / CMSIS-NN |

---

## 3. Enumerated speedup methods

### 3.1 Always-on four-accumulator float dots

**Where:** `include/netkit_loop_unroll.hpp` —
`NetkitLoopUnroll::dot_contiguous` / `dot_strided` / `dot_strided_b_offset`;
exposed as `NetkitUtil::DotProductF32`. Used by dense FC, `MatMulImpl`, 1×1
conv, 3×3 direct conv channel dots, partial im2col, depthwise row dots.

**Idea.** A single accumulator reduction

$$
s \leftarrow s + a_t b_t
$$

is **latency-bound** on pipelined FPUs: each FMA waits on the previous
write-back. Four independent partial sums

$$
\begin{aligned}
s_0 &\leftarrow s_0 + a_t b_t, &
s_1 &\leftarrow s_1 + a_{t+1} b_{t+1}, \\
s_2 &\leftarrow s_2 + a_{t+2} b_{t+2}, &
s_3 &\leftarrow s_3 + a_{t+3} b_{t+3},
\end{aligned}
$$

then

$$
s = (s_0 + s_1) + (s_2 + s_3)
$$

expose ILP so the FPU can keep several multiply-adds in flight. Remainder
length $\lvert n\rvert \bmod 4$ is a scalar cleanup loop.

**Why always on.** Code-size cost is a handful of instructions on the hottest
float path; it is **not** gated by `NETKIT_LOOP_UNROLL`.

**Applies to:** float32 only.  
**Portability:** Correct on any IEEE-754 `float` target. Helps most when a
hardware FPU has multi-cycle FMA latency (Cortex-M4F, RV32F, desktop). Soft-float
remains correct but stays expensive. Assumes natural `float` alignment
($\operatorname{alignof}(\texttt{float})=4$ on ordinary ABIs).

---

### 3.2 Cache-friendly NHWC loop order (default, no flag)

**Where:** float and int8 spatial ops in `reference_kernel.cpp`,
`conv_*_kernel.cpp`, `quant_ops.cpp`; narrative in [KERNELS.md](KERNELS.md).

**Idea.** Activations are NHWC. Looping

$$
\text{for } oh,\ ow,\ c
$$

writes

$$
\mathrm{out}[(oh\cdot W_{\mathrm{out}}+ow)\cdot C + c]
$$

with **sequential channel stores**, which use cache lines / tightly coupled
SRAM better than channel-outer loops. Inner reductions over input channels
$i_c$ walk contiguous NHWC channels at a fixed $(i_h,i_w)$.

Other low-cost conventions (always on):

| Pattern | Benefit |
|---------|---------|
| Row-level padding skip | If $i_h$ is OOB, skip the whole $k_w$ row — fewer branches |
| Inlined $\mathrm{idx}(h,w,c)$ | Avoid helper call overhead on hot paths |
| Contiguous `MatAdd` when `stride[1]==1` | One linear pass instead of 2D indexing |
| Structured `for` (no hot-path `while`) | Predictable control flow |

**Portability:** Layout-dependent (NHWC). No word-size trick beyond 32-bit
indices. Helps any cache or TCM hierarchy; still good practice on tiny MCUs
with no cache (sequential SRAM access).

---

### 3.3 Float Conv2D specialized lowering

**Where:** `conv_dispatch.cpp` + `conv1x1_kernel.cpp`,
`conv_direct_kernel.cpp`, `conv_depthwise_kernel.cpp`, im2col modules.

Hard rules (compiler-style policy, not user knobs except im2col — §3.6):

| Case | Path |
|------|------|
| $1\times 1$, stride 1 | Always **direct** channel dots — never im2col |
| Depthwise | Always **direct** — never im2col |
| $3\times 3$, stride 1, pad 0 | Unrolled 9 contiguous channel dots |
| Generic / padded | Nested $k_h,k_w$ + channel dots; optional input-stationary |

**Input-stationary + HWIO.** At load, `RepackConv2dWeights` builds a second
weight view in HWIO $[k_h,k_w,C_{\mathrm{in}},C_{\mathrm{out}}]$ beside the
stored OIHW blob. When $C_{\mathrm{out}}$ is large enough, the direct path can
stream each input activation $x$ and scatter-add

$$
y_{oc} \leftarrow y_{oc} + x \cdot W_{\mathrm{hwio}}[\ldots, oc]
$$

across output channels, skipping exact zeros in $x$. That improves reuse of
input taps when the working set is weight-heavy.

**Portability:** Float-only; needs arena space for the HWIO copy. Zero-skip is
a value-dependent branch (harmless for correctness; benefit varies).

---

### 3.4 Float depthwise: clipped windows + cross-row accumulators

**Where:** `conv_depthwise_kernel.cpp`. Knob `NETKIT_DW_ROW_ACCUM` (default
**on**).

**Idea.** For each output $(oh,ow,c)$, precompute the valid kernel row/column
ranges so interior pixels avoid per-tap bounds checks. Each kernel row is a
strided float dot (NHWC channel stride $= C$) via the 4-accumulator helper.
Across rows, partial sums are optionally round-robined into four cross-row
accumulators (same ILP idea as §3.1).

**Portability:** Float depthwise only. Same 32-bit / FPU assumptions as §3.1.

---

### 3.5 Float specialists (pool, FC)

| Specialist | Role |
|------------|------|
| Dense FC row-major | Contiguous `DotProductF32` + fused bias / ReLU-family clamp |
| Max-pool $2\times 2$, stride 2, pad 0 | Four-tap channel loop without generic nest |
| Generic pool | Nested window with bounds; still NHWC $oh\to ow\to c$ |

These avoid general machinery when shapes match common CNN stems.

---

### 3.6 Optional im2col → GEMM (`NETKIT_IM2COL`)

**Where:** `conv_im2col_policy.hpp`, `im2col_partial.cpp`, `im2col_full.cpp`,
`im2col_quant.cpp`. Default **`NETKIT_IM2COL=0`** (direct only).

| Value | Strategy |
|------:|----------|
| $0$ | Direct loops only (product default) |
| $1$ | **Partial** im2col: one patch per output pixel + per-filter dots |
| $2$ | **Full** im2col matrix + float `MatMul` (or int8 per-OC MAC) |

Volume heuristics (when mode allows):

- Partial warrants if $\mathrm{patch}\times\mathrm{spatial} \ge 2048$,
- Full warrants if $\ge 32768$ and mode $= 2$.

For float, full im2col reuses the 4-accumulator `MatMulImpl`. For int8, dots
stay scalar $\mathrm{int32}$ MACs (§3.8). Workspace is arena-backed; if scratch
does not fit, QuantOps degrades full → partial → direct.

**Tradeoff.** Can help large float CNNs on reference-only MPU/MCU builds; costs
RAM and gather bandwidth. CMSIS-NN / ESP-NN / XNNPACK **ignore** this knob.

**Portability:** Same 32-bit types; larger models may need arena headroom that
tiny MCUs lack — hence default $0$.

---

### 3.7 Experimental 4× body unroll (`NETKIT_LOOP_UNROLL`)

**Where:** `NetkitLoopUnroll::for_count` wrapping elementwise mul/add/scale and
some float activations / BN / Softmax loops. **Off by default.**

Duplicates the loop body four times:

$$
i,\ i+1,\ i+2,\ i+3
$$

to expose ILP and help the compiler. Independent of CMSIS
`ARM_MATH_LOOPUNROLL`.

**Tradeoff.** Grows `.text` / flash. Prefer measuring the link map; more
plausible on **MPU** than flash-tight MCU. Does **not** gate the always-on
4-accumulator dots (§3.1).

**Portability:** Pure C++; fine on any target, but flash-limited devices may
fail to link.

---

### 3.8 Int8 QuantOps: $\mathrm{int32}$ MAC + integer requant

**Where:** `quant_ops.cpp`, `quant_integer.hpp`, `im2col_quant.cpp`. Compiled
on host/MPU always; on MCU class builds only when
`NETKIT_REFERENCE_QUANT_LOOPS=1` (otherwise flash-trimmed under
`NETKIT_MCU_ACCEL_ONLY` / `NETKIT_MCU_CMSIS_ONLY`).

**Hot path.** For filter weight $w$, activation $x$, and zero-point offset
$z_x$ (when not folded):

$$
\mathrm{acc} = \sum_i w_i\,(x_i + z_x) \in \mathbb{Z}_{32},
$$

then TFLite-compatible requant

$$
y = \mathrm{sat}_{\,[y_{\min},y_{\max}]}\!\left(
  \mathrm{round}\big(\mathrm{acc}\cdot m \cdot 2^{s-31}\big) + z_y
\right),
$$

with quantized multiplier $m\in\mathbb{Z}_{32}$ and shift $s$. Intermediate
products use $\mathrm{int64}$ widen (`SaturatingRoundingDoublingHighMul`,
`MultiplyByQuantizedMultiplier`). **No float** in the inner MAC/requant once
multipliers are prepared.

**Structural opts (int8 conv):**

1. **Bias fold** — precompute
   $\mathrm{bias}'_{oc} = b_{oc} + z_x\sum_i w_{oc,i}$ so the interior uses a
   plain MAC.
2. **Valid output range** — border pixels keep bounds checks; interior does
   not.
3. **Shape specialists** — e.g. $3\times 3$ RGB stem (gather 27 taps, reuse
   across $C_{\mathrm{out}}$), $3\times 3$ / $5\times 5$ tap-unrolled
   depthwise, `alignas(16)` patch buffers for gather+flat MAC.
4. **Prepared** per-channel multipliers / shifts / clamp bounds — avoid float
   scale math in the loop.

**Asymmetry with float.** Int8 `DotProductS8` uses a **single** $\mathrm{int32}$
accumulator (scalar loop), not a 4-way split. Int8×int8 → int32 is already
cheap on 32-bit ALUs; the win is integer-only requant and less branching, not
FPU ILP.

**Portability limits (int8):**

- Requires 32-bit MAC accumulation and 64-bit widen for requant.
- `alignas(16)` patches prefer 16-byte alignment (still correct if
  over-aligned; not a SIMD dependency).
- MCU production usually uses **CMSIS-NN** instead; keep reference loops only
  for A/B or RISC MCU (`REFERENCE_QUANT_LOOPS=1`).

---

### 3.9 Portable helpers (`NetkitUtil`)

`CopyF32` / `CopyInt8` (`memcpy`), scalar `ArgMax*`, and elementwise
mul/add/scale (optional unroll). **No CMSIS-DSP.** Keeps board staging and
reference fallbacks free of vendor DSP libraries.

---

## 4. What stays deliberately simple

| Area | Behavior |
|------|----------|
| Softmax / GELU / Sigmoid / Tanh | Scalar `expf` / `tanh` math |
| Softmax max reduction | Scalar scan |
| LayerNorm / GRN | Scalar (GRN uses `double` sum-of-squares per channel) |
| Generic max/avg pool | Nested windows (except float $2\times 2$ s2 p0) |
| Int8 FC / pool / flatten | Straightforward NHWC scalar |
| Default product flags | `NETKIT_IM2COL=0`, `NETKIT_LOOP_UNROLL=0` |
| Float32 on Arm MCU | Reference only by policy — production MCU is int8 + CMSIS-NN |
| ISA vector code in reference TUs | None |

Further reference-kernel work is planned along the same themes (cache/line use,
fewer branches) without requiring a vector ISA.

---

## 5. Portability summary

### Safe / intended platforms

- 32-bit and 64-bit Arm (Cortex-M with soft/hard float, Cortex-A)
- 32-bit and 64-bit RISC-V (with or without F extension)
- Desktop x86_64 / aarch64 (reference path when XNNPACK is off)

### Fragile or unsupported

| Risk | Why |
|------|-----|
| 8/16-bit MCUs | Types and MAC width assume $\ge 32$-bit |
| No hardware or soft `int64` | Int8 requant |
| Flash ≪ code growth from `NETKIT_LOOP_UNROLL=1` or fat specialists | Link failure |
| Arena too small for im2col / HWIO repack | Must stay on direct (`IM2COL=0`) or larger arena |
| Big-endian `.nk` load without swap | Format is LE |
| Expecting SIMD from reference alone | Use XNNPACK (cpu/MPU) or CMSIS-NN (Arm MCU) |

### Endianness and alignment (short)

- **Computation** after load: ordinary C++ arithmetic — host endian.
- **Model bytes:** little-endian `.nk` ([NK_FILE_SPECIFICATION.md](NK_FILE_SPECIFICATION.md)).
- **Alignment:** rely on arena allocation; int8 gather patches request 16-byte
  alignment for friendly access patterns on 32/64-bit hosts.

---

## 6. Flag cheat-sheet

| Flag | Default | Affects generic path | Notes |
|------|---------|----------------------|-------|
| `NETKIT_IM2COL` | $0$ | Float ref + int8 QuantOps Conv2D | Ignored by CMSIS-NN / ESP-NN / XNNPACK |
| `NETKIT_LOOP_UNROLL` | $0$ | Float elementwise / some activations | Experimental; grows `.text` |
| `NETKIT_DW_ROW_ACCUM` | $1$ | Float depthwise cross-row ILP | — |
| `NETKIT_REFERENCE_QUANT_LOOPS` | $0$ | Keep int8 QuantOps bodies on MCU class | Needed for RISC MCU int8 / reference A/B |
| `NETKIT_MCU_ACCEL_ONLY` (`NETKIT_MCU_CMSIS_ONLY`) | derived | Strip QuantOps ref when loops off (CMSIS or ESP production) | Float `ReferenceKernel` still links |

Guidance: leave **`NETKIT_IM2COL=0`** and **`NETKIT_LOOP_UNROLL=0`** unless
profiling a reference-only build. See
[BUILD_TARGETS.md](BUILD_TARGETS.md#netkit_im2col-guidance).

---

## 7. Relationship to accelerated backends

| Backend | Role vs generics |
|---------|------------------|
| **XNNPACK** | Production LayerFast on cpu / MPU — ISA microkernels; generics are fallback |
| **CMSIS-NN** | Production int8 on Arm MCU; generics for float MCU and CMSIS rejects |
| **ESP-NN** | Production int8 on Espressif MCU; float always uses generics (no ESP float API) |
| **Generics** | Correctness baseline, RISC MCU production, host “XNNPACK OFF” peer, accel fallback |

Peer benches that disable XNNPACK compare against this stack (and, for TF Lite,
often `BUILTIN_REF`). See [STATUS.md](STATUS.md).

---

## 8. Key source map

| Path | Role |
|------|------|
| `include/reference_kernel.hpp`, `src/reference_kernel.cpp` | Float op surface + pool/FC/activations |
| `include/netkit_loop_unroll.hpp` | 4-acc dots + optional unroll |
| `include/netkit_util.hpp`, `src/netkit_util.cpp` | memcpy / ArgMax / elementwise |
| `include/conv_im2col_policy.hpp` | im2col volume policy |
| `src/conv_dispatch.cpp` | Float Conv lowering |
| `src/conv1x1_kernel.cpp`, `conv_direct_kernel.cpp`, `conv_depthwise_kernel.cpp` | Direct specialists |
| `src/im2col_{partial,full,quant}.cpp` | im2col paths |
| `include/quant_integer.hpp`, `include/quant_ops.hpp`, `src/quant_ops.cpp` | Int8 integer path |

---

## 9. Takeaway

Generic kernels are **portable C++ tuned for 32-bit+ machines**: NHWC-friendly
loops, always-on multi-accumulator float reductions, shape specialists, optional
im2col / unroll, and integer-only int8 MAC+requant. They intentionally avoid
ISA-specific SIMD so the same sources serve desktop reference builds, Arm MCU
fallback, Espressif float, and RISC MCU production. The cost of that portability is that peak
host/MPU performance still belongs to **XNNPACK**, peak Arm MCU int8
to **CMSIS-NN**, and peak Espressif MCU int8 to **ESP-NN** — with the understanding that almost every
speedup above assumes at least a 32-bit word, native or soft `float` /
`int64` where those paths run, and little-endian model load.
