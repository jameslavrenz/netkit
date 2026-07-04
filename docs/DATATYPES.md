# Data Types and Numeric Precision

Part of the netkit roadmap — see [PHILOSOPHY.md](PHILOSOPHY.md) (Phase 2 packager will emit quantized `.nk` payloads).

## Float32 only (today)

**All inference in netkit today uses IEEE-754 single precision (`float`, 32-bit).**

| Component | Type |
|-----------|------|
| Tensor payload (inference) | `float` / `DataType::Float32` |
| `.nk` weight/bias payloads | little-endian float32 |
| Activations, matmul, conv | float32 math (`expf`, `tanhf`, etc.) |
| CLI / C API inputs and outputs | float32 |
| Regression expected values | float32 |

There is **no float64 (double) inference path**. CLI values are parsed with `strtof` / `ParseFloat` and stored as float32.

The `DataType` / `nk_dtype_t` enums list `Int8`, `UInt8`, and `Int16` for future tensor metadata — **these are not used for inference yet**.

## Planned (roadmap)

Quantized and reduced-precision paths are planned for **Phase 2** (Python packager + runtime decode kernels):

| Type | Status | Intended use |
|------|--------|--------------|
| **float16** | Planned | Half-precision weights/activations where hardware supports FP16 |
| **int16** | Planned | Wider quantized weights; intermediate precision |
| **int8** | Planned | Standard post-training or QAT deployment on MCU/MPU |
| **int4** | Planned | Aggressive edge quantization (kernel and layout TBD) |

When added, expect:

- Extended `.nk` format versions with per-tensor dtype tags
- Packager-side quantization, calibration, and layout selection ([PHILOSOPHY.md](PHILOSOPHY.md))
- Runtime load paths and kernels per dtype (possibly fused in Phase 2)
- Updated regression suites with tolerance policies per type

Until then, **export scripts and models must emit float32** — see [NK_FORMAT.md](NK_FORMAT.md).

## API surface

Both APIs expose float-only data accessors:

| C++ | C |
|-----|---|
| `tensor_data_f32()` | `nk_tensor_data_f32()` |
| `nk_tensor_fill` | fill tensor elements from `float*` (C); `TensorFactory::Fill(tensor, std::span<const float>)` (C++) |
| `nk_model_run(..., const float* input, ..., float* output, ...)` | same |

Do not assume `double` or integer tensor payloads work for forward passes.

## Related docs

- [NK_FORMAT.md](NK_FORMAT.md) — `.nk` weight layout (float32)
- [ARENA.md](ARENA.md) — weight loading into arena (`alignof(float)`)
- [API.md](API.md) — overview
