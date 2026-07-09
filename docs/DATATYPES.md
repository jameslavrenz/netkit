# Data Types and Numeric Precision

Part of the netkit roadmap — see [PHILOSOPHY.md](PHILOSOPHY.md). **Float32** is the default inference path. **Int8** post-training quantization is implemented for MNIST **CNN** and **MLP** on MCU (CMSIS-NN); broader dtype support is Phase 2.

## Float32 (default)

**Most inference in netkit uses IEEE-754 single precision (`float`, 32-bit).**

| Component | Type |
|-----------|------|
| Tensor payload (inference) | `float` / `DataType::Float32` |
| `.nk` weight/bias payloads | little-endian float32 |
| Activations, matmul, conv | float32 math (`expf`, `tanhf`, etc.) |
| CLI / C API inputs and outputs | float32 |
| Regression expected values | float32 |

There is **no float64 (double) inference path**. CLI values are parsed with `strtof` / `ParseFloat` and stored as float32.

## Int8 (MNIST CNN + MLP — implemented)

Int8 inference is available for MNIST **CNN** and **MLP** on **MCU + CMSIS-NN**:

| Component | Type |
|-----------|------|
| `.nk` quant payload | int8 weights, int32 biases, per-layer `QuantLayerParams` |
| Activations | int8 (CMSIS-NN conv/pool/FC + int8 softmax) |
| CNN export | `make export-mnist-cnn-int8` |
| MLP export | `make export-mnist-mlp-int8` |
| MCU firmware (CNN) | [boards/nucleo-f446re-cnn-int8](../boards/nucleo-f446re-cnn-int8/README.md) — **10/10** @ ~95 ms |
| MCU firmware (MLP) | [boards/nucleo-f446re-mlp-int8](../boards/nucleo-f446re-mlp-int8/README.md) — **10/10** @ ~3.4 ms (CMSIS) / ~15 ms (reference) |

TFLite input-quant alignment (layer 0) is optional when matching `benchmark/tflm/generated/mnist_*_int8.tflite` exists. Weight and hidden-layer output scales are calibrated from netkit float weights.

Host desktop builds do not run the CMSIS-NN quant forward path (`NETKIT_CMSIS_NN_ALLOWED=0`); use Python `forward_quantized_cnn` or flash MCU firmware for int8 validation.

The `DataType` / `nk_dtype_t` enums list `Int8`, `UInt8`, and `Int16` for future tensor metadata beyond the MNIST CNN path.

## Planned (roadmap)

Quantized and reduced-precision paths are planned for **Phase 2** (Python packager + runtime decode kernels):

| Type | Status | Intended use |
|------|--------|--------------|
| **float16** | Planned | Half-precision weights/activations where hardware supports FP16 |
| **int16** | Planned | Wider quantized weights; intermediate precision |
| **int8** | **MNIST CNN + MLP MCU** | Post-training quant export + CMSIS-NN kernels; broader models planned |
| **int4** | Planned | Aggressive edge quantization (kernel and layout TBD) |

When added, expect:

- Extended `.nk` format versions with per-tensor dtype tags
- Packager-side quantization, calibration, and layout selection ([PHILOSOPHY.md](PHILOSOPHY.md))
- Runtime load paths and kernels per dtype (possibly fused in Phase 2)
- Updated regression suites with tolerance policies per type

Until broader dtype support lands, **export scripts for non-CNN models must emit float32** — see [NK_FORMAT.md](NK_FORMAT.md).

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
