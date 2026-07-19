# Data Types and Numeric Precision

Part of the netkit roadmap — see [PHILOSOPHY.md](PHILOSOPHY.md) and [STATUS.md](STATUS.md). **Float32** and **int8** inference are **complete** on cpu, Arm MCU, Arm MPU, **Espressif MCU** (ESP-NN int8 / reference float), **RISC MPU** (XNNPACK), and **RISC MCU** (NMSIS-NN int8 / reference float). float16 / int16 / int4 remain Phase 2.

## Float32 (default)

**Most inference in netkit uses IEEE-754 single precision (`float`, 32-bit).**

| Component | Type |
|-----------|------|
| Tensor payload (inference) | `float` / `DataType::Float32` |
| `.nk` weight/bias payloads | little-endian float32 |
| Activations, matmul, conv | float32 math (`expf`, `tanhf`, etc.) |
| CLI / C API inputs and outputs | float32 (`nk_model_run`); int8 models use `nk_model_run_int8` |
| Regression expected values | float32 |

There is **no float64 (double) inference path**. CLI values are parsed with `strtof` / `ParseFloat` and stored as float32.

## Int8 (implemented — MNIST + ImageNet fixtures)

Int8 inference is available end-to-end (int8 in → int8 out) for MNIST **CNN** / **MLP** and ImageNet MobileNetV4 fixtures:

| Path | Backend |
|------|---------|
| Arm MCU | CMSIS-NN (NUCLEO-F446RE boards) |
| Espressif MCU | ESP-NN (`mcu_esp` + `NETKIT_ARCH=ESP32*`) |
| cpu / MPU (incl. RISC MPU) | XNNPACK qs8 when enabled; else QuantOps integer reference |
| RISC MCU | NMSIS-NN (`mcu_risc` + `NETKIT_ARCH=N300|RV32*…`) |

| Component | Type |
|-----------|------|
| `.nk` quant payload | int8 weights, int32 biases, per-layer `QuantLayerParams`; optional per-channel weight scales (`QUAN_FLAG_PER_CHANNEL_WEIGHTS`) |
| Weight scales | Per-output-channel (TFLite-style) when the QUAN flag is set; else one `weight_scale` per layer |
| Activations | int8 per-tensor (CMSIS-NN / ESP-NN / NMSIS-NN or integer reference kernels; int8 softmax) |
| Device I/O | **int8 in → int8 out** (no float quant / dequant on MCU or in C++) |
| CNN export | `make export-mnist-cnn-int8` |
| MLP export | `make export-mnist-mlp-int8` |
| MCU firmware (CNN) | [boards/nucleo-f446re-cnn-int8](../boards/nucleo-f446re-cnn-int8/README.md) — **10/10** @ **95.3 ms** CMSIS / **336 ms** reference (`NETKIT_EMBED=1`; vs TFLM / microTVM in [STATUS.md](STATUS.md)) |
| MCU firmware (MLP) | [boards/nucleo-f446re-mlp-int8](../boards/nucleo-f446re-mlp-int8/README.md) — **10/10** @ ~3.4 ms (CMSIS) / ~15 ms (reference) |

**On-device / host C++ rule:** int8 inference stays integer end-to-end on MCU, MPU, and CPU. There is **no C++ float→int8 quantization or dequantization** — float→int8 happens only in Python at `.nk` / test-vector export time; scale metadata lives in the model; kernels apply TFLite-style multiply-by-quantized-multiplier. Interpreting outputs (dequantized confidence, float probabilities) is done **offline in Python** (e.g. `benchmark/tools/parse_mcu_cnn_int8_log.py`), never in firmware or `libnetkit`.

**Kernel fusion (int8):** CMSIS-NN fuses **ReLU / ReLU6** output clamps into conv, depthwise, and FC via `QuantInteger::QuantClamp` (ReLU6 uses quantized `6.0`). Softmax, Sigmoid, Tanh, and LeakyReLU remain separate follow-up kernels — CMSIS-NN has no fused Softmax API. For **classification benches** (MNIST MCU firmware), AOT/`MLPNetwork`/`CmsisQuantPlan` support `--omit-final-softmax` / `SetOmitFinalSoftmax(true)` / `runtime.omit_final_softmax`: the final Dense Softmax is skipped and logits are written; `argmax(logits) == argmax(softmax(logits))`. Float max-pool can fuse ReLU/ReLU6 through CMSIS `cmsis_nn_activation` when the layer carries an activation tag. UIB int8 depthwise now tries CMSIS-NN before the reference loop. Residual Add is fused as a conv/FC epilogue where the graph allows it (`Conv2dForward` optional residual; `Conv2dNhwcQuant` + `ResidualAddS8`; ResNet `MatAddThenRelu`) — CMSIS still has no native conv+add, so the epilogue uses `arm_add_f32` / `arm_elementwise_add_s8` after the conv. Quant AOT lowering covers CNN primitives, MLP dense, avg-pool, and MobileNetV4 UIB composites (`forward_quant`); float AOT also lowers ResNet BasicBlock and ConvNeXt V2 blocks.

TFLite input-quant alignment (layer 0) is optional when matching `benchmark/tflm/generated/mnist_*_int8.tflite` exists. Weight and hidden-layer output scales are calibrated from netkit float weights (Python export). ImageNet MobileNetV4 int8 PTQ emits per-channel weight scales so CMSIS-NN / XNNPACK `qc8w` / reference requant match TFLite’s per-axis weights.

Host desktop builds do not run the CMSIS-NN / ESP-NN / NMSIS-NN quant forward paths (`NETKIT_*_NN_ALLOWED=0` on cpu). On **cpu / any MPU**, int8 LayerFast uses **XNNPACK qs8 / qs8_qc8w** when `NETKIT_XNNPACK=1`, else netkit integer reference loops. Python `forward_quantized_cnn` / `forward_quantized_mlp` and Arm MCU firmware remain the CMSIS-NN validation paths; Espressif uses ESP-NN (`mcu_esp`); RISC-V MCU uses NMSIS-NN (`mcu_risc`). Host regression and ImageNet/MNIST int8 benches load **native int8** inputs (Python-prequantized; TCAS uses `FLAG_HAS_INT8_TESTS`) via `nk_model_run_int8` / int8 tensor views — never float→int8 inside C++. Float models stay float32 end-to-end.

The `DataType` / `nk_dtype_t` enums list `Int8`, `UInt8`, and `Int16` for future tensor metadata beyond the current int8 fixtures.

## Planned (roadmap)

Additional reduced-precision paths are planned for **Phase 2** (Python packager + runtime decode kernels):

| Type | Status | Intended use |
|------|--------|--------------|
| **float16** | Planned | Half-precision weights/activations where hardware supports FP16 |
| **int16** | Planned | Wider quantized weights; intermediate precision |
| **int8** | **Complete** (MNIST + ImageNet fixtures) | PTQ `.nk` + CMSIS-NN (Arm MCU) / ESP-NN (Espressif) / NMSIS-NN (RISC-V MCU) / XNNPACK qs8 (cpu/MPU) / QuantOps reference; broader models planned |
| **int4** | Planned | Aggressive edge quantization (kernel and layout TBD) |

When float16 / int16 / int4 land, expect:

- Extended `.nk` format versions with per-tensor dtype tags
- Packager-side quantization, calibration, and layout selection ([PHILOSOPHY.md](PHILOSOPHY.md))
- Runtime load paths and kernels per dtype (possibly fused in Phase 2)
- Updated regression suites with tolerance policies per type

Float32 remains the default for models without a dedicated int8 export — see [NK_FORMAT.md](NK_FORMAT.md) and [STATUS.md](STATUS.md).

## API surface

Float32 models use float I/O; int8 models use int8 I/O:

| C++ / C | Role |
|---------|------|
| `nk_model_run(..., const float* input, ..., float* output, ...)` | float32 models only |
| `nk_model_run_int8(..., const int8_t* input, ..., int8_t* output, ...)` | int8 models only |
| AOT `Model::forwardInt8` | MCU int8 firmware path |

Do not assume float inputs work for quantized models — quantize in Python at export time.

## Related docs

- [NK_FORMAT.md](NK_FORMAT.md) — `.nk` weight layout (float32)
- [ARENA.md](ARENA.md) — weight loading into arena (`alignof(float)`)
- [API.md](API.md) — overview
