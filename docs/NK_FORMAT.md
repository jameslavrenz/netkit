# netkit Binary Model Format (`.nk`)

Version **3** — single-file inference bundle for embedded runtimes. Produced by the Python package in [`python/`](../python/); consumed by the C++ loader (`NkLoader`) and `./netkit run` / `./netkit inspect`. Load at runtime (interpreter) or embed with `python -m netkit aot` (compiled path) — [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

**Canonical byte-level specification:** [NK_FILE_SPECIFICATION.md](NK_FILE_SPECIFICATION.md) (header offsets, layer record sizes, payload alignment, TCAS layout, `xxd` / CLI inspection).

## Overview

| Section | Contents |
|---------|----------|
| Header | Magic, version, network kind, input shape, layer/tensor counts, payload sizes |
| Layer descriptors | One record per layer (dense, conv2d, depthwise_conv2d, max_pool2d, avg_pool2d, batch_norm2d, convnextv2_block, flatten) |
| Tensor catalog | Shape + dtype for each weight tensor, then each bias tensor |
| Weight payload | All weight tensors concatenated (float32, little-endian) |
| Bias payload | All bias tensors concatenated (float32, little-endian) |

Weights and biases are **split into separate sections** within the `.nk` file (weights payload first, then biases payload).

Optional **embedded regression tests** (flag `kFlagHasTests`) append a `TCAS` section after the bias payload so a single `.nk` file carries model weights and test cases.

## Format limits (v3)

| Constant | Value | Notes |
|----------|------:|-------|
| `kMaxLayers` | 100 | Layer descriptor array in loader and op list |
| `kMaxTensorCatalog` | 128 | Weight + bias tensor descriptors |
| `kMaxTestCases` | 16 | Embedded TCAS cases per file |
| `kMaxCaseFloats` | 16384 | Max input or expected floats per TCAS case (e.g. 56×56×3 = 9408) |

Python mirror: `python/netkit/format.py` (`MAX_LAYERS`, `MAX_TENSOR_CATALOG`, `MAX_CASE_FLOATS`).

## File layout

```
┌──────────────────────────────────────┐
│ Header (48 bytes)                    │
├──────────────────────────────────────┤
│ Layer descriptors (variable)         │
├──────────────────────────────────────┤
│ Weight tensor catalog (24 bytes × Nw)│
├──────────────────────────────────────┤
│ Bias tensor catalog (24 bytes × Nb)  │
├──────────────────────────────────────┤
│ Weight payload (float32 × Σ W)       │
├──────────────────────────────────────┤
│ Bias payload (float32 × Σ b)         │
├──────────────────────────────────────┤
│ Test section (optional, TCAS)        │
└──────────────────────────────────────┘
```

## Header (48 bytes, little-endian)

| Offset | Type | Field |
|--------|------|-------|
| 0 | `char[4]` | Magic `"NKIT"` |
| 4 | `uint32` | Format version (`3`) |
| 8 | `uint8` | Network kind (`1`=MLP, `2`=CNN) |
| 9 | `uint8` | Input rank (1–4) |
| 10 | `uint16` | Flags (`0x0001` = embedded tests) |
| 12 | `uint32[4]` | Input shape (unused dims `0`) |
| 28 | `uint32` | Layer count |
| 32 | `uint32` | Weight tensor count |
| 36 | `uint32` | Bias tensor count |
| 40 | `uint32` | Weight payload bytes |
| 44 | `uint32` | Bias payload bytes |

### Input shapes

| Network | `input_rank` | Shape meaning |
|---------|--------------|---------------|
| MLP | 2 | `[batch, features]` |
| CNN | 3 | `[height, width, channels]` NHWC |

## Layer descriptors

Each layer starts with **`uint8 kind` + 3 reserved bytes**, then kind-specific fields:

| Kind | Value | Extra fields |
|------|------:|--------------|
| `dense` | 1 | `units u32`, `activation u8`, pad×3, `alpha f32` |
| `conv2d` | 2 | `kernel u32`, `stride u32`, `filters u32`, `activation u8`, `pad_h u8`, `pad_w u8`, `pad_extra u8`, `alpha f32` |
| `depthwise_conv2d` | 7 | `kernel_h u32`, `stride u32`, `channels u32`, `activation u8`, `pad_h u8`, `pad_w u8`, `kernel_w u8`, `alpha f32` |
| `max_pool2d` | 3 | `pool_size u32`, `stride u32`, `pad_h u8`, `pad_w u8`, `reserved u16` |
| `flatten` | 4 | (none) |
| `avg_pool2d` | 5 | `pool_size u32`, `stride u32`, `pad_h u8`, `pad_w u8`, `reserved u16` |
| `batch_norm2d` | 6 | `channels u32`, `reserved u32` |
| `convnextv2_block` | 8 | `channels u32`, `reserved u32`, `eps f32` |
| `mobilenetv4_uib` | 9 | `in_channels u32`, `out_channels u32`, `start_dw u8`, `middle_dw u8`, `stride u8`, `middle_dw_downsample u8`, `expand_ratio f32`, `reserved u32` |
| `resnet_basic_block` | 10 | `in_channels u32`, `out_channels u32`, `stride u8`, `reserved u8[3]` |
| `layernorm2d` | 11 | `channels u32`, `eps f32`, `reserved u32` |
| `yolox_decoupled_head` | 12 | `in_channels u32`, `hidden_dim u32`, `num_classes u32`, `num_convs u8`, `reserved u8[3]` |

See [CONVNEXTV2.md](CONVNEXTV2.md), [MOBILENETV4.md](MOBILENETV4.md), and [YOLOX.md](YOLOX.md) for fused block details.

Weight/bias tensor pairs (in layer order, W then B each):

1. depthwise `[C,7,7]` + bias `[C]`
2. LayerNorm scale `[C]` + bias `[C]`
3. expand pointwise `[4C,C]` + bias `[4C]`
4. GRN gamma `[4C]` + beta `[4C]`
5. project pointwise `[C,4C]` + bias `[C]`

### Activation enum (`uint8`)

`0=none`, `1=relu`, `2=sigmoid`, `3=tanh`, `4=leaky_relu`, `5=relu6`, `6=softmax`

## Tensor catalog entry (24 bytes)

| Field | Type |
|-------|------|
| `rank` | `uint8` |
| `dtype` | `uint8` (`1=float32`) |
| pad | `uint16` |
| `dims` | `uint32[4]` (unused dims `0`) |
| `num_elements` | `uint32` |

Example MLP first-layer weight with `2 → 2` units: `rank=2`, `dtype=float32`, `dims=[2,2]` (`[out_features, in_features]`), `num_elements=4`.

Example depthwise conv weight: `rank=3`, `dims=[C,Kh,Kw]` per channel.

**1D depthwise:** there is no separate `depthwise_conv1d` kind. Use `depthwise_conv2d` with independent `kernel_h` / `kernel_w`. For a sequence axis of length `T` with `W=1`, set input shape `[T,1,C]` and e.g. `kernel_h=5`, `kernel_w=1`, `pad_h=2`, `pad_w=0` (weights `[C,5,1]`).

Example conv weight: `rank=4`, `dims=[O,Kh,Kw,I]` in **netkit** layout `[out, kernel, kernel, in_channels]`.

### Asymmetric padding and non-square pools

ONNX `pads` are four values `(top, left, bottom, right)`. The `.nk` format stores **top/left** in `pad_h` / `pad_w` and packs **extra bottom/right** into a single byte so symmetric padding stays backward compatible.

**Conv2D `pad_extra` byte** (field named `kernel_w` in depthwise layers — do not reuse there):

| Nibble | Meaning |
|--------|---------|
| low 4 bits | `pad_h_end - pad_h` (extra bottom padding, 0–15) |
| high 4 bits | `pad_w_end - pad_w` (extra right padding, 0–15) |

Decode: `pad_h_end = pad_h + (pad_extra & 0xF)`, `pad_w_end = pad_w + ((pad_extra >> 4) & 0xF)`.

Output height/width use per-side padding: `(size + pad_before + pad_after - kernel) // stride + 1`.

**Pool `reserved u16`** (max/avg pool):

| Bits | Meaning |
|------|---------|
| low 8 bits | `pool_w` when different from `pool_size` (`pool_h`); `0` means square kernel |
| bits 8–11 | extra bottom padding (`pad_h_end - pad_h`) |
| bits 12–15 | extra right padding (`pad_w_end - pad_w`) |

Python encode/decode: `python/netkit/pad_encoding.py` (`encode_pad_extra`, `encode_pool_reserved`). C++: `include/nk_op_detail.hpp` (`DecodeConvPadExtra`, `DecodePoolMeta`).

## Payload

- **Weights:** concatenated in layer order (conv/dense layers only)
- **Biases:** concatenated in the same layer order
- All values IEEE-754 **float32**, little-endian

### Weight load policy (runtime)

Same `.nk` bytes on disk or in flash. Weights **always** stay in the blob: buffer/AOT bind views into the payload; optional POSIX **`mmap`** file load when `NETKIT_USE_MMAP=1` (CPU default on macOS/Linux; off for MCU/MPU unless opted in). Arena holds structs + activations only. For buffer load, `data` must outlive the network. Misaligned payloads return a load error. See [ARENA.md](ARENA.md#weight-storage-always-flashblob-backed).

## Embedded regression tests (optional)

When header `flags & 0x0001` is set, a test section follows the bias payload:

| Field | Type | Description |
|-------|------|-------------|
| Magic | `char[4]` | `"TCAS"` |
| Case count | `uint32` | Number of test cases (≤ 16) |
| Tolerance | `float32` | Per-output absolute tolerance |

Each case:

| Field | Type | Description |
|-------|------|-------------|
| Name length | `uint8` | 1–127 |
| Name | `char[len]` | UTF-8, padded to 4-byte alignment |
| Label | `int32` | Expected class (`-1` = none) |
| Input count | `uint32` | Input float count |
| Input | `float32 × count` | Flat input tensor |
| Output count | `uint32` | Expected output float count |
| Expected | `float32 × count` | Reference outputs |

Runtime inference ignores the test section. `NkRegression::RunModelTests()` loads embedded cases via `NkLoader::ReadTestSuite()`. Hand-check **inputs** are defined in `python/netkit/regression_data.py`; `make embed-tests` computes expected outputs from each model's weights via the NumPy reference forward pass. MNIST export scripts embed cases at write time.

## Tooling

### Python (convert ONNX → `.nk`)

```bash
pip install -e python
python -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk
python -m netkit inspect models/test_mlp.nk
make export-nk    # convert all bundled regression models
```

### C++ runtime

```bash
./netkit inspect models/test_mlp.nk
./netkit run models/test_mlp.nk --input 1,2
```

Headers: `include/nk_format.hpp`, `include/nk_loader.hpp`

## Related docs

- [ONNX.md](ONNX.md) — supported ONNX ops for conversion
- [DATATYPES.md](DATATYPES.md) — float32-only inference today
