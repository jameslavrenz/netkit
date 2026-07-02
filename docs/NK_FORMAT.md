# netkit Binary Model Format (`.nk`)

Version **2** — single-file inference bundle for embedded runtimes. Produced by the Python package in [`python/`](../python/); consumed by the C++ loader (`NkLoader`) and `./netkit run` / `./netkit inspect`.

## Overview

| Section | Contents |
|---------|----------|
| Header | Magic, version, network kind, input shape, layer/tensor counts, payload sizes |
| Layer descriptors | One record per layer (dense, conv2d, max_pool2d, flatten) |
| Tensor catalog | Shape + dtype for each weight tensor, then each bias tensor |
| Weight payload | All weight tensors concatenated (float32, little-endian) |
| Bias payload | All bias tensors concatenated (float32, little-endian) |

Weights and biases are **split into separate sections** within the `.nk` file (weights payload first, then biases payload).

Optional **embedded regression tests** (flag `kFlagHasTests`) append a `TCAS` section after the bias payload so a single `.nk` file carries model weights and test cases.

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
| 4 | `uint32` | Format version (`2`) |
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
| `conv2d` | 2 | `kernel u32`, `stride u32`, `filters u32`, `activation u8`, pad×3, `alpha f32` |
| `max_pool2d` | 3 | `pool_size u32`, `stride u32` |
| `flatten` | 4 | (none) |

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

Example conv weight: `rank=4`, `dims=[O,Kh,Kw,I]` in **netkit** layout `[out, kernel, kernel, in_channels]`.

## Payload

- **Weights:** concatenated in layer order (conv/dense layers only)
- **Biases:** concatenated in the same layer order
- All values IEEE-754 **float32**, little-endian

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

Runtime inference ignores the test section. `NkRegression::RunModelTests()` loads embedded cases via `NkLoader::ReadTestSuite()`. Hand-check suites are defined in `python/netkit/regression_data.py`; MNIST export scripts embed cases when training.

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
