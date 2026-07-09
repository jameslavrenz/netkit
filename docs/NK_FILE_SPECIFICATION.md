# netkit `.nk` File Specification

**Format version:** 3  
**Magic:** `NKIT` (ASCII)  
**Byte order:** little-endian for all multi-byte integers and IEEE-754 float32 payloads  
**Reference implementations:** `include/nk_format.hpp`, `src/nk_loader.cpp`, `python/netkit/format.py`

A `.nk` file is a single binary bundle containing everything needed for on-device inference: network architecture, weight/bias tensor metadata, float32 coefficients, and (optionally) embedded regression test cases. Runtime inference ignores the optional test section.

Related docs: [NK_FORMAT.md](NK_FORMAT.md) (overview), [CLI.md](CLI.md) (inspect/run commands), [ONNX.md](ONNX.md) (conversion).

---

## 1. File layout (top level)

```
Offset (conceptual)   Section                          Size
─────────────────────────────────────────────────────────────────
0                     File header                      48 bytes (fixed)
48                    Layer descriptors                variable (see §3)
48 + L                Weight tensor catalog            24 × Nw bytes
48 + L + 24·Nw        Bias tensor catalog              24 × Nb bytes
payload_offset        Alignment padding (optional)     0–3 zero bytes
payload_offset        Weight payload                   weights_bytes
+ weights_bytes       Bias payload                     biases_bytes
+ biases_bytes        Test section (optional, TCAS)    variable (see §7)
```

Where:

- `L` = sum of byte sizes of all layer descriptors (§3).
- `Nw` = `num_weight_tensors` from the header.
- `Nb` = `num_bias_tensors` from the header.
- `payload_offset` = smallest offset ≥ end of metadata such that `payload_offset % 4 == 0` (§6).

**Total file size (minimum, no tests):**

```
file_size ≥ payload_offset + weights_bytes + biases_bytes
```

When header flag `0x0001` is set, a `TCAS` section may follow immediately after the bias payload.

---

## 2. File header (48 bytes)

All offsets are from the start of the file.

| Offset | Size | Type | Field | Description |
|-------:|-----:|------|-------|-------------|
| 0 | 4 | `char[4]` | `magic` | Must be `N`, `K`, `I`, `T` (`0x4E 0x4B 0x49 0x54`) |
| 4 | 4 | `uint32` | `version` | Format version; must be **3** |
| 8 | 1 | `uint8` | `network_kind` | `1` = MLP, `2` = CNN |
| 9 | 1 | `uint8` | `input_rank` | 1–4 |
| 10 | 2 | `uint16` | `flags` | Bit `0x0001` = embedded test section (`TCAS`) follows payload |
| 12 | 16 | `uint32[4]` | `input_shape` | Shape dimensions; unused entries are `0` |
| 28 | 4 | `uint32` | `num_layers` | Layer descriptor count (≤ 100) |
| 32 | 4 | `uint32` | `num_weight_tensors` | Weight catalog entries (≤ 128) |
| 36 | 4 | `uint32` | `num_bias_tensors` | Bias catalog entries (≤ 128) |
| 40 | 4 | `uint32` | `weights_bytes` | Weight payload size in bytes |
| 44 | 4 | `uint32` | `biases_bytes` | Bias payload size in bytes |

**C struct packing equivalent:** `"<4sIBBH4IIIIII"` (Python `struct` notation).

### Input shape conventions

| `network_kind` | Typical `input_rank` | `input_shape` meaning |
|----------------|---------------------:|------------------------|
| MLP (`1`) | 2 | `[batch, features]` |
| CNN (`2`) | 3 | `[height, width, channels]` (NHWC) |

Example MLP with batch 1 and 2 features: `input_rank=2`, `input_shape=[1, 2, 0, 0]`.

Example CNN 28×28×1: `input_rank=3`, `input_shape=[28, 28, 1, 0]`.

---

## 3. Layer descriptors

Layer records are stored **in forward execution order**, starting at file offset **48**.

Every layer begins with a **4-byte prefix**:

| Offset (within layer) | Size | Field |
|----------------------:|-----:|-------|
| 0 | 1 | `kind` (`uint8`, see table below) |
| 1 | 3 | `reserved` (must be `0`) |

Kind-specific fields follow immediately after the prefix.

### 3.1 Layer kind values and record sizes

| Kind name | `kind` | Total record size (prefix + body) |
|-----------|-------:|----------------------------------:|
| `dense` | 1 | **16 bytes** |
| `conv2d` | 2 | **24 bytes** |
| `max_pool2d` | 3 | **16 bytes** |
| `flatten` | 4 | **4 bytes** (prefix only) |
| `avg_pool2d` | 5 | **16 bytes** |
| `batch_norm2d` | 6 | **12 bytes** |
| `depthwise_conv2d` | 7 | **24 bytes** |
| `convnextv2_block` | 8 | **16 bytes** |
| `mobilenetv4_uib` | 9 | **24 bytes** |
| `resnet_basic_block` | 10 | **16 bytes** |
| `layernorm2d` | 11 | **16 bytes** |
| `yolox_decoupled_head` | 12 | **20 bytes** |

**Metadata size after header:**

```
L = Σ layer_record_size[i]   for i in 0 .. num_layers-1
```

### 3.2 Kind-specific field layouts

Fields below are **after** the 4-byte prefix. All integers are little-endian.

#### `dense` (kind 1) — 12 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `units` |
| 4 | 1 | `uint8` | `activation` (§5) |
| 5 | 3 | — | padding (zero) |
| 8 | 4 | `float32` | `alpha` (leaky_relu scale; ignored for other activations) |

#### `conv2d` (kind 2) — 20 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `kernel_size` (square kernel K×K) |
| 4 | 4 | `uint32` | `stride` |
| 8 | 4 | `uint32` | `filters` (output channels) |
| 12 | 1 | `uint8` | `activation` |
| 13 | 1 | `uint8` | `pad_h` (top padding) |
| 14 | 1 | `uint8` | `pad_w` (left padding) |
| 15 | 1 | `uint8` | `pad_extra` (asymmetric end padding; §4) |
| 16 | 4 | `float32` | `alpha` |

#### `depthwise_conv2d` (kind 7) — 20 body bytes

Same layout as conv2d, but field meanings differ:

| Field | Meaning |
|-------|---------|
| `kernel_size` | `kernel_h` |
| `filters` | input/output channel count |
| `kernel_w` (byte at offset 15) | **kernel width** when ≠ `kernel_h`; may also encode extra padding (see §4) |

#### `max_pool2d` (kind 3) / `avg_pool2d` (kind 5) — 12 body bytes each

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `pool_size` (`pool_h`; square unless `reserved` encodes `pool_w`) |
| 4 | 4 | `uint32` | `stride` |
| 8 | 1 | `uint8` | `pad_h` (top) |
| 9 | 1 | `uint8` | `pad_w` (left) |
| 10 | 2 | `uint16` | `reserved` (non-square pool + asymmetric pad; §4) |

#### `flatten` (kind 4)

No body fields.

#### `batch_norm2d` (kind 6) — 8 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `channels` |
| 4 | 4 | `uint32` | `reserved` (zero) |

#### `convnextv2_block` (kind 8) — 12 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `channels` |
| 4 | 4 | `uint32` | `reserved` (zero) |
| 8 | 4 | `float32` | `eps` |

#### `mobilenetv4_uib` (kind 9) — 20 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `in_channels` |
| 4 | 4 | `uint32` | `out_channels` |
| 8 | 1 | `uint8` | `start_dw_kernel` |
| 9 | 1 | `uint8` | `middle_dw_kernel` |
| 10 | 1 | `uint8` | `stride` |
| 11 | 1 | `uint8` | `middle_dw_downsample` |
| 12 | 4 | `float32` | `expand_ratio` |
| 16 | 4 | `uint32` | `reserved` (zero) |

#### `resnet_basic_block` (kind 10) — 12 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `in_channels` |
| 4 | 4 | `uint32` | `out_channels` |
| 8 | 1 | `uint8` | `stride` |
| 9 | 3 | — | padding (zero) |

#### `layernorm2d` (kind 11) — 12 body bytes

Same layout as `convnextv2_block` (`channels`, `reserved`, `eps`).

#### `yolox_decoupled_head` (kind 12) — 16 body bytes

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 4 | `uint32` | `in_channels` |
| 4 | 4 | `uint32` | `hidden_dim` |
| 8 | 4 | `uint32` | `num_classes` |
| 12 | 1 | `uint8` | `num_convs` |
| 13 | 3 | — | padding (zero) |

Fused blocks (`convnextv2_block`, `mobilenetv4_uib`, `resnet_basic_block`, `yolox_decoupled_head`) expand to multiple weight/bias tensors in catalog order — see feature guides ([CONVNEXTV2.md](CONVNEXTV2.md), [MOBILENETV4.md](MOBILENETV4.md), [RESNET18.md](RESNET18.md), [YOLOX.md](YOLOX.md)).

---

## 4. Asymmetric padding and non-square pools

ONNX `pads` are four values `(top, left, bottom, right)`. The `.nk` format stores **top/left** in `pad_h` / `pad_w` and packs extra bottom/right into auxiliary bytes so symmetric models stay compact.

### Conv2D `pad_extra` byte (conv2d only — not depthwise kernel width)

| Nibble | Meaning |
|--------|---------|
| low 4 bits | `pad_h_end - pad_h` (extra bottom, 0–15) |
| high 4 bits | `pad_w_end - pad_w` (extra right, 0–15) |

Decode:

```
pad_h_end = pad_h + (pad_extra & 0x0F)
pad_w_end = pad_w + ((pad_extra >> 4) & 0x0F)
```

Output spatial size (per axis): `(size + pad_before + pad_after - kernel) // stride + 1`.

### Pool `reserved u16` (max/avg pool)

| Bits | Meaning |
|------|---------|
| 0–7 | `pool_w` when different from `pool_size` (`pool_h`); `0` means square |
| 8–11 | extra bottom padding (`pad_h_end - pad_h`) |
| 12–15 | extra right padding (`pad_w_end - pad_w`) |

Encode/decode: `python/netkit/pad_encoding.py`; C++: `include/nk_op_detail.hpp`.

---

## 5. Activation enum (`uint8`)

| Value | Name |
|------:|------|
| 0 | `none` |
| 1 | `relu` |
| 2 | `sigmoid` |
| 3 | `tanh` |
| 4 | `leaky_relu` |
| 5 | `relu6` |
| 6 | `softmax` |

Used in `dense`, `conv2d`, and `depthwise_conv2d` layer records.

---

## 6. Tensor catalog (24 bytes per entry)

After all layer descriptors, the file contains **`num_weight_tensors`** weight descriptors, then **`num_bias_tensors`** bias descriptors. Order matches concatenation order in the payload.

| Offset | Size | Type | Field |
|-------:|-----:|------|-------|
| 0 | 1 | `uint8` | `rank` (1–4) |
| 1 | 1 | `uint8` | `dtype` (`1` = float32) |
| 2 | 2 | `uint16` | padding (zero) |
| 4 | 16 | `uint32[4]` | `dims` (unused dims `0`) |
| 20 | 4 | `uint32` | `num_elements` (product of active dims) |

**Struct equivalent:** `"<BBH4II"`.

### Weight layout conventions

| Layer type | Typical weight shape (dims) |
|------------|----------------------------|
| Dense | `[out_features, in_features]` rank 2 |
| Conv2D | `[out, Kh, Kw, in]` rank 4 |
| Depthwise | `[C, Kh, Kw]` rank 3 |

Bias tensors are usually rank 1: `[channels_or_units]`.

**Catalog byte size:**

```
catalog_bytes = 24 × (num_weight_tensors + num_bias_tensors)
```

### Payload alignment padding

If `(48 + L + catalog_bytes) % 4 != 0`, the writer inserts **0–3 zero bytes** so the weight payload starts on a 4-byte boundary. The loader records the aligned start as `payload_offset`.

```
payload_offset = round_up(48 + L + catalog_bytes, 4)
```

Weight and bias payloads must be readable as `float32` arrays on little-endian hosts. Buffer/AOT load binds flash views and requires `payload_offset % 4 == 0`.

---

## 7. Weight and bias payload

| Region | Start offset | Size |
|--------|-------------|------|
| Weights | `payload_offset` | `weights_bytes` |
| Biases | `payload_offset + weights_bytes` | `biases_bytes` |

- All values are **IEEE-754 binary32**, little-endian.
- Weights for all layers are concatenated first; biases follow in the same logical layer order.
- `weights_bytes / 4` should equal the sum of `num_elements` in the weight catalog; same for biases.

Layers without learnable parameters (pool, flatten) contribute no tensors.

---

## 8. Embedded test section (`TCAS`, optional)

Present when header `flags & 0x0001`. Starts immediately after the bias payload.

### 8.1 Section header

| Field | Size | Description |
|-------|-----|-------------|
| Magic | 4 | `"TCAS"` |
| Case count | 4 | `uint32` (≤ 16) |
| Tolerance | 4 | `float32` absolute per-output tolerance |

### 8.2 Each test case

| Field | Size | Description |
|-------|-----|-------------|
| Name length | 1 | `uint8` (1–127) |
| Name | `len` | UTF-8 bytes |
| Name padding | 0–3 | zero bytes so `(1 + len + pad) % 4 == 0` |
| Label | 4 | `int32` expected class index (`-1` = none) |
| Input count | 4 | `uint32` |
| Input data | `4 × input_count` | float32 values |
| Output count | 4 | `uint32` |
| Expected output | `4 × output_count` | float32 reference values |

Runtime forward pass **does not read** this section. Desktop regression uses `NkRegression::RunModelTests()` / `nk_run_model_tests()`.

---

## 9. Format limits (version 3)

| Constant | Value |
|----------|------:|
| `kMaxLayers` / `NK_MAX_LAYERS` | 100 |
| `kMaxTensorCatalog` | 128 |
| `kMaxTestCases` | 16 |
| `kMaxCaseFloats` | 16384 |
| `kMaxCaseNameLen` | 127 |
| Supported `dtype` | float32 only (`1`) |

---

## 10. Worked offset example

For a minimal MLP (`test_mlp.nk` class):

- Header: 48 bytes
- 2 layers: `dense` (16) + `dense` (16) → L = 32
- 2 weight + 2 bias catalog entries → catalog = 96 bytes
- Meta end = 48 + 32 + 96 = 176 (already 4-byte aligned)
- `payload_offset = 176`
- Weights at 176, biases at `176 + weights_bytes`

Verify with:

```bash
xxd -g 1 -l 48 models/test_mlp.nk          # header
xxd -s 176 -l 32 models/test_mlp.nk        # first floats in weight payload
```

---

## 11. Inspecting `.nk` files

### 11.1 Human-readable summary (recommended)

Build the desktop CLI (`make`, `NETKIT_TARGET=cpu`), then:

```bash
./netkit inspect models/test_mlp.nk
```

Prints a boxed **Network Summary**: model name, network type (MLP/CNN), format version, input shape, numbered layer list, input/output element counts, and total weight float count.

Example output:

```
=====================================================
Network Summary
=====================================================

Name        : test_mlp
Type        : MLP
Version     : 3

Input Shape : [1, 2]

Layers (2)
-----------------------------------------------------
  [0] Dense units=2 activation=relu
  [1] Dense units=2 activation=none
-----------------------------------------------------
Input elements : 2
Output elements: 2
Weight floats  : 12
=====================================================
```

**Arena sizing probe** (load model, run zero-input forward, print memory high-water):

```bash
./netkit inspect models/mnist_cnn.nk --full
```

Adds lines such as:

```
Arena (4194304 bytes capacity):
  after load:           … bytes
  after forward (zero): … bytes
  remaining:            … bytes
```

Use this before choosing firmware arena size. C API equivalent: `nk_arch_print()` (summary) and `nk_inspect_model()` (`--full`).

**Path resolution:** if `models/foo.nk` is not found in the current directory, the CLI tries `../models/foo.nk`.

### 11.2 Python inspector (header + tensor catalog)

Detailed dump including per-tensor shapes (no forward pass):

```bash
pip install -e python
python -m netkit inspect models/test_mlp.nk
```

Example output:

```
netkit binary model (.nk)
  file:            models/test_mlp.nk
  format version:  3
  network:         mlp
  input rank:      2
  input shape:     [1, 2]
  layers:          2
  weight tensors:  2 (32 bytes)
  bias tensors:    2 (16 bytes)

Layer stack:
  [0] {'kind': 'dense', 'units': 2, ...}
  ...

Weight tensor catalog:
  weight[0]: {'rank': 2, 'dtype': 'float32', 'shape': [2, 2], ...}
  ...
```

### 11.3 Raw hex dump (`xxd`, `hexdump`, `od`)

Inspect magic, version, and header fields:

```bash
# macOS / Linux — first 48 bytes (header)
xxd -g 1 -l 48 models/test_mlp.nk

# Show ASCII + hex (magic "NKIT" at offset 0)
xxd -g 4 models/test_mlp.nk | head -20

# GNU hexdump alternative
hexdump -C -n 128 models/test_mlp.nk

# POSIX od
od -A x -t x1z -N 48 models/test_mlp.nk
```

Expected header start:

```
00000000: 4e 4b 49 54 03 00 00 00  ...  NKIT....  (magic + version 3)
```

Jump to payload (replace `176` with your computed `payload_offset`):

```bash
xxd -s 176 -g 4 -l 64 models/test_mlp.nk
```

Search for embedded tests:

```bash
grep -abo TCAS models/test_mlp.nk    # print byte offset of TCAS magic
xxd -s <offset> models/test_mlp.nk
```

### 11.4 Quick inference (sanity check)

```bash
./netkit run models/test_mlp.nk --input 1,2
```

Loads the same file, validates input size, runs forward, and prints labeled input/output tensors.

---

## 12. Producing `.nk` files

```bash
pip install -e python
python -m netkit convert model.onnx -o model.nk
make export-nk    # regenerate bundled regression models
```

Embed in firmware:

```bash
python -m netkit aot models/test_mlp.nk -o build/aot/
```

See [GETTING_STARTED.md](GETTING_STARTED.md) and [python/README.md](../python/README.md).

---

## 13. Version history

| Version | Notes |
|--------:|-------|
| 3 | Current: depthwise conv, fused blocks (ConvNeXt, UIB, ResNet, YOLOX head), asymmetric padding, optional `TCAS` tests, up to 100 layers |

Older versions are rejected by the loader (`UnsupportedVersion`).

---

## 14. Implementation index

| Component | Location |
|-----------|----------|
| C++ constants / structs | `include/nk_format.hpp` |
| C++ loader / parser | `include/nk_loader.hpp`, `src/nk_loader.cpp` |
| CLI inspect / run | `src/cli.cpp` |
| Python writer / reader | `python/netkit/format.py`, `python/netkit/writer.py`, `python/netkit/inspect.py` |
| Padding encode/decode | `python/netkit/pad_encoding.py`, `include/nk_op_detail.hpp` |
