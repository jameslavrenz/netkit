# Model File Format

netkit models are **file bundles**: a JSON architecture descriptor and a companion binary weight file. All numeric values are **float32** (IEEE-754 single precision, little-endian on disk). Quantized types (float16, int16, int8, int4) are planned — see [DATATYPES.md](DATATYPES.md).

## Bundle layout

Given a base name `my_model`:

| File | Required | Purpose |
|------|----------|---------|
| `my_model.json` | Yes | Architecture (layers, activations, input shape) |
| `my_model.bin` | Yes | Raw float32 weights in layer order |
| `my_model.vectors.json` | No | Regression test cases for `./netkit test` |

The `.bin` path is derived from the `.json` path by replacing the extension (see `ModelLoader::JsonPathToBinPath` / `nk_json_path_to_bin_path`).

## JSON schema (version 1)

Top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Must be `1` |
| `network` | string | `"mlp"` or `"cnn"` |
| `input` | array of integers | Input tensor shape |
| `layers` | array of objects | One entry per layer, in order |

### MLP (`network: "mlp"`)

**Input shape:** 2D `[batch, features]`. Batch is typically `1` for inference.

**Layer object (`type: "dense"`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"dense"` |
| `units` | integer | Yes | Output feature count |
| `activation` | string | No | Default `"none"` |
| `alpha` | number | No | Leaky ReLU slope; default `0.01` |

Example:

```json
{
  "version": 1,
  "network": "mlp",
  "input": [1, 2],
  "layers": [
    { "type": "dense", "units": 2, "activation": "relu" },
    { "type": "dense", "units": 2, "activation": "none" }
  ]
}
```

**Forward pass:** For layer *i*, compute `output = activation(input @ W + b)` where `W` has shape `[in_features, out_features]` row-major and `in_features` comes from the previous layer (or the input shape for layer 0).

### CNN (`network: "cnn"`)

**Input shape:** 3D NHWC `[height, width, channels]`.

CNN pipelines may mix spatial layers and a dense classification head. Supported layer types:

#### `conv2d`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `"conv2d"` |
| `kernel_size` | integer | Yes | Square kernel (k×k) |
| `stride` | integer | Yes | Stride (valid convolution only; no padding) |
| `filters` | integer | Yes | Output channel count |
| `activation` | string | No | Default `"none"` |
| `alpha` | number | No | Leaky ReLU slope; default `0.01` |

#### `max_pool2d`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `"max_pool2d"` |
| `pool_size` | integer | No | Window size (default `2`) |
| `stride` | integer | No | Stride (default `pool_size`) |

Output size: `(input - pool_size) / stride + 1` per spatial dimension. Channel count unchanged.

#### `flatten`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `"flatten"` |

Converts NHWC tensor to `[1, H×W×C]` for dense layers. No weights.

#### `dense` (CNN head)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `"dense"` |
| `units` | integer | Yes | Output feature count |
| `activation` | string | No | Default `"none"` |
| `alpha` | number | No | Leaky ReLU slope; default `0.01` |

Example (pure conv stack):

```json
{
  "version": 1,
  "network": "cnn",
  "input": [3, 3, 2],
  "layers": [
    { "type": "conv2d", "kernel_size": 2, "stride": 1, "filters": 2, "activation": "relu" },
    { "type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 1, "activation": "none" }
  ]
}
```

Example (MNIST tutorial-style head — see [MNIST_CNN.md](MNIST_CNN.md)):

```json
{
  "version": 1,
  "network": "cnn",
  "input": [28, 28, 1],
  "layers": [
    { "type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 32, "activation": "relu" },
    { "type": "max_pool2d", "pool_size": 2, "stride": 2 },
    { "type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 64, "activation": "relu" },
    { "type": "max_pool2d", "pool_size": 2, "stride": 2 },
    { "type": "flatten" },
    { "type": "dense", "units": 128, "activation": "relu" },
    { "type": "dense", "units": 10, "activation": "softmax" }
  ]
}
```

**Spatial output size** for conv/pool: `(input - kernel_size) / stride + 1` (integer division). Convolutions are **valid-only** — no padding.

**Tensor layout:** NHWC for spatial tensors; dense head uses `[batch, features]`. Flattened CLI/C API input order is row-major over `[H, W, C]`.

### Activations

Supported `activation` strings (MLP and CNN):

| Value | Description |
|-------|-------------|
| `none` | Linear (identity) |
| `relu` | max(0, x) |
| `sigmoid` | σ(x) |
| `tanh` | tanh(x) |
| `leaky_relu` | x if x > 0 else α·x (use optional `alpha`) |
| `relu6` | min(max(0, x), 6) |
| `softmax` | Softmax over all elements in the tensor |

## Weight file (`.bin`)

- Format: contiguous **float32**, little-endian
- No header or metadata — size must match the architecture exactly
- Loaded with `ModelLoader::LoadWeightsBin` / `nk_load_weights_bin`

### MLP weight order

For each dense layer, in sequence:

1. **Weights** `W[in × out]` row-major (each row is one input feature's connections to all outputs)
2. **Bias** `b[out]`

Where `in` is the previous layer's output size (or input features for layer 0).

### CNN weight order

For each conv2d layer, in sequence:

1. **Weights** `W[out_channels × kernel × kernel × in_channels]` — see `Conv2D` in `conv2d.hpp`
2. **Bias** `b[out_channels]`

For each dense layer in a CNN pipeline (after flatten), same as MLP: `W[in×out]` row-major, then `b[out]`.

Only `conv2d` and `dense` layers consume weights; `max_pool2d` and `flatten` have none.

### Generating weights

Hand-crafted models in this repo use [`tools/write_hand_models.py`](../tools/write_hand_models.py):

```bash
python3 tools/write_hand_models.py
```

This writes `models/mlp_hand.bin` and `models/cnn_hand.bin`. Use `./netkit inspect <model.json> --full` to verify `expected weight floats` matches the `.bin` file size ÷ 4.

## Path resolution

Loaders try the path as given, then `../<path>` relative to the current working directory. This allows running the CLI or examples from subdirectories during development.

## Arena memory (not in JSON)

Model JSON does **not** include an arena size or memory budget. Loaders derive memory needs from:

- Weight count (layers × dimensions → `.bin` size)
- Input shape in JSON (batch / spatial dims)
- Two **ping-pong activation buffers** sized to the largest intermediate tensor (allocated at load)

Your application chooses a byte buffer and passes it to `Arena::init(memory, size)` or `nk_arena_init()`. Use `./netkit inspect <model.json> --full` or `nk_inspect_model()` to measure high-water, then allocate static storage with margin. Full guide: [ARENA.md](ARENA.md).

| Typical model | Suggested starting buffer |
|---------------|---------------------------|
| Hand test MLP/CNN | 64 KiB (`Arena::kDefaultCapacity`) |
| MNIST MLP | ~2 MiB (see [MNIST.md](MNIST.md)) |
| MNIST CNN | ~4 MiB (see [MNIST_CNN.md](MNIST_CNN.md)) |

## Inspecting a model

```bash
./netkit inspect models/test_mlp.json
./netkit inspect models/test_mlp.json --full
```

Default mode prints a boxed network summary. `--full` adds weight file summary and arena high-water marks after load and a zero-input forward pass — use that to size embedded memory buffers. The CLI uses a 64 KiB arena; MNIST models may overflow `--full` on the CLI even though they run fine in the test suite with larger buffers.

See also [VECTORS_TESTS.md](VECTORS_TESTS.md) for adding regression cases.
