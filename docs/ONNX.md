# ONNX Import

ONNX is a **host-side** format only. The Python packager reads ONNX and writes **`.nk`**; the C++ runtime loads `.nk` only.

There is **no ONNX reader in C++**. Parity between converted models and their source graphs is tested in Python (`python/tests/test_onnx_parity.py`) using ONNX Runtime as the reference.

## Python packager

```bash
pip install -e python
python -m netkit convert models/my_model.onnx -o models/my_model.nk
make export-nk    # all bundled regression models

./netkit run models/my_model.nk --input 1,2,3
```

See [python/README.md](../python/README.md) and [NK_FORMAT.md](NK_FORMAT.md).

## Supported ONNX operators

| ONNX op | netkit layer | Notes |
|---------|--------------|-------|
| `Gemm` | `dense` | float32 weights/bias initializers; `transB` supported |
| `Conv` | `conv2d` | NCHW weights → netkit `[O,Kh,Kw,I]`; **valid conv, no padding** |
| `MaxPool` | `max_pool2d` | square kernel from `kernel_shape` |
| `Flatten` | `flatten` | CNN head — test ONNX sidecars transpose NCHW→NHWC before flatten to match runtime |
| `Relu` | activation | fused when immediately after Gemm/Conv |
| `Softmax` | activation | fused when immediately after final Gemm |

## Input layouts

| netkit network | ONNX graph input | `.nk` runtime input shape |
|----------------|------------------|----------------------------|
| MLP | `[batch, features]` | same |
| CNN | `[N, C, H, W]` (NCHW) | `[H, W, C]` NHWC |

At inference time, feed CNN inputs in **NHWC flatten order** (same as existing netkit CNN models). The converter reorders conv weights; it does **not** transpose runtime inputs.

## Limitations (v1)

- **Float32 only** — other ONNX `TensorProto` types are rejected
- **No external data** — weights must be embedded in the `.onnx` file (`raw_data` or `float_data`)
- **Linear graphs** — no branches, skip connections, or subgraphs
- **No `Pad`** — padded convolutions are not supported (matches netkit valid conv)
- **Square kernels** — `Conv` / `MaxPool` use one `kernel_shape` value for height and width

PyTorch/TensorFlow exports often include `MatMul`, `Add`, `BatchNormalization`, or `Reshape` nodes — re-export or simplify the graph (e.g. `torch.onnx.export` on an `nn.Sequential`) or extend the converter.

## Testing

| Suite | What it validates |
|-------|-------------------|
| C++ `make test-cpp` / `make test-c` | **`.nk` loader + inference** against embedded `TCAS` cases in each model (69 cases) |
| Python `make test-python` | **`.nk` runtime vs ONNX Runtime** on embedded inputs (69 cases) |

```bash
make                          # build netkit CLI
pip install -e python   # onnx + onnxruntime
make test                     # C++ embedded + Python ONNX parity
```

Regenerate bundled ONNX sidecars from committed `.nk` files:

```bash
pip install onnx numpy
python3 tools/export_onnx_test_models.py
```

## Related docs

- [NK_FORMAT.md](NK_FORMAT.md) — binary model layout and embedded tests
- [TESTING.md](TESTING.md) — full test matrix
- [CLI.md](CLI.md) — full CLI reference
