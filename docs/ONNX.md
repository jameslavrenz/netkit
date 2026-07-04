# ONNX Import

ONNX is a **host-side** format only. The Python packager reads ONNX and writes **`.nk`**; the C++ runtime loads `.nk` only.

There is **no ONNX reader in C++**. Parity between converted models and their source graphs is tested in Python (`python/tests/test_onnx_parity.py`) using ONNX Runtime as the reference.

## Python packager

```bash
pip install -e python
python -m netkit convert models/my_model.onnx -o models/my_model.nk
python -m netkit aot models/my_model.nk -o build/aot   # optional: embed .nk in C/C++ for firmware
make export-nk    # all bundled regression models

./netkit run models/my_model.nk --input 1,2,3
```

See [python/README.md](../python/README.md) and [NK_FORMAT.md](NK_FORMAT.md).

## Supported ONNX operators

| ONNX op | netkit layer | Notes |
|---------|--------------|-------|
| `Gemm` | `dense` | float32 weights/bias initializers; `transB` supported |
| `Conv` | `conv2d` or `depthwise_conv2d` | NCHW weights → netkit `[O,Kh,Kw,I]` or depthwise `[C,Kh,Kw]` when `group == in_channels`; per-side symmetric `pads` on import (see [Limitations](#limitations-v1)); fuses trailing activations |
| `MaxPool` | `max_pool2d` | square kernel from `kernel_shape`; symmetric `pads` |
| `AveragePool` / `AvgPool` | `avg_pool2d` | square kernel from `kernel_shape`; symmetric `pads` |
| `GlobalAveragePool` | `avg_pool2d` | emitted as `pool_size = H`, `stride = 1` (square spatial dims only) |
| `BatchNormalization` | `batch_norm2d` | folded to per-channel `scale` + `bias` tensors |
| `Flatten` | `flatten` | CNN head — test ONNX sidecars transpose NCHW→NHWC before flatten to match runtime |
| `Relu` | activation | fused when immediately after Gemm/Conv |
| `Sigmoid` / `Tanh` / `LeakyRelu` | activation | fused when immediately after Gemm/Conv |
| `Clip` | activation | fused as ReLU6 when min=0, max=6 |
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
- **Sequential primitive graphs** — no generic `Add` / skip branches unless **composite fusion** recognizes them (ResNet BasicBlock on `convert`; see below). Full ResNet / MobileNet / ConvNeXt backbones are best packed from timm via `python -m netkit pack`
- **ONNX padding import** — `pads` must be per-side symmetric (top = bottom, left = right). The runtime supports independent `pad_h` and `pad_w` (including non-square depthwise kernels); ONNX import rejects per-side asymmetric pads (e.g. top ≠ bottom)
- **Square pool kernels** — `MaxPool` / `AvgPool` use one `kernel_shape` value for height and width on import (runtime `.nk` layers can specify non-square depthwise kernels)

PyTorch/TensorFlow exports often include `MatMul`, `Add`, `Reshape`, or extra `Pad` nodes — re-export or simplify the graph (e.g. `torch.onnx.export` on an `nn.Sequential`), enable composite fusion, or use `netkit pack` for supported timm backbones.

### Composite block fusion

**ONNX `convert` (default):** when the graph has `Add` nodes, `python -m netkit convert` fuses matching **ResNet BasicBlock** subgraphs into layer kind `resnet_basic_block` (residual add + two conv branches). Disable with `--no-fuse`.

```bash
python -m netkit convert model.onnx --no-fuse   # primitive layers only
```

**Timm `pack` (no ONNX round-trip):** ResNet-18, MobileNetV4-Conv-Small, and ConvNeXt V2-Atto emit fused composite layers (`resnet_basic_block`, `mobilenetv4_uib`, `convnextv2_block`) directly from PyTorch checkpoints:

```bash
python -m netkit pack --arch resnet18 -o models/my_resnet18.nk --height 56 --width 56 --num-classes 10
python -m netkit pack --arch convnextv2_atto -o models/my_convnextv2_atto.nk --height 32 --width 32 --num-classes 10
python -m netkit pack --arch mobilenetv4_small -o models/my_mobilenetv4_small.nk --height 56 --width 56 --num-classes 10
```

Requires `pip install -e "python[train]"` (torch + timm).

## Testing

| Suite | What it validates |
|-------|-------------------|
| C++ `make test-cpp` / `make test-c` | **`.nk` loader + inference** against embedded `TCAS` cases in each model (59 cases) |
| Python `make test-python` | **`.nk` runtime vs ONNX Runtime** on embedded inputs (49 cases); **AOT compile** tests (C/C++ from `.nk`, requires `make lib`) |

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
