# ONNX Import

ONNX is a **host-side** format only. The Python packager reads ONNX and writes **`.nk`**; the C++ runtime loads `.nk` only.

There is **no ONNX reader in C++**. Parity between converted models and their source graphs is tested in Python (`python/tests/test_onnx_parity.py`) using ONNX Runtime as the reference.

## Python packager

```bash
pip install -e python
python -m netkit convert models/my_model.onnx -o models/my_model.nk
python -m netkit convert models/my_model.onnx --no-optimize   # skip BN fold / dense merge
python -m netkit aot models/my_model.nk -o build/aot   # optional: embed .nk in C/C++ for firmware
make export-nk    # all bundled regression models

./netkit run models/my_model.nk --input 1,2,3
```

See [python/README.md](../python/README.md) and [NK_FORMAT.md](NK_FORMAT.md).

## Supported ONNX operators

| ONNX op | netkit layer | Notes |
|---------|--------------|-------|
| `Gemm` | `dense` | float32 weights/bias initializers; `transB` supported |
| `MatMul` (+ optional `Add` bias) | `dense` | MLP graphs; weight initializer `[in, out]` |
| `Conv` | `conv2d` or `depthwise_conv2d` | NCHW weights → netkit `[O,Kh,Kw,I]` or depthwise `[C,Kh,Kw]` when `group == in_channels`; other `group > 1` expands to dense `conv2d`; per-side asymmetric `pads` on import for conv and square-kernel depthwise; fuses trailing activations |
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
- **Sequential primitive graphs** — no generic `Add` / skip branches unless **composite fusion** recognizes them (ResNet BasicBlock, MobileNetV4 UIB on `convert`, or packager `nk_fuse` after `--no-fuse` import). ConvNeXt V2 GRN residuals still need `netkit pack` or future ONNX fusion
- **MobileNet ONNX** — timm exports fold BatchNorm into Conv weights; import and packager UIB fusion synthesize identity BN tensors for conv-only stacks. Full backbone may leave stem/head as primitives when subgraphs do not match UIB patterns
- **ONNX padding import** — `pads` may be per-side asymmetric (top ≠ bottom); encoded in `.nk` for `conv2d`, square-kernel `depthwise_conv2d`, and pools. Non-square depthwise (e.g. 5×1) with asymmetric pads is still rejected
- **Grouped `Conv`** — `group > 1` non-depthwise convs expand to dense `conv2d` at import (no native grouped layer)
- **Non-square pool kernels on import** — `MaxPool` / `AvgPool` accept non-square `kernel_shape`; encoded in pool `reserved` metadata

PyTorch/TensorFlow exports often include `MatMul`, `Add`, `Reshape`, or extra `Pad` nodes — re-export or simplify the graph (e.g. `torch.onnx.export` on an `nn.Sequential`), enable composite fusion, or use `netkit pack` for supported timm backbones.

### Composite block fusion

**ONNX `convert` (default):** when the graph has `Add` nodes, `python -m netkit convert` fuses matching **ResNet BasicBlock** and **MobileNetV4 UIB** (residual add) subgraphs into composite layers. Stride-2 UIB blocks without skip fuse at the project conv. Disable with `--no-fuse`.

**Packager fuse (`nk_fuse`):** after import, `optimize_nk(..., fuse_composite=True)` (default during `convert`) can rebuild composite blocks from primitive layers — ResNet BasicBlock, ConvNeXt V2 block, and MobileNet UIB (including timm conv-only exports via identity BN). This enables **ONNX → primitive `.nk` → optimize → fuse** when `--no-fuse` skips ONNX-side fusion. Packager fuse runs **before** optimize verification so branched ResNet primitive stacks (main + shortcut convs) do not need to forward sequentially; conv input channels are taken from ONNX weight shapes / `.nk` weight catalogs when they differ from the running tensor depth (stride-2 shortcut convs).

```bash
python -m netkit convert model.onnx --no-fuse   # import ResNet blocks as conv+bn primitives
# convert still runs packager fuse during optimize (default)
python -m netkit convert model.onnx --no-fuse --no-optimize   # fully primitive output
```

Fixtures: `models/import_resnet_basic_block.{onnx,nk}` (timm export → primitive import → packager fuse), `models/import_mobilenet_uib.{onnx,nk}` (stride-2 UIB without skip), `models/import_mobilenet_uib_skip.{onnx,nk}` (UIB with residual Add), and `models/import_convnextv2_block.{onnx,nk}` (timm ConvNeXt block → ONNX-side `convnextv2_block` fusion).

**Composite fusion diagnostics:** pass `--verbose-fuse` to `python -m netkit convert` to log ONNX Add-node and packager layer fusion attempts that do not match a composite pattern.

**Full-backbone ONNX round-trip** (timm `forward_features` → convert → ORT vs `nk_infer`) ships in CI via `models/import_*_backbone.{onnx,nk}` — ResNet-18, MobileNetV4-Conv-Small, and ConvNeXt V2-Atto (6 cases in `PARITY_PAIRS`, tolerance `1e-3`). Regenerate with `make export-import-parity` (`tools/write_import_parity_models.py` → `build_import_*_backbone()`).

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
| C++ `make test-cpp` / `make test-c` | **`.nk` loader + inference** against embedded `TCAS` cases in each model (86 cases) |
| Python `make test-python-full` | **`.nk` runtime vs ONNX Runtime** on embedded inputs (82 cases); **AOT compile** tests (C/C++ from `.nk`, requires `make lib`) |

```bash
make                          # build netkit CLI
pip install -e python   # onnx + onnxruntime
make test                     # default: C++ embedded + fast Python
make test-full                # full ONNX parity (manual)
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
