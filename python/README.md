# netkit-tools (Python)

Python tooling for **netkit** — a multi-modal (voice, image, vision) inference engine with an embedded-first design for MCU, MPU, and NPU targets. Converts ONNX models into binary **`.nk`** files for the C++26 / C23 runtime, and optionally AOT-embeds `.nk` bytes into C/C++ source for firmware.

**Role in netkit:** Phase 1 serializer (ONNX → `.nk`) for the **`NkOpsResolver` interpreter** path, and AOT embedder (`.nk` → C++26 / C23) for the **compiled** path. Use `convert --optimize` / `aot --optimize` to fold and fuse the graph before embedding — fewer runtime layer dispatches. Phase 2 adds more packager-side compilation (layout, broader quantization); Kalman estimation and tracking are planned in the backend — see [docs/PHILOSOPHY.md](../docs/PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

Supported ONNX ops: `Gemm`, `Conv` (symmetric padding), `MaxPool` / `AveragePool` (symmetric padding), `GlobalAveragePool`, `BatchNormalization`, `Flatten`, and fused activations (`Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Clip`→ReLU6, `Softmax`). Details: [docs/ONNX.md](../docs/ONNX.md).

## Install

```bash
pip install -e python
```

Requires **numpy** and **onnx**. Training/export scripts additionally need PyTorch and timm (backbone packing):

```bash
pip install -e "python[train]"
```

The `[train]` extra installs `torch`, `timm`, and `onnxscript` (used by `python -m netkit pack`, MNIST export scripts, and ONNX fuse tests).

## Usage

```bash
# ONNX -> .nk
python -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk

# Inspect header + tensor catalog
python -m netkit inspect models/test_mlp.nk

# AOT embed .nk in firmware source (default: C++26)
python -m netkit aot models/test_mlp.nk -o build/aot
python -m netkit aot models/test_mlp.nk -o build/aot --no-lower          # embed .nk + loader
python -m netkit aot models/test_mlp.nk -o build/aot --language c
python -m netkit aot models/test_mlp.nk -o build/aot --main   # optional smoke main
python -m netkit aot models/mlp_hand.nk -o build/aot --target mcu --arena-headroom 15
# Weights always flash/blob-backed (coefs in .rodata)
python -m netkit aot models/cnn_extended_ops.nk -o build/aot --optimize   # fewer runtime ops

# Convert all bundled regression models (from repo root)
make export-nk

# Pack timm backbone checkpoints to .nk (ResNet-18, ConvNeXt V2-Atto, MobileNetV4 Small)
python -m netkit pack --arch resnet18 -o models/my_resnet18.nk --height 56 --width 56 --num-classes 10
python -m netkit pack --arch convnextv2_atto -o models/my_convnextv2_atto.nk --height 32 --width 32 --num-classes 10
python -m netkit pack --arch mobilenetv4_small -o models/my_mobilenetv4_small.nk --height 56 --width 56 --num-classes 10
```

Typical pipeline:

```text
model.onnx  →  python -m netkit convert  →  model.nk  →  python -m netkit aot  →  model_aot.{hpp,cpp}
```

Link AOT output with `libnetkit.a` — see [docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md#5-aot-compile-embed-nk-in-firmware).

## Python API

```python
from netkit import compile_aot, convert_onnx_to_nk, AotLanguage, optimize_nk

convert_onnx_to_nk("models/test_mlp.onnx", "models/test_mlp.nk")
result = compile_aot("models/test_mlp.nk", "build/aot", language=AotLanguage.CPP)
result = compile_aot("models/mlp_hand.nk", "build/aot", arena_headroom_percent=15)
result = compile_aot("models/cnn_extended_ops.nk", "build/aot", optimize=True)
# result.arena_bytes_recommended — static arena size for firmware
```

## Testing

```bash
pip install -e python   # onnx + onnxruntime for parity tests
make lib                # required for AOT compile tests
make test-python        # fast Python subset (from repo root; same as in make test)
make test-python-full   # ONNX parity (82) + AOT tests
python -m unittest python.tests.test_onnx_convert_ops  # padding, avg pool, batch norm, fusion
python -m unittest python.tests.test_aot_compile
```

See [docs/TESTING.md](../docs/TESTING.md) and [docs/ONNX.md](../docs/ONNX.md).

## C++ runtime

```bash
./netkit inspect models/test_mlp.nk
./netkit run models/test_mlp.nk --input 1,2
```

See [docs/NK_FORMAT.md](../docs/NK_FORMAT.md) for the binary layout. Getting started: [docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md).

### Depthwise conv

`depthwise_conv2d` layers require explicit `kernel_h` and `kernel_w` (weights `[C, Kh, Kw]`). For 1D along time on `[T, 1, C]` NHWC input, use e.g. `kernel_h=5`, `kernel_w=1`, `pad_h=2`, `pad_w=0`.
