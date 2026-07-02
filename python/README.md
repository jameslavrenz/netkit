# netkit-tools (Python)

Convert ONNX models into binary **`.nk`** files for the C++ runtime, and optionally AOT-embed `.nk` bytes into C/C++ source for firmware.

**Role in netkit:** Phase 1 serializer (ONNX → `.nk`) and AOT embedder (`.nk` → C++26 / C23). Phase 2 adds packager optimizations (fusion at export, layout, quantization) — see [docs/PHILOSOPHY.md](../docs/PHILOSOPHY.md).

Supported ONNX ops: `Gemm`, `Conv` (symmetric padding), `MaxPool` / `AveragePool` (symmetric padding), `GlobalAveragePool`, `BatchNormalization`, `Flatten`, and fused activations (`Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Clip`→ReLU6, `Softmax`). Details: [docs/ONNX.md](../docs/ONNX.md).

## Install

```bash
pip install -e python
```

Requires **numpy** and **onnx**. Training/export scripts additionally need PyTorch:

```bash
pip install -e "python[train]"
```

## Usage

```bash
# ONNX -> .nk
python -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk

# Inspect header + tensor catalog
python -m netkit inspect models/test_mlp.nk

# AOT embed .nk in firmware source (default: C++26)
python -m netkit aot models/test_mlp.nk -o build/aot
python -m netkit aot models/test_mlp.nk -o build/aot --language c
python -m netkit aot models/test_mlp.nk -o build/aot --main   # optional smoke main

# Convert all bundled regression models (from repo root)
make export-nk
```

Typical pipeline:

```text
model.onnx  →  python -m netkit convert  →  model.nk  →  python -m netkit aot  →  model_aot.{hpp,cpp}
```

Link AOT output with `libnetkit.a` — see [docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md#5-aot-compile-embed-nk-in-firmware).

## Python API

```python
from netkit import compile_aot, convert_onnx_to_nk, AotLanguage

convert_onnx_to_nk("models/test_mlp.onnx", "models/test_mlp.nk")
result = compile_aot("models/test_mlp.nk", "build/aot", language=AotLanguage.CPP)
```

## Testing

```bash
pip install -e python   # onnx + onnxruntime for parity tests
make lib                # required for AOT compile tests
make test-python        # ONNX parity (69) + AOT tests (from repo root)
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
