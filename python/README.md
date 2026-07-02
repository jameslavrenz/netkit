# netkit-tools (Python)

Convert ONNX models into binary **`.nk`** files for the C++ runtime.

**Role in netkit:** Phase 1 serializer (ONNX → `.nk`). Phase 2 will add compiler-style optimizations here (fusion, layout, quantization) — see [docs/PHILOSOPHY.md](../docs/PHILOSOPHY.md).

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

# Convert all bundled regression models (from repo root)
make export-nk
```

## Testing

```bash
pip install -e python   # onnx + onnxruntime for parity tests
make test-python        # .nk vs ONNX Runtime, 69 cases (from repo root, after make)
```

See [docs/TESTING.md](../docs/TESTING.md) and [docs/ONNX.md](../docs/ONNX.md).

## C++ runtime

```bash
./netkit inspect models/test_mlp.nk
./netkit run models/test_mlp.nk --input 1,2
```

See [docs/NK_FORMAT.md](../docs/NK_FORMAT.md) for the binary layout. Getting started: [docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md).
