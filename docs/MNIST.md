# MNIST MLP Test

End-to-end regression test: real MNIST handwritten digits (28×28 grayscale) through a trained two-layer MLP in netkit.

## Architecture

| Stage | Shape | Activation |
|-------|-------|------------|
| Input | `[1, 784]` | — (pixels ÷ 255) |
| Dense hidden | 128 units | ReLU |
| Dense output | 10 units | Softmax (digit probabilities) |

Weights and **10 embedded regression cases** live in `models/mnist_mlp.nk` (~398 KiB model + test payloads). Trained offline with `tools/export_mnist_mlp.py`: **Adam**, cross-entropy, **60,000 images**, **40 epochs** → **98.06%** test accuracy.

## Test assets

```
models/
├── mnist_mlp.onnx    # source graph (ONNX parity)
└── mnist_mlp.nk      # runtime model + 10 embedded TCAS cases
```

Each embedded case:

1. Holds 784 float32 input pixels and 10 reference softmax outputs
2. Runs via `NkRegression::RunModelTests("models/mnist_mlp.nk")`
3. Compares all outputs within tolerance (`0.0001`)
4. Checks **classification** (argmax vs label)

The MNIST suite uses a **2 MiB** dedicated arena in `src/nk_regression.cpp`. See [ARENA.md](ARENA.md).

## Regenerate

Requires **PyTorch** (`pip install -e "python[train]"`). MNIST data from CSV sibling path or IDX download into `data/mnist/`.

```bash
make export-mnist    # train + write mnist_mlp.nk with embedded cases
make export-onnx-test
make test
```

Commit `models/mnist_mlp.nk` and `models/mnist_mlp.onnx` after regenerating so tests stay offline (no training at test time).

## Running

Part of `make test` / `./netkit test` — see [TESTING.md](TESTING.md).
