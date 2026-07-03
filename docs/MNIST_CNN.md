# MNIST CNN Regression Test

Trained **784→CNN→10** classifier on MNIST using a stack common in Keras/TensorFlow tutorials. Runs as part of `make test` alongside the [MNIST MLP suite](MNIST.md).

## Architecture

| Stage | Config | Activation |
|-------|--------|------------|
| Input | 28×28×1 (NHWC) | — |
| Conv2D | 3×3, 32 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Conv2D | 3×3, 64 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Flatten | 5×5×64 → 1600 | — |
| Dense | 128 units | ReLU |
| Dense | 10 units | Softmax |

## Accuracy

| Metric | netkit (committed export) | Typical baseline |
|--------|---------------------------|------------------|
| Test accuracy | **99.02%** | ~98.5–99.3% (Keras/TF CNN tutorials) |
| Training set | **60,000** images | 60,000 |
| Optimizer | Adam, lr=0.001, **20 epochs** | Similar |

The [MNIST MLP suite](MNIST.md) on the same data reaches **98.06%** test accuracy.

## Files

| Path | Purpose |
|------|---------|
| `models/mnist_cnn.onnx` | Source graph (ONNX parity) |
| `models/mnist_cnn.nk` | Runtime model + 10 embedded TCAS cases |
| `tools/export_mnist_cnn.py` | Train + export script |

The MNIST CNN suite uses a **4 MiB** dedicated arena in `src/nk_regression.cpp`. See [ARENA.md](ARENA.md).

## Running

Part of `make test` / `./netkit test` — see [TESTING.md](TESTING.md).

## Regenerating

```bash
make export-mnist-cnn
make export-onnx-test
make test
```

Commit `models/mnist_cnn.nk` and `models/mnist_cnn.onnx` after regenerating so tests stay offline (no training at test time).
