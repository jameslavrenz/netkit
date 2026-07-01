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

This is the standard “two conv blocks + pooling + dense head” recipe used in introductory MNIST CNN examples (TensorFlow/Keras docs, Colab tutorials, etc.).

## Accuracy vs published results

| Metric | netkit (committed export) | Typical baseline |
|--------|---------------------------|------------------|
| Test accuracy | **99.02%** | ~98.5–99.3% (Keras/TF CNN tutorials) |
| Training set | **60,000** images | 60,000 |
| Optimizer | Adam, lr=0.001, **20 epochs** | Similar |

For comparison, the [MNIST MLP suite](MNIST.md) on the same data reaches **98.06%** test accuracy — the CNN matches or exceeds published tutorial numbers.

Recorded in `models/mnist_cnn/training_meta.json`.

## Files

| Path | Purpose |
|------|---------|
| `models/mnist_cnn.json` | Architecture JSON |
| `models/mnist_cnn.bin` | float32 weights (~900 KiB) |
| `models/mnist_cnn/manifest.json` | 10 regression cases |
| `models/mnist_cnn/case_*.input.bin` | Flattened 28×28 NHWC input |
| `models/mnist_cnn/case_*.expected.bin` | 10 softmax outputs |
| `tools/export_mnist_cnn.py` | Train + export script |

Each case loads binary input/expected tensors, compares all 10 softmax outputs within tolerance, and checks argmax vs label. Console output shows predicted class, winner probability, and runner-ups above `0.01` (see [TESTING.md](TESTING.md)).

The MNIST CNN suite uses a **4 MiB** dedicated arena in `src/test_mnist.cpp` (~900 KiB weights plus ping-pong activation buffers for conv feature maps). Arena size is not in JSON — see [ARENA.md](ARENA.md).

## Running

MNIST CNN tests run automatically in `make test` / `./netkit test` — see [TESTING.md](TESTING.md).

```cpp
merge(run_mnist_cnn_tests());  // in src/test.cpp, after MNIST MLP
```

## Regenerating

```bash
make export-mnist-cnn
```

Requires **numpy**. Uses the same MNIST sources as the MLP export (`../python/mnist/*.csv` or downloaded IDX files).

## Engine support added for this test

CNN JSON models may now include:

- `max_pool2d` — 2×2 max pooling (valid, no padding)
- `flatten` — NHWC → 1×features before dense layers
- `dense` — fully connected head (ReLU / softmax activations)

See [MODEL_FORMAT.md](MODEL_FORMAT.md).
