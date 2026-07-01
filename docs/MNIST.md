# MNIST MLP Test

End-to-end regression test: real MNIST handwritten digits (28×28 grayscale) through a trained two-layer MLP in netkit.

## Architecture

| Stage | Shape | Activation |
|-------|-------|--------------|
| Input | `[1, 784]` | — (pixels ÷ 255) |
| Dense hidden | 128 units | ReLU |
| Dense output | 10 units | Softmax (digit probabilities) |

Weights and biases are **float32** in `models/mnist_mlp.bin` (~398 KiB). Trained offline with `tools/export_mnist_mlp.py`: **Adam**, cross-entropy, **60,000 images**, **40 epochs** → **98.06%** test accuracy (best-in-class for this architecture). Metrics in `models/mnist/training_meta.json`.

## Test assets

```
models/
├── mnist_mlp.json          # architecture
├── mnist_mlp.bin           # float32 weights
└── mnist/
    ├── manifest.json       # 10 cases + tolerance
    ├── training_meta.json  # accuracy + training hyperparams
    ├── case_000.input.bin  # 784 float32 pixels
    ├── case_000.expected.bin # 10 float32 reference outputs
    └── ...
```

Each case:

1. Loads a 784-float input from MNIST test set
2. Runs `./netkit test` MNIST suite (or `run_mnist_tests()`)
3. Compares all 10 softmax outputs to reference within tolerance (`0.05` default)
4. Checks **classification** (argmax vs label)

Console output shows the predicted digit, winner probability, and any runner-up classes above `0.01` — not all ten neurons (most are `0.0000` after softmax).

The MNIST suite uses a **2 MiB** dedicated arena in `src/test_mnist.cpp` (not 64 KiB). Weights alone need ~400 KiB; load also reserves two ping-pong activation buffers (2 × largest hidden tensor). Arena size is **not** in `mnist_mlp.json` — the test harness picks a safe static buffer. See [ARENA.md](ARENA.md).

## Regenerate model and cases

Requires **numpy**. MNIST data from (in order):

1. `../python/mnist/mnist_{train,test}.csv` (sibling repo path), or
2. Download IDX gzip files from Google CVDF mirror into `data/mnist/`

```bash
make export-mnist
# or: python3 tools/export_mnist_mlp.py
make test-cpp
```

Commit updated `models/mnist_mlp.*` and `models/mnist/*` after regenerating so CI stays offline.

## Running

MNIST tests run automatically as part of `make test` / `./netkit test` — see [TESTING.md](TESTING.md). They are wired in `src/test.cpp`:

```cpp
merge(run_mnist_tests());  // after hand MLP/CNN vector suites
```

Both C++ and C API suites call the same path via `run_all_tests()` / `nk_run_all_tests()`.
