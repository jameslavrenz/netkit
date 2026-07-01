# Testing

netkit uses **Make** as the primary build and test driver (no CMake required). All regression tests run through `./netkit test` and the C API harness `tests/test_c_api`.

## Quick commands

```bash
make              # netkit CLI + libnetkit.a (default)
make build-all    # netkit + examples + C API test binary
make test         # full regression (C++ then C API)
make test-cpp     # ./netkit test only
make test-c       # ./tests/test_c_api only
make clean        # remove objects and binaries
make rebuild      # clean + make
```

## Regression suites

Both `make test-cpp` and `make test-c` exercise the **same 36 inference cases** via `run_all_tests()` / `nk_run_all_tests()`:

| Suite | Cases | Source | Description |
|-------|------:|--------|-------------|
| Hand MLP vectors | 9 | `models/test_mlp.vectors.json`, `models/mlp_hand.vectors.json` | Small hand-checked MLP forwards |
| Hand CNN vectors | 7 | `models/test_cnn.vectors.json`, `models/cnn_4x4_single.vectors.json`, `models/cnn_hand.vectors.json` | Small hand-checked CNN forwards (pure conv) |
| MNIST MLP | 10 | `models/mnist/manifest.json` | Trained 784→128→10 MLP (98.06% test acc) |
| MNIST CNN | 10 | `models/mnist_cnn/manifest.json` | Conv+pool+flatten+dense CNN (99.02% test acc) |

**Total: 36 passed** when healthy (`16` hand vector + `10` MNIST MLP + `10` MNIST CNN).

| Doc | Contents |
|-----|----------|
| [VECTORS_TESTS.md](VECTORS_TESTS.md) | Hand `*.vectors.json` format |
| [MNIST.md](MNIST.md) | MNIST MLP bundle |
| [MNIST_CNN.md](MNIST_CNN.md) | MNIST CNN bundle |

## Arena buffers in tests

All regression paths use an arena; only the **backing buffer size** varies:

| Harness | Source | Arena size | Models |
|---------|--------|------------|--------|
| Hand vector tests | `src/vectors_loader.cpp` | **64 KiB** | `*.vectors.json` hand models |
| MNIST MLP | `src/test_mnist.cpp` | **2 MiB** | `mnist_mlp.json` |
| MNIST CNN | `src/test_mnist.cpp` | **4 MiB** | `mnist_cnn.json` |
| C API smoke / unit tests | `tests/test_c_api.c` | **64 KiB** | hand models + parse/load smoke |
| CLI `run` / `inspect` | `src/cli.cpp` | **64 KiB** | hand models (MNIST may overflow `--full`) |

The test code does not read arena size from JSON — constants are chosen so weights + ping-pong activation buffers fit. See [ARENA.md](ARENA.md) for sizing your own firmware buffer.

## C++ API suite (`make test-cpp`)

Entry: `./netkit test` → `run_all_tests()` in `src/test.cpp`.

Sections printed in order:

1. **MLP TESTS** — hand `*.vectors.json` files  
2. **CNN TESTS** — hand `*.vectors.json` files  
3. **MNIST MLP TESTS** — `run_mnist_tests()` in `src/test_mnist.cpp`  
4. **MNIST CNN TESTS** — `run_mnist_cnn_tests()` in `src/test_mnist.cpp`

## Test output

**Hand vector cases** print the input tensor, then a per-output line (`out[i]: actual=… expected=…`) so small models show meaningful numeric checks. See [VECTORS_TESTS.md](VECTORS_TESTS.md).

**MNIST cases** print predicted class, winner softmax probability, and any runner-up outputs above `0.01` (most classes stay at `0.0000` after softmax and are omitted from the log). All 10 outputs are still compared internally within tolerance.

## C API suite (`make test-c`)

Entry: `./tests/test_c_api` (C23).

| Phase | What it covers |
|-------|----------------|
| Arena | init, aligned alloc, reset, capacity |
| Tensor / ops | create, matmul, activations |
| Parse architecture | MLP and CNN JSON metadata |
| Model load / run | `nk_model_load` + `nk_model_run` on hand MLP/CNN |
| Hybrid CNN | `nk_parse_architecture` + `nk_cnn_load` on `mnist_cnn.json` |
| Full regression | `nk_run_all_tests()` — same **36** inference cases as C++ |

The C API regression path uses the same C++ runner internally (`nk_run_all_tests` → `run_all_tests`), so MNIST CNN (conv / pool / flatten / dense) is covered without retraining in CI.

## Adding tests

| Kind | How |
|------|-----|
| Hand vector case | Edit `models/*.vectors.json`, register file in `src/test.cpp` if new bundle |
| MNIST MLP case | `make export-mnist` (requires numpy) |
| MNIST CNN case | `make export-mnist-cnn` (requires numpy) |

Always run `make test` before committing.

## Regenerating MNIST assets

Weights are **committed** so CI never trains. Regenerate only when architecture or training changes:

```bash
make export-mnist       # MLP — full 60k, 40 epochs (~8s)
make export-mnist-cnn   # CNN — full 60k, 20 epochs (~18 min)
make export-mnist-all   # both
```

Requires **numpy**. Uses `../python/mnist/*.csv` when present, else downloads IDX files to `data/mnist/`.

## CI

GitHub Actions (`.github/workflows/ci.yml`): `make`, `make test`, example smoke tests, CLI smoke tests. All model weights and case files are in the repo — no network or Python in CI.
