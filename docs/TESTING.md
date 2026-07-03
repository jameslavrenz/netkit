# Testing

netkit uses **GNU Make** as the primary build and test driver. **CMake** is optional (`cmake -B cmake-build && cmake --build cmake-build`) with the same flags — see [BUILD_TARGETS.md](BUILD_TARGETS.md). C++ regression tests run through `./netkit test` and the C API harness `tests/test_c_api`. ONNX parity runs in Python.

**GitHub Actions** (`.github/workflows/ci.yml`) is **manual only** (`workflow_dispatch`) — see [CI](#ci). Run locally with `make test` before pushing.

## Quick commands

```bash
make              # NETKIT_TARGET=cpu (default): netkit CLI + libnetkit.a
make build-all    # cpu: netkit + examples + C API test binary; mcu/mpu: lib + examples + embedded_smoke
make test         # C++ embedded regression + Python ONNX parity (cpu only)
make test-cpp     # ./netkit test only (73 embedded .nk cases)
make test-c       # ./tests/test_c_api only
make test-python  # ONNX parity (69) + AOT compile tests; requires libnetkit.a
make clean        # remove objects and binaries
make rebuild      # clean + make

# Optional CMSIS backend parity (after make cmsis-init)
make NETKIT_CMSIS_DSP=1 test-cpp
make test-embedded-smoke-matrix   # MCU CM4/M33 + CMSIS-NN; MPU DSP-only profiles (included in manual CI job)

# Optional CMake build + test
cmake -B cmake-build && cmake --build cmake-build
./cmake-build/netkit test
```

Embedded runtime-only builds: `make NETKIT_TARGET=mcu lib` or `make NETKIT_TARGET=mpu lib` — see [BUILD_TARGETS.md](BUILD_TARGETS.md). Full regression requires `NETKIT_TARGET=cpu`. MCU/MPU bring-up smoke: `make test-embedded-smoke-matrix` — see [Embedded smoke](#embedded-smoke-mcupu). New users: [GETTING_STARTED.md](GETTING_STARTED.md).

## C++ regression (`.nk` loader + inference)

Both `make test-cpp` and `make test-c` exercise the **same 73 embedded cases** via `run_all_tests()` / `nk_run_all_tests()`:

| Suite | Cases | Source | Description |
|-------|------:|--------|-------------|
| Hand MLP | 9 | `models/test_mlp.nk`, `models/mlp_hand.nk` | Small hand-checked MLP forwards |
| Hand CNN | 7 | `models/test_cnn.nk`, `models/cnn_4x4_single.nk`, `models/cnn_hand.nk` | Small hand-checked CNN forwards |
| MNIST MLP | 10 | `models/mnist_mlp.nk` | Trained 784→128→10 MLP (98.06% test acc) |
| MNIST CNN | 10 | `models/mnist_cnn.nk` | Conv+pool+flatten+dense CNN (99.02% test acc) |
| Op matrix | 17 | `models/op_matrix_mlp.nk`, `models/op_matrix_cnn.nk`, `models/cnn_extended_ops.nk`, `models/deep_mlp.nk` | Activation sweep, padded conv/pool, avg pool, batch norm |
| Fashion-MNIST MLP | 10 | `models/fashion_mnist_mlp.nk` | Trained 784→128→10 MLP |
| Fashion-MNIST CNN | 10 | `models/fashion_mnist_cnn.nk` | Conv+pool+flatten+dense CNN |

**Total: 73 passed** when healthy (`16` hand + `10` MNIST MLP + `10` MNIST CNN + `17` op matrix + `20` Fashion-MNIST).

These tests validate **`.nk` parsing, weight loading, and forward inference** against reference outputs embedded in each file (`TCAS` section). See [NK_FORMAT.md](NK_FORMAT.md).

## Python ONNX parity

`make test-python` runs `python/tests/test_onnx_parity.py`: replays embedded inputs through **`tools/nk_infer`** and **ONNX Runtime** on the matching `.onnx` file (69 cases).

Requires **onnxruntime** for parity and **`make lib`** for AOT compile tests.

```bash
pip install -e python   # adds onnxruntime
make lib
make test-python
```

## AOT compile tests

`python/tests/test_aot_compile.py` generates C++26 and C23 sources from hand `.nk` models, compiles them against `libnetkit.a`, and checks outputs against the NumPy reference forward pass (embedded TCAS inputs). Generated headers are checked for arena sizing constants; an MCU-target compile (`-DNETKIT_TARGET_MCU=1`) is exercised against `mlp_hand.nk`.

Models exercised: `test_mlp.nk`, `cnn_4x4_single.nk`, `mlp_hand.nk`, `cnn_hand.nk`. With `--optimize` / `optimize=True`, `cnn_extended_ops.nk` is also checked end-to-end (optimized graph embedded, runtime parity preserved).

`python/tests/test_nk_optimize.py` covers individual graph passes (BN folding, conv+BN fusion, linear dense merge) with numeric checks against the reference forward pass.

## Hand model coverage (MCU paths)

| Harness | In CI job? | Models | What it checks |
|---------|:----------:|--------|----------------|
| `make test` / `make test-cpp` | Yes | `test_mlp.nk`, `mlp_hand.nk`, `test_cnn.nk`, `cnn_4x4_single.nk`, `cnn_hand.nk` (+ MNIST / op-matrix / Fashion-MNIST) | Full `.nk` load + forward vs embedded TCAS expected outputs (**73 cases**) |
| `make test-c` | Yes | same via `nk_run_all_tests()` | C API parity with C++ regression |
| `tests/embedded_smoke` / `make test-embedded-smoke-matrix` | Yes | `test_mlp.nk`, `cnn_4x4_single.nk` | Lean MCU/MPU runtime on host (`NETKIT_HOST_SMOKE=1`); CI job runs `./tools/run_embedded_smoke.sh` |

Hand-checked models (`mlp_hand`, `cnn_hand`) are fully validated in **`make test`**. Embedded smoke uses the smaller `test_mlp` / `cnn_4x4_single` fixtures for fast firmware bring-up.

## Embedded smoke (MCU/MPU)

`tests/embedded_smoke.c` validates the **lean firmware runtime** without `NETKIT_DESKTOP` APIs (`nk_run_all_tests`, CLI, etc.). It uses a caller-owned static arena (`NK_ARENA_DEFAULT_CAPACITY`), parses `test_mlp.nk` and `cnn_4x4_single.nk`, and runs `nk_model_load` + `nk_model_run` with fixed expected outputs.

For **`mlp_hand.nk`** and **`cnn_hand.nk`**, use **`make test`** (embedded TCAS regression).

```bash
make cmsis-init   # required for CMSIS profiles
make NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 embedded-smoke
./tests/embedded_smoke

# Full matrix (mcu/mpu × reference + CMSIS × CM4/M33/A32 arch flags)
make test-embedded-smoke-matrix
```

Host execution exercises linking and inference paths before on-device bring-up. The matrix sets `NETKIT_HOST_SMOKE=1` so CMSIS-DSP uses the portable `__GNUC_PYTHON__` path on the host (no CMSIS-Core headers). On hardware, link with your toolchain `-mcpu` flags and `NETKIT_ARCH=...` without `NETKIT_HOST_SMOKE`.

| Doc | Contents |
|-----|----------|
| [NK_FORMAT.md](NK_FORMAT.md) | `.nk` layout + embedded regression tests |
| [ONNX.md](ONNX.md) | Python converter + parity testing |
| [MNIST.md](MNIST.md) | MNIST MLP model |
| [MNIST_CNN.md](MNIST_CNN.md) | MNIST CNN model |

## Arena buffers in tests

All C++ regression paths use an arena; only the **backing buffer size** varies:

| Harness | Source | Arena size | Models |
|---------|--------|------------|--------|
| Hand tests | `src/nk_regression.cpp` | **64 KiB** | hand `.nk` models |
| MNIST MLP | `src/nk_regression.cpp` | **2 MiB** | `mnist_mlp.nk` |
| MNIST CNN | `src/nk_regression.cpp` | **4 MiB** | `mnist_cnn.nk` |
| C API smoke / unit tests | `tests/test_c_api.c` | **64 KiB** | hand models + parse/load smoke |
| CLI `run` / `inspect` | `src/cli.cpp` | model-sized heap (cpu default) | all models |

The test code does not read arena size from the model file — constants are chosen so weights + ping-pong activation buffers fit. See [ARENA.md](ARENA.md) for sizing your own firmware buffer.

## C++ API suite (`make test-cpp`)

Entry: `./netkit test` → `run_all_tests()` in `src/test.cpp`.

Sections printed in order:

1. **MLP TESTS** — hand `.nk` models with embedded cases  
2. **CNN TESTS** — hand `.nk` models with embedded cases  
3. **MNIST MLP TESTS** — `models/mnist_mlp.nk`  
4. **MNIST CNN TESTS** — `models/mnist_cnn.nk`  
5. **OP MATRIX TESTS** — `models/op_matrix_mlp.nk`, `models/op_matrix_cnn.nk`, `models/cnn_extended_ops.nk`, `models/deep_mlp.nk`  
6. **FASHION-MNIST MLP TESTS** — `models/fashion_mnist_mlp.nk`  
7. **FASHION-MNIST CNN TESTS** — `models/fashion_mnist_cnn.nk`

## Test output

**Hand cases** print the input tensor, then a per-output line (`out[i]: actual=… expected=…`) so small models show meaningful numeric checks.

**MNIST cases** print predicted class, winner softmax probability, and any runner-up outputs above `0.01`. All outputs are compared internally within tolerance.

## C API suite (`make test-c`)

Entry: `./tests/test_c_api` (C23).

| Phase | What it covers |
|-------|----------------|
| Arena | init, aligned alloc, reset, capacity |
| Tensor / ops | create, matmul, activations |
| Parse architecture | MLP and CNN `.nk` metadata |
| Model load / run | `nk_model_load` + `nk_model_run` on hand MLP/CNN |
| Hybrid CNN | `nk_parse_architecture` + `nk_cnn_load` on `mnist_cnn.nk` |
| Full regression | `nk_run_all_tests()` — same **73** embedded cases as C++ |

The C API regression path uses the same C++ runner internally (`nk_run_all_tests` → `run_all_tests`).

## Adding tests

| Kind | How |
|------|-----|
| Hand case | Add to `python/netkit/regression_data.py`, run `make embed-tests`, register `.nk` in `src/test.cpp` if new bundle |
| ONNX parity case | Add matching `models/<name>.onnx`, convert with `make export-nk`, add pair to `PARITY_PAIRS` in `python/tests/test_onnx_parity.py` |
| MNIST MLP case | `make export-mnist` (requires PyTorch: `pip install -e "python[train]"`) |
| MNIST CNN case | `make export-mnist-cnn` (requires PyTorch) |
| Op matrix models | `make export-op-matrix` (requires numpy) |
| Fashion-MNIST MLP | `make export-fashion-mnist` (requires PyTorch) |
| Fashion-MNIST CNN | `make export-fashion-mnist-cnn` (requires PyTorch) |

Always run `make test` before committing.

## Regenerating models

Weights and embedded tests are **committed** so `make test` never trains. Regenerate only when architecture or training changes:

```bash
make export-mnist       # MLP — full 60k, 40 epochs (~8s)
make export-mnist-cnn   # CNN — full 60k, 20 epochs (~18 min)
make export-mnist-all   # both + refresh ONNX from .nk
make export-op-matrix   # synthetic activation/deep-chain models + ONNX
make export-fashion-mnist       # Fashion-MNIST MLP (~30 epochs)
make export-fashion-mnist-cnn   # Fashion-MNIST CNN (~15 epochs)
make export-fashion-mnist-all   # both Fashion-MNIST models + ONNX
make export-nk          # ONNX → .nk + embed hand tests
make embed-tests        # re-embed hand tests from regression_data.py
```

Requires **PyTorch** for training scripts (`pip install -e "python[train]"`). NumPy is used for IDX I/O and packing only. MNIST data from CSV sibling path or IDX download into `data/mnist/`.

## CI

GitHub Actions does **not** run on push or pull request. Trigger manually from the repo **Actions** tab (**CI** → **Run workflow**), or from the CLI:

```bash
gh workflow run ci.yml
gh run watch    # optional: wait for the run you just started
```

The **`build-and-test`** job on `ubuntu-latest` uses **host Clang** only:

1. `make cmsis-init` — fetch CMSIS-NN and CMSIS-DSP
2. `make` — default desktop build
3. `make NETKIT_HOST_SMOKE=1 NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 NETKIT_CMSIS_DSP=1 lib` — CMSIS MCU link smoke (host)
4. `make NETKIT_CMSIS_DSP=1 test-cpp` — CMSIS-DSP parity
5. `make test` — full C++ embedded + C API + Python ONNX parity + AOT compile + convert ops + trim-lib check
6. Example and CLI smoke tests
7. CMake configure + build smoke test (`./cmake-build/netkit test`)
8. `./tools/run_embedded_smoke.sh` — MCU/MPU + `NETKIT_ARCH` + CMSIS host smoke matrix

Model weights and embedded test cases are in the repo — no training in CI.

## Recommended local validation

Pushes do **not** trigger CI. Before pushing, run locally:

```bash
make cmsis-init
make test
./tools/run_embedded_smoke.sh    # same embedded-smoke matrix as CI
```

`make test-embedded-smoke-matrix` is equivalent to the script above.
