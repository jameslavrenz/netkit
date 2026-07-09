# Testing

netkit uses **GNU Make** as the primary build and test driver. **CMake** is optional (`cmake -B cmake-build && cmake --build cmake-build`) with the same flags — see [BUILD_TARGETS.md](BUILD_TARGETS.md). C++ regression tests run through `./netkit test` (interpreter path) and the C API harness `tests/test_c_api`. ONNX parity and **AOT compile** tests run in Python. See [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

**GitHub Actions** — fast **`CI`** workflow on push to `main` and pull requests (`make test`); full suite via manual **`Test full`** workflow only — see [CI](#ci). Run `make test-full` locally before release or when changing ONNX import/fusion.

## Quick commands

```bash
make              # NETKIT_TARGET=cpu (default): netkit CLI + libnetkit.a
make build-all    # cpu: netkit + examples + C API test binary; mcu/mpu: lib + examples + embedded_smoke
make test         # default: C++/C embedded regression + fast Python (~1 min; cpu only)
make test-full    # full suite incl. ONNX parity (82) + backbone tests (manual / pre-release)
make test-cpp     # ./netkit test only (88 embedded .nk cases)
make test-c       # ./tests/test_c_api only
make test-python  # fast Python subset (same as in make test)
make test-python-full  # ONNX parity (82) + AOT compile tests; requires libnetkit.a
make clean        # remove objects and binaries
make rebuild      # clean + make

# Optional CMSIS backend parity (after make cmsis-init)
make NETKIT_CMSIS_DSP=1 test-cpp
make test-embedded-smoke-matrix   # MCU CM4/M33 + CMSIS-NN; MPU DSP-only profiles (local only)

# Optional CMake build + test
cmake -B cmake-build && cmake --build cmake-build
./cmake-build/netkit test
```

Embedded runtime-only builds: `make NETKIT_TARGET=mcu lib` or `make NETKIT_TARGET=mpu lib` — see [BUILD_TARGETS.md](BUILD_TARGETS.md). Full regression requires `NETKIT_TARGET=cpu`. MCU/MPU bring-up smoke: `make test-embedded-smoke-matrix` — see [Embedded smoke](#embedded-smoke-mcupu). New users: [GETTING_STARTED.md](GETTING_STARTED.md).

## C++ regression (`.nk` loader + inference)

Both `make test-cpp` and `make test-c` exercise the **same 88 embedded cases** via `run_all_tests()` / `nk_run_all_tests()`:

| Suite | Cases | Source | Description |
|-------|------:|--------|-------------|
| Hand MLP | 9 | `models/test_mlp.nk`, `models/mlp_hand.nk` | Small hand-checked MLP forwards |
| Hand CNN | 10 | `models/test_cnn.nk`, `models/cnn_4x4_single.nk`, `models/cnn_hand.nk`, `models/convnextv2_atto_block.nk`, `models/mobilenetv4_small_uib.nk`, `models/resnet18_basic_block.nk` | Hand-checked CNN + ConvNeXt V2 block + MobileNetV4 UIB + ResNet BasicBlock — [CONVNEXTV2.md](CONVNEXTV2.md), [MOBILENETV4.md](MOBILENETV4.md), [RESNET18.md](RESNET18.md) |
| MNIST MLP | 10 | `models/mnist_mlp.nk` | Trained 784→128→10 MLP (98.06% test acc) |
| MNIST CNN | 10 | `models/mnist_cnn.nk` | Conv+pool+flatten+dense CNN (99.02% test acc) |
| Op matrix | 17 | `models/op_matrix_mlp.nk`, `models/op_matrix_cnn.nk`, `models/cnn_extended_ops.nk`, `models/deep_mlp.nk` | Activation sweep, padded conv/pool, avg pool, batch norm |
| ONNX import extensions | 27 | `models/import_*.nk` | Asymmetric conv/depthwise, UIB/ResNet/ConvNeXt ONNX import — [ONNX.md](ONNX.md) |
| ONNX import backbones | 3 | `models/import_*_backbone.nk` | Full timm `forward_features` round-trip (ResNet-18, MobileNetV4 Small, ConvNeXt V2-Atto) — [ONNX.md](ONNX.md) |
| MobileNetV4 Small | 1 | `models/mobilenetv4_small.nk` | Full MNv4-Conv-Small backbone (22 layers, 56×56×3) — [MOBILENETV4.md](MOBILENETV4.md) |
| ResNet-18 | 1 | `models/resnet18.nk` | Full ResNet-18 backbone (13 layers, 56×56×3) — [RESNET18.md](RESNET18.md) |
| ConvNeXt V2-Atto | 1 | `models/convnextv2_atto.nk` | Full ConvNeXt V2-Atto backbone (24 layers, 32×32×3) — [CONVNEXTV2.md](CONVNEXTV2.md) |

**Total: 88 passed** when healthy (`19` hand + `10` MNIST MLP + `10` MNIST CNN + `17` op matrix + `27` ONNX import extensions + `1` MobileNetV4 Small + `2` YOLOX (full + head-only) + `1` ResNet-18 + `1` ConvNeXt V2-Atto).

These tests validate **`.nk` parsing, weight loading, and forward inference** against reference outputs embedded in each file (`TCAS` section). See [NK_FORMAT.md](NK_FORMAT.md).

## Python ONNX parity

`make test-python-full` (or `make test-full`) runs `python/tests/test_onnx_parity.py`: replays embedded inputs through **`tools/nk_infer`** and **ONNX Runtime** on the matching `.onnx` file (82 cases). Skipped in default `make test` / `make test-python`.

Requires **onnxruntime** for parity and **`make lib`** for AOT compile tests.

```bash
pip install -e python   # adds onnxruntime
make lib
make test-python-full   # or make test-full
```

## PyTorch backbone pack parity (timm)

With train extras installed (`pip install -e "python[train]"`), Python tests pack random-init **timm** backbones to flat `.nk` weights and compare forwards:

| Test | What it checks |
|------|----------------|
| `python/tests/test_torch_backbone_pack.py` | Packed weights vs NumPy reference vs timm (ResNet-18, ConvNeXt V2-Atto, MobileNetV4 Small) |
| `python/tests/test_torch_backbone_runtime_parity.py` | Same pack path, then **C++ runtime** (`tools/nk_infer`) vs timm and NumPy reference on random inputs |

Requires **`make tools/nk_infer`** (model-sized heap arena via `nk_recommended_arena_bytes`). Not part of the 59 embedded cases — uses ephemeral `.nk` files in a temp directory. Skipped in default `make test`; run `make test-full`.

## AOT compile tests

`python/tests/test_aot_compile.py` generates C++26 and C23 sources from hand `.nk` models, compiles them against `libnetkit.a`, and checks outputs against the NumPy reference forward pass (embedded TCAS inputs). Generated headers are checked for arena sizing constants; an MCU-target compile (`-DNETKIT_TARGET_MCU=1`) is exercised against `mlp_hand.nk`.

Models exercised: `test_mlp.nk`, `cnn_4x4_single.nk`, `mlp_hand.nk`, `cnn_hand.nk`. With `--optimize` / `optimize=True`, `cnn_extended_ops.nk` is also checked end-to-end (optimized graph embedded, runtime parity preserved).

`python/tests/test_nk_optimize.py` covers individual graph passes (BN folding, conv+BN fusion, linear dense merge) with numeric checks against the reference forward pass.

## Hand model coverage (MCU paths)

| Harness | In CI job? | Models | What it checks |
|---------|:----------:|--------|----------------|
| `make test` / `make test-cpp` / `make test-c` | Yes | hand + MNIST + op-matrix + import extensions + MobileNetV4 / YOLOX / ResNet-18 / ConvNeXt V2 backbones | C++/C `.nk` load + forward vs embedded TCAS (**88 cases**) + fast Python (AOT, unit tests) |
| `make test-full` / `make test-python-full` | No | + ONNX sidecars + timm backbone pack/runtime | Full ONNX parity (**82 cases**) + backbone tests — run manually before release |
| `tests/embedded_smoke` / `make test-embedded-smoke-matrix` | No | `test_mlp.nk`, `cnn_4x4_single.nk` | Lean MCU/MPU runtime on host (`NETKIT_HOST_SMOKE=1`); run locally via `./tools/run_embedded_smoke.sh` |

Hand-checked models (`mlp_hand`, `cnn_hand`) are fully validated in **`make test`**. Embedded smoke uses `test_mlp` and `cnn_4x4_single` for fast firmware bring-up.

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

Host execution exercises linking and inference paths before on-device bring-up. Smoke loads two bundled models: tiny MLP and CNN hand fixtures. The matrix sets `NETKIT_HOST_SMOKE=1` so CMSIS-DSP uses the portable `__GNUC_PYTHON__` path on the host (no CMSIS-Core headers). On hardware, link with your toolchain `-mcpu` flags and `NETKIT_ARCH=...` without `NETKIT_HOST_SMOKE`.

| Doc | Contents |
|-----|----------|
| [NK_FORMAT.md](NK_FORMAT.md) | `.nk` layout + embedded regression tests |
| [ONNX.md](ONNX.md) | Python converter + parity testing |
| [MNIST.md](MNIST.md) | MNIST MLP model |
| [MNIST_CNN.md](MNIST_CNN.md) | MNIST CNN model |
| [RESNET18.md](RESNET18.md) | ResNet BasicBlock + full ResNet-18 backbone |
| [CONVNEXTV2.md](CONVNEXTV2.md) | ConvNeXt V2 block + LayerNorm2d + full Atto backbone |
| [MOBILENETV4.md](MOBILENETV4.md) | MobileNetV4 UIB + full MNv4-Conv-Small backbone |

## Arena buffers in tests

All C++ regression paths use an arena; only the **backing buffer size** varies:

| Harness | Source | Arena size | Models |
|---------|--------|------------|--------|
| Hand tests | `src/nk_regression.cpp` | **64 KiB** | hand `.nk` models |
| MNIST MLP | `src/nk_regression.cpp` | **2 MiB** | `mnist_mlp.nk` |
| MNIST CNN | `src/nk_regression.cpp` | **64 MiB** heap default | `mnist_cnn.nk` |
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

## Test output

**Hand cases** (≤64 outputs) print the input tensor and a per-output line (`out[i]: actual=… expected=…`). Larger models (MNIST, full backbones) print only `PASS`/`FAIL` unless `NETKIT_REGRESSION_VERBOSE=1`. GitHub Actions sets `GITHUB_ACTIONS=true`, which keeps C++ logs compact automatically.

**MNIST cases** print predicted class, winner softmax probability, and any runner-up outputs above `0.01` when output logging is enabled. All outputs are compared internally within tolerance.

## C API suite (`make test-c`)

Entry: `./tests/test_c_api` (C23).

| Phase | What it covers |
|-------|----------------|
| Arena | init, aligned alloc, reset, capacity |
| Tensor / ops | create, matmul, activations |
| Parse architecture | MLP and CNN `.nk` metadata |
| Model load / run | `nk_model_load` + `nk_model_run` on hand MLP/CNN |
| Hybrid CNN | `nk_parse_architecture` + `nk_cnn_load` on `mnist_cnn.nk` |
| Full regression | `nk_run_all_tests()` — same **88** embedded cases as C++ (`run_all_tests()` in `src/test.cpp`) |

The C API regression path uses the same C++ runner internally (`nk_run_all_tests` → `run_all_tests`).

## Adding tests

| Kind | How |
|------|-----|
| Hand case | Add input to `python/netkit/regression_data.py` (`HAND_CASE_INPUTS`), run `make embed-tests`, register `.nk` in `src/test.cpp` if new bundle |
| ONNX parity case | Add matching `models/<name>.onnx`, convert with `make export-nk`, add pair to `PARITY_PAIRS` in `python/tests/test_onnx_parity.py` |
| MNIST MLP case | `make export-mnist` (requires PyTorch: `pip install -e "python[train]"`) |
| MNIST CNN case | `make export-mnist-cnn` (requires PyTorch) |
| Op matrix models | `make export-op-matrix` (requires numpy) |

Always run `make test` before committing. Run `make test-full` when changing ONNX import, fusion, or backbone pack paths.

## Regenerating models

Weights and embedded tests are **committed** so `make test` never trains. Regenerate only when architecture or training changes:

```bash
make export-mnist       # MLP — full 60k, 40 epochs (~8s)
make export-mnist-cnn   # CNN — full 60k, 20 epochs (~18 min)
make export-mnist-all   # both + refresh ONNX from .nk
make export-op-matrix   # synthetic activation/deep-chain models + ONNX
make export-nk          # ONNX → .nk + embed hand tests
make embed-tests        # re-embed hand tests (inputs from regression_data.py; expected from reference forward)
```

Requires **PyTorch** for training scripts (`pip install -e "python[train]"`). NumPy is used for IDX I/O and packing only. MNIST data from CSV sibling path or IDX download into `data/mnist/`.

## CI

**CI** (`.github/workflows/ci.yml`) runs on **push to `main`**, **pull requests**, and manual dispatch. It runs the default fast suite (`make test`).

**Test full** (`.github/workflows/test-full.yml`) is **manual only** — use before release or when changing ONNX import/fusion:

```bash
gh workflow run test-full.yml
gh run watch    # optional: wait for the run you just started
```

Re-run the fast workflow manually if needed:

```bash
gh workflow run ci.yml
```

The **`build-and-test`** job on `ubuntu-latest` uses **host Clang**. The checkout fetches the CMSIS submodules (`submodules: recursive`), so the Make build runs the **CMSIS-DSP** float32 path (`NETKIT_CMSIS_DSP=1`). CMSIS-NN stays off — it is Cortex-M (MCU) only and cannot link on the host `cpu` target:

1. `make` — initial build (cpu profile: CMSIS-DSP on)
2. `make NETKIT_CMSIS_DSP=1 rebuild test` — default C++ embedded + C API + fast Python suite, CMSIS-DSP kernels
3. Example and CLI smoke tests
4. CMake configure + build smoke test (`./cmake-build/netkit test`, Release, `-DNETKIT_CMSIS_DSP=OFF`) — cross-checks the **reference-kernel** path in the same run

The **`Test full`** workflow uses the same setup (`submodules: recursive`, `make NETKIT_CMSIS_DSP=1 rebuild test-full`). CMSIS-NN and on-device paths are still validated **locally only** (`make NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 ...`, `make test-embedded-smoke-matrix`, NUCLEO CNN int8 flash + UART capture) — not in CI.

**CI build notes**

- When `GITHUB_ACTIONS=true`, the Makefile adds `-O2` for **`NETKIT_TARGET=cpu` only** so full-backbone C++ regression (ResNet-18, MobileNetV4, ConvNeXt V2-Atto) finishes in reasonable time on Linux runners. Local `make test` stays debug-oriented (`-g`, no default `-O2`).
- The job timeout is 45 minutes.
- C++ regression truncates large embedded-case input dumps to 256 values and prints load/forward progress lines so long-running cases stay visible in logs.

Model weights and embedded test cases are in the repo — no training in CI.

## Recommended local validation

Pushes to `main` and pull requests run **CI** automatically. Before pushing, you can also run locally:

```bash
make cmsis-init
make test          # default fast suite (same as CI)
make test-full     # optional: full ONNX/backbone parity before release
./tools/run_embedded_smoke.sh    # optional MCU/MPU + CMSIS host smoke matrix
```

`make test-embedded-smoke-matrix` is equivalent to the script above.
