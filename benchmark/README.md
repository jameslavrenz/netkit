# netkit vs TFLM MNIST benchmarks

Side-by-side invoke latency on the same 10 MNIST test vectors per model (from TCAS cases in `models/mnist_*.nk`): **one correctly-classified test image per digit 0–9**, sorted by label.

**Fair comparison policy:** benchmarks use the **interpreter** path only — netkit loads `.nk` models via `NkLoader` and calls `forward()` / `forward_quantized()`; TFLM uses `MicroInterpreter::Invoke()`. **AOT lowered firmware is not included** in `compare.sh` or `make -C benchmark/netkit run-all` (use `run-aot*` targets separately for deployment profiling).

**Kernel defaults:** Conv2D strategy is the single `NETKIT_IM2COL` knob (`0` = direct, `1` = partial im2col, `2` = full im2col + GEMM) for **float reference** and **int8 QuantOps**. **Default `0` on cpu / MCU / MPU** — leave it there unless profiling MCU or reference-only MPU builds. `NETKIT_IM2COL=1` can give a small bump when XNNPACK is off; CMSIS-NN / XNNPACK ignore the knob. Prefer at most `1`; safest is `0`. See [docs/BUILD_TARGETS.md](../docs/BUILD_TARGETS.md#netkit_im2col-guidance). `NETKIT_LOOP_UNROLL` stays **off** (`0`) in all benchmark builds.

## Methodology

Each benchmark:

1. Runs **100 outer iterations**
2. Each iteration invokes all **10 test images**
3. **Discards the first invoke** (image 0) as warmup
4. Records the average of images 1–9 for that run
5. Reports **mean** invoke latency across the 100 run averages

Only `MLPNetwork::forward()` / `CNNNetwork::forward()` / `MicroInterpreter::Invoke()` is timed.

Profile builds use `forward_timed()` (NETKIT) or TFLM `MicroProfiler` (TFLM) to accumulate per-op means over the same 100×10 schedule.

## Compiler flags (fair comparison)

Benchmarks use **two** host flag profiles, selected by peer:

| Peer | Profile file | Opt | Notes |
|------|--------------|-----|-------|
| **TFLM Micro** (MNIST `compare.sh`) | `benchmark/common/tflm_host_flags.mk` | `-O2` | Mirrors TFLM Micro host: `TF_LITE_DISABLE_X86_NEON`, `-fno-exceptions` / `-fno-rtti`, section GC, TFLM warnings |
| **TF Lite / LiteRT** (ImageNet MPU) | `benchmark/common/tflite_host_flags.mk` | `-O3 -DNDEBUG` | Mimics LiteRT pip opt (`bazel -c opt --copt=-O3`); SIMD on; **no** TFLM Micro extras |

LiteRT-matched profile (ImageNet):

| Setting | Value | Why |
|---------|--------|-----|
| Compiler / linker | host `gcc` / `g++` | Same driver names as TFLM host; Darwin = Apple clang (matches LiteRT macOS wheels); Linux = GNU g++ |
| Opt | `-O3 -DNDEBUG` | LiteRT `ci/build_pip_package_with_bazel.sh` passes `--copt=-O3` on top of `-c opt` |
| C++ dialect | `-std=gnu++20` | LiteRT uses `gnu++17`; netkit needs C++20 (`std::span`) so keep GNU dialect at C++20 |
| Permissive | `-fpermissive` (Darwin/Linux) | LiteRT `.bazelrc` `build:macos` / `build:linux` |
| Darwin arm64 link | `-ld_classic` | LiteRT wheel script `--linkopt=-ld_classic` on `arm64` |
| Exceptions / RTTI | toolchain default (on) | LiteRT `libLiteRt.dylib` uses exceptions; do not force `-fno-*` |
| SIMD | enabled | No `TF_LITE_DISABLE_X86_NEON` |
| netkit-only | `-DNETKIT_*` | mmap/target/im2col knobs |

**Same compiler for host peers:** netkit (`BENCH_FLAG_PROFILE=tflite`) and the LiteRT wheel path use `gcc`/`g++`. On Darwin that is Apple clang (LiteRT’s actual macOS compiler); do not use Homebrew `g++-N`. Host/CPU TVM benches are not maintained.

XNNPACK itself is built once via `./tools/fetch_xnnpack.sh` as CMake **Release** (`-O3 -DNDEBUG`, same `gcc`/`g++`), pinned to the **same commit** LiteRT 2.1.6 embeds (`c2e81f01…`; override with `NETKIT_XNNPACK_PIN`).

ImageNet int8 XNNPACK runtime also mirrors the LiteRT delegate defaults used by the peer bench (`num_threads=1`):

| Setting | Value | TF Lite source |
|---------|--------|----------------|
| Head reduce | `xnn_define_static_reduce_v2` (MEAN, KEEP_DIMS) | int8 model uses `MEAN`; `AVERAGE_POOL_2D` is float-only in the delegate |
| Runtime flags | `XNN_FLAG_DONT_SPIN_WORKERS` | always set in `xnnpack_delegate.cc` |
| Workspace | shared `xnn_create_workspace` | delegate creates one workspace per instance |
| Threadpool | `nullptr` | delegate only creates a pool when `num_threads > 1` |
| Weights cache finalize | soft, then hard fallback | same as `TfLiteXNNPackDelegateFinalizeWeightsCache` |

ImageNet targets pass `BENCH_FLAG_PROFILE=tflite` so netkit is not accidentally compared under TFLM's Micro flags. MNIST / `compare.sh` keep the TFLM profile.

The main repo `libnetkit.a` (clang, debug) is **not** used by these benchmarks.

## Host A/B suite (netkit vs TF Lite)

Primary fair host peer for **float32** and **int8**:

| key | Model |
|-----|--------|
| `cnn` | **MNIST CNN** — digit classifier |
| `cnn_dw` | **MNIST DS-CNN** — depthwise-separable digit peer |
| `imagenet` | **MobileNetV4-Conv-Small** on ImageNet (10-class fixture) |

```bash
./tools/build_onnxruntime_litert_matched.sh          # once (XNNPACK EP + LiteRT-matched flags)
python3 benchmark/tools/export_host_onnx_assets.py --all
python3 benchmark/tools/run_host_ab_suite_int8.py
python3 benchmark/tools/run_host_ab_suite_float32.py
```

Three-way: **netkit vs TF Lite vs ONNX Runtime**, XNNPACK ON/OFF. Prebuild + discarded first process + order swaps (`nk→tf→ort` / `ort→tf→nk`). ORT is source-built with the same `gcc`/`g++` / `-O3` / `-fpermissive` / Darwin `-ld_classic` policy as LiteRT-matched netkit (`tools/build_onnxruntime_litert_matched.sh`); stock pip ORT lacks the XNNPACK EP. See [`benchmark/onnxruntime/`](onnxruntime/).

**Ref-mode caveat:** TF Lite OFF is `BUILTIN_REF` (slowest CPU path), so that column is netkit reference vs TF Lite’s slow reference. ORT OFF still runs **MLAS** and stays faster on all six — but with XNNPACK ON, optimized TF Lite builtins and MLAS are moot. **MLAS is not needed for netkit.** Numbers: [docs/STATUS.md](../docs/STATUS.md#host-three-way-suite-netkit-vs-tf-lite-vs-onnx-runtime). Suite infographics (MCU / MPU / CPU): [linkedin/](linkedin/).

## MPU A/B (Raspberry Pi Zero 2 W)

Same models and XNNPACK ON/OFF policy, cross-built for `linux/aarch64` and run over SSH vs TF Lite on the Pi. Guide: [`boards/pi-zero-2w/README.md`](../boards/pi-zero-2w/README.md).

```bash
./tools/fetch_xnnpack.sh
./tools/build_mpu_pi_aarch64.sh
NETKIT_PI_PASS='...' ./tools/run_mpu_pi_float32_ab.sh
NETKIT_PI_PASS='...' ./tools/run_mpu_pi_int8_ab.sh
```

## Run comparison (recommended)

```bash
make export-mnist export-mnist-cnn   # if models not present
make -C benchmark/tflm export-assets # once, or after retraining

./benchmark/compare.sh
```

`compare.sh` builds and runs **10 variants**, prints ASCII comparison tables, and writes PNG exports:

| Output | Contents |
|--------|----------|
| `benchmark/mnist_latency_comparison.png` | MLP + CNN mean invoke (NETKIT ref/CMSIS vs TFLM) |
| `benchmark/mnist_mlp_profile_comparison.png` | MLP per-op profile (NETKIT vs TFLM) |
| `benchmark/mnist_cnn_profile_comparison.png` | CNN per-op profile (NETKIT vs TFLM) |

Variants executed (in order):

1. NETKIT MLP reference
2. NETKIT CNN reference
5. TFLM MLP
6. TFLM CNN
7. NETKIT MLP profile
8. NETKIT CNN profile
9. TFLM MLP profile
10. TFLM CNN profile

Speedup in tables is **TFLM mean ÷ NETKIT mean** (e.g. `5× faster` = NETKIT is five times faster).

## Run individually

```bash
make -C benchmark/netkit run              # netkit interpreter MLP + CNN (reference)
make -C benchmark/tflm run                # TFLM interpreter MLP + CNN (needs GNU make)
make -C benchmark/netkit run-aot          # optional: lowered AOT only (not in compare.sh)
make -C benchmark/netkit run-mlp-profile  # NETKIT MLP per-layer profile
make -C benchmark/netkit run-cnn-profile  # NETKIT CNN per-layer profile
make -C benchmark/tflm run-mlp-profile    # TFLM MLP per-op MicroProfiler breakdown
make -C benchmark/tflm run-cnn-profile    # TFLM CNN per-op MicroProfiler breakdown
```

Or per model/backend:

```bash
make -C benchmark/netkit run-mlp
make -C benchmark/netkit run-mlp-cmsis
make -C benchmark/tflm run-mlp
make -C benchmark/netkit run-cnn
make -C benchmark/netkit run-cnn-cmsis
make -C benchmark/tflm run-cnn
```

## MobileNetV4-small (depthwise-heavy, host only)

`models/mobilenetv4_small.nk` (CNN, input **56×56×3**, 10 classes, 12 UIB blocks) is a larger, **depthwise-heavy** float32 model used to exercise `ConvDepthwiseForward`. It is host/CPU only (the model + activations are too large for the target MCU). It reuses the 10 embedded MNIST images (one per class), upsampled 28×28×1 → 56×56×3, and times **every** invoke over 10 images × 30 loops (300 invokes), reporting the cold first invoke plus warm median/min/mean/stddev.

```bash
make -C benchmark/netkit run-mobilenetv4         # netkit float32 reference kernels
make -C benchmark/netkit run-mobilenetv4-int8     # netkit int8 reference kernels
make -C benchmark/tflm run-mobilenetv4           # TFLM float32 (after export-mobilenetv4)
make -C benchmark/tflm run-mobilenetv4-int8        # TFLM int8 (after export-mobilenetv4-int8)
./tools/run_mobilenetv4_4way_benchmark.sh        # all four, writes benchmark/mobilenetv4_4way_results.txt
```

Int8 netkit loads `models/mobilenetv4_small_int8.nk` (UIB composite quant op, per-tensor symmetric int8). Int8 TFLM uses `generated/mobilenetv4_small_int8.tflite` with prequantized `mobilenetv4_int8_test_images.{h,cc}`.

This benchmark is **not** part of `compare.sh` / `run-all` (separate MobileNetV4 harness) and accuracy is not meaningful (fixture weights) — it measures invoke latency only.

## ImageNet MobileNetV4 (pretrained, host / MPU)

Pretrained `mobilenetv4_conv_small` (224×224×3, 1000 classes) compared across netkit, TFLM, and TF Lite / LiteRT on the same 10 ImageNet images.

**Compiler policy:** netkit ImageNet targets use `BENCH_FLAG_PROFILE=tflite` to mimic LiteRT's opt wheel (`-O3 -DNDEBUG`, host `gcc`/`g++`, no TFLM Micro `-fno-exceptions` / disabled-NEON flags).

```bash
./tools/fetch_xnnpack.sh   # once
make -C benchmark/netkit run-mobilenetv4-imagenet-xnnpack   # netkit XNNPACK, tflite flags
make -C benchmark/tflite run-mobilenetv4-imagenet           # TF Lite XNNPACK, 1 thread
make -C benchmark/tflite run-mobilenetv4-imagenet-ref       # TF Lite builtin-ref
make -C benchmark/tflm run-mobilenetv4-imagenet             # TFLM (still TFLM -O2 host flags)
```

### ImageNet int8

Same 10 images, int8 end-to-end. Quantize/dequant is **Python-only** (export / offline); C++ and interpreters feed/consume int8 and ArgMax on int8 logits.

| Runtime | Target | Notes |
|---------|--------|-------|
| netkit | `make -C benchmark/netkit run-mobilenetv4-imagenet-int8` | XNNPACK qs8 default; fixtures from `--quant-source nk` |
| TFLM | `make -C benchmark/tflm run-mobilenetv4-imagenet-int8` | Host builtin int8 kernels (no XNNPACK) |
| TF Lite | `make -C benchmark/tflite run-mobilenetv4-imagenet-int8` | LiteRT XNNPACK; quantize inputs in the Python bench |

Export: `make -C benchmark/tflm export-mobilenetv4-imagenet-int8` (TFLite + TFLM fixtures) and `python3 tools/write_mobilenetv4_imagenet_int8.py` (`.nk`). Netkit fixtures: `python3 benchmark/tflm/tools/export_imagenet_mnv4_int8_test_images.py --quant-source nk`.

Recent order-averaged A/B (XNNPACK ON/OFF, float32 + int8): [docs/STATUS.md](../docs/STATUS.md).

## Depthwise MNIST CNN (host XNNPACK peer)

Separable PW/DW tutorial CNN (`models/mnist_cnn_dw.nk` / `_int8.nk`) vs TF Lite:

```bash
make -C benchmark/netkit run-cnn-dw-xnnpack
make -C benchmark/netkit run-cnn-dw-int8-xnnpack
make -C benchmark/tflite run-cnn-dw
make -C benchmark/tflite run-cnn-dw-int8
```

## Models

| Model | netkit file | Architecture |
|-------|-------------|--------------|
| MLP | `models/mnist_mlp.nk` | 784 → 128 ReLU → 10 softmax |
| CNN | `models/mnist_cnn.nk` | Conv32/Pool/Conv64/Pool/Flatten/Dense128/Dense10 |
| CNN int8 | `models/mnist_cnn_int8.nk` | Same topology, int8 weights + quant params |
| CNN DW | `models/mnist_cnn_dw.nk` | PW32→DW32→Pool→PW64→DW64→Pool→Dense128→Dense10 |
| CNN DW int8 | `models/mnist_cnn_dw_int8.nk` | Same separable topology, int8 |

Shared test vectors: `benchmark/tflm/generated/mnist_*_test_images.{h,cc}`

Each binary prints machine-readable summary lines:

```
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=reference mean_us=... runs=100
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=reference mean_us=... runs=100
BENCHMARK_SUMMARY runtime=tflm model=mlp backend=reference mean_us=... runs=100
PROFILE_SUMMARY runtime=netkit model=cnn kind=op tag=Conv2D mean_us=... pct=...
PROFILE_SUMMARY runtime=tflm model=cnn kind=op tag=Conv2D mean_us=... pct=...
```

Re-render PNG tables from a saved log:

```bash
python3 benchmark/tools/render_benchmark_tables.py --log /path/to/compare.log --out-dir benchmark
```

## Interpreting results (host vs MCU)

These benchmarks run on the **host desktop** (macOS/Linux), not on Cortex-M firmware.

| Label in compare output | What it actually is on host |
|-------------------------|----------------------------|
| NETKIT (reference) | Reference kernels only |
| NETKIT (xnnpack) | XNNPACK LayerFast when enabled |
| TFLM reference | TFLM reference kernels (not CMSIS-NN on desktop) |

**CMSIS-NN is only enabled on MCU + Cortex-M builds** (`NETKIT_CMSIS_NN_ALLOWED`). On the host, TFLM does not use CMSIS-NN either — both stacks exercise portable reference conv paths. The large CNN gap on Apple Silicon is therefore **not** a direct preview of Cortex-M4F ratios: on MCU, TFLM typically links CMSIS-NN while netkit can use CMSIS-NN for conv/pool/FC when the case is supported.

Use these numbers for **relative regression tracking on the same machine** and for **per-op breakdown** (where Conv2D dominates the CNN gap on host). For firmware SLA, re-run on the target board or an cycle-accurate simulator with the intended `NETKIT_TARGET` and CMSIS flags.

See also [docs/KERNELS.md](../docs/KERNELS.md) for reference conv optimizations (HWIO repack, input-stationary, im2col) that apply when CMSIS-NN is unavailable or falls back.

## On-device MCU benchmarks

Host `compare.sh` numbers are not a direct preview of MCU ratios. Index of all UART logs:
[`mcu_ab_logs/README.md`](mcu_ab_logs/README.md). Gallery: [root README](../README.md#peer-benchmarks-mcu--mpu--cpu).

### NUCLEO-F446RE

**Canonical three-way A/B** (netkit vs TFLM vs microTVM): [`mcu_ab_logs/mcu_int8_ab_results.txt`](mcu_ab_logs/mcu_int8_ab_results.txt) and [docs/STATUS.md](../docs/STATUS.md#mcu-nucleo-f446re).

### Int8 latency (10×10, discard first invoke; all 10/10)

netkit = interpreter embed (`NETKIT_EMBED=1`). Matched −O2/−flto toolchain. No XNNPACK on MCU.

**CMSIS-NN**

| Model | netkit | TFLM | microTVM |
|-------|-------:|-----:|---------:|
| MNIST CNN | **95.3 ms** | 95.5 ms | 112.3 ms |
| MNIST DS-CNN | **58.3 ms** | 61.4 ms | 86.4 ms |

**Reference (CMSIS-NN off)**

| Model | netkit | TFLM | microTVM |
|-------|-------:|-----:|---------:|
| MNIST CNN | **336.2 ms** | 2593.5 ms | 343.0 ms |
| MNIST DS-CNN | **140.3 ms** | 826.8 ms | 236.0 ms |

| Firmware | Model | Role |
|----------|-------|------|
| [nucleo-f446re-cnn-int8](../boards/nucleo-f446re-cnn-int8/README.md) | MNIST CNN int8 | netkit |
| [nucleo-f446re-tflm-cnn-int8](../boards/nucleo-f446re-tflm-cnn-int8/README.md) | MNIST CNN int8 | TFLM |
| [nucleo-f446re-tvm-cnn-int8](../boards/nucleo-f446re-tvm-cnn-int8/README.md) | MNIST CNN int8 | microTVM AOT |
| [nucleo-f446re-cnn-dw-int8](../boards/nucleo-f446re-cnn-dw-int8/README.md) | MNIST DS-CNN int8 | netkit |
| [nucleo-f446re-tflm-cnn-dw-int8](../boards/nucleo-f446re-tflm-cnn-dw-int8/README.md) | MNIST DS-CNN int8 | TFLM |
| [nucleo-f446re-tvm-cnn-dw-int8](../boards/nucleo-f446re-tvm-cnn-dw-int8/README.md) | MNIST DS-CNN int8 | microTVM AOT |
| [nucleo-f446re](../boards/nucleo-f446re/README.md) | MNIST MLP f32 | CMSIS-NN / reference lowered AOT |

Shared **float** test vectors: `benchmark/tflm/generated/mnist_*_test_images.{h,cc}`

**Int8** benchmarks use separate prequantized vectors: `benchmark/tflm/generated/mnist_cnn_int8_test_images.{h,cc}` (export via `export_int8_test_images.py`; quant params from TFLite input tensor). No float→int8 conversion at test time.

```bash
make export-mnist-cnn-int8
python3 benchmark/tflm/tools/export_int8_test_images.py
make -C boards/nucleo-f446re-cnn-int8 NETKIT_EMBED=1 && cd boards/nucleo-f446re-cnn-int8 && ./scripts/flash.sh
```

Board default `make` is **quant lowered**. Use `NETKIT_EMBED=1` for the TFLM-fair interpreter numbers above.

```bash
./scripts/monitor.sh   # press RESET
```

MCU firmware prints **raw int8 softmax** only (`pred_i8`, `out_i8=...` on `DIGIT_SUMMARY` lines). Dequantized confidence for netkit vs TFLM comparison is computed offline:

```bash
python3 benchmark/tools/parse_mcu_cnn_int8_log.py uart.log
python3 benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit_uart.log tflm_uart.log
```

Uses TFLite int8 softmax output spec (`scale=1/256`, `zero_point=-128`).

### Seeed XIAO ESP32C3 / ESP32-P4-Function-EV

Espressif MCU peers vs TFLM (matched `-O3` C++). Canonical logs and tables:

| Board | Results |
|-------|---------|
| XIAO ESP32C3 | [`mcu_ab_logs/xiao_esp32c3/`](mcu_ab_logs/xiao_esp32c3/) · [STATUS](../docs/STATUS.md#mcu-seeed-xiao-esp32c3) |
| XIAO ESP32-S3 | [`mcu_ab_logs/xiao_esp32s3/`](mcu_ab_logs/xiao_esp32s3/) (int8 ESP-NN / int8 ref / float32) · [STATUS](../docs/STATUS.md#mcu-seeed-xiao-esp32s3) |
| ESP32-P4-Function-EV | [`mcu_ab_logs/esp32_p4_ev/`](mcu_ab_logs/esp32_p4_ev/) (int8 ESP-NN / int8 ref / float32) · [STATUS](../docs/STATUS.md#mcu-espressif-esp32-p4-function-ev) |

Float embed bug on **P4 and S3**: [KNOWN_ISSUES KI-001](../docs/KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device).

## Layout

```
benchmark/
  compare.sh                 # full comparison + PNG export
  tools/
    render_benchmark_tables.py
    export_aot_assets.py
    parse_mcu_cnn_int8_log.py   # offline DIGIT_SUMMARY → dequantized confidence
  common/                    # shared stats + profile headers + host flags
  netkit/                    # netkit bench Makefile + mains
  tflite/                    # LiteRT / TF Lite Python peers (ImageNet + MNIST XNNPACK)
  tflm/                      # TFLM wrapper (clone via tools/fetch_tflm.sh)
  mnist_*_comparison.png     # generated by compare.sh (committed as samples)
```

TFLM setup: [tflm/README.md](tflm/README.md). Peer-result snapshot: [docs/STATUS.md](../docs/STATUS.md).
