# netkit vs TFLM MNIST benchmarks

Side-by-side invoke latency on the same 10 MNIST test vectors per model (from TCAS cases in `models/mnist_*.nk`): **one correctly-classified test image per digit 0–9**, sorted by label.

**Fair comparison policy:** benchmarks use the **interpreter** path only — netkit loads `.nk` models via `NkLoader` and calls `forward()` / `forward_quantized()`; TFLM uses `MicroInterpreter::Invoke()`. **AOT lowered firmware is not included** in `compare.sh` or `make -C benchmark/netkit run-all` (use `run-aot*` targets separately for deployment profiling).

**Kernel defaults:** float Conv2D strategy is the single `NETKIT_IM2COL` knob (`0` = direct, `1` = partial im2col, `2` = full im2col + GEMM). The host/CPU benchmark uses the default **`0` (direct loops)**; override with `NETKIT_IM2COL=1` (partial) or `NETKIT_IM2COL=2` (full). `NETKIT_LOOP_UNROLL` stays **off** (`0`) in all benchmark builds. Int8 quantized inference uses CMSIS-NN kernels, not float im2col.

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

| Peer | Profile file | Opt | SIMD / NEON |
|------|--------------|-----|-------------|
| **TFLM Micro** (MNIST `compare.sh`) | `benchmark/common/tflm_host_flags.mk` | `-O2` | `TF_LITE_DISABLE_X86_NEON` (TFLM host default) |
| **TF Lite / LiteRT** (ImageNet MPU) | `benchmark/common/tflite_host_flags.mk` | `-O3 -DNDEBUG` | enabled (matches LiteRT `darwin_*-opt` / XNNPACK Release) |

Shared across both profiles:

| Setting | Value |
|---------|--------|
| Compiler | host `c++` / `cc` (Apple Clang or GCC) |
| Link | `-lm` (+ XNNPACK libs when enabled) |
| Exceptions / RTTI | `-fno-exceptions -fno-rtti` |

ImageNet targets pass `BENCH_FLAG_PROFILE=tflite` so netkit is not accidentally compared under TFLM's `-O2` + disabled-NEON settings. MNIST / `compare.sh` keep the TFLM profile.

The only intentional C++ dialect deviation is `-std=c++20` instead of TFLM's `-std=c++17`, because netkit's runtime API uses `std::span`.

The main repo `libnetkit.a` (clang, debug) is **not** used by these benchmarks.

## Run comparison (recommended)

```bash
make export-mnist export-mnist-cnn   # if models not present
make -C benchmark/tflm export-assets # once, or after retraining
./tools/fetch_cmsis_dsp.sh           # once, for CMSIS-DSP netkit variant

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
3. NETKIT MLP CMSIS-DSP
4. NETKIT CNN CMSIS-DSP
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
make -C benchmark/netkit run-cmsis        # netkit interpreter MLP + CNN (CMSIS-DSP)
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
make -C benchmark/netkit run-mobilenetv4-cmsis   # netkit float32 CMSIS-DSP
make -C benchmark/tflm run-mobilenetv4           # TFLM float32 (after export-mobilenetv4)
make -C benchmark/tflm run-mobilenetv4-int8        # TFLM int8 (after export-mobilenetv4-int8)
./tools/run_mobilenetv4_4way_benchmark.sh        # all four, writes benchmark/mobilenetv4_4way_results.txt
```

Int8 netkit loads `models/mobilenetv4_small_int8.nk` (UIB composite quant op, per-tensor symmetric int8). Int8 TFLM uses `generated/mobilenetv4_small_int8.tflite` with prequantized `mobilenetv4_int8_test_images.{h,cc}`.

This benchmark is **not** part of `compare.sh` / `run-all` (separate MobileNetV4 harness) and accuracy is not meaningful (fixture weights) — it measures invoke latency only.

## ImageNet MobileNetV4 (pretrained, host / MPU)

Pretrained `mobilenetv4_conv_small` (224×224×3, 1000 classes) compared across netkit, TFLM, and TF Lite / LiteRT on the same 10 ImageNet images.

**Compiler policy:** netkit ImageNet targets use `BENCH_FLAG_PROFILE=tflite` (`-O3 -DNDEBUG`, SIMD enabled) to match LiteRT's opt build — **not** the TFLM Micro `-O2` + `TF_LITE_DISABLE_X86_NEON` profile.

```bash
./tools/fetch_xnnpack.sh   # once
make -C benchmark/netkit run-mobilenetv4-imagenet-xnnpack   # netkit XNNPACK, tflite flags
make -C benchmark/netkit run-mobilenetv4-imagenet-cmsis     # netkit CMSIS-DSP, tflite flags
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

## Models

| Model | netkit file | Architecture |
|-------|-------------|--------------|
| MLP | `models/mnist_mlp.nk` | 784 → 128 ReLU → 10 softmax |
| CNN | `models/mnist_cnn.nk` | Conv32/Pool/Conv64/Pool/Flatten/Dense128/Dense10 |
| CNN int8 | `models/mnist_cnn_int8.nk` | Same topology, int8 weights + quant params |

Shared test vectors: `benchmark/tflm/generated/mnist_*_test_images.{h,cc}`

Each binary prints machine-readable summary lines:

```
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=reference mean_us=... runs=100
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=cmsis-dsp mean_us=... runs=100
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
| NETKIT (without CMSIS-DSP) | Reference kernels only |
| NETKIT (with CMSIS-DSP) | CMSIS-DSP vector ops + reference layer ops |
| TFLM reference | TFLM reference kernels (not CMSIS-NN on desktop) |

**CMSIS-NN is only enabled on MCU + Cortex-M builds** (`NETKIT_CMSIS_NN_ALLOWED`). On the host, TFLM does not use CMSIS-NN either — both stacks exercise portable reference conv paths. The large CNN gap on Apple Silicon is therefore **not** a direct preview of Cortex-M4F ratios: on MCU, TFLM typically links CMSIS-NN while netkit can use CMSIS-NN for conv/pool/FC when the case is supported.

Use these numbers for **relative regression tracking on the same machine** and for **per-op breakdown** (where Conv2D dominates the CNN gap on host). For firmware SLA, re-run on the target board or an cycle-accurate simulator with the intended `NETKIT_TARGET` and CMSIS flags.

See also [docs/KERNELS.md](../docs/KERNELS.md) for reference conv optimizations (HWIO repack, input-stationary, im2col) that apply when CMSIS-NN is unavailable or falls back.

## On-device MCU benchmarks (NUCLEO-F446RE)

Host `compare.sh` numbers are not a direct preview of Cortex-M ratios. For firmware SLA, use the board firmware targets:

| Firmware | Model | Backend | Notes |
|----------|-------|---------|-------|
| [boards/nucleo-f446re](../boards/nucleo-f446re/README.md) | MNIST MLP f32 | CMSIS-DSP lowered AOT | ~10.7 ms, 10/10 |
| [boards/nucleo-f446re-cnn-int8](../boards/nucleo-f446re-cnn-int8/README.md) | MNIST CNN int8 | CMSIS-NN interpreter embed | ~95 ms (94.9–97.0 ms typical), 10/10; 64 KiB arena, ~334 KiB flash / ~75 KiB SRAM |
| [boards/nucleo-f446re-tflm-cnn-int8](../boards/nucleo-f446re-tflm-cnn-int8/README.md) | MNIST CNN int8 | TFLite Micro | comparison baseline |

Shared **float** test vectors: `benchmark/tflm/generated/mnist_*_test_images.{h,cc}`

**Int8** benchmarks use separate prequantized vectors: `benchmark/tflm/generated/mnist_cnn_int8_test_images.{h,cc}` (export via `export_int8_test_images.py`; quant params from TFLite input tensor). No float→int8 conversion at test time.

```bash
make export-mnist-cnn-int8
python3 benchmark/tflm/tools/export_int8_test_images.py
make -C boards/nucleo-f446re-cnn-int8 && cd boards/nucleo-f446re-cnn-int8 && ./scripts/flash.sh
```

Default `make` builds **interpreter embed** (`--no-lower`). For quant lowered deployment profiling: `make NETKIT_LOWERED=1`.

```bash
./scripts/monitor.sh   # press RESET
```

MCU firmware prints **raw int8 softmax** only (`pred_i8`, `out_i8=...` on `DIGIT_SUMMARY` lines). Dequantized confidence for netkit vs TFLM comparison is computed offline:

```bash
python3 benchmark/tools/parse_mcu_cnn_int8_log.py uart.log
python3 benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit_uart.log tflm_uart.log
```

Uses TFLite int8 softmax output spec (`scale=1/256`, `zero_point=-128`).

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
  tflm/                      # TFLM wrapper (clone via tools/fetch_tflm.sh)
  mnist_*_comparison.png     # generated by compare.sh (committed as samples)
```

TFLM setup: [tflm/README.md](tflm/README.md).
