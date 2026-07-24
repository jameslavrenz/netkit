# Platform and dtype status

Snapshot of what works today, what was measured, and what is still open. Companion to [PHILOSOPHY.md](PHILOSOPHY.md), [BUILD_TARGETS.md](BUILD_TARGETS.md), [PLATFORMS.md](PLATFORMS.md) (per-device setup), [DATATYPES.md](DATATYPES.md), and **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** (bugs / anomalies / deferred).

## Dtypes

| Dtype | Status | Notes |
|-------|--------|-------|
| **float32** | **Complete** | Default path on cpu / MCU / MPU; ImageNet MobileNetV4, MNIST CNN / DS-CNN / MLP, fused blocks |
| **int8** | **Complete** | End-to-end int8 I/O (no C++ float↔int8); MNIST CNN / DS-CNN / MLP Arm MCU (CMSIS-NN); Espressif MCU via ESP-NN; RISC-V MCU via NMSIS-NN; host/MPU via XNNPACK qs8 or QuantOps reference; ImageNet MNv4 int8 |
| float16 / int16 / int4 | Planned (Phase 2) | See [DATATYPES.md](DATATYPES.md) |

## Target maturity

| Target | Role | Kernels | Maturity |
|--------|------|---------|----------|
| **cpu** | Desktop / CI / peer benches | XNNPACK (default); reference fallback; **no MLAS** | **Done** — float32 + int8 |
| **mcu_arm** | Arm Cortex-M firmware | CMSIS-NN (int8 production); float32 via reference; XNNPACK **forbidden** | **Done** — float32 + int8 (NUCLEO-F446RE) |
| **mpu_arm** | Arm Cortex-A / RTOS-class | XNNPACK (default); CMSIS-NN off | **Done** — float32 + int8 |
| **mpu_risc** | RISC-V MPU | XNNPACK (default); CMSIS-NN **forbidden** | **Done** — float32 + int8; same XNNPACK LayerFast stack as other MPUs (XNNPACK has strong RISC-V MPU support) |
| **mcu_risc** | Non-Espressif RISC-V MCU (Nuclei / RV32) | NMSIS-NN (int8 production); float32 via reference (NMSIS-NN has no float API); CMSIS + XNNPACK + ESP-NN **forbidden** | **Done** — float32 + int8 runtime (host smoke via `NETKIT_HOST_SMOKE=1`); on-device peer benches TBD |
| **mcu_esp** | Espressif MCU — Xtensa **and** RISC-V (ESP32 / S3 / C3 / C6 / P4) | ESP-NN (int8 production); float32 via reference (ESP-NN has no float API); XNNPACK **forbidden** | **Done** — float32 + int8 runtime; on-device peers: [XIAO ESP32C3](../boards/xiao-esp32c3/README.md) int8 · [XIAO ESP32-S3](../boards/xiao-esp32s3/README.md) int8+float32 · [ESP32-P4-Function-EV](../boards/esp32-p4-function-ev/README.md) int8+float32 vs TFLM (matched `-O3`); ImageNet skipped — C3: [int8](../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ab_results.txt) · S3: [all rounds](../benchmark/mcu_ab_logs/xiao_esp32s3/esp32s3_all_ab_results.txt) · P4: [all rounds](../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_all_ab_results.txt) |

**Policy reminder:** XNNPACK is default on cpu and all MPUs, never on MCU. CMSIS-NN is Arm MCU only (production int8). ESP-NN is **all Espressif MCUs** (production int8) — including RISC-V C3/C6/P4; do not use `mcu_risc` for ESP32*. NMSIS-NN is **non-Espressif** RISC-V MCU only (production int8; same CMSIS-style Try* / plan wiring). Targets follow vendor + backend, not ISA alone — [PLATFORMS.md — Target ≠ CPU ISA](PLATFORMS.md#target--cpu-isa). **CMSIS-DSP is not used.** Float32 on MCU uses reference kernels (CMSIS float LayerFast exists on Arm; ESP-NN / NMSIS-NN float Try* always miss). `NETKIT_IM2COL` defaults to **0** on all targets (see [BUILD_TARGETS.md](BUILD_TARGETS.md#netkit_im2col-guidance)).

## Host file mmap

| Host OS | Status | Implementation |
|---------|--------|----------------|
| **macOS / Linux** | **Complete** | POSIX `mmap` (`MAP_PRIVATE`) |
| **Windows** | **Complete** | Win32 `CreateFileMapping` / `MapViewOfFile` (`FILE_MAP_COPY`) |
| **MCU** | **Forbidden** | Use `Load*FromBuffer` / flash; `NETKIT_MMAP=1` is forced off |

Default **on** for cpu + any MPU; opt out with `NETKIT_MMAP=0` on RTOS / bare-metal MPU. See [ARENA.md](ARENA.md) and [BUILD_TARGETS.md](BUILD_TARGETS.md).

## Peer highlights (wins)

Fair A/B across **MCU · MPU · CPU**. Peers: **TFLM** (MCU), **microTVM** (NUCLEO), **TF Lite** (Pi Zero + host), **ONNX Runtime** (host). One-panel LinkedIn card: [`benchmark/linkedin/netkit_linkedin_wins.png`](../benchmark/linkedin/netkit_linkedin_wins.png) (regenerate: `benchmark/linkedin/render_wins_summary.py`).

| Story | Result |
|-------|--------|
| **MCU int8 reference** (QuantOps vs TFLM ref) | Up to **9.9×** faster — ESP32-S3 MNIST CNN **112 ms** vs **1113 ms**. Also NUCLEO **7.7×**, P4 **6.3×**, C3 **5.3×** (CNN). |
| **MCU vs microTVM** (NUCLEO CMSIS-NN) | Faster than microTVM AOT — DS-CNN **58.3 ms** vs **86.4 ms** (**1.5×**); CNN **95.3** vs **112.3 ms**. |
| **MCU vendor NN** (CMSIS-NN / ESP-NN) | **Parity** with TFLM when both use the same accel (~**1.00–1.05×**). |
| **MCU float32** (lowered AOT; ESP-NN has no float API) | S3 DS-CNN **2.6×**, S3/P4 CNN **1.7×** vs TFLM (all 10/10). ImageNet skipped on Espressif MCU (flash). |
| **MPU Pi Zero 2 W vs TF Lite** | XNNPACK ≈ parity; int8 reference up to **2.7×** (CNN / DS-CNN / MobileNetV4-Small). |
| **CPU Apple M4 vs TF Lite / ORT** | XNNPACK ON: ≈ TF Lite; beats ORT XNNPACK EP on all six (up to **4.9×** float / **2.5×** int8). Binary ~**9×** smaller than LiteRT (~1.3 vs ~12.4 MiB). |

## Host three-way suite (netkit vs TF Lite vs ONNX Runtime)

Fair CPU peer suite on host (Apple Silicon). **Not TFLM** — TF Lite / LiteRT Python benches + ONNX Runtime on matching model assets. **TVM is not a host/CPU peer**; use TVM only for MCU / microTVM.

**CMSIS-NN is MCU-only.** Host accel ON/OFF:

| Mode | netkit | TF Lite | ONNX Runtime |
|------|--------|---------|--------------|
| **xnn** (accel ON) | XNNPACK | XNNPACK | `XNNPACKExecutionProvider` |
| **ref** (accel OFF) | reference | `BUILTIN_REF` (intentionally slow) | `CPUExecutionProvider` (**MLAS**) |

**Models:** MNIST CNN, MNIST DS-CNN, MobileNetV4-Conv-Small ImageNet — **float32 and int8**.

Fairness: LiteRT-matched flags for netkit (`BENCH_FLAG_PROFILE=tflite`); prebuild; discard 1st process; order swaps `nk→tf→ort` / `ort→tf→nk`; 1 thread; `NETKIT_IM2COL=0`. Same host drivers: `gcc`/`g++` (Darwin = Apple clang). **ORT** is built from source with those same flags plus `--use_xnnpack` (`./tools/build_onnxruntime_litert_matched.sh`); stock pip wheels lack the XNNPACK EP. TF Lite remains the published LiteRT wheel. ORT int8 models are QDQ (float graph I/O).

```bash
./tools/build_onnxruntime_litert_matched.sh          # once
python3 benchmark/tools/export_host_onnx_assets.py --all
python3 benchmark/tools/run_host_ab_suite_float32.py
python3 benchmark/tools/run_host_ab_suite_int8.py
```

Results: `benchmark/host_ab_suite_results_{int8,float32}.txt`. ORT harness: [`benchmark/onnxruntime/`](../benchmark/onnxruntime/).

### Results (host Apple Silicon, Jul 2026)

Absolute warm latency (µs). MNIST = `mean_us`; ImageNet = `warm_mean_us`. Full tables: `benchmark/host_ab_suite_results_{float32,int8}.txt`.

#### FLOAT32

| model | mode | netkit | TF Lite | ORT | TF÷nk | ORT÷nk |
|-------|------|-------:|--------:|----:|------:|-------:|
| MNIST CNN | xnn | 30.5 | 50.0 | 149.9 | 1.64× | 4.91× |
| MNIST DS-CNN | xnn | 28.6 | 32.3 | 44.0 | 1.13× | 1.54× |
| MobileNetV4-Small ImageNet | xnn | 1059 | 1083 | 4989 | 1.02× | 4.71× |
| MNIST CNN | ref | 483 | 1061 | 84.1 | 2.20× | 0.17× |
| MNIST DS-CNN | ref | 298 | 428 | 78.2 | 1.44× | 0.26× |
| MobileNetV4-Small ImageNet | ref | 32069 | 61587 | 7444 | 1.92× | 0.23× |

#### INT8

| model | mode | netkit | TF Lite | ORT | TF÷nk | ORT÷nk |
|-------|------|-------:|--------:|----:|------:|-------:|
| MNIST CNN | xnn | 28.8 | 20.5 | 36.0 | 0.71× | 1.25× |
| MNIST DS-CNN | xnn | 22.3 | 21.4 | 25.4 | 0.96× | 1.14× |
| MobileNetV4-Small ImageNet | xnn | 670 | 691 | 1681 | 1.03× | 2.51× |
| MNIST CNN | ref | 137 | 546 | 37.0 | 3.98× | 0.27× |
| MNIST DS-CNN | ref | 205 | 450 | 31.8 | 2.20× | 0.16× |
| MobileNetV4-Small ImageNet | ref | 7420 | 28315 | 1635 | 3.82× | 0.22× |

**Flash/RAM** (runtime image only): netkit XNN ~1.3 MiB / 192 KiB; LiteRT ~12.4 MiB / 752 KiB; ORT ~28 MiB / 160 KiB. netkit reference ~200 KiB / 16 KiB.

**Takeaways (Jul 2026 host Apple Silicon):**

- **XNNPACK ON (production peer):** netkit ≈ TF Lite on all six models; netkit beats ORT XNNPACK EP on all six (about 1.1–4.9×). TF Lite edges MNIST CNN int8 clearly and DS-CNN int8 by a hair; netkit leads or ties the rest. This is the column that matters for host shipping.
- **XNNPACK OFF — TF Lite:** OFF uses `BUILTIN_REF` (TF Lite’s slowest CPU path), not optimized builtins. So that comparison is netkit reference vs TF Lite’s slow reference — netkit wins all six. TF Lite’s optimized CPU kernels also route through **XNNPACK**, so enabling them is not a distinct peer test from XNNPACK ON (same as ORT MLAS becoming moot under XNNPACK).
- **XNNPACK OFF — ORT:** ORT never drops to a slow reference; OFF still runs **MLAS** (`CPUExecutionProvider`) and stays faster than netkit reference on all six. That is a separate optimized CPU stack, not an apples-to-apples “ref” peer.
- **With XNNPACK ON, TF Lite optimized builtins and ORT MLAS are moot.** Host production for netkit and TF Lite is XNNPACK; ORT’s competitive XNNPACK path is already covered (and slower here). **MLAS is not needed for netkit** — integrating it would only chase the non-production OFF column.
- ORT int8 assets are QDQ (float graph I/O).

### MPU — Raspberry Pi Zero 2 W (aarch64, Jul 2026)

Same fairness policy as the host suite (discard 1st process, order swaps, 1 thread). Cross-built on host (`tools/build_mpu_pi_aarch64.sh`), lean payload over SSH (`tools/run_mpu_pi_{float32,int8}_ab.sh`). Setup guide: [`boards/pi-zero-2w/README.md`](../boards/pi-zero-2w/README.md). Raw logs (local): `benchmark/host_ab_suite_results_{float32,int8}_pi_zero2w.txt`.

**FLOAT32** (order-avg latency; TF÷nk):

| model | XNNPACK | netkit | TF Lite | TF÷nk |
|-------|---------|--------|---------|-------|
| MNIST CNN | ON | 1.66 ms | 1.78 ms | 1.07× |
| MNIST DS-CNN | ON | 1.22 ms | 1.33 ms | 1.09× |
| MobileNetV4-Small ImageNet | ON | 100.0 ms | 100.7 ms | 1.01× |
| MNIST CNN | OFF | 14.4 ms | 22.3 ms | 1.55× |
| MNIST DS-CNN | OFF | 7.7 ms | 12.1 ms | 1.57× |
| MobileNetV4-Small ImageNet | OFF | 1056 ms | 1342 ms | 1.27× |

ImageNet float32 top-1: **9/10** both runtimes.

**INT8** (order-avg latency; TF÷nk):

| model | XNNPACK | netkit | TF Lite | TF÷nk |
|-------|---------|--------|---------|-------|
| MNIST CNN | ON | 1.09 ms | 1.11 ms | 1.02× |
| MNIST DS-CNN | ON | 0.61 ms | 0.62 ms | 1.01× |
| MobileNetV4-Small ImageNet | ON | 70.1 ms | 70.0 ms | ~1.00× |
| MNIST CNN | OFF | 5.55 ms | 15.2 ms | 2.74× |
| MNIST DS-CNN | OFF | 5.29 ms | 13.7 ms | 2.59× |
| MobileNetV4-Small ImageNet | OFF | 348 ms | 783 ms | 2.25× |

MNIST int8 top-1: **10/10**. ImageNet int8 XNNPACK: **8/10** both. ImageNet int8 **reference**: netkit **7/10**, TF Lite **8/10** — the extra netkit miss is a **retrain-only** issue (weights / quant calibration), **deferred** (not a runtime parity bug; XNNPACK path already matches).

### `NETKIT_IM2COL` note (from earlier host sweep)

With XNNPACK ON, im2col does not move the needle (accelerated path ignores it). With XNNPACK OFF, **`NETKIT_IM2COL=1` (partial)** can give a **small** float CNN reference bump on MPU/cpu; **`2` (full)** was not a clear win. **Default and recommendation: leave `NETKIT_IM2COL=0`.** At most try `1` on MCU or reference-only MPU builds.

### MCU (NUCLEO-F446RE)

Canonical results file: [`benchmark/mcu_ab_logs/mcu_int8_ab_results.txt`](../benchmark/mcu_ab_logs/mcu_int8_ab_results.txt) (UART logs in the same directory; index: [`mcu_ab_logs/README.md`](../benchmark/mcu_ab_logs/README.md)). Methodology: 10×10; discard first invoke. Matched toolchain: `mcu_tflm_toolchain.mk` (−O2 CORE/KERNEL/THIRD_PARTY, −flto, shared linker). **netkit** numbers below are **interpreter embed** (`NETKIT_EMBED=1`). Gain = TFLM÷netkit. Flash/RAM after MCU **no-heap** reclaim.

**Latency — CMSIS-NN** (all 10/10, no XNNPACK):

| Model | netkit embed | TFLM | microTVM AOT |
|-------|-------------:|-----:|-------------:|
| MNIST CNN | **95.3 ms** | 95.5 ms | 112.3 ms |
| MNIST DS-CNN | **58.3 ms** | 61.4 ms | 86.4 ms |

**Latency — reference kernels** (CMSIS-NN off; all 10/10):

| Model | netkit embed | TFLM | microTVM C AOT |
|-------|-------------:|-----:|---------------:|
| MNIST CNN | **336.2 ms** | 2593.5 ms | 343.0 ms |
| MNIST DS-CNN | **140.3 ms** | 826.8 ms | 236.0 ms |

TFLM÷netkit gain (CMSIS): CNN **1.00×**, DS-CNN **1.05×**. Reference: CNN **7.71×**, DS-CNN **5.89×**. microTVM CMSIS path still keeps per-channel FC on C (conv/pool/softmax via CMSIS-NN).

**Flash / RAM** (`NETKIT_ARENA_KB=96`):

| Model | Mode | netkit flash | TFLM flash | flash gain | netkit RAM | TFLM RAM | RAM gain |
|-------|------|-------------:|-----------:|-----------:|-----------:|---------:|---------:|
| MNIST CNN | CMSIS-NN | **346.7 KiB** | 354.6 KiB | **1.02×** | 107.1 KiB | 126.1 KiB | **1.18×** |
| MNIST CNN | reference | **327.9 KiB** | 324.2 KiB | 0.99× | 107.1 KiB | 126.1 KiB | **1.18×** |
| MNIST DS-CNN | CMSIS-NN | **332.3 KiB** | 360.5 KiB | **1.09×** | 107.1 KiB | 126.1 KiB | **1.18×** |
| MNIST DS-CNN | reference | **313.6 KiB** | 317.9 KiB | **1.01×** | 107.1 KiB | 126.1 KiB | **1.18×** |

Boards: `nucleo-f446re-cnn-int8` / `nucleo-f446re-tflm-cnn-int8` / `nucleo-f446re-tvm-cnn-int8`; DS-CNN twins `*-cnn-dw-int8`. MLP int8 CMSIS ~3.4 ms (10/10). XNNPACK: **none** on MCU ELFs.

**MCU heap policy:** forbidden forever (`NETKIT_HEAP_ARENA` error; aborting `new`/`malloc`; static arena only).

**Float32 MNIST CNN / DS-CNN on this MCU:** deferred — models exceed 512 KiB flash (~850–911 KiB); on-device digit peers remain int8.

### MCU (Seeed XIAO ESP32C3)

Canonical results: [`esp32c3_int8_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ab_results.txt) (ESP-NN) · [`esp32c3_int8_ref_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ref_ab_results.txt) (reference) — index: [`mcu_ab_logs/README.md`](../benchmark/mcu_ab_logs/README.md). Chip: ESP32-C3 @ 160 MHz (RISC-V) · `NETKIT_TARGET=mcu_esp` + `NETKIT_ARCH=ESP32C3`. Methodology matches NUCLEO: 10×10, discard first invoke; order swaps `nk→tflm` / `tflm→nk`. Compiler match: netkit C++ uses the same speed flags as `esp-tflite-micro` (`-O3`, no RTTI/exceptions) — [`mcu_esp_tflm_match_compile.cmake`](../boards/xiao-esp32c3/mcu_esp_tflm_match_compile.cmake). ImageNet / MobileNetV4 skipped (weights exceed the factory app partition).

**Latency — ESP-NN** (all 10/10; no FPU — int8 path only for peer A/B; netkit = **interpreter embed**, arena 64 KiB CNN / 96 KiB DS-CNN):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | 252.0 ms | **251.4 ms** | 1.00× |
| MNIST DS-CNN | 87.7 ms | **87.5 ms** | 1.00× |

**Note:** quant lowered AOT should be faster than embed but measured a hair slower under ESP-NN — [KNOWN_ISSUES KI-002](KNOWN_ISSUES.md#ki-002--xiao-esp32c3-quant-lowered-aot-slower-than-interpreter-embed). Peer default stays interpreter embed.

**Latency — reference (ESP-NN off)** (all 10/10; netkit = embed + QuantOps):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **226.8 ms** | 1205.5 ms | 5.32× |
| MNIST DS-CNN | **85.8 ms** | 392.3 ms | 4.57× |

**Flash / RAM** (PlatformIO size check, ESP-NN embeds):

| Firmware | Flash | RAM (static) |
|----------|------:|-------------:|
| netkit CNN | 420 KiB | 75 KiB |
| netkit DS-CNN | 405 KiB | 107 KiB |
| TFLM CNN | 432 KiB | 106 KiB |
| TFLM DS-CNN | 433 KiB | 106 KiB |

Boards: [`xiao-esp32c3/`](../boards/xiao-esp32c3/README.md) index — `xiao-esp32c3-cnn-int8`, `xiao-esp32c3-cnn-dw-int8`, TFLM twins `xiao-esp32c3-tflm-*`. Runners: `run_esp_int8_ab.sh` / `run_esp_int8_ref_ab.sh`.

### MCU (Espressif ESP32-P4-Function-EV)

Canonical (all rounds): [`esp32_p4_ev_all_ab_results.txt`](../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_all_ab_results.txt) · per-round: [`int8 ESP-NN`](../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_int8_ab_results.txt) · [`int8 reference`](../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_int8_ref_ab_results.txt) · [`float32`](../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_float32_ab_results.txt). Chip: **ESP32-P4 @ 360 MHz** (RISC-V, **FPU**) · companion **ESP32-C6 on this kit is WiFi-only** (not the inference peer). `NETKIT_TARGET=mcu_esp`. Same 10×10 + order-swap methodology as C3/NUCLEO; matched `-O3` C++ via [`mcu_esp_tflm_match_compile.cmake`](../boards/esp32-p4-function-ev/mcu_esp_tflm_match_compile.cmake). **P4 PIE ESP-NN asm is disabled** under PlatformIO gas (portable/generic_opt instead). ImageNet skipped (flash). Index: [`mcu_ab_logs/README.md`](../benchmark/mcu_ab_logs/README.md).

**Round 1 — int8 ESP-NN portable** (all 10/10; netkit = interpreter embed):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **78.9 ms** | 79.3 ms | 1.00× |
| MNIST DS-CNN | **40.3 ms** | 41.1 ms | 1.02× |

**Round 2 — int8 reference (ESP-NN off)** (all 10/10; netkit = embed + QuantOps):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **77.1 ms** | 485.4 ms | 6.29× |
| MNIST DS-CNN | **39.6 ms** | 172.0 ms | 4.34× |

**Round 3 — float32 reference** (all 10/10; ESP-NN has no float API; netkit = **lowered AOT**):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **97.5 ms** | 166.4 ms | 1.71× |
| MNIST DS-CNN | **74.8 ms** | 102.6 ms | 1.37× |

**Known issue — float32 interpreter embed wrong on P4 and S3:** [KNOWN_ISSUES.md KI-001](KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device). Published float peers use **lowered AOT**, not `--no-lower` embed.

Boards: [`esp32-p4-function-ev/`](../boards/esp32-p4-function-ev/README.md) — float: `esp32-p4-function-ev-cnn`, `-cnn-dw`, TFLM twins; int8: `-cnn-int8`, `-cnn-dw-int8`, TFLM twins. Runners: `run_esp_int8_ab.sh` / `run_esp_int8_ref_ab.sh` / `run_esp_float32_ab.sh`. Gallery: [README.md — Peer benchmarks](../README.md#mcu--esp32-p4-function-ev--360-mhz-fpu).

### MCU (Seeed XIAO ESP32-S3)

Canonical (all rounds): [`esp32s3_all_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32s3/esp32s3_all_ab_results.txt) · board index: [`xiao-esp32s3/README.md`](../boards/xiao-esp32s3/README.md). Chip: **ESP32-S3 @ 240 MHz** (Xtensa, **FPU**) · bring-up unit: **16 MB PSRAM** / **32 MB OPI flash**. `NETKIT_TARGET=mcu_esp` + `NETKIT_ARCH=ESP32S3`. Same 10×10 methodology as C3/P4/NUCLEO; matched `-O3` C++ via [`mcu_esp_tflm_match_compile.cmake`](../boards/xiao-esp32s3/mcu_esp_tflm_match_compile.cmake). Default int8 path uses **ESP-NN S3 asm** (`NETKIT_ESP_NN_USE_S3_ASM=1`; plan scratch from `esp_nn_get_*_scratch_size`). ImageNet skipped. Flash via OpenOCD builtin JTAG (esptool hard-reset often lands in download mode).

**Round 1 — int8 ESP-NN S3 asm** (all 10/10; netkit = interpreter embed):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **34.7 ms** | 34.9 ms | 1.00× |
| MNIST DS-CNN | **31.4 ms** | 31.7 ms | 1.01× |

**Round 2 — int8 reference (ESP-NN off)** (all 10/10; netkit = embed + QuantOps):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **112.1 ms** | 1113.0 ms | 9.93× |
| MNIST DS-CNN | **64.3 ms** | 362.8 ms | 5.64× |

**Round 3 — float32 reference** (all 10/10; ESP-NN has no float API; netkit = **lowered AOT**):

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **308.2 ms** | 525.6 ms | 1.71× |
| MNIST DS-CNN | **63.4 ms** | 166.4 ms | 2.62× |

**Known issue — float32 interpreter embed wrong on S3 (same as P4):** [KNOWN_ISSUES.md KI-001](KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device). S3 `--no-lower` CNN float: **2/10**, many zero logits — [`cnn_f32_netkit_embed.txt`](../benchmark/mcu_ab_logs/xiao_esp32s3/cnn_f32_netkit_embed.txt). Published float peers use lowered AOT.

Boards: [`xiao-esp32s3/`](../boards/xiao-esp32s3/README.md) — int8: `-cnn-int8`, `-cnn-dw-int8`, TFLM twins; float: `-cnn`, `-cnn-dw`, TFLM twins. Gallery: [README.md — Peer benchmarks](../README.md#mcu--xiao-esp32s3--240-mhz-fpu).

### YOLOX detection (host CPU, float32)

`benchmark/host_ab_suite_results_yolox_f32.txt` — YOLOX MNv4-PAFPN 320 warm mean:

| Mode | netkit | TF Lite | TF÷nk |
|------|--------|---------|------:|
| XNNPACK ON | 18.0 ms | 18.3 ms | 1.02× |
| Reference | 119.7 ms | 260.9 ms | 2.18× |

## What “done” means here

- **Arm MCU / MPU:** production-oriented paths exist (CMSIS-NN on MCU; XNNPACK on MPU/cpu) with float32 and int8 models, benches, and docs.
- **Espressif MCU (`mcu_esp`):** **fully functional** for float32 + int8 — ESP-NN for int8 (CMSIS-style), float32 on reference; host ANSI smoke via `NETKIT_HOST_SMOKE=1`. On-device peers: **XIAO ESP32C3** int8 + **XIAO ESP32-S3** + **ESP32-P4-Function-EV** CNN/DS-CNN int8 **and float32** vs TFLM done (see above).
- **RISC MPU (`mpu_risc`):** **fully functional** for float32 + int8 via the same **XNNPACK** LayerFast stack as Arm MPU / cpu. XNNPACK has strong RISC-V coverage on MPU-class cores; CMSIS-NN is correctly unavailable.
- **RISC MCU (`mcu_risc`):** **fully functional** for float32 + int8 — **NMSIS-NN** for int8 (CMSIS-NN analogue for Nuclei / RISC-V MCU), float32 on reference; host smoke via `NETKIT_HOST_SMOKE=1`. On-device peer benches TBD. Override with `NETKIT_NMSIS_NN=0` for generic-only.

## Tests / CI summary

Canonical detail: [TESTING.md](TESTING.md). C ↔ C++ core runtime: [API_PARITY.md](API_PARITY.md).

| Suite | Command | Cases / scope | CI |
|-------|---------|---------------|----|
| C++ / C embedded regression | `make test` / `make test-cpp` / `make test-c` | **89** `.nk` TCAS cases (both languages) + fast Python | Yes (`CI` on push/PR) |
| Full Python | `make test-full` / `make test-python-full` | ONNX parity **82** + timm backbone pack/runtime + AOT | Manual (`gh workflow run test-full.yml`) |
| Embedded smoke matrix | `make test-embedded-smoke-matrix` | MCU/MPU host smoke (CMSIS / ESP-NN / NMSIS-NN) | Local only |
| Peer A/B (on-device) | board runners under `boards/` | NUCLEO · C3 · S3 · P4 · Pi Zero · host ORT | Not in CI (hardware) |

## Open / next

Full tracker with IDs, symptoms, and workarounds: **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**.

Highlights:

- **KI-001** — Espressif float32 interpreter embed ~2/10 on **P4 and S3** (float peers use lowered AOT)
- **KI-002** — XIAO ESP32C3 quant lowered slower than embed under ESP-NN
- **KI-003** — ESP32-P4 ESP-NN PIE asm disabled under PlatformIO gas
- **KI-004** — YOLOX detection accuracy (training / calibration)
- **KI-D04** — `mcu_risc` on-device peer A/B still TBD (S3 / KI-D03 done)
