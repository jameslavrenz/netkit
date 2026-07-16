# Platform and dtype status

Snapshot of what works today, what was measured, and what is still open. Companion to [PHILOSOPHY.md](PHILOSOPHY.md), [BUILD_TARGETS.md](BUILD_TARGETS.md), and [DATATYPES.md](DATATYPES.md).

## Dtypes

| Dtype | Status | Notes |
|-------|--------|-------|
| **float32** | **Complete** | Default path on cpu / MCU / MPU; ImageNet MobileNetV4, MNIST CNN/MLP, fused blocks |
| **int8** | **Complete** | End-to-end int8 I/O (no C++ float↔int8); MNIST CNN/MLP MCU (CMSIS-NN); host/MPU via XNNPACK qs8 or QuantOps reference; ImageNet MNv4 int8 |
| float16 / int16 / int4 | Planned (Phase 2) | See [DATATYPES.md](DATATYPES.md) |

## Target maturity

| Target | Role | Kernels | Maturity |
|--------|------|---------|----------|
| **cpu** | Desktop / CI / peer benches | XNNPACK (default); reference fallback | **Done** — float32 + int8 |
| **mcu_arm** | Arm Cortex-M firmware | CMSIS-NN (int8 production); float32 via reference; XNNPACK **forbidden** | **Done** — float32 + int8 (NUCLEO-F446RE) |
| **mpu_arm** | Arm Cortex-A / RTOS-class | XNNPACK (default); CMSIS-NN off | **Done** — float32 + int8 |
| **mpu_risc** | RISC-V MPU | XNNPACK (default); CMSIS-NN **forbidden** | **Done** — float32 + int8; same XNNPACK LayerFast stack as other MPUs (XNNPACK has strong RISC-V MPU support) |
| **mcu_risc** | RISC-V MCU | Generic / reference kernels only; CMSIS + XNNPACK **forbidden** | **Done** — float32 + int8 on **fast generic** kernels; a RISC MCU NN library (CMSIS-NN–class) is planned later |

**Policy reminder:** XNNPACK is default on cpu and all MPUs, never on MCU. CMSIS-NN is Arm MCU only (production int8). **CMSIS-DSP is not used.** Float32 on MCU uses reference kernels only (no optimized float32 MCU plan). `NETKIT_IM2COL` defaults to **0** on all targets (see [BUILD_TARGETS.md](BUILD_TARGETS.md#netkit_im2col-guidance)).

## Host file mmap

| Host OS | Status | Implementation |
|---------|--------|----------------|
| **macOS / Linux** | **Complete** | POSIX `mmap` (`MAP_PRIVATE`) |
| **Windows** | **Complete** | Win32 `CreateFileMapping` / `MapViewOfFile` (`FILE_MAP_COPY`) |
| **MCU** | **Forbidden** | Use `Load*FromBuffer` / flash; `NETKIT_MMAP=1` is forced off |

Default **on** for cpu + any MPU; opt out with `NETKIT_MMAP=0` on RTOS / bare-metal MPU. See [ARENA.md](ARENA.md) and [BUILD_TARGETS.md](BUILD_TARGETS.md).

## Host A/B suite (preliminary)

Fair CPU peer suite vs TF Lite / LiteRT (`benchmark/tools/run_host_ab_suite_{int8,float32}.py`):

**Models**

| key | What it is |
|-----|------------|
| `cnn` | **MNIST CNN** — digit classifier |
| `cnn_dw` | **MNIST DS-CNN** — depthwise-separable digit peer |
| `imagenet` | **MobileNetV4-Conv-Small** on ImageNet (10-class fixture) |

- Prebuild netkit binaries (untimed); discard first process per timed slot; order swaps (nk→TF, TF→nk)
- LiteRT-matched `-O3` flags; `NETKIT_IM2COL=0`
- **Latency** metric: MNIST CNN/DS-CNN `mean_us` (discard run 0 + image 0 each run); MobileNetV4-Small ImageNet `warm_mean_us` (discard full first image pass)
- **Flash / RAM**: MCU-style **runtime image only** — netkit bench ELF `__TEXT`/`__DATA` minus hard-coded test-image `.o` fixtures; TF Lite = core LiteRT CPU libs the same way. **Models and bench fixture images are excluded** (production would not embed those test vectors).
- Ratio column is always **TF ÷ netkit** (>1 ⇒ netkit faster / smaller)
- Modes: **XNNPACK ON** (both sides) and **XNNPACK OFF** (both reference). No MLP; no TF builtin-NEON-only peer.

```bash
python3 benchmark/tools/run_host_ab_suite_int8.py
python3 benchmark/tools/run_host_ab_suite_float32.py
```

Results: `benchmark/host_ab_suite_results_{int8,float32}.txt`, summary PDF `benchmark/host_ab_suite_results.pdf`. Suite infographics (tracked): [benchmark/linkedin/](../benchmark/linkedin/).

### Preliminary results (host Apple Silicon, Jul 2026)

Flash/RAM = **runtime image only** (`size` TEXT≈flash, DATA≈static RAM). Models and hard-coded bench fixture images excluded.

**Absolute runtime sizes (same LiteRT libs for all models):**

| mode | netkit flash | netkit RAM | TF Lite flash | TF Lite RAM |
|------|--------------|------------|---------------|-------------|
| XNNPACK ON | 1.31–1.32 MiB | 191.8 KiB | 12.41 MiB | 752.0 KiB |
| XNNPACK OFF | 193–200 KiB | 15.8 KiB | 12.41 MiB | 752.0 KiB |

#### INT8 — latency / flash / ram (TF÷netkit)

| model | XNNPACK | latency | flash | ram |
|-------|---------|---------|-------|-----|
| MNIST CNN | ON | 1.02× | 9.4× | 3.9× |
| MNIST DS-CNN | ON | 1.03× | 9.4× | 3.9× |
| MNv4-Small ImageNet | ON | 1.05× | 9.4× | 3.9× |
| MNIST CNN | OFF | 3.78× | 63.5× | 47.7× |
| MNIST DS-CNN | OFF | 2.17× | 63.5× | 47.7× |
| MNv4-Small ImageNet | OFF | 3.78× | 65.6× | 47.7× |

#### FLOAT32 — latency / flash / ram (TF÷netkit)

| model | XNNPACK | latency | flash | ram |
|-------|---------|---------|-------|-----|
| MNIST CNN | ON | 1.03× | 9.4× | 3.9× |
| MNIST DS-CNN | ON | 1.09× | 9.4× | 3.9× |
| MNv4-Small ImageNet | ON | 1.08× | 9.4× | 3.9× |
| MNIST CNN | OFF | 1.99× | 65.8× | 47.7× |
| MNIST DS-CNN | OFF | 1.63× | 65.8× | 47.7× |
| MNv4-Small ImageNet | OFF | 1.88× | 63.6× | 47.7× |

**Takeaways:** With XNNPACK ON, netkit is slightly ahead on every model (float and int8; TF÷nk ≈ 1.02–1.09×). With XNNPACK OFF (TF `BUILTIN_REF` vs netkit reference), netkit is clearly ahead. **Runtime flash/RAM favor netkit** — ~1.3 MiB TEXT (XNN) or ~194–200 KiB (reference) vs ~12.4 MiB LiteRT CPU libs. Absolute MobileNetV4-Small ImageNet warm means: float32 ~1.09 ms (netkit XNN) vs ~1.17 ms (TF); int8 ~0.68 ms vs ~0.71 ms.

### MPU — Raspberry Pi Zero 2 W (aarch64, Jul 2026)

Same fairness policy as the host suite (discard 1st process, order swaps, 1 thread). Cross-built on host (`tools/build_mpu_pi_aarch64.sh`), lean payload over SSH (`tools/run_mpu_pi_{float32,int8}_ab.sh`). Raw UART-style logs (local): `benchmark/host_ab_suite_results_{float32,int8}_pi_zero2w.txt`.

**FLOAT32** (order-avg latency; TF÷nk):

| model | XNNPACK | netkit | TF Lite | TF÷nk |
|-------|---------|--------|---------|-------|
| MNIST CNN | ON | 1.66 ms | 1.78 ms | 1.07× |
| MNIST DS-CNN | ON | 1.22 ms | 1.33 ms | 1.09× |
| MNv4-Small ImageNet | ON | 100.0 ms | 100.7 ms | 1.01× |
| MNIST CNN | OFF | 14.4 ms | 22.3 ms | 1.55× |
| MNIST DS-CNN | OFF | 7.7 ms | 12.1 ms | 1.57× |
| MNv4-Small ImageNet | OFF | 1056 ms | 1342 ms | 1.27× |

ImageNet float32 top-1: **9/10** both runtimes.

**INT8** (order-avg latency; TF÷nk):

| model | XNNPACK | netkit | TF Lite | TF÷nk |
|-------|---------|--------|---------|-------|
| MNIST CNN | ON | 1.09 ms | 1.11 ms | 1.02× |
| MNIST DS-CNN | ON | 0.61 ms | 0.62 ms | 1.01× |
| MNv4-Small ImageNet | ON | 70.1 ms | 70.0 ms | ~1.00× |
| MNIST CNN | OFF | 5.55 ms | 15.2 ms | 2.74× |
| MNIST DS-CNN | OFF | 5.29 ms | 13.7 ms | 2.59× |
| MNv4-Small ImageNet | OFF | 348 ms | 783 ms | 2.25× |

MNIST int8 top-1: **10/10**. ImageNet int8 XNNPACK: **8/10** both. ImageNet int8 **reference**: netkit **7/10**, TF Lite **8/10** — the extra netkit miss is a **retrain-only** issue (weights / quant calibration), **deferred** (not a runtime parity bug; XNNPACK path already matches).

### `NETKIT_IM2COL` note (from earlier host sweep)

With XNNPACK ON, im2col does not move the needle (accelerated path ignores it). With XNNPACK OFF, **`NETKIT_IM2COL=1` (partial)** can give a **small** float CNN reference bump on MPU/cpu; **`2` (full)** was not a clear win. **Default and recommendation: leave `NETKIT_IM2COL=0`.** At most try `1` on MCU or reference-only MPU builds.

### MCU (NUCLEO-F446RE)

UART A/B logs + tables: [`benchmark/mcu_ab_logs/`](../benchmark/mcu_ab_logs/) (10×10 methodology; discard first invoke). Matched toolchain: `mcu_tflm_toolchain.mk` (−O2 CORE/KERNEL/THIRD_PARTY, −flto, shared linker). **netkit** numbers below are **interpreter embed** (`NETKIT_EMBED=1`). Gain = TFLM÷netkit. Flash/RAM after MCU **no-heap** reclaim.

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

### YOLOX detection (host CPU, float32)

`benchmark/host_ab_suite_results_yolox_f32.txt` — YOLOX MNv4-PAFPN 320 warm mean:

| Mode | netkit | TF Lite | TF÷nk |
|------|--------|---------|------:|
| XNNPACK ON | 18.0 ms | 18.3 ms | 1.02× |
| Reference | 119.7 ms | 260.9 ms | 2.18× |

## What “done” means here

- **Arm MCU / MPU:** production-oriented paths exist (CMSIS-NN on MCU; XNNPACK on MPU/cpu) with float32 and int8 models, benches, and docs.
- **RISC MPU (`mpu_risc`):** **fully functional** for float32 + int8 via the same **XNNPACK** LayerFast stack as Arm MPU / cpu. XNNPACK has strong RISC-V coverage on MPU-class cores; CMSIS-NN is correctly unavailable.
- **RISC MCU (`mcu_risc`):** **fully functional** for float32 + int8 on **generic / reference kernels** only (CMSIS + XNNPACK forbidden). Those portable kernels are already fast and are the same fallbacks used when accelerators are off elsewhere. A dedicated RISC MCU NN library (analogous to Arm **CMSIS-NN**) is planned as a future acceleration layer; generics remain the production path until then.

## Open / next

- **YOLOX detection accuracy** — runtime and host latency path land; more training / calibration needed for mAP
- **Deeper float AOT specialization** (optional) — fused/specialized codegen beyond calling shared `Kernels` / composite `::forward` APIs; not required for correctness
- RISC-V MCU NN kernels (CMSIS-NN–class accelerator; optional — generic path stays the default until it lands)
- Broader int8 model coverage beyond MNIST + ImageNet MNv4 fixtures
- float16 / int16 / int4 (Phase 2)
- Voice modality fixtures
- **Deferred:** Pi ImageNet int8 reference top-1 gap (netkit 7/10 vs TF Lite 8/10) — retrain / recalibrate only
- **Deferred:** float32 MNIST CNN / DS-CNN on NUCLEO-F446RE — models exceed 512 KiB flash; use int8 on-device peers
