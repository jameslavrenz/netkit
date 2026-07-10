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
| **cpu** | Desktop / CI / peer benches | XNNPACK (default) + optional CMSIS-DSP helpers; reference fallback | **Done** — float32 + int8 |
| **mcu_arm** | Arm Cortex-M firmware | CMSIS-NN + CMSIS-DSP; XNNPACK **forbidden** | **Done** — float32 + int8 (NUCLEO-F446RE) |
| **mpu_arm** | Arm Cortex-A / RTOS-class | XNNPACK (default) + CMSIS-DSP helpers; CMSIS-NN off | **Done** — float32 + int8 |
| **mpu_risc** | RISC-V MPU | XNNPACK (default); CMSIS-DSP/NN **forbidden** | **Mostly done** — same portable XNNPACK + generic path as other MPUs |
| **mcu_risc** | RISC-V MCU | Generic / reference kernels only; CMSIS + XNNPACK **forbidden** | **Works today** — no RISC-specific optimized kernels yet; **generic fallbacks are fast** and suitable until ISA-tuned kernels land |

**Policy reminder:** XNNPACK is default on cpu and all MPUs, never on MCU. CMSIS-NN is Arm MCU only. CMSIS-DSP is Arm ISA (and optional host cpu), never on RISC.

## Host file mmap

| Host OS | Status | Implementation |
|---------|--------|----------------|
| **macOS / Linux** | **Complete** | POSIX `mmap` (`MAP_PRIVATE`) |
| **Windows** | **Complete** | Win32 `CreateFileMapping` / `MapViewOfFile` (`FILE_MAP_COPY`) |
| **MCU** | **Forbidden** | Use `Load*FromBuffer` / flash; `NETKIT_MMAP=1` is forced off |

Default **on** for cpu + any MPU; opt out with `NETKIT_MMAP=0` on RTOS / bare-metal MPU. See [ARENA.md](ARENA.md) and [BUILD_TARGETS.md](BUILD_TARGETS.md).

## Recent peer benches (CPU host)

Methodology: LiteRT-matched `-O3` flags for fair TF Lite peers; order swaps (netkit↔TF Lite) to reduce cold-start bias. Primary ImageNet metric: `warm_mean_us`.

### ImageNet MobileNetV4-Conv-Small @224 (order-averaged)

| dtype | accel | netkit | TF Lite | speedup (TF÷netkit) |
|-------|--------|--------|---------|---------------------|
| float32 | XNNPACK ON (DSP+XNNPACK / XNNPACK) | ~1.11 ms | ~1.19 ms | ~1.07× |
| float32 | OFF (reference / builtin-ref) | ~32.9 ms | ~62.4 ms | ~1.90× |
| int8 | XNNPACK ON | ~0.71 ms | ~0.71 ms | ~1.00× (tie) |
| int8 | OFF (reference / builtin-ref) | ~8.0 ms | ~28.2 ms | ~3.5× |

Top-1 on the 10-image fixture: float 9/10 both; int8 ON 8/10 both.

### MNIST (host XNNPACK peers)

| Model | dtype | Result (order-averaged) |
|-------|--------|-------------------------|
| Tutorial CNN | float32 | ~tied (~31 µs class) |
| Tutorial CNN | int8 | TF Lite slightly ahead (~19 vs ~21 µs) |
| MLP | float32 | netkit ahead (~1.2–1.5 vs ~2.3 µs) |
| MLP | int8 | ~tied (~1.3 µs) |
| Depthwise-separable CNN | float32 | ~tied (~28.5 µs) |
| Depthwise-separable CNN | int8 | netkit ~1.05× (~20.9 vs ~22.0 µs) |

Reproduce: `make -C benchmark/netkit run-*-xnnpack` and `make -C benchmark/tflite run-*` (see [benchmark/README.md](../benchmark/README.md)).

### MCU (NUCLEO-F446RE)

| Board | Result |
|-------|--------|
| MNIST CNN int8 (CMSIS-NN) | 10/10 @ ~95 ms (10×10 methodology) |
| MNIST MLP int8 (CMSIS-NN) | 10/10 @ ~3.4 ms |
| XNNPACK in MCU ELF | **None** — `nm` shows no `xnn*` / XNNPACK symbols on nucleo CNN int8 firmware |

## What “done” means here

- **Arm MCU / MPU:** production-oriented paths exist (CMSIS-NN on MCU; XNNPACK on MPU/cpu) with float32 and int8 models, benches, and docs.
- **RISC MPU:** uses the same XNNPACK LayerFast stack as other MPUs; CMSIS is correctly unavailable.
- **RISC MCU:** builds and runs on **generic reference kernels** only. Those kernels are the portable fallback used everywhere else when accelerators are off — they are already competitive on CPU “OFF” peers. Dedicated RISC-V vector / DSP microkernels are **not** implemented yet.

## Open / next

- RISC-V MCU-optimized kernels (optional; generic path remains the default)
- Broader int8 model coverage beyond MNIST + ImageNet MNv4 fixtures
- float16 / int16 / int4 (Phase 2)
- Voice modality fixtures; Kalman estimation (Phase 3)
