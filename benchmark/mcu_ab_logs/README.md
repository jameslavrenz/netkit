# MCU A/B logs

On-device peer captures for NUCLEO-F446RE, XIAO ESP32C3, and ESP32-P4-Function-EV.
Published gallery: [README.md — Peer benchmarks](../../README.md#peer-benchmarks-mcu--mpu--cpu) ·
[STATUS.md](../../docs/STATUS.md) · open issues: [KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md).

## NUCLEO-F446RE

Canonical **netkit vs TFLM vs microTVM** int8 results for MNIST CNN and DS-CNN.

| Artifact | Contents |
|----------|----------|
| **[mcu_int8_ab_results.txt](mcu_int8_ab_results.txt)** | Latency + flash/RAM tables (source of truth) |
| `*_cmsis.log` / `*_ref.log` | UART captures per runtime / model / backend |
| [docs/STATUS.md](../../docs/STATUS.md#mcu-nucleo-f446re) | Published summary |

## Latency summary (10×10, discard first invoke; all 10/10)

**CMSIS-NN**

| Model | netkit embed | TFLM | microTVM AOT |
|-------|-------------:|-----:|-------------:|
| MNIST CNN | **95.3 ms** | 95.5 ms | 112.3 ms |
| MNIST DS-CNN | **58.3 ms** | 61.4 ms | 86.4 ms |

**Reference (CMSIS-NN off)**

| Model | netkit embed | TFLM | microTVM C AOT |
|-------|-------------:|-----:|---------------:|
| MNIST CNN | **336.2 ms** | 2593.5 ms | 343.0 ms |
| MNIST DS-CNN | **140.3 ms** | 826.8 ms | 236.0 ms |

Board: STM32F446RE @ 180 MHz. Matched toolchain (`mcu_tflm_toolchain.mk`). No XNNPACK on MCU.

## Seeed XIAO ESP32C3

Canonical **netkit vs TFLM** int8 (ESP-NN) for MNIST CNN and DS-CNN @ 160 MHz.

| Artifact | Contents |
|----------|----------|
| **[xiao_esp32c3/esp32c3_int8_ab_results.txt](xiao_esp32c3/esp32c3_int8_ab_results.txt)** | ESP-NN-on latency table |
| **[xiao_esp32c3/esp32c3_int8_ref_ab_results.txt](xiao_esp32c3/esp32c3_int8_ref_ab_results.txt)** | ESP-NN-off (reference) latency table |
| `xiao_esp32c3/*.log` / `xiao_esp32c3/ref_*.log` | UART captures (order swaps) |

Methodology: 10×10; discard first invoke; `nk→tflm` / `tflm→nk` swaps. ImageNet skipped (flash).
Published summary: [docs/STATUS.md — MCU (Seeed XIAO ESP32C3)](../../docs/STATUS.md#mcu-seeed-xiao-esp32c3).

**Latency summary (ESP-NN, interpreter embed, 10×10, all 10/10)**

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | 252.0 ms | **251.4 ms** | 1.00× |
| MNIST DS-CNN | 87.7 ms | **87.5 ms** | 1.00× |

Quant lowered AOT should be faster than embed but measured a hair slower on this board under ESP-NN — investigate before making lowered the peer default (see [STATUS](../../docs/STATUS.md#mcu-seeed-xiao-esp32c3)).

**Reference (ESP-NN off, 10×10, all 10/10)**

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **226.8 ms** | 1205.5 ms | 5.32× |
| MNIST DS-CNN | **85.8 ms** | 392.3 ms | 4.57× |

Board: ESP32-C3 @ 160 MHz. Matched C++ flags with `esp-tflite-micro` (`-O3`). No FPU (soft-float); peer A/B is int8 only.

## Seeed ESP32-P4-Function-EV

Canonical **netkit vs TFLM** for MNIST CNN and DS-CNN @ 360 MHz (FPU). Three published rounds:

| Artifact | Contents |
|----------|----------|
| **[esp32_p4_ev/esp32_p4_ev_all_ab_results.txt](esp32_p4_ev/esp32_p4_ev_all_ab_results.txt)** | All three rounds side-by-side |
| **[esp32_p4_ev/esp32_p4_ev_int8_ab_results.txt](esp32_p4_ev/esp32_p4_ev_int8_ab_results.txt)** | int8 · ESP-NN ON (portable) |
| **[esp32_p4_ev/esp32_p4_ev_int8_ref_ab_results.txt](esp32_p4_ev/esp32_p4_ev_int8_ref_ab_results.txt)** | int8 · ESP-NN OFF (reference) |
| **[esp32_p4_ev/esp32_p4_ev_float32_ab_results.txt](esp32_p4_ev/esp32_p4_ev_float32_ab_results.txt)** | float32 · reference (no ESP-NN float API) |
| `esp32_p4_ev/*.log` / `ref_*.log` / `f32_*.log` | UART captures (order swaps) |

Methodology: 10×10; discard first invoke; `nk→tflm` / `tflm→nk` swaps. ImageNet skipped (flash).
Published summary: [docs/STATUS.md — MCU (Espressif ESP32-P4-Function-EV)](../../docs/STATUS.md#mcu-espressif-esp32-p4-function-ev).

**Round 1 — int8 ESP-NN ON** (portable; P4 PIE asm off under PIO gas; all 10/10)

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **78.9 ms** | 79.3 ms | 1.00× |
| MNIST DS-CNN | **40.3 ms** | 41.1 ms | 1.02× |

**Round 2 — int8 ESP-NN OFF** (reference; all 10/10)

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **77.1 ms** | 485.4 ms | 6.29× |
| MNIST DS-CNN | **39.6 ms** | 172.0 ms | 4.34× |

**Round 3 — float32 reference** (ESP-NN N/A; netkit lowered AOT; all 10/10)

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **97.5 ms** | 166.4 ms | 1.71× |
| MNIST DS-CNN | **74.8 ms** | 102.6 ms | 1.37× |

Board: ESP32-P4 @ 360 MHz. Matched `-O3` C++. Companion ESP32-C6 on kit is WiFi-only.
