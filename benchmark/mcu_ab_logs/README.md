# MCU A/B logs

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
| **[xiao_esp32c3/esp32c3_int8_ab_results.txt](xiao_esp32c3/esp32c3_int8_ab_results.txt)** | Latency table + methodology |
| `xiao_esp32c3/*.log` | UART captures (order swaps) |

Methodology: 10×10; discard first invoke; `nk→tflm` / `tflm→nk` swaps. ImageNet skipped (flash).
Published summary: [docs/STATUS.md — MCU (Seeed XIAO ESP32C3)](../../docs/STATUS.md#mcu-seeed-xiao-esp32c3).

**Latency summary (ESP-NN, 10×10, all 10/10)**

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | 254.6 ms | **253.2 ms** | 0.99× |
| MNIST DS-CNN | 88.5 ms | **87.5 ms** | 0.99× |

Board: ESP32-C3 @ 160 MHz. Matched C++ flags with `esp-tflite-micro` (`-O3`). No FPU (soft-float); peer A/B is int8 only.
