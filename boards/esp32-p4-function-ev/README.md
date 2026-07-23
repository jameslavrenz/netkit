# Seeed ESP32-P4-Function-EV — shared board support

Common ESP-IDF / PlatformIO pieces for ESP32-P4-Function-EV firmwares under
`boards/esp32-p4-function-ev-*`.

| Item | Value |
|------|--------|
| Chip | **ESP32-P4** · RISC-V · **360 MHz** (inference target) |
| Companion | ESP32-C6 on this kit is **WiFi-only** — not used for netkit/TFLM peers |
| netkit profile | `mcu_esp` + `NETKIT_ARCH=ESP32P4` + **ESP-NN** (portable opt) |
| Console | USB-UART bridge (CH34x), not USB-Serial-JTAG |
| Peer deploy | int8: **interpreter embed**; float32: **lowered AOT** (embed float broken on-device) |
| Arena | int8: **64 KiB** CNN / **96 KiB** DS-CNN; float32 lowered: ~173 KiB CNN (heap) |

**Target ≠ ISA:** this is Espressif RISC-V → still `mcu_esp`, not `mcu_risc`.
See [PLATFORMS.md — Target ≠ CPU ISA](../../docs/PLATFORMS.md#target--cpu-isa).

**ESP-NN note:** PlatformIO’s `riscv32-esp` gas (esp-14.2.0) rejects some P4 PIE
immediates (`esp.vldbc.8.ip …,1`). Peers use ESP-NN **portable / generic_opt**
kernels (same policy in `third_party/ESP-NN` and managed `espressif__esp-nn`).
Opt-in PIE later with `NETKIT_ESP_NN_USE_P4_ASM=1` when the toolchain accepts it.

## Firmwares

| Directory | Runtime | Model |
|-----------|---------|-------|
| [`../esp32-p4-function-ev-cnn/`](../esp32-p4-function-ev-cnn/) | netkit | MNIST CNN float32 (lowered AOT) |
| [`../esp32-p4-function-ev-cnn-dw/`](../esp32-p4-function-ev-cnn-dw/) | netkit | MNIST DS-CNN float32 (lowered AOT) |
| [`../esp32-p4-function-ev-tflm-cnn/`](../esp32-p4-function-ev-tflm-cnn/) | TFLM | MNIST CNN float32 |
| [`../esp32-p4-function-ev-tflm-cnn-dw/`](../esp32-p4-function-ev-tflm-cnn-dw/) | TFLM | MNIST DS-CNN float32 |
| [`../esp32-p4-function-ev-cnn-int8/`](../esp32-p4-function-ev-cnn-int8/README.md) | netkit | MNIST CNN int8 |
| [`../esp32-p4-function-ev-cnn-dw-int8/`](../esp32-p4-function-ev-cnn-dw-int8/README.md) | netkit | MNIST DS-CNN int8 |
| [`../esp32-p4-function-ev-tflm-cnn-int8/`](../esp32-p4-function-ev-tflm-cnn-int8/README.md) | TFLM (ESP-NN) | MNIST CNN int8 |
| [`../esp32-p4-function-ev-tflm-cnn-dw-int8/`](../esp32-p4-function-ev-tflm-cnn-dw-int8/README.md) | TFLM (ESP-NN) | MNIST DS-CNN int8 |

## Peer A/B (all rounds published)

Methodology: **10×10**, discard first invoke; order swaps `nk→tflm` / `tflm→nk`. All **10/10**.
All-rounds rollup: [`esp32_p4_ev_all_ab_results.txt`](../../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_all_ab_results.txt) ·
canonical: [STATUS.md](../../docs/STATUS.md#mcu-espressif-esp32-p4-function-ev).

### Round 1 — int8, ESP-NN ON (portable)

[`scripts/run_esp_int8_ab.sh`](scripts/run_esp_int8_ab.sh) · [`esp32_p4_ev_int8_ab_results.txt`](../../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_int8_ab_results.txt)

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **78.9 ms** | 79.3 ms | 1.00× |
| MNIST DS-CNN | **40.3 ms** | 41.1 ms | 1.02× |

### Round 2 — int8, ESP-NN OFF (reference)

[`scripts/run_esp_int8_ref_ab.sh`](scripts/run_esp_int8_ref_ab.sh) · [`esp32_p4_ev_int8_ref_ab_results.txt`](../../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_int8_ref_ab_results.txt) · `PIO_ENV=esp32_p4_ev_ref`

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **77.1 ms** | 485.4 ms | 6.29× |
| MNIST DS-CNN | **39.6 ms** | 172.0 ms | 4.34× |

### Round 3 — float32 (reference; ESP-NN has no float API)

P4 has an FPU. Runner: [`scripts/run_esp_float32_ab.sh`](scripts/run_esp_float32_ab.sh) ·
[`esp32_p4_ev_float32_ab_results.txt`](../../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_float32_ab_results.txt).

**Deploy note:** float peers use **lowered AOT** (not interpreter embed).

#### Known issue — float32 interpreter embed (P4 **and** S3)

| | |
|--|--|
| **What** | `--no-lower` float embed mispredicts (~**2/10**); many exact-zero logits; on P4 path can be ~2× too fast vs correct |
| **Also on** | [XIAO ESP32-S3](../xiao-esp32s3/README.md) — same failure class (reproduced 2026-07-23) |
| **Works** | Same embed **10/10 on host**; **lowered AOT** **10/10 on P4 and S3**; int8 embed OK on both |
| **Workaround** | Ship / bench float with lowered AOT (no `--no-lower`) |
| **Ruled out** | Arena size, 64 KiB stack, `-O3`, flash XIP (RAM blob copy), PSRAM + `esp_cache_msync` |
| **Next leads** | Float bind / `RepackConv2dWeights` vs lowered static weights; on-device weight CRCs; optional DRAM force-copy / repack-off |
| **Tracker** | [KNOWN_ISSUES.md KI-001](../../docs/KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device) |

| Model | netkit | TFLM | Gain (TFLM÷netkit) |
|-------|-------:|-----:|-------------------:|
| MNIST CNN | **97.5 ms** | 166.4 ms | 1.71× |
| MNIST DS-CNN | **74.8 ms** | 102.6 ms | 1.37× |

**Compiler match:** netkit C++ uses the same speed flags as `esp-tflite-micro`
(`-O3`, `-fno-rtti`, `-fno-exceptions`, `-fno-threadsafe-statics`,
`-fno-unwind-tables`) via [`mcu_esp_tflm_match_compile.cmake`](mcu_esp_tflm_match_compile.cmake).
ESP-NN C stays at IDF `-O2` on both sides. IDF global: `CONFIG_COMPILER_OPTIMIZATION_PERF`.

**ImageNet / MobileNetV4:** not run on this part — int8 weights alone are ~2.5–3.8 MiB,
above the default 1 MiB factory app partition.
