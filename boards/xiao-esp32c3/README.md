# Seeed XIAO ESP32C3 — shared board support

Common ESP-IDF / PlatformIO pieces for XIAO ESP32C3 firmwares under
`boards/xiao-esp32c3-*`.

| Item | Value |
|------|--------|
| Chip | ESP32-C3 · RISC-V · 160 MHz |
| netkit profile | `mcu_esp` + `NETKIT_ARCH=ESP32C3` + **ESP-NN** |
| Console | USB Serial/JTAG |

**Target ≠ ISA:** this is Espressif RISC-V → still `mcu_esp`, not `mcu_risc`.
See [PLATFORMS.md — Target ≠ CPU ISA](../../docs/PLATFORMS.md#target--cpu-isa).

## Firmwares

| Directory | Runtime | Model |
|-----------|---------|-------|
| [`../xiao-esp32c3-mlp-int8/`](../xiao-esp32c3-mlp-int8/README.md) | netkit | MNIST MLP int8 |
| [`../xiao-esp32c3-cnn-int8/`](../xiao-esp32c3-cnn-int8/README.md) | netkit | MNIST CNN int8 |
| [`../xiao-esp32c3-cnn-dw-int8/`](../xiao-esp32c3-cnn-dw-int8/README.md) | netkit | MNIST DS-CNN int8 |
| [`../xiao-esp32c3-tflm-cnn-int8/`](../xiao-esp32c3-tflm-cnn-int8/README.md) | TFLM (ESP-NN) | MNIST CNN int8 |
| [`../xiao-esp32c3-tflm-cnn-dw-int8/`](../xiao-esp32c3-tflm-cnn-dw-int8/README.md) | TFLM (ESP-NN) | MNIST DS-CNN int8 |

Peer A/B (order swaps + MCU 10×10 methodology):
[`scripts/run_esp_int8_ab.sh`](scripts/run_esp_int8_ab.sh).

**Compiler match:** netkit C++ uses the same speed flags as `esp-tflite-micro`
(`-O3`, `-fno-rtti`, `-fno-exceptions`, `-fno-threadsafe-statics`,
`-fno-unwind-tables`) via [`mcu_esp_tflm_match_compile.cmake`](mcu_esp_tflm_match_compile.cmake).
ESP-NN C stays at IDF `-O2` on both sides. IDF global: `CONFIG_COMPILER_OPTIMIZATION_PERF`.

**ImageNet / MobileNetV4:** not run on this part — int8 weights alone are ~2.5–3.8 MiB,
above the default 1 MiB factory app partition (and far above NUCLEO’s 512 KiB flash).
