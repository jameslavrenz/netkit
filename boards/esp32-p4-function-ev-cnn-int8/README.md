# ESP32-P4-Function-EV — MNIST CNN int8 (netkit)

| Item | Value |
|------|--------|
| Target | `mcu_esp` + `ESP32P4` + ESP-NN |
| Model | MNIST CNN int8, interpreter embed |
| Arena | 64 KiB (`NK_ARENA_DEFAULT_CAPACITY`) |
| Method | **10×10**, discard first invoke each run (same as NUCLEO MCU peers) |
| Published | ESP-NN **252.0 ms** vs TFLM **251.4 ms**; reference **226.8 ms** vs TFLM **1205.5 ms** (10/10) — [STATUS](../../docs/STATUS.md#mcu-seeed-esp32-p4-function-ev) |

```bash
make -C ../.. esp-nn-init
make -C ../../benchmark/tflm export-cnn-int8   # once
cd boards/esp32-p4-function-ev-cnn-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
# ESP-NN off (QuantOps reference):
PIO_ENV=esp32_p4_ev_ref make flash monitor
```

Peer vs TFLM: [`../esp32-p4-function-ev-tflm-cnn-int8/`](../esp32-p4-function-ev-tflm-cnn-int8/README.md).  
A/B runners: [`run_esp_int8_ab.sh`](../esp32-p4-function-ev/scripts/run_esp_int8_ab.sh) / [`run_esp_int8_ref_ab.sh`](../esp32-p4-function-ev/scripts/run_esp_int8_ref_ab.sh).
